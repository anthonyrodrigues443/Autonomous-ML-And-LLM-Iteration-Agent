"""Persisted user config — the defaults a user picks once via ``iterate setup``.

Stored at ``$XDG_CONFIG_HOME/iterate/config.toml`` (default ``~/.config/iterate``)
and read by the CLI. Precedence everywhere is: **explicit flag > this file >
built-in default**, so a user keeps their choices without retyping flags but can
always override per run.

Two sets of model settings live here. The keys at the top are the harness, the model
that runs the loop. ``[providers.<name>]`` holds a key and a base URL for each provider
a prompt run may call as the model under test, and ``[prompt] allowed`` names the ones
it may call now. A provider saved but not allowed keeps its key in the file and is
never handed to a run.

This is separate from `config.py` `Settings`, which reads process/env config (and
a project-local ``.env``); this file is the user's personal cross-project defaults.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import stat
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

# The flat keys `load_user_config` returns; load ignores any other. The tables are read
# by `load_prompt_settings`.
PERSISTED_KEYS = frozenset(
    {"backend", "model", "base_url", "api_key", "e2b_api_key", "compute", "install"}
)
# The harness. Kept at the top level, where every version before v0.7 reads them; a
# [harness] table written by hand is read too, and wins.
HARNESS_KEYS = ("backend", "model", "base_url", "api_key")
_PROVIDER_KEYS = ("api_key", "base_url")
# Long enough that a draft this old belongs to no save still running.
_STALE_SECONDS = 60.0
# A provider's name is a bare TOML key in its table header.
_NAME = re.compile(r"[a-z0-9][a-z0-9_-]*")


class ConfigError(ValueError):
    """The saved file cannot be read as it stands. The message names the file and
    what is wrong in it."""


class _Keep:
    """Leave what the file holds."""


KEEP = _Keep()


@dataclass(frozen=True)
class SavedProvider:
    """One provider a prompt run may call: its key and, where it has no public
    endpoint of its own, its base URL."""

    name: str
    api_key: str | None = field(default=None, repr=False)
    base_url: str | None = None


@dataclass(frozen=True)
class PromptSettings:
    """The providers a prompt run may call. ``allowed`` is None when the user saved
    no list. ``providers`` holds the allowed ones only."""

    allowed: tuple[str, ...] | None = None
    providers: Mapping[str, SavedProvider] = field(default_factory=dict)
    # Saved and not allowed: named so the user can see them, never loaded.
    held_back: tuple[str, ...] = ()


def config_dir() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME")
    return (Path(base) if base else Path.home() / ".config") / "iterate"


def config_path() -> Path:
    return config_dir() / "config.toml"


def cache_dir() -> Path:
    """Everything iterate caches outside a project. A relative XDG_CACHE_HOME is
    ignored, as the XDG spec says. Imports nothing heavy."""
    base = os.environ.get("XDG_CACHE_HOME", "")
    root = Path(base) if base and Path(base).is_absolute() else Path.home() / ".cache"
    return root / "iterate"


def exists() -> bool:
    return config_path().exists()


def _read() -> dict[str, Any]:
    path = config_path()
    if not path.exists():
        return {}
    _owner_only(path)
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        raise ConfigError(f"{path} is not valid TOML: {exc}. Fix it, or remove it") from exc
    except OSError as exc:
        raise ConfigError(f"{path} cannot be opened ({type(exc).__name__})") from exc


def load_user_config() -> dict[str, Any]:
    """The saved defaults as a flat dict (only recognized keys), or ``{}`` if unset."""
    data = _read()
    flat = {k: v for k, v in data.items() if k in PERSISTED_KEYS and not isinstance(v, dict)}
    harness = data.get("harness")
    if isinstance(harness, dict):
        table = {k: v for k, v in harness.items() if k in HARNESS_KEYS and not isinstance(v, dict)}
        if table.get("backend", flat.get("backend")) != flat.get("backend"):
            # Another backend's: the key and the address at the top are not its.
            flat = {k: v for k, v in flat.items() if k not in HARNESS_KEYS}
        flat |= table
    return flat


def load_prompt_settings(allowed: Iterable[str] | None = None) -> PromptSettings:
    """The providers a prompt run may call. ``allowed`` is this run's own list and
    replaces the saved one. Keys of providers outside the list are not returned."""
    data = _read()
    saved = _tables(data.get("providers"))
    names = _names(allowed) if allowed is not None else _saved_list(data.get("prompt"))
    providers = {name: _provider(name, saved[name]) for name in (names or ()) if name in saved}
    held_back = tuple(sorted(n for n in saved if names is None or n not in names))
    return PromptSettings(allowed=names, providers=providers, held_back=held_back)


def load_saved_providers() -> dict[str, SavedProvider]:
    """Every saved provider, allowed or not. For `iterate setup`, which rewrites the
    file and must keep what it was not asked about. A run never calls this."""
    saved = _tables(_read().get("providers"))
    return {name: _provider(name, entry) for name, entry in saved.items()}


def _saved_list(prompt: Any) -> tuple[str, ...] | None:
    """The saved list, or None when the file holds none. A list that cannot be read is
    refused: read as no list, it would let every provider through."""
    if not isinstance(prompt, dict) or "allowed" not in prompt:
        return None
    listed = prompt["allowed"]
    if isinstance(listed, str):
        listed = [listed]
    if not isinstance(listed, list) or not all(isinstance(n, str) for n in listed):
        raise ConfigError(
            f"{config_path()}: allowed under [prompt] has to be a list of names, "
            f'such as ["openai", "groq"]'
        )
    return _names(listed)


def _tables(providers: Any) -> dict[str, dict[str, Any]]:
    """Each `[providers.<name>]` table under its name in lower case, as the list holds it."""
    if not isinstance(providers, dict):
        return {}
    tables: dict[str, dict[str, Any]] = {}
    for name, entry in providers.items():
        if not isinstance(entry, dict):
            raise ConfigError(
                f"{config_path()}: {name} under [providers] has to be a table, "
                f'[providers.{name}] with api_key = "..." under it'
            )
        tables[_name(str(name))] = entry
    return tables


def save_user_config(
    values: dict[str, Any],
    *,
    allowed: Iterable[str] | _Keep | None = KEEP,
    providers: Mapping[str, SavedProvider] | _Keep = KEEP,
) -> Path:
    """Write the recognized, non-empty values to the config file, readable by its
    owner only. ``allowed`` as None saves no list, so each run names its own provider.
    Either one left out keeps what the file holds."""
    listed = load_prompt_settings().allowed if isinstance(allowed, _Keep) else allowed
    entries = load_saved_providers() if isinstance(providers, _Keep) else providers
    kept = {k: v for k, v in values.items() if k in PERSISTED_KEYS and v not in (None, "")}
    # Every plain key comes before the first table, or it would belong to that table.
    lines = ["# the harness: the model that runs the loop"]
    lines += [_toml_line(k, kept[k]) for k in HARNESS_KEYS if k in kept]
    lines += ["", "# where generated code runs"]
    lines += [_toml_line(k, kept[k]) for k in sorted(kept) if k not in HARNESS_KEYS]
    if listed is not None:
        names = ", ".join(_quoted(n) for n in _names(listed))
        lines += ["", "# prompt runs: the only providers the model under test may come from"]
        lines += ["[prompt]", f"allowed = [{names}]"]
    for name in sorted(entries):
        entry = entries[name]
        body = [_toml_line(k, getattr(entry, k)) for k in _PROVIDER_KEYS if getattr(entry, k)]
        lines += ["", f"[providers.{_name(name)}]", *body]
    text = "\n".join(lines) + "\n"
    tomllib.loads(text)
    path = config_path()
    _write_owner_only(path, text)
    return path


def _provider(name: str, entry: dict[str, Any]) -> SavedProvider:
    return SavedProvider(
        name=name,
        api_key=str(entry["api_key"]) if entry.get("api_key") else None,
        base_url=str(entry["base_url"]) if entry.get("base_url") else None,
    )


def _name(name: str) -> str:
    cleaned = name.strip().lower()
    if not _NAME.fullmatch(cleaned):
        raise ConfigError(f"{name!r} is not a provider name: letters, digits, - and _ only")
    return cleaned


def _names(names: Iterable[str]) -> tuple[str, ...]:
    """Cleaned, in the order given, each once."""
    return tuple(dict.fromkeys(_name(n) for n in names if n and n.strip()))


def _write_owner_only(path: Path, text: str) -> None:
    """Replace the file in one step with a copy only its owner can read. The keys never
    sit on disk under a wider mode, as they would if the file were written and then
    narrowed. A config that is a link into a dotfiles folder stays a link."""
    path = Path(os.path.realpath(path))
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    draft = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    # A save that was killed half way left its draft behind, with every key in it.
    for stale in path.parent.glob(f".{path.name}.*.tmp"):
        ours = stale.name[len(path.name) + 2 : -4].isdigit()
        with contextlib.suppress(OSError):
            if ours and (stale == draft or time.time() - stale.stat().st_mtime > _STALE_SECONDS):
                stale.unlink()
    fd = os.open(draft, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(draft, path)
    except BaseException:
        with contextlib.suppress(OSError):
            draft.unlink()
        raise


def keep_beside(path: Path, word: str) -> Path | None:
    """A copy of the file as it is, beside it and readable by its owner only, under its
    name with ``word`` added. None when there is no file to copy."""
    if not path.is_file():
        return None
    copy = path.with_name(f"{path.name}.{word}")
    with contextlib.suppress(FileNotFoundError):
        copy.unlink()
    fd = os.open(copy, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(path.read_bytes())
    return copy


def _owner_only(path: Path) -> None:
    """Narrow a file written before v0.7, which anyone on the machine could read.
    Windows has no such mode bits, and a file that is not ours is left alone."""
    if os.name == "nt":
        return
    with contextlib.suppress(OSError):
        mode = stat.S_IMODE(path.stat().st_mode)
        if path.is_file() and mode & 0o077:
            path.chmod(mode & 0o700)


def _quoted(value: Any) -> str:
    """A TOML basic string. JSON's escapes are a subset of TOML's, and JSON leaves only
    DEL unescaped among the characters TOML refuses."""
    return json.dumps(str(value), ensure_ascii=False).replace("\x7f", "\\u007f")


def _toml_line(key: str, value: Any) -> str:
    if isinstance(value, bool):
        return f"{key} = {'true' if value else 'false'}"
    return f"{key} = {_quoted(value)}"


__all__ = [
    "HARNESS_KEYS",
    "KEEP",
    "PERSISTED_KEYS",
    "ConfigError",
    "PromptSettings",
    "SavedProvider",
    "cache_dir",
    "config_dir",
    "config_path",
    "exists",
    "keep_beside",
    "load_prompt_settings",
    "load_saved_providers",
    "load_user_config",
    "save_user_config",
]
