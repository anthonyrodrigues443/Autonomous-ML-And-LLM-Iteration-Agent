"""Persisted user config — the defaults a user picks once via ``iterate setup``.

Stored at ``$XDG_CONFIG_HOME/iterate/config.toml`` (default ``~/.config/iterate``)
and read by the CLI. Precedence everywhere is: **explicit flag > this file >
built-in default**, so a user keeps their choices without retyping flags but can
always override per run.

Only the small set the setup wizard asks about lives here (backend, model, API
keys, compute venue, local-install consent); everything else stays a per-run flag.
This is separate from `config.py` `Settings`, which reads process/env config (and
a project-local ``.env``); this file is the user's personal cross-project defaults.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

# The only keys persisted. The wizard never writes anything else, and load ignores
# unknown keys, so the file stays a small, predictable surface.
PERSISTED_KEYS = frozenset({"backend", "model", "api_key", "e2b_api_key", "compute", "install"})


def config_dir() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME")
    return (Path(base) if base else Path.home() / ".config") / "iterate"


def config_path() -> Path:
    return config_dir() / "config.toml"


def exists() -> bool:
    return config_path().exists()


def load_user_config() -> dict[str, Any]:
    """The saved defaults as a dict (only recognized keys), or ``{}`` if unset."""
    path = config_path()
    if not path.exists():
        return {}
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    return {key: value for key, value in data.items() if key in PERSISTED_KEYS}


def save_user_config(values: dict[str, Any]) -> Path:
    """Write the recognized, non-empty values to the config file. Returns its path."""
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        _toml_line(key, values[key])
        for key in sorted(values)
        if key in PERSISTED_KEYS and values[key] not in (None, "")
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _toml_line(key: str, value: Any) -> str:
    if isinstance(value, bool):
        return f"{key} = {'true' if value else 'false'}"
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'{key} = "{escaped}"'


__all__ = [
    "PERSISTED_KEYS",
    "config_dir",
    "config_path",
    "exists",
    "load_user_config",
    "save_user_config",
]
