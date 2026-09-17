"""What a local kernel may open: its own folder, the files its run gave it, the
interpreter it runs on, and iterate's weights folder. macOS enforces it with
sandbox-exec; no other platform enforces anything yet.
"""

from __future__ import annotations

import functools
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from iterate import userconfig

SANDBOX_EXEC = "/usr/bin/sandbox-exec"
_SYSTEM = (
    "/System",
    "/usr/lib",
    "/usr/share",
    "/usr/bin",
    "/usr/libexec",
    "/bin",
    "/sbin",
    "/Library/Apple",
    "/Library/Preferences",
    "/private/etc",
    "/private/var/db",
    "/dev",
)
# Package-manager library trees: libomp for lightgbm and xgboost, openssl and sqlite
# for a Homebrew Python. Their var/ and etc/ hold user data and stay closed.
_LIBRARIES = (
    "/opt/homebrew/opt",
    "/opt/homebrew/Cellar",
    "/opt/homebrew/lib",
    "/opt/homebrew/Frameworks",
    "/usr/local/opt",
    "/usr/local/Cellar",
    "/usr/local/lib",
    "/opt/local/lib",
)
_SQLITE_SIDES = ("", "-journal", "-wal", "-shm")
_INSTALLER = re.compile(
    r"pip[\d.]*|uvx?|conda|mamba|micromamba|poetry|pipx|pipenv|pdm|rye|pixi|hatch"
)
# sandbox_apply refuses any profile with a deny rule inside an outer sandbox, while
# a bare (allow default) still applies there, so the probe must carry a deny.
_PROBE = '(version 1)(allow default)(deny file-read-data (literal "/private/var/empty/iterate"))'
# Inherited overrides would send torch or Hugging Face past the weights folder.
WEIGHTS_OVERRIDES = (
    "HF_HUB_CACHE",
    "HUGGINGFACE_HUB_CACHE",
    "TRANSFORMERS_CACHE",
    "HF_DATASETS_CACHE",
)


@dataclass(frozen=True)
class Confinement:
    """What a cell may open besides its own folder.

    ``reads`` are opened read-only. ``weights`` is read-write, and torch and Hugging
    Face download into it. ``files`` are sqlite files opened read-write with their
    sidecars. No interpreter path that holds a ``protected`` path is opened."""

    reads: tuple[Path, ...] = ()
    weights: Path | None = None
    files: tuple[Path, ...] = ()
    protected: tuple[Path, ...] = ()


@functools.cache
def sandbox_available() -> bool:
    if sys.platform != "darwin" or not Path(SANDBOX_EXEC).exists():
        return False
    try:
        probe = subprocess.run(
            [SANDBOX_EXEC, "-p", _PROBE, "/usr/bin/true"],
            capture_output=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return probe.returncode == 0


def weights_dir() -> Path:
    return userconfig.cache_dir() / "weights"


def weights_env(folder: Path) -> dict[str, str]:
    # matplotlib and fontconfig cache fonts under HOME by default. A cache a cell cannot
    # write is rebuilt every session, and fontconfig prints an error per font file.
    return {
        "TORCH_HOME": str(folder / "torch"),
        "HF_HOME": str(folder / "huggingface"),
        "MPLCONFIGDIR": str(folder / "matplotlib"),
        "XDG_CACHE_HOME": str(folder / "cache"),
    }


def installer_paths() -> frozenset[str]:
    """Installer programs by name, in every PATH folder and beside the interpreter. A
    name check: it does not stop the interpreter's own pip."""
    folders = (
        *os.environ.get("PATH", "").split(os.pathsep),
        os.path.join(sys.prefix, "bin"),
        os.path.join(sys.base_prefix, "bin"),
    )
    interpreter = _real(sys.executable)
    found: set[str] = set()
    for folder in dict.fromkeys(f for f in folders if f):
        try:
            with os.scandir(folder) as entries:
                named = [e.path for e in entries if _INSTALLER.fullmatch(e.name)]
        except OSError:
            continue
        found.update(
            real
            for path in named
            if os.path.isfile(path)
            and os.access(path, os.X_OK)
            and (real := _real(path)) != interpreter
        )
    return frozenset(found)


def _real(path: str | Path) -> str:
    return os.path.realpath(path)


def _holds(outer: str, inner: str) -> bool:
    return inner == outer or inner.startswith(outer.rstrip("/") + "/")


# The interpreter's TLS trust store can sit outside every library tree (Homebrew's
# etc/), and without it no HTTPS download verifies.
_CHILD = (
    "import json, os, site, ssl, sys; v = ssl.get_default_verify_paths(); "
    "print(json.dumps([*sys.path, site.ENABLE_USER_SITE and site.getusersitepackages(), "
    "v.cafile, v.capath, v.openssl_capath, os.path.dirname(v.openssl_cafile)]))"
)


@functools.cache
def _import_path(pythonpath: str | None) -> tuple[str, ...]:
    done = subprocess.run(
        [sys.executable, "-c", _CHILD], capture_output=True, text=True, cwd="/", check=True
    )
    return tuple(p for p in json.loads(done.stdout) if p)


def interpreter_reads(protected: tuple[Path, ...] = ()) -> list[str]:
    """Where the kernel's interpreter imports and verifies TLS from, asked of that
    interpreter, minus any entry that holds the home folder or a protected path."""
    import iterate

    shielded = [_real(Path.home()), *(_real(p) for p in protected)]
    entries = [
        *_import_path(os.environ.get("PYTHONPATH")),
        sys.prefix,
        sys.base_prefix,
        str(Path(iterate.__file__).parent),
    ]
    kept: list[str] = []
    for entry in entries:
        real = _real(entry)
        if real not in kept and not any(_holds(real, s) for s in shielded):
            kept.append(real)
    return kept


def _darwin_dir(name: str) -> str:
    return subprocess.run(
        ["getconf", name], capture_output=True, text=True, check=False
    ).stdout.strip()


def _metal() -> tuple[str, ...]:
    if sys.platform != "darwin":
        return ()
    found = (
        (_darwin_dir("DARWIN_USER_TEMP_DIR"), "com.apple.MetalPerformanceShadersGraph"),
        (_darwin_dir("DARWIN_USER_CACHE_DIR"), "com.apple.metal"),
    )
    return tuple(str(Path(base, name)) for base, name in found if base)


def _q(path: str | Path) -> str:
    text = _real(path)
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _subpaths(paths: tuple[str | Path, ...]) -> str:
    return " ".join(f"(subpath {_q(p)})" for p in paths)


def profile(workdir: Path, scratch: Path, confinement: Confinement) -> str:
    weights = (confinement.weights,) if confinement.weights is not None else ()
    writable = (workdir, scratch, *weights, *_metal())
    readable = (
        *_SYSTEM,
        *_LIBRARIES,
        *interpreter_reads(confinement.protected),
        *confinement.reads,
        *writable,
    )
    literals = " ".join(
        f"(literal {_q(_real(p) + side)})" for p in confinement.files for side in _SQLITE_SIDES
    )
    programs = " ".join(f"(literal {_q(p)})" for p in sorted(installer_paths()))
    rules = [
        "(version 1)",
        "(allow default)",
        "(deny file-read-data)",
        f'(allow file-read-data (literal "/") {_subpaths(readable)} {literals})',
        "(deny file-write*)",
        f'(allow file-write* (literal "/dev/null") {_subpaths(writable)} {literals})',
        "(deny appleevent-send)",
        f"(deny process-exec {_subpaths(writable)} {programs})",
    ]
    return "\n".join(rules) + "\n"


__all__ = [
    "SANDBOX_EXEC",
    "WEIGHTS_OVERRIDES",
    "Confinement",
    "installer_paths",
    "interpreter_reads",
    "profile",
    "sandbox_available",
    "weights_dir",
    "weights_env",
]
