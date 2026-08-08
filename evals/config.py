"""Sweep conditions and the version list, read from `config.toml`.

The split between this file and the command line is deliberate. Everything that
changes what a NUMBER MEANS lives here, in a tracked file: the model, the
iteration budget, the patience, how many repeats. Everything that only chooses
WHICH ROWS to fill lives in flags.

If the budget were a flag, one sweep would run v0.4 at ten iterations and the next
would run v0.5 at fifteen, the table would show v0.5 ahead, and the table would be
fiction. Conditions are therefore fingerprinted and stamped onto every recorded
cell, so results measured under different conditions can be told apart later
instead of being averaged into a lie.
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

EVALS_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVALS_DIR.parent
CONFIG_PATH = EVALS_DIR / "config.toml"
DATASETS_DIR = EVALS_DIR / "datasets"
WORK_DIR = EVALS_DIR / ".work"
STORE_PATH = EVALS_DIR / "results.db"
REPORT_PATH = EVALS_DIR / "RESULTS.md"

# The working tree, as opposed to a released version pulled from PyPI.
DEV_VERSION = "dev"


@dataclass(frozen=True)
class Conditions:
    """Everything held constant across a sweep."""

    model: str
    backend: str
    max_iterations: int
    patience: int
    repeats: int
    timeout_minutes: int
    sweep_threads: int

    def fingerprint(self) -> str:
        """A short hash of the conditions.

        Stamped on every cell. Two cells with different fingerprints measured the
        same thing under different rules and must never be compared, which the
        report enforces by grouping on this value rather than trusting the dates.
        """
        blob = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]

    def as_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


@dataclass(frozen=True)
class EvalConfig:
    versions: list[str]
    conditions: Conditions


_DEFAULT_CONDITIONS = Conditions(
    model="gemma4:12b",
    backend="ollama",
    max_iterations=10,
    patience=3,
    repeats=3,
    timeout_minutes=90,
    sweep_threads=4,
)


def load(path: Path | None = None) -> EvalConfig:
    """Read config.toml. Missing keys fall back to the defaults above, so adding a
    condition later does not invalidate an existing config file — it changes the
    fingerprint, which is exactly the signal that old cells are no longer
    comparable."""
    config_path = path or CONFIG_PATH
    raw: dict[str, object] = {}
    if config_path.exists():
        with config_path.open("rb") as handle:
            raw = tomllib.load(handle)

    conditions_raw = raw.get("conditions")
    conditions_map = conditions_raw if isinstance(conditions_raw, dict) else {}
    conditions = Conditions(
        model=str(conditions_map.get("model", _DEFAULT_CONDITIONS.model)),
        backend=str(conditions_map.get("backend", _DEFAULT_CONDITIONS.backend)),
        max_iterations=int(
            conditions_map.get("max_iterations", _DEFAULT_CONDITIONS.max_iterations)
        ),
        patience=int(conditions_map.get("patience", _DEFAULT_CONDITIONS.patience)),
        repeats=int(conditions_map.get("repeats", _DEFAULT_CONDITIONS.repeats)),
        timeout_minutes=int(
            conditions_map.get("timeout_minutes", _DEFAULT_CONDITIONS.timeout_minutes)
        ),
        sweep_threads=int(conditions_map.get("sweep_threads", _DEFAULT_CONDITIONS.sweep_threads)),
    )

    versions_raw = raw.get("versions")
    versions_map = versions_raw if isinstance(versions_raw, dict) else {}
    tracked = versions_map.get("track")
    versions = [str(v) for v in tracked] if isinstance(tracked, list) and tracked else [DEV_VERSION]

    return EvalConfig(versions=versions, conditions=conditions)


__all__ = [
    "CONFIG_PATH",
    "DATASETS_DIR",
    "DEV_VERSION",
    "EVALS_DIR",
    "REPORT_PATH",
    "REPO_ROOT",
    "STORE_PATH",
    "WORK_DIR",
    "Conditions",
    "EvalConfig",
    "load",
]
