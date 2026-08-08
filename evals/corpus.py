"""The dataset corpus — a folder per dataset, discovered rather than hardcoded.

    evals/datasets/churn/
        dataset.toml   tracked: target column, metric, where the data came from
        data.csv       gitignored: the bytes stay on the machine that has them

Splitting it this way keeps the repo honest about what the corpus IS without
carrying 2.6MB of diamonds around. Anyone can read the registry, see exactly which
datasets a published table was measured on, and fetch them from the recorded
source. Adding a dataset is dropping a folder, with no shared file to edit.

A spec may point `data` somewhere else in the repo instead, which is how the
datasets already sitting in `examples/` join the corpus without being duplicated.
Those paths resolve from the repo root, one rule with no special cases.

Every entry carries a content hash of its bytes, and that hash is stored on every
result. Swap the file behind a name and the old numbers do not silently keep
counting: they are results for a different dataset that happens to share a folder.
"""

from __future__ import annotations

import hashlib
import tomllib
from dataclasses import dataclass
from typing import TYPE_CHECKING

from evals.config import DATASETS_DIR, REPO_ROOT

if TYPE_CHECKING:
    from pathlib import Path

_DEFAULT_DATA_FILE = "data.csv"
# Enough of a sha256 to make a collision a non-issue while staying readable in a
# table cell. Same length the package uses for its own data fingerprints.
_HASH_CHARS = 16


@dataclass(frozen=True)
class Dataset:
    """One corpus entry."""

    name: str
    path: Path
    target: str
    metric: str
    source: str = ""
    notes: str = ""

    @property
    def available(self) -> bool:
        """False when the CSV is not on this machine.

        A missing dataset is a skip with a message, never a crash. Most of the
        corpus is gitignored, so a fresh clone legitimately has almost none of it.
        """
        return self.path.is_file()

    def content_hash(self) -> str:
        """Fingerprint of the file's bytes.

        Hashes the raw file rather than a parsed frame: the question this answers is
        "is this the same file the old numbers were measured on", and parsing would
        paper over exactly the kind of change (an encoding fix, a re-export) that
        makes results incomparable.
        """
        digest = hashlib.sha256()
        with self.path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()[:_HASH_CHARS]


class BadDatasetSpecError(Exception):
    """A dataset.toml is missing something the harness cannot invent."""


def _load_one(spec_path: Path) -> Dataset:
    with spec_path.open("rb") as handle:
        raw = tomllib.load(handle)

    missing = [key for key in ("target", "metric") if not raw.get(key)]
    if missing:
        raise BadDatasetSpecError(f"{spec_path}: missing {', '.join(missing)}")

    declared = str(raw.get("data", "")).strip()
    path = (REPO_ROOT / declared) if declared else (spec_path.parent / _DEFAULT_DATA_FILE)

    return Dataset(
        name=spec_path.parent.name,
        path=path,
        target=str(raw["target"]),
        metric=str(raw["metric"]),
        source=str(raw.get("source", "")),
        notes=str(raw.get("notes", "")),
    )


def load(datasets_dir: Path | None = None) -> list[Dataset]:
    """Every dataset folder, in name order. Includes unavailable ones so the caller
    can report what is missing rather than pretending the corpus is smaller."""
    root = datasets_dir or DATASETS_DIR
    if not root.is_dir():
        return []
    return sorted(
        (_load_one(spec) for spec in root.glob("*/dataset.toml")),
        key=lambda dataset: dataset.name,
    )


def select(names: list[str] | None, datasets_dir: Path | None = None) -> list[Dataset]:
    """The named datasets, or all of them. Raises on a name that does not exist,
    rather than silently sweeping fewer datasets than the caller asked for."""
    everything = load(datasets_dir)
    if not names:
        return everything
    by_name = {dataset.name: dataset for dataset in everything}
    unknown = [name for name in names if name not in by_name]
    if unknown:
        raise BadDatasetSpecError(
            f"unknown dataset(s): {', '.join(unknown)}. known: {', '.join(by_name) or 'none'}"
        )
    return [by_name[name] for name in names]


__all__ = ["BadDatasetSpecError", "Dataset", "load", "select"]
