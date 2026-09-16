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
Those paths resolve from the repo root, one rule with no special cases. A `holdout`
beside it names the user's own sealed holdout, resolved the same way.

Every entry carries a content hash of its bytes, and that hash is stored on every
result. Swap the file behind a name and the old numbers do not silently keep
counting: they are results for a different dataset that happens to share a folder.
"""

from __future__ import annotations

import hashlib
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from evals.config import DATASETS_DIR, REPO_ROOT

if TYPE_CHECKING:
    from iterate.adapters.data.tabular import TabularDataset

_DEFAULT_DATA_FILE = "data.csv"
# Enough of a sha256 to make a collision a non-issue while staying readable in a
# table cell. Same length the package uses for its own data fingerprints.
_HASH_CHARS = 16
# Only a vision dataset needs saying: a prompt dataset is known by its task line and
# everything else is tabular.
FAMILIES = ("", "tabular", "prompt", "vision")


@dataclass(frozen=True)
class Dataset:
    """One corpus entry."""

    name: str
    path: Path
    target: str
    metric: str
    source: str = ""
    notes: str = ""
    # Present only on a PROMPT dataset: the one-line job description a run is given.
    # Its presence is what tells the harness to sweep prompt techniques for this
    # dataset's ceiling rather than model families.
    task: str = ""
    # "vision" for a CSV of image paths; the ceiling is then a sweep of recipes.
    family: str = ""
    holdout: Path | None = None

    @property
    def is_prompt_task(self) -> bool:
        return bool(self.task.strip())

    @property
    def is_vision(self) -> bool:
        return self.family.strip() == "vision"

    @property
    def missing(self) -> list[Path]:
        return [p for p in (self.path, self.holdout) if p is not None and not p.is_file()]

    @property
    def available(self) -> bool:
        """False when a CSV is not on this machine.

        A missing dataset is a skip with a message, never a crash. Most of the
        corpus is gitignored, so a fresh clone legitimately has almost none of it.
        """
        return not self.missing

    def content_hash(self) -> str:
        """Fingerprint of the file's bytes.

        Hashes the raw file rather than a parsed frame: the question this answers is
        "is this the same file the old numbers were measured on", and parsing would
        paper over exactly the kind of change (an encoding fix, a re-export) that
        makes results incomparable.
        """
        # Without a holdout the key must stay the one every stored result was keyed by.
        digest = hashlib.sha256()
        with self.path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        if self.holdout is not None:
            digest.update(hashlib.sha256(self.holdout.read_bytes()).digest())
        if self.is_vision:
            # The CSV holds only paths, so an image swapped behind it would keep the key.
            digest.update(_image_digest(self.path, self.target).encode())
            if self.holdout is not None:
                digest.update(_image_digest(self.holdout, self.target, self.path).encode())
        return digest.hexdigest()[:_HASH_CHARS]


def load_data(dataset: Dataset, *, task: str | None = None) -> TabularDataset:
    """The split the CLI makes from the same files: the user's own when a holdout is named."""
    from iterate.adapters.data.tabular import load_csv, load_split

    if dataset.holdout is not None:
        return load_split(dataset.path, dataset.holdout, target=dataset.target, task=task)
    return load_csv(dataset.path, target=dataset.target, task=task)


def _image_digest(csv: Path, target: str, paths_from: Path | None = None) -> str:
    """Every image the CSV names, in row order, by its bytes; a missing one as missing.
    Relative paths resolve beside `paths_from`, the CSV `prepare_images` is given."""
    import pandas as pd

    from iterate.adapters.data.images import detect_image_column

    frame = pd.read_csv(csv)
    features = [c for c in frame.columns if c != target]
    column = detect_image_column(frame, features, paths_from or csv)
    if column is None:
        raise BadDatasetSpecError(f"{csv}: a vision dataset needs one column of image paths")
    digest = hashlib.sha256()
    for value in frame[column.column].astype(str):
        path = Path(value) if Path(value).is_absolute() else column.root / value
        try:
            digest.update(hashlib.sha256(path.read_bytes()).digest())
        except OSError:
            digest.update(b"missing")
    return digest.hexdigest()


class BadDatasetSpecError(Exception):
    """A dataset.toml is missing something the harness cannot invent."""


def _load_one(spec_path: Path) -> Dataset:
    with spec_path.open("rb") as handle:
        raw = tomllib.load(handle)

    missing = [key for key in ("target", "metric") if not raw.get(key)]
    if missing:
        raise BadDatasetSpecError(f"{spec_path}: missing {', '.join(missing)}")

    family = str(raw.get("family", "")).strip()
    if family not in FAMILIES:
        raise BadDatasetSpecError(
            f"{spec_path}: family {family!r} is not one of {', '.join(f for f in FAMILIES if f)}"
        )

    task = str(raw.get("task", "")).strip()
    if family == "prompt" and not task:
        raise BadDatasetSpecError(f"{spec_path}: family 'prompt' needs a task line")
    if task and family in ("tabular", "vision"):
        raise BadDatasetSpecError(
            f"{spec_path}: a task line makes this a prompt dataset, not {family}"
        )

    declared = str(raw.get("data", "")).strip()
    path = (REPO_ROOT / declared) if declared else (spec_path.parent / _DEFAULT_DATA_FILE)
    held = str(raw.get("holdout", "")).strip()

    return Dataset(
        name=spec_path.parent.name,
        path=path,
        target=str(raw["target"]),
        metric=str(raw["metric"]),
        source=str(raw.get("source", "")),
        notes=str(raw.get("notes", "")),
        task=str(raw.get("task", "")),
        family=family,
        holdout=(REPO_ROOT / held) if held else None,
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


__all__ = ["BadDatasetSpecError", "Dataset", "load", "load_data", "select"]
