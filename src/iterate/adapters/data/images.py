"""Image datasets behind the tabular seam.

A vision dataset is a CSV whose one feature column holds image paths, or a folder
with one subfolder per class. Either way the split, the sealed holdout, scoring and
memory are the tabular ones, unchanged. This module recognises the path column,
reads a class-folder tree into the same shape, makes paths absolute so a kernel in
its own working directory can open them, fingerprints the image bytes so the data
version covers what the model actually sees, copies every image under a name
derived from its bytes so nothing the kernel sees can carry a class, and describes
the images for the supervisor. It never decodes pixels: headers only.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import median
from typing import TYPE_CHECKING

import pandas as pd

from iterate.adapters.data.tabular import (
    DEFAULT_SEED,
    DEFAULT_TEST_SIZE,
    TabularDataset,
    dataset_from_frames,
    looks_like_classification,
    split_frame,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

log = logging.getLogger(__name__)

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"})
IMAGE_COLUMN = "image"
LABEL_COLUMN = "label"
# Subfolder pairs that mean the folder carries its own split.
SPLIT_FOLDERS = (("train", "test"), ("train", "val"), ("train", "holdout"))
_DETECT_SAMPLE = 20
# One fixed timestamp on every copy: a write time is a side channel.
_FIXED_MTIME = 0


@dataclass(frozen=True)
class ImageColumn:
    """Which feature column holds the paths, and what relative paths resolve against."""

    column: str
    root: Path


def detect_image_column(
    frame: pd.DataFrame, features: Sequence[str], csv_path: Path
) -> ImageColumn | None:
    """The feature column holding image files, or None for a tabular frame.

    Strict: exactly one feature column, every non-null value ends in an image
    suffix, and a sample of them exists on disk.
    """
    if len(features) != 1:
        return None
    column = features[0]
    values = frame[column].dropna().astype(str)
    if values.empty or not values.map(lambda v: Path(v).suffix.lower() in IMAGE_SUFFIXES).all():
        return None
    root = csv_path.resolve().parent
    if not all(_locate(v, root).is_file() for v in values.head(_DETECT_SAMPLE)):
        return None
    return ImageColumn(column=column, root=root)


def _locate(value: str, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def resolve_paths(dataset: TabularDataset, column: ImageColumn) -> TabularDataset:
    """The same dataset with every image path absolute.

    The kernel runs in a temporary directory, so a relative path would resolve
    against the wrong place inside a session.
    """

    def absolute(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        out[column.column] = out[column.column].map(lambda v: str(_locate(str(v), column.root)))
        return out

    return replace(
        dataset,
        train_features=absolute(dataset.train_features),
        test_features=absolute(dataset.test_features),
    )


# ─── folders ──────────────────────────────────────────────────────────────


def split_folders(folder: Path) -> tuple[Path, Path] | None:
    """(train, holdout) subfolders when the folder carries its own split, else None."""
    names = {p.name.lower(): p for p in folder.iterdir() if p.is_dir()}
    for train, holdout in SPLIT_FOLDERS:
        if train in names and holdout in names:
            return names[train], names[holdout]
    return None


def frame_from_folder(folder: Path) -> pd.DataFrame:
    """One row per image under ``<folder>/<class>/<file>``; the label is the class folder."""
    rows: list[dict[str, str]] = []
    class_dirs = sorted(p for p in folder.iterdir() if p.is_dir() and not p.name.startswith("."))
    for class_dir in class_dirs:
        for file in sorted(class_dir.iterdir()):
            if (
                file.is_file()
                and not file.name.startswith(".")
                and file.suffix.lower() in IMAGE_SUFFIXES
            ):
                rows.append({IMAGE_COLUMN: str(file.resolve()), LABEL_COLUMN: class_dir.name})
    if not rows:
        raise ValueError(f"{folder}: no images found under class folders")
    return pd.DataFrame(rows)


def load_image_folder(
    folder: str | Path, *, test_size: float = DEFAULT_TEST_SIZE, seed: int = DEFAULT_SEED
) -> TabularDataset:
    """A class-folder tree as a dataset: split here, unless it carries its own split.

    With ``train/`` and ``test/`` (or ``val/``, ``holdout/``) inside, those two are
    the split and any other subfolder is ignored, with a warning.
    """
    root = Path(folder)
    if (pair := split_folders(root)) is not None:
        ignored = sorted(
            p.name
            for p in root.iterdir()
            if p.is_dir() and not p.name.startswith(".") and p not in pair
        )
        if ignored:
            log.warning(
                "%s: using %s and %s; ignoring %s", root, pair[0].name, pair[1].name, ignored
            )
        return dataset_from_frames(
            frame_from_folder(pair[0]), frame_from_folder(pair[1]), LABEL_COLUMN, seed=seed
        )
    return split_frame(frame_from_folder(root), LABEL_COLUMN, test_size=test_size, seed=seed)


def load_image_split(
    train_folder: str | Path, holdout_folder: str | Path, *, seed: int = DEFAULT_SEED
) -> TabularDataset:
    """Two class-folder trees the user split themselves."""
    return dataset_from_frames(
        frame_from_folder(Path(train_folder)),
        frame_from_folder(Path(holdout_folder)),
        LABEL_COLUMN,
        seed=seed,
    )


# ─── bytes ────────────────────────────────────────────────────────────────


def file_hashes(paths: Iterable[str]) -> dict[str, str | None]:
    """sha256 of every file's bytes, keyed by path; None where the file is missing."""
    out: dict[str, str | None] = {}
    for p in paths:
        try:
            out[p] = hashlib.sha256(Path(p).read_bytes()).hexdigest()
        except OSError:
            out[p] = None
    return out


def _all_paths(dataset: TabularDataset, column: ImageColumn) -> list[str]:
    return [
        str(p)
        for p in (*dataset.train_features[column.column], *dataset.test_features[column.column])
    ]


def content_hash(
    dataset: TabularDataset, column: ImageColumn, hashes: dict[str, str | None]
) -> str:
    """A data version covering the CSV and the image bytes behind it, in frame order."""
    digest = hashlib.sha256(dataset.data_hash.encode())
    for p in _all_paths(dataset, column):
        digest.update((hashes.get(p) or "missing").encode())
    return digest.hexdigest()[:16]


def materialise(
    dataset: TabularDataset,
    column: ImageColumn,
    hashes: dict[str, str | None],
    into: Path,
) -> tuple[TabularDataset, Path]:
    """Copy every image into ``into/<version>/`` under a name derived from its bytes.

    The kernel only ever sees these paths. The name is the file's own sha256, which
    carries nothing the bytes do not; there is no suffix, since a format could be a
    label; every copy gets one fixed mtime; a missing file maps to an opaque path
    that does not exist. Copies are written to a temporary name and renamed, so an
    interrupted copy never sits in a slot.
    """
    version = content_hash(dataset, column, hashes)
    target_dir = into / version
    target_dir.mkdir(parents=True, exist_ok=True)
    mapping: dict[str, str] = {}
    for source in dict.fromkeys(_all_paths(dataset, column)):
        digest = hashes.get(source)
        if digest is None:
            mapping[source] = str(
                target_dir / f"missing-{hashlib.sha256(source.encode()).hexdigest()[:16]}"
            )
            continue
        destination = target_dir / digest[:16]
        if not destination.exists():
            partial = target_dir / f".{digest[:16]}.part"
            shutil.copyfile(source, partial)
            os.utime(partial, (_FIXED_MTIME, _FIXED_MTIME))
            os.replace(partial, destination)
        mapping[source] = str(destination)

    def rewritten(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        out[column.column] = out[column.column].map(lambda v: mapping[str(v)])
        return out

    return (
        replace(
            dataset,
            train_features=rewritten(dataset.train_features),
            test_features=rewritten(dataset.test_features),
            data_hash=version,
        ),
        target_dir,
    )


# ─── profile ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ImageProfile:
    """Host-computed facts about the images, for the supervisor and the coder."""

    n_train: int
    n_test: int
    column: str
    classes: int
    class_balance: tuple[tuple[str, float], ...]
    target_spread: tuple[float, float, float, float] | None
    widths: tuple[int, int, int]
    heights: tuple[int, int, int]
    portrait_share: float
    modes: tuple[tuple[str, float], ...]
    formats: tuple[tuple[str, float], ...]
    unreadable: int
    shared_across_split: int

    def render(self) -> str:
        w, h = self.widths, self.heights
        if self.target_spread is not None:
            mean, std, low, high = self.target_spread
            target = (
                "Target: a number, so this is regression. "
                f"Spread: mean={mean:.4g}, std={std:.4g}, min={low:.4g}, max={high:.4g}."
            )
        else:
            target = (
                f"Classes: {self.classes}. Class balance: "
                + ", ".join(f"{c!r}: {p:.0%}" for c, p in self.class_balance[:6])
                + "."
            )
        lines = [
            f"Rows: {self.n_train} train / {self.n_test} test (sealed holdout). "
            "Every row is one image.",
            f"Image column: {self.column!r} (absolute paths). {target}",
            f"Width min/median/max: {w[0]}/{w[1]}/{w[2]}; height {h[0]}/{h[1]}/{h[2]}; "
            f"{self.portrait_share:.0%} portrait.",
            "Modes: "
            + ", ".join(f"{m} {p:.0%}" for m, p in self.modes)
            + ". Formats: "
            + ", ".join(f"{f} {p:.0%}" for f, p in self.formats)
            + ".",
            f"Unreadable files: {self.unreadable}. Byte-identical images in both splits: "
            f"{self.shared_across_split}.",
        ]
        return "\n".join(lines)


def profile_images(
    dataset: TabularDataset, column: ImageColumn, hashes: dict[str, str | None]
) -> ImageProfile:
    """Describe the TRAINING images from file headers only.

    The holdout contributes nothing but its hashes, for the count of images that
    sit byte-identical in both splits, which the split itself cannot see.
    """
    from PIL import Image, UnidentifiedImageError

    train_paths = [str(p) for p in dataset.train_features[column.column]]
    widths: list[int] = []
    heights: list[int] = []
    modes: Counter[str] = Counter()
    formats: Counter[str] = Counter()
    unreadable = sum(1 for p in train_paths if hashes.get(p) is None)
    for p in train_paths:
        if hashes.get(p) is None:
            continue
        try:
            with Image.open(p) as im:
                widths.append(im.width)
                heights.append(im.height)
                modes[im.mode] += 1
                formats[im.format or "unknown"] += 1
        except (UnidentifiedImageError, OSError):
            unreadable += 1

    seen = len(widths)
    train_digests = {hashes[p] for p in train_paths if hashes.get(p)}
    test_digests = {
        hashes[str(p)] for p in dataset.test_features[column.column] if hashes.get(str(p))
    }
    classification = looks_like_classification(dataset.train_target)
    counts = (
        dataset.train_target.astype(str).value_counts(normalize=True)
        if classification
        else dataset.train_target.iloc[0:0]
    )
    spread = None
    if not classification:
        target = dataset.train_target.astype(float)
        spread = (
            float(target.mean()),
            float(target.std()),
            float(target.min()),
            float(target.max()),
        )

    def extent(values: list[int]) -> tuple[int, int, int]:
        return (min(values), int(median(values)), max(values)) if values else (0, 0, 0)

    def shares(counter: Counter[str]) -> tuple[tuple[str, float], ...]:
        return tuple((k, n / seen) for k, n in counter.most_common()) if seen else ()

    portrait = sum(1 for w, h in zip(widths, heights, strict=True) if h > w)
    return ImageProfile(
        n_train=dataset.n_train,
        n_test=dataset.n_test,
        column=column.column,
        classes=int(dataset.train_target.nunique()) if classification else 0,
        class_balance=tuple((str(k), float(v)) for k, v in counts.items()),
        target_spread=spread,
        widths=extent(widths),
        heights=extent(heights),
        portrait_share=(portrait / seen) if seen else 0.0,
        modes=shares(modes),
        formats=shares(formats),
        unreadable=unreadable,
        shared_across_split=len(train_digests & test_digests),
    )


__all__ = [
    "IMAGE_COLUMN",
    "IMAGE_SUFFIXES",
    "LABEL_COLUMN",
    "SPLIT_FOLDERS",
    "ImageColumn",
    "ImageProfile",
    "content_hash",
    "detect_image_column",
    "file_hashes",
    "frame_from_folder",
    "load_image_folder",
    "load_image_split",
    "materialise",
    "profile_images",
    "resolve_paths",
    "split_folders",
]
