"""The canonical data folder: a linked dataset laid out so a person can browse it.

    <out>/<name>/
        raw_files/        a copy of everything the user gave, never touched again
        train/            hard links into raw_files; one subfolder per class, or flat
        holdout/          the same, for the sealed holdout
        train.csv         image (relative to this folder), label
        holdout.csv
        link.json         the plan that produced it
    <out>/plans/<name>-<hash>.json
                          a plan a person said yes to, keyed by the folder's contents,
                          so the same data never goes through the Linker twice

The train and holdout folders name the class in their paths on purpose: they are
for people. The kernel never reads them; it gets the byte-named copies the image
adapter makes from the CSVs.
"""

from __future__ import annotations

import contextlib
import filecmp
import hashlib
import json
import math
import os
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from iterate.adapters.data.images import file_hashes
from iterate.adapters.data.linking import (
    LINK_VERSION,
    MAX_DEPTH,
    SKIP_DIRS,
    TABLE_SUFFIXES,
    LinkedFrames,
    LinkError,
    confine,
)
from iterate.adapters.data.tabular import DEFAULT_SEED, DEFAULT_TEST_SIZE, split_frame
from iterate.schemas.link import LinkPlan

if TYPE_CHECKING:
    from collections.abc import Mapping

RAW = "raw_files"
TRAIN = "train"
HOLDOUT = "holdout"
TRAIN_CSV = "train.csv"
HOLDOUT_CSV = "holdout.csv"
PLAN_JSON = "link.json"
PLANS = "plans"
ROLES = ("train", "holdout")


@dataclass(frozen=True)
class Workspace:
    root: Path
    train_csv: Path
    holdout_csv: Path
    plan_file: Path


@dataclass(frozen=True)
class Sides:
    """Both sides of the split as ``write`` lays them out, and the holdout images
    left out because their bytes sit in the training side."""

    frames: LinkedFrames
    dropped: list[str] = field(default_factory=list)
    dropped_labels: list[str] = field(default_factory=list)


def _ignore(directory: str, names: list[str]) -> set[str]:
    return {n for n in names if n.startswith(".") or n in SKIP_DIRS}


def _files_under(source: Path) -> list[Path]:
    """Every visible file to the inventory's depth, following symlinked folders once
    each, so a link back up the tree ends the walk instead of looping."""
    out: list[Path] = []
    seen = {os.path.realpath(source)}
    for directory, dirs, files in os.walk(source, followlinks=True):
        base = Path(directory)
        depth = len(base.relative_to(source).parts)
        kept = []
        for d in sorted(dirs):
            if d.startswith(".") or d in SKIP_DIRS or depth >= MAX_DEPTH:
                continue
            real = os.path.realpath(base / d)
            if real in seen:
                continue
            seen.add(real)
            kept.append(d)
        dirs[:] = kept
        out.extend(base / f for f in sorted(files) if not f.startswith("."))
    return out


def workspace_name(sources: list[Path], plan_: LinkPlan, frames: LinkedFrames) -> str:
    """Stable per source folder, plan and linked labels, so a re-run on the same data
    lands on the same workspace and an edited label file gets a new one."""
    digest = hashlib.sha256()
    for source in sources:
        for p in _files_under(source):
            digest.update(f"{p.relative_to(source).as_posix()}:{p.stat().st_size}".encode())
    digest.update(plan_.model_dump_json(exclude={"notes"}).encode())  # notes are commentary
    digest.update(frames.train.to_csv(index=False).encode())
    if frames.holdout is not None:
        digest.update(frames.holdout.to_csv(index=False).encode())
    return f"{sources[0].name}-{digest.hexdigest()[:8]}"


def _link_or_copy(source: Path, destination: Path) -> None:
    """A hard link, or a copy across filesystems. A file already in the slot must be
    this file; anything else is a stale workspace and is refused, never reused."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.samefile(source) or filecmp.cmp(source, destination, shallow=False):
            return
        raise LinkError(f"{destination} already holds a different file; delete the workspace")
    try:
        os.link(source, destination)
    except OSError:
        shutil.copyfile(source, destination)


def _safe_folder(label: object) -> str:
    """A class label as a folder name: no separators, no dot-prefix, nothing empty."""
    name = str(label).replace("/", "_").replace("\\", "_").strip()
    if name in ("", ".", ".."):
        name = "_"
    return "_" + name[1:] if name.startswith(".") else name


def _place(
    frame: pd.DataFrame, split_dir: Path, *, by_class: bool, raw_of: dict[str, Path], root: Path
) -> pd.DataFrame:
    """Lay one split out under its folder and return its CSV rows."""
    taken: set[str] = set()
    rows: list[dict[str, object]] = []
    for image, label in zip(frame["image"], frame["label"], strict=True):
        source = raw_of[str(image)]
        folder = split_dir / _safe_folder(label) if by_class else split_dir
        destination = folder / source.name
        n = 1
        while destination.as_posix().casefold() in taken:
            destination = folder / f"{source.stem}_{n}{source.suffix}"
            n += 1
        taken.add(destination.as_posix().casefold())
        _link_or_copy(source, destination)
        rows.append({"image": destination.relative_to(root).as_posix(), "label": label})
    return pd.DataFrame(rows, columns=["image", "label"])


def _raw_of(frames: LinkedFrames, sources: list[Path], targets: list[Path]) -> dict[str, Path]:
    """Where each linked image's copy lands, keyed by its original path."""
    out: dict[str, Path] = {}
    parts = [frames.train] + ([frames.holdout] if frames.holdout is not None else [])
    for part in parts:
        for image in part["image"]:
            original = Path(str(image))
            for source, target in zip(sources, targets, strict=True):
                if original.is_relative_to(source):
                    out[str(image)] = target / original.relative_to(source)
                    break
            else:
                raise LinkError(f"{original} is not under any of the given folders")
    return out


def _check_relations(sources: list[Path], out: Path) -> None:
    for source in sources:
        if out.is_relative_to(source) or source.is_relative_to(out):
            raise LinkError(
                f"the output folder {out} and the data folder {source} contain each other; "
                "pass --out somewhere else"
            )


def _check_class_floor(frame: pd.DataFrame, test_size: float) -> None:
    """The stratified split's own rules, said before any copy: two images per class,
    and enough images that each side of the split holds one per class."""
    counts = frame["label"].value_counts()
    thin = sorted(str(c) for c, n in counts.items() if n < 2)
    if thin:
        raise LinkError(
            f"classes with a single image cannot be split: {thin}; drop them, add images, or "
            "pass --train and --holdout"
        )
    n_holdout = math.ceil(test_size * len(frame))
    if min(n_holdout, len(frame) - n_holdout) < len(counts):
        raise LinkError(
            f"{len(counts)} classes need at least {len(counts)} images on each side of the "
            f"split and {len(frame)} images give {n_holdout} holdout; add images, about five "
            "per class, or pass --train and --holdout"
        )


def _paths(frames: LinkedFrames) -> list[str]:
    parts = [frames.train] + ([frames.holdout] if frames.holdout is not None else [])
    return list(dict.fromkeys(str(p) for part in parts for p in part["image"]))


def sides(
    plan_: LinkPlan,
    frames: LinkedFrames,
    *,
    hashes: Mapping[str, str | None] | None = None,
    seed: int = DEFAULT_SEED,
    test_size: float = DEFAULT_TEST_SIZE,
) -> Sides:
    """Both sides as ``write`` lays them out: the user's as given, ours made here.
    What is checked before the pause is what is written after it. With ``hashes``
    a holdout image whose bytes sit in training is left out of our split, since a
    model would score it from memory."""
    if frames.holdout is not None:
        return Sides(frames)
    if plan_.task == "classification":
        _check_class_floor(frames.train, test_size)
    try:
        split = split_frame(frames.train, "label", test_size=test_size, seed=seed, task=plan_.task)
    except ValueError as exc:
        raise LinkError(
            f"the 80/20 split cannot be made: {exc}; pass --train and --holdout"
        ) from exc
    train = split.train_features.assign(label=split.train_target.to_numpy())
    holdout = split.test_features.assign(label=split.test_target.to_numpy())
    if hashes is None:
        return Sides(LinkedFrames(train, holdout))
    seen = {d for p in train["image"] if (d := hashes.get(str(p))) is not None}
    twin = holdout["image"].map(lambda p: hashes.get(str(p)) in seen).to_numpy(dtype=bool)
    if twin.all():
        raise LinkError(
            "every holdout image is a byte copy of a training image, so the split leaves no "
            "holdout rows; add images that are not copies, or pass --train and --holdout"
        )
    return Sides(
        LinkedFrames(train, holdout.loc[~twin].reset_index(drop=True)),
        [str(p) for p in holdout.loc[twin, "image"]],
        [str(v) for v in holdout.loc[twin, "label"]],
    )


def write(
    plan_: LinkPlan,
    frames: LinkedFrames,
    *,
    sources: list[Path],
    out: Path,
    seed: int = DEFAULT_SEED,
    test_size: float = DEFAULT_TEST_SIZE,
    both: Sides | None = None,
) -> Workspace:
    """Write the canonical folder. Idempotent: an existing workspace is reused as is.
    ``both`` is the split the pause showed; without it the same split is made here."""
    sources = [s.resolve() for s in sources]
    for source in sources:
        confine(source)
    out = out.resolve()
    _check_relations(sources, out)
    if both is None:
        hashes = None if frames.holdout is not None else file_hashes(_paths(frames))
        both = sides(plan_, frames, hashes=hashes, seed=seed, test_size=test_size)
    train, holdout = both.frames.train, both.frames.holdout
    assert holdout is not None  # sides

    root = out / workspace_name(sources, plan_, both.frames)
    ws = Workspace(
        root=root,
        train_csv=root / TRAIN_CSV,
        holdout_csv=root / HOLDOUT_CSV,
        plan_file=root / PLAN_JSON,
    )
    if ws.train_csv.exists() and ws.holdout_csv.exists() and ws.plan_file.exists():
        return ws

    raw = root / RAW
    targets = [raw] if len(sources) == 1 else [raw / role for role in ROLES[: len(sources)]]
    raw_of = _raw_of(both.frames, sources, targets)
    for source, target in zip(sources, targets, strict=True):
        try:
            shutil.copytree(
                source, target, ignore=_ignore, copy_function=shutil.copyfile, dirs_exist_ok=True
            )
        except shutil.Error as exc:
            errors = exc.args[0]
            src, _, why = errors[0]
            raise LinkError(
                f"{source}: {len(errors)} file(s) could not be copied into {target}, e.g. "
                f"{src}: {why}"
            ) from exc

    by_class = plan_.task == "classification"
    train_rows = _place(train, root / TRAIN, by_class=by_class, raw_of=raw_of, root=root)
    holdout_rows = _place(holdout, root / HOLDOUT, by_class=by_class, raw_of=raw_of, root=root)
    train_rows.to_csv(ws.train_csv, index=False)
    holdout_rows.to_csv(ws.holdout_csv, index=False)
    ws.plan_file.write_text(
        json.dumps({"sources": [str(s) for s in sources], "plan": plan_.model_dump()}, indent=2),
        encoding="utf-8",
    )
    return ws


# ─── the plans a person said yes to ───────────────────────────────────────


def _feed(digest: hashlib._Hash, part: bytes) -> None:
    digest.update(f"{len(part)}:".encode())
    digest.update(part)


def plan_key(sources: list[Path]) -> str:
    """Stable per source folders as the link sees them: every file's path and size,
    every table's bytes, and the ladder's version, so an edited label file or a
    new rule is a different key. A file that cannot be read is part of the key,
    not a reason to stop."""
    digest = hashlib.sha256()
    _feed(digest, LINK_VERSION.encode())
    for source in (s.resolve() for s in sources):
        _feed(digest, str(source).encode())
        for p in _files_under(source):
            rel = p.relative_to(source).as_posix()
            try:
                _feed(digest, f"{rel}:{p.stat().st_size}".encode())
                if p.suffix.lower() in TABLE_SUFFIXES:
                    _feed(digest, p.read_bytes())
            except OSError:
                _feed(digest, f"{rel}:unreadable".encode())
    return f"{sources[0].name}-{digest.hexdigest()[:12]}"


def plan_path(sources: list[Path], *, out: Path) -> Path:
    return out.resolve() / PLANS / f"{plan_key(sources)}.json"


def remember_plan(
    plans: list[LinkPlan], *, sources: list[Path], out: Path, drop_twins: bool = False
) -> Path:
    """Keep the plans a person accepted, one per source folder, and whether they
    asked for the byte twins to come out of their holdout."""
    path = plan_path(sources, out=out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "sources": [str(s.resolve()) for s in sources],
                "plans": [p.model_dump() for p in plans],
                "drop_twins": drop_twins,
                "accepted": datetime.now(UTC).isoformat(timespec="seconds"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def recall_drop(sources: list[Path], *, out: Path) -> bool:
    """Whether the remembered yes for these folders came with a drop of the twins."""
    path = plan_path(sources, out=out)
    try:
        return bool(json.loads(path.read_text(encoding="utf-8")).get("drop_twins"))
    except (OSError, ValueError, AttributeError, RecursionError):
        return False


def recall_plan(sources: list[Path], *, out: Path) -> list[LinkPlan] | None:
    """The plans accepted for these folders as they are now, or None. Anything that
    does not read back as one plan per folder is treated as nothing remembered."""
    path = plan_path(sources, out=out)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        plans = [LinkPlan.model_validate(p) for p in data["plans"]]
    except (OSError, ValueError, KeyError, TypeError, RecursionError):
        return None
    return plans if len(plans) == len(sources) else None


def forget_plan(sources: list[Path], *, out: Path) -> None:
    with contextlib.suppress(FileNotFoundError):
        plan_path(sources, out=out).unlink()


__all__ = [
    "HOLDOUT_CSV",
    "PLAN_JSON",
    "TRAIN_CSV",
    "Sides",
    "Workspace",
    "forget_plan",
    "plan_key",
    "plan_path",
    "recall_drop",
    "recall_plan",
    "remember_plan",
    "sides",
    "workspace_name",
    "write",
]
