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
import os
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from iterate.adapters.data.linking import SKIP_DIRS, TABLE_SUFFIXES, LinkedFrames, LinkError
from iterate.adapters.data.tabular import DEFAULT_SEED, DEFAULT_TEST_SIZE, split_frame
from iterate.schemas.link import LinkPlan

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


def _ignore(directory: str, names: list[str]) -> set[str]:
    return {n for n in names if n.startswith(".") or n in SKIP_DIRS}


def _files_under(source: Path) -> list[Path]:
    """Every visible file, following symlinked folders the way the inventory does."""
    out: list[Path] = []
    for directory, dirs, files in os.walk(source, followlinks=True):
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d not in SKIP_DIRS)
        base = Path(directory)
        out.extend(base / f for f in sorted(files) if not f.startswith("."))
    return out


def workspace_name(sources: list[Path], plan_: LinkPlan, frames: LinkedFrames) -> str:
    """Stable per source folder, plan and linked labels, so a re-run on the same data
    lands on the same workspace and an edited label file gets a new one."""
    digest = hashlib.sha256()
    for source in sources:
        for p in _files_under(source):
            digest.update(f"{p.relative_to(source).as_posix()}:{p.stat().st_size}".encode())
    digest.update(plan_.model_dump_json().encode())
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


def _check_class_floor(frame: pd.DataFrame) -> None:
    counts = frame["label"].value_counts()
    thin = sorted(str(c) for c, n in counts.items() if n < 2)
    if thin:
        raise LinkError(
            f"classes with a single image cannot be split: {thin}; drop them, add images, or "
            "pass --train and --holdout"
        )


def write(
    plan_: LinkPlan,
    frames: LinkedFrames,
    *,
    sources: list[Path],
    out: Path,
    seed: int = DEFAULT_SEED,
    test_size: float = DEFAULT_TEST_SIZE,
) -> Workspace:
    """Write the canonical folder. Idempotent: an existing workspace is reused as is."""
    sources = [s.resolve() for s in sources]
    out = out.resolve()
    _check_relations(sources, out)
    if frames.holdout is None and plan_.task == "classification":
        _check_class_floor(frames.train)

    root = out / workspace_name(sources, plan_, frames)
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
    raw_of = _raw_of(frames, sources, targets)
    for source, target in zip(sources, targets, strict=True):
        shutil.copytree(
            source, target, ignore=_ignore, copy_function=shutil.copyfile, dirs_exist_ok=True
        )

    train, holdout = frames.train, frames.holdout
    if holdout is None:
        split = split_frame(train, "label", test_size=test_size, seed=seed, task=plan_.task)
        train = split.train_features.assign(label=split.train_target.to_numpy())
        holdout = split.test_features.assign(label=split.test_target.to_numpy())

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


def plan_key(sources: list[Path]) -> str:
    """Stable per source folders as the link sees them: every file's path and size,
    and every table's bytes, so an edited label file is a different key."""
    digest = hashlib.sha256()
    for source in (s.resolve() for s in sources):
        digest.update(str(source).encode())
        for p in _files_under(source):
            digest.update(f"{p.relative_to(source).as_posix()}:{p.stat().st_size}".encode())
            if p.suffix.lower() in TABLE_SUFFIXES:
                digest.update(p.read_bytes())
    return f"{sources[0].name}-{digest.hexdigest()[:12]}"


def plan_path(sources: list[Path], *, out: Path) -> Path:
    return out.resolve() / PLANS / f"{plan_key(sources)}.json"


def remember_plan(plans: list[LinkPlan], *, sources: list[Path], out: Path) -> Path:
    """Keep the plans a person accepted, one per source folder."""
    path = plan_path(sources, out=out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "sources": [str(s.resolve()) for s in sources],
                "plans": [p.model_dump() for p in plans],
                "accepted": datetime.now(UTC).isoformat(timespec="seconds"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def recall_plan(sources: list[Path], *, out: Path) -> list[LinkPlan] | None:
    """The plans accepted for these folders as they are now, or None. Anything that
    does not read back as one plan per folder is treated as nothing remembered."""
    path = plan_path(sources, out=out)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        plans = [LinkPlan.model_validate(p) for p in data["plans"]]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return plans if len(plans) == len(sources) else None


def forget_plan(sources: list[Path], *, out: Path) -> None:
    with contextlib.suppress(FileNotFoundError):
        plan_path(sources, out=out).unlink()


__all__ = [
    "HOLDOUT_CSV",
    "PLAN_JSON",
    "TRAIN_CSV",
    "Workspace",
    "forget_plan",
    "plan_key",
    "plan_path",
    "recall_plan",
    "remember_plan",
    "workspace_name",
    "write",
]
