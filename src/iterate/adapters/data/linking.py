"""The rules ladder: from a folder with anything inside to frames of image path
plus label, with no model involved.

An inventory walks the folder, then the ladder tries shapes in a fixed order and
the first plan that builds a frame with enough coverage wins: explicit flags, a
table joined to the images on an exact key, class folders. Anything the rules
cannot prove is refused with the reason: a plan that resolves too few rows, two
tables or two key columns that both fit, a split value nobody named. The Linker (a
later piece) can only add a plan where this ladder found none, and its plan goes
through the same `apply`.
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from iterate.adapters.data.tabular import looks_like_classification
from iterate.schemas.link import KeyMethod, LinkPlan, Shape, Source, Task

if TYPE_CHECKING:
    from collections.abc import Sequence

log = logging.getLogger(__name__)

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"})
TABLE_SUFFIXES = frozenset({".csv", ".tsv"})
CONTAINER_SUFFIXES = frozenset(
    {".parquet", ".h5", ".hdf5", ".npz", ".npy", ".mat", ".tfrecord", ".tfrecords", ".bson"}
)
SKIP_DIRS = frozenset({"__MACOSX", ".ipynb_checkpoints"})
TRAIN_NAMES = ("train", "training")
HOLDOUT_NAMES = (
    "test",
    "testing",
    "holdout",
    "val",
    "valid",
    "validation",
    "eval",
    "evaluation",
    "dev",
)
SPLIT_COLUMNS = frozenset(
    {"split", "subset", "set", "partition", "is_train", "is_training", "fold"}
)
KEY_NAMES = (
    "image",
    "image_id",
    "image_name",
    "image_path",
    "img",
    "filename",
    "file_name",
    "file",
    "path",
    "id",
    "name",
)
TARGET_NAMES = (
    "label",
    "labels",
    "class",
    "class_name",
    "classname",
    "class_id",
    "target",
    "category",
    "species",
    "breed",
    "dx",
    "diagnosis",
    "y",
    "score",
    "rating",
    "age",
    "price",
    "count",
)
KEY_METHODS: tuple[KeyMethod, ...] = ("path", "basename", "stem", "stem_int")
ACCEPT = 0.98
PAUSE = 0.90
MAX_DEPTH = 6


class LinkError(ValueError):
    """The folder cannot be read as a labelled image dataset; the message says why."""


@dataclass
class Inventory:
    """What a folder holds, after wrapper folders are collapsed."""

    root: Path
    images: list[Path] = field(default_factory=list)
    tables: list[Path] = field(default_factory=list)
    containers: list[Path] = field(default_factory=list)
    split_pair: tuple[Path, Path] | None = None
    collapsed: list[str] = field(default_factory=list)
    ignored: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class LinkedFrames:
    """The plan applied: frames of ``image`` (absolute path) and ``label``."""

    train: pd.DataFrame
    holdout: pd.DataFrame | None


# ─── inventory ────────────────────────────────────────────────────────────


def _visible(p: Path) -> bool:
    return not p.name.startswith(".") and p.name not in SKIP_DIRS


def _collapse_wrappers(root: Path) -> tuple[Path, list[str]]:
    """Descend while the folder holds exactly one visible subfolder and no data files."""
    collapsed: list[str] = []
    while True:
        entries = [p for p in root.iterdir() if _visible(p)]
        dirs = [p for p in entries if p.is_dir()]
        data = [
            p
            for p in entries
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES | TABLE_SUFFIXES
        ]
        if len(dirs) == 1 and not data:
            root = dirs[0]
            collapsed.append(root.name)
            continue
        return root, collapsed


def _split_pair(root: Path) -> tuple[tuple[Path, Path] | None, list[str]]:
    """The train and holdout folders when the root carries its own split, and any
    third split-named folder that is then left out."""
    names = {p.name.lower(): p for p in root.iterdir() if p.is_dir() and _visible(p)}
    train = next((names[n] for n in TRAIN_NAMES if n in names), None)
    holdout = next((names[n] for n in HOLDOUT_NAMES if n in names), None)
    if train is None or holdout is None:
        return None, []
    ignored = [
        names[n].name
        for n in (*TRAIN_NAMES, *HOLDOUT_NAMES)
        if n in names and names[n] not in (train, holdout)
    ]
    return (train, holdout), ignored


def inventory(folder: str | Path) -> Inventory:
    """Walk a folder to depth six; hidden entries and archive litter are skipped."""
    root = Path(folder).resolve()
    if not root.is_dir():
        raise LinkError(f"{root} is not a folder")
    root, collapsed = _collapse_wrappers(root)
    pair, ignored = _split_pair(root)
    inv = Inventory(root=root, collapsed=collapsed, split_pair=pair, ignored=ignored)

    def walk(directory: Path, depth: int) -> None:
        if depth > MAX_DEPTH:
            return
        for p in sorted(directory.iterdir()):
            if not _visible(p):
                continue
            suffix = p.suffix.lower()
            if p.is_dir():
                walk(p, depth + 1)
            elif suffix in IMAGE_SUFFIXES:
                inv.images.append(p)
            elif suffix in TABLE_SUFFIXES:
                inv.tables.append(p)
            elif suffix in CONTAINER_SUFFIXES:
                inv.containers.append(p)

    walk(root, 0)
    return inv


# ─── the key index ────────────────────────────────────────────────────────


def _keys_for(image: Path, root: Path, table_dir: Path) -> dict[KeyMethod, set[str]]:
    rel = image.relative_to(root).as_posix()
    keys: dict[KeyMethod, set[str]] = {
        "path": {rel, rel.lower()},
        "basename": {image.name, image.name.lower()},
        "stem": {image.stem, image.stem.lower()},
        "stem_int": set(),
    }
    with contextlib.suppress(ValueError):
        keys["path"].add(image.relative_to(table_dir).as_posix())
    if image.stem.isdigit():
        keys["stem_int"].add(str(int(image.stem)))
    return keys


def _index(
    images: Sequence[Path], root: Path, table_dir: Path
) -> dict[KeyMethod, dict[str, list[Path]]]:
    index: dict[KeyMethod, dict[str, list[Path]]] = {m: {} for m in KEY_METHODS}
    for image in images:
        for method, keys in _keys_for(image, root, table_dir).items():
            for key in keys:
                index[method].setdefault(key, []).append(image)
    return index


def _key_strings(values: pd.Series) -> list[str]:
    """Cell text to look up; blank cells give an empty string, integral floats give
    the integer, because one blank in an id column turns the whole column float."""
    if pd.api.types.is_float_dtype(values):
        numbers = values.dropna()
        if bool(((numbers % 1) == 0).all()):
            return ["" if pd.isna(v) else str(int(v)) for v in values]
    return ["" if pd.isna(v) else str(v).strip() for v in values]


def _resolve(
    values: pd.Series, index: dict[str, list[Path]], method: KeyMethod
) -> tuple[list[Path | None], float]:
    """One image per row or None; coverage is the share of rows that resolved to an
    image no other row resolved to."""
    out: list[Path | None] = []
    for v in _key_strings(values):
        candidates = (index.get(v) or index.get(v.lower()) or []) if v else []
        if method == "stem_int" and v.isdigit():
            candidates = index.get(str(int(v))) or candidates
        if method == "path" and v and not candidates:
            candidates = index.get(v.replace("\\", "/").lstrip("./")) or []
        out.append(candidates[0] if len(candidates) == 1 else None)
    counts: dict[Path, int] = {}
    for p in out:
        if p is not None:
            counts[p] = counts.get(p, 0) + 1
    out = [p if p is not None and counts[p] == 1 else None for p in out]
    hits = sum(1 for p in out if p is not None)
    return out, hits / max(1, len(out))


# ─── the ladder ───────────────────────────────────────────────────────────


def _read_table(path: Path) -> pd.DataFrame:
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    return pd.read_csv(path, sep=sep, encoding_errors="replace")


def _task_of(values: pd.Series) -> Task:
    return "classification" if looks_like_classification(values) else "regression"


def _onehot_block(frame: pd.DataFrame, exclude: set[str]) -> list[str]:
    """Binary 0/1 columns that sum to exactly one per row: a one-hot label."""
    binary = [
        c
        for c in frame.columns
        if c not in exclude
        and pd.api.types.is_numeric_dtype(frame[c])
        and set(frame[c].dropna().unique()) <= {0, 1}
    ]
    if len(binary) < 2:
        return []
    return binary if bool((frame[binary].sum(axis=1) == 1).all()) else []


def _pick_target(frame: pd.DataFrame, exclude: set[str]) -> tuple[str | None, list[str]]:
    lowered = {str(c).lower(): str(c) for c in frame.columns if c not in exclude}
    for name in TARGET_NAMES:
        if name in lowered:
            return lowered[name], []
    block = _onehot_block(frame, exclude)
    if block:
        return None, block
    rest = [str(c) for c in frame.columns if c not in exclude]
    return (rest[0], []) if len(rest) == 1 else (None, [])


def _split_column(frame: pd.DataFrame) -> str | None:
    """A column that names the split, or None. A value that is neither a train name
    nor a holdout name is refused rather than guessed."""
    for c in frame.columns:
        if str(c).lower() not in SPLIT_COLUMNS:
            continue
        values = {str(v).lower() for v in frame[c].dropna().unique()}
        if not (values & set(TRAIN_NAMES) and values & set(HOLDOUT_NAMES)):
            continue
        stray = sorted(values - set(TRAIN_NAMES) - set(HOLDOUT_NAMES))
        if stray:
            raise LinkError(
                f"column {c!r} names the split but also holds {stray}; rename those rows "
                "to a train or a test value, or drop the column"
            )
        if len(values & set(HOLDOUT_NAMES)) > 1:
            raise LinkError(
                f"column {c!r} holds more than one holdout name "
                f"({sorted(values & set(HOLDOUT_NAMES))}); merge them into one first"
            )
        return str(c)
    return None


def _looks_like_a_label_table(table: Path) -> bool:
    """A table with a label-looking column claims the labels, even when no key resolves."""
    try:
        frame = _read_table(table)
    except (OSError, ValueError):
        return False
    lowered = {str(c).lower() for c in frame.columns}
    return bool(lowered & set(TARGET_NAMES)) or bool(_onehot_block(frame, set()))


def _key_dtype_ok(series: pd.Series) -> bool:
    if pd.api.types.is_string_dtype(series) or pd.api.types.is_integer_dtype(series):
        return True
    return bool(pd.api.types.is_float_dtype(series) and ((series.dropna() % 1) == 0).all())


@dataclass(frozen=True)
class _Candidate:
    plan: LinkPlan
    resolved: tuple[Path | None, ...]


def _plan_from_table(
    inv: Inventory,
    table: Path,
    *,
    key: str | None,
    target: str | None,
    source: Source,
) -> LinkPlan | None:
    frame = _read_table(table)
    if len(frame) < 2:
        return None
    if key is not None and key not in frame.columns:
        raise LinkError(f"{table.name} has no column {key!r}; columns are {list(frame.columns)}")
    if target is not None and target not in frame.columns:
        raise LinkError(f"{table.name} has no column {target!r}; columns are {list(frame.columns)}")
    index = _index(inv.images, inv.root, table.parent)
    named = [c for c in frame.columns if str(c).lower() in KEY_NAMES]
    columns = [key] if key else named + [c for c in frame.columns if c not in named]
    if target is not None:
        columns = [c for c in columns if c != target]
    candidates: list[_Candidate] = []
    for column in columns:
        if not _key_dtype_ok(frame[column]):
            continue
        for method in KEY_METHODS:
            resolved, coverage = _resolve(frame[column], index[method], method)
            if coverage < PAUSE:
                continue
            chosen, block = (target, []) if target else _pick_target(frame, {column})
            if chosen is None and not block:
                continue
            labelled = frame[block].notna().all(axis=1) if block else frame[chosen].notna()
            pairs = list(zip(resolved, labelled, strict=True))
            linked = [p if ok else None for p, ok in pairs]
            unlabelled = sum(1 for p, ok in pairs if p is not None and not ok)
            coverage = sum(1 for p in linked if p is not None) / len(linked)
            if coverage < PAUSE:
                continue
            split_col = _split_column(frame)
            first = str(frame[column].iloc[0]).lower()
            shape: Shape = (
                "csv_of_paths"
                if method == "path" and first.endswith(tuple(IMAGE_SUFFIXES))
                else "table_join"
            )
            notes: list[str] = []
            if coverage < ACCEPT:
                notes.append(f"{coverage:.1%} of rows resolved to an image; the rest are dropped")
            if unlabelled:
                notes.append(f"{unlabelled} row(s) have no label and are dropped")
            candidates.append(
                _Candidate(
                    LinkPlan(
                        shape=shape,
                        source=source,
                        labels_file=str(table),
                        key_column=str(column),
                        key_to_file=method,
                        target_column=chosen,
                        onehot_columns=block,
                        task=_task_of(frame[chosen]) if chosen else "classification",
                        split="column" if split_col else ("folders" if inv.split_pair else "ours"),
                        split_column=split_col,
                        coverage=coverage,
                        notes=notes,
                    ),
                    tuple(linked),
                )
            )
            break
    if not candidates:
        return None
    best = max(candidates, key=lambda c: c.plan.coverage)
    for other in candidates:
        if other is best or other.plan.key_column == best.plan.key_column:
            continue
        if any(
            a is not None and b is not None and a != b
            for a, b in zip(best.resolved, other.resolved, strict=True)
        ):
            raise LinkError(
                f"{table.name}: columns {best.plan.key_column!r} and {other.plan.key_column!r} "
                "both name the images but point at different files; pass --key to say which"
            )
    return best.plan


def _class_dirs(root: Path) -> list[Path]:
    return [p for p in sorted(root.iterdir()) if p.is_dir() and _visible(p)]


def _class_of(image: Path, root: Path) -> str | None:
    """The top-level folder under ``root`` an inventoried image sits in, or None."""
    try:
        parts = image.relative_to(root).parts
    except ValueError:
        return None
    return parts[0] if len(parts) >= 2 else None


def _plan_class_folders(inv: Inventory) -> LinkPlan | None:
    """One folder per class, images directly inside; a class folder that holds
    folders of its own is not a class folder the rules can name."""
    roots = list(inv.split_pair) if inv.split_pair else [inv.root]
    for r in roots:
        dirs = _class_dirs(r)
        if len(dirs) < 2:
            return None
        for d in dirs:
            members = [p for p in inv.images if _class_of(p, r) == d.name]
            if not members:
                return None
            if any(p.parent != d for p in members):
                raise LinkError(
                    f"{d}: a class folder with folders inside; flatten it, or pass --labels"
                )
    notes = (
        [f"ignoring folder(s) {inv.ignored}; only the train and holdout pair is read"]
        if inv.ignored
        else []
    )
    return LinkPlan(
        shape="class_folders",
        task="classification",
        split="folders" if inv.split_pair else "ours",
        coverage=1.0,
        notes=notes,
    )


def plan(
    inv: Inventory,
    *,
    labels: Path | None = None,
    key: str | None = None,
    target: str | None = None,
) -> LinkPlan:
    """The first shape that builds a frame with enough coverage; a LinkError otherwise.

    Flags beat rules: with ``labels`` only that table is tried, and ``key`` and
    ``target`` name its columns. Without flags every table is tried; two tables that
    fit equally well are a tie, and a tie is refused rather than guessed.
    """
    if not inv.images:
        if inv.containers:
            kinds = sorted({p.suffix for p in inv.containers})
            raise LinkError(
                f"{inv.root}: the data is inside {', '.join(kinds)} files, not image files; "
                "unpack it to images first"
            )
        raise LinkError(f"{inv.root}: no image files found")

    if labels is not None:
        found = _plan_from_table(inv, labels.resolve(), key=key, target=target, source="flags")
        if found is None:
            raise LinkError(
                f"{labels.name}: no column resolves at least {PAUSE:.0%} of its rows to an "
                f"image under {inv.root}"
            )
        return found

    fits: list[tuple[Path, LinkPlan]] = []
    for table in inv.tables:
        found = _plan_from_table(inv, table, key=None, target=target, source="rules")
        if found is not None:
            fits.append((table, found))
    if fits:
        top = max(p.coverage for _, p in fits)
        tied = [t.name for t, p in fits if p.coverage == top]
        if len(tied) > 1:
            raise LinkError(
                f"{inv.root}: {' and '.join(tied)} both link the images equally well; pass "
                "--labels to say which one holds the labels"
            )
        return next(p for _, p in fits if p.coverage == top)
    claimants = [t.name for t in inv.tables if _looks_like_a_label_table(t)]
    if claimants:
        raise LinkError(
            f"{inv.root}: {', '.join(claimants)} looks like the label table but no column "
            f"resolves at least {PAUSE:.0%} of its rows to an image. Pass --labels <csv> "
            "with --key <column> and --target <column> to say how"
        )
    folders = _plan_class_folders(inv)
    if folders is not None:
        return folders
    seen = f"{len(inv.images)} images and {len(inv.tables)} table(s)"
    raise LinkError(
        f"{inv.root}: {seen}, and no rule links them. Pass --labels <csv> with --key <column> "
        "and --target <column> to say how, or lay the images out as one folder per class"
    )


# ─── apply ────────────────────────────────────────────────────────────────


def _frame_from_class_dirs(inv: Inventory, root: Path) -> pd.DataFrame:
    rows = [
        {"image": str(p), "label": label}
        for p in inv.images
        if (label := _class_of(p, root)) is not None
    ]
    return pd.DataFrame(rows, columns=["image", "label"])


def _under(path: str, folder: Path) -> bool:
    return path.startswith(str(folder) + "/")


def apply(plan_: LinkPlan, inv: Inventory) -> LinkedFrames:
    """The only path from a plan to frames, whichever built the plan. A split with an
    empty side is refused: it means the table never reached one of the folders."""
    if plan_.shape == "class_folders":
        if inv.split_pair:
            frames = LinkedFrames(
                _frame_from_class_dirs(inv, inv.split_pair[0]),
                _frame_from_class_dirs(inv, inv.split_pair[1]),
            )
        else:
            frames = LinkedFrames(_frame_from_class_dirs(inv, inv.root), None)
        return _checked(frames, plan_, inv)

    assert plan_.labels_file  # the schema
    assert plan_.key_column
    assert plan_.key_to_file
    table = Path(plan_.labels_file)
    frame = _read_table(table)
    index = _index(inv.images, inv.root, table.parent)[plan_.key_to_file]
    resolved, _ = _resolve(frame[plan_.key_column], index, plan_.key_to_file)
    if plan_.onehot_columns:
        label = (
            frame[plan_.onehot_columns]
            .idxmax(axis=1)
            .where(frame[plan_.onehot_columns].notna().all(axis=1))
        )
    else:
        assert plan_.target_column  # the schema
        label = frame[plan_.target_column]
    out = pd.DataFrame(
        {"image": [str(p) if p else None for p in resolved], "label": label.to_numpy()}
    )
    if plan_.split_column:
        marks = frame[plan_.split_column].astype(str).str.lower()
        out["split"] = marks.map(lambda v: "train" if v in TRAIN_NAMES else "holdout").to_numpy()
    out = out.dropna(subset=["image", "label"]).reset_index(drop=True)

    if plan_.split == "column":
        train = out[out["split"] == "train"].drop(columns="split").reset_index(drop=True)
        holdout = out[out["split"] == "holdout"].drop(columns="split").reset_index(drop=True)
        return _checked(LinkedFrames(train, holdout), plan_, inv)
    if plan_.split == "folders" and inv.split_pair:
        train_dir, holdout_dir = inv.split_pair
        in_train = out["image"].map(lambda v: _under(str(v), train_dir))
        in_holdout = out["image"].map(lambda v: _under(str(v), holdout_dir))
        return _checked(
            LinkedFrames(
                out[in_train].reset_index(drop=True), out[in_holdout].reset_index(drop=True)
            ),
            plan_,
            inv,
        )
    return _checked(LinkedFrames(out, None), plan_, inv)


def _checked(frames: LinkedFrames, plan_: LinkPlan, inv: Inventory) -> LinkedFrames:
    if frames.train.empty:
        raise LinkError(f"{inv.root}: the link leaves no training rows")
    if frames.holdout is not None and frames.holdout.empty:
        where = plan_.split_column or (inv.split_pair[1].name if inv.split_pair else "holdout")
        raise LinkError(
            f"{inv.root}: the split leaves no holdout rows; nothing in the table reaches {where!r}"
        )
    return frames


def render(plan_: LinkPlan, inv: Inventory, frames: LinkedFrames) -> str:
    """The block a person reads before saying yes."""
    where = f"data: {inv.root}"
    if inv.collapsed:
        where += f" (inside {'/'.join(inv.collapsed)})"
    if plan_.shape == "class_folders":
        labels = f"labels: the folder names ({frames.train['label'].nunique()} classes)"
    else:
        how = {
            "path": "a path",
            "basename": "the file name",
            "stem": "the file name without its extension",
            "stem_int": "the file number",
        }[plan_.key_to_file or "path"]
        target = (
            ", ".join(plan_.onehot_columns) + " (one-hot)"
            if plan_.onehot_columns
            else repr(plan_.target_column)
        )
        labels = (
            f"labels: {Path(plan_.labels_file or '').name}, column {plan_.key_column!r} is "
            f"{how}, target {target}"
        )
    n_hold = 0 if frames.holdout is None else len(frames.holdout)
    split = {
        "ours": f"split: none given, {len(frames.train)} images split here 80/20",
        "folders": f"split: yours, from the folders, {len(frames.train)} train / {n_hold} holdout",
        "column": (
            f"split: yours, from column {plan_.split_column!r}, "
            f"{len(frames.train)} train / {n_hold} holdout"
        ),
    }[plan_.split]
    lines = [where, labels, f"task: {plan_.task}", split]
    if plan_.shape == "class_folders":
        lines.append(f"images: {len(inv.images)}")
    else:
        lines.append(f"coverage: {plan_.coverage:.1%} of the table's rows found their image")
    lines.extend(f"note: {n}" for n in plan_.notes)
    if plan_.source != "rules":
        lines.append(f"linked by: {plan_.source}")
    return "\n".join(lines)


__all__ = [
    "ACCEPT",
    "CONTAINER_SUFFIXES",
    "IMAGE_SUFFIXES",
    "KEY_METHODS",
    "PAUSE",
    "SKIP_DIRS",
    "TABLE_SUFFIXES",
    "Inventory",
    "LinkError",
    "LinkedFrames",
    "apply",
    "inventory",
    "plan",
    "render",
]
