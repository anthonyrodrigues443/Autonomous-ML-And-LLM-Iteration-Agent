"""Deterministic checks on a linked dataset, before a run and before any model.

Every check reads the source images and tables and reports; none relabels or
moves anything, and none stops the run. The one leak a rule proves, the same
bytes on both sides of the split, comes out of the holdout: by the split itself
when the split is ours, at the person's request when it is theirs. The label
audit and the spot check that look at images with a model follow the target and
are not here.
"""

from __future__ import annotations

import hashlib
import io
import time
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from iterate.adapters.data import linking
from iterate.adapters.data.linking import Inventory, LinkedFrames, TableRows, class_of
from iterate.schemas.monitor import DETAILS, EXAMPLES, DataReport, Finding, Severity

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from iterate.adapters.data.workspace import Sides
    from iterate.schemas.link import LinkPlan, Split

DETAIL_FLOOR = 4.0
IMBALANCE = 20
ORPHANS_WARN = 0.02
GROUP_NAMES = (
    "id",
    "subject",
    "patient",
    "user",
    "person",
    "session",
    "group",
    "source",
    "photographer",
    "device",
    "camera",
    "site",
    "study",
    "family",
    "owner",
    "batch",
)
GROUP_MIN_VALUES = 20
GROUP_ROWS_PER_VALUE = (2.0, 50.0)
GROUP_WARN = 0.10
RIVAL_SHARE = 0.5
LOOKALIKE_BUDGET = 2 * 1024**3  # bytes of images; over it the thumbnail pass is skipped
MONITOR_VERSION = "1"
REPORT_JSON = "monitor.json"


def _rel(path: str, inventories: Sequence[Inventory]) -> str:
    p = Path(path)
    for inv in inventories:
        if p.is_relative_to(inv.root):
            return p.relative_to(inv.root).as_posix()
    return path


def _size(path: str) -> int:
    try:
        return Path(path).stat().st_size
    except OSError:
        return 0


def _rows_of(plan_: LinkPlan, inv: Inventory) -> TableRows | None:
    if plan_.shape == "class_folders":
        return None
    assert plan_.labels_file  # the schema
    assert plan_.key_column
    assert plan_.key_to_file
    return linking.table_rows(
        inv,
        Path(plan_.labels_file),
        plan_.key_column,
        plan_.key_to_file,
        target_column=plan_.target_column,
        onehot_columns=plan_.onehot_columns,
    )


def _dhash(data: bytes) -> tuple[int, float] | None:
    """A 64-bit difference hash from a 9x8 grey thumbnail, the sign of each
    horizontal step, and the thumbnail's spread. Every format is decoded in full, so
    a copy in another container hashes the same; a flat tile has no spread and is
    left out of the search."""
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(io.BytesIO(data)) as im:
            small = im.convert("L").resize((9, 8), Image.Resampling.BILINEAR)
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        return None
    a = np.asarray(small, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return int(np.packbits(bits).view(">u8")[0]), float(a.std())


class Monitor:
    """The checks, with the hashes kept between rounds of the pause: a correction
    changes the plan, never the bytes. Each file is read once for both hashes."""

    def __init__(self) -> None:
        self._sha: dict[str, str | None] = {}
        self._thumb: dict[str, tuple[int, float] | None] = {}
        self._spent = 0.0

    def _hash(self, paths: Iterable[str], *, thumbs: bool) -> None:
        started = time.perf_counter()
        try:
            self._read(paths, thumbs=thumbs)
        finally:
            self._spent += time.perf_counter() - started

    def _read(self, paths: Iterable[str], *, thumbs: bool) -> None:
        for p in dict.fromkeys(paths):
            if p in self._sha and (not thumbs or p in self._thumb):
                continue
            try:
                data = Path(p).read_bytes()
            except OSError:
                self._sha[p] = None
                self._thumb[p] = None
                continue
            self._sha[p] = hashlib.sha256(data).hexdigest()
            if thumbs:
                self._thumb[p] = _dhash(data)

    def hashes(self, paths: Iterable[str]) -> dict[str, str | None]:
        """The byte hash of every path, read once; the thumbnail comes from the same
        read when the images fit the budget."""
        wanted = list(dict.fromkeys(paths))
        self._hash(wanted, thumbs=sum(_size(p) for p in wanted) <= LOOKALIKE_BUDGET)
        return {p: self._sha[p] for p in wanted}

    def check(
        self, plans: Sequence[LinkPlan], inventories: Sequence[Inventory], both: Sides
    ) -> DataReport:
        """Seven findings, always, so a pass is a line a person reads. ``both`` is the
        split as ``workspace.sides`` made it."""
        frames = both.frames
        assert frames.holdout is not None  # workspace.sides
        paths = list(dict.fromkeys(map(str, (*frames.train["image"], *frames.holdout["image"]))))
        size = sum(_size(p) for p in paths)
        thumbs = size <= LOOKALIKE_BUDGET
        self._hash(paths, thumbs=thumbs)  # its own time goes to _spent
        started = time.perf_counter()
        rows = [_rows_of(p, inv) for p, inv in zip(plans, inventories, strict=True)]
        split: Split = plans[0].split if len(plans) == 1 else "folders"
        left_out = list(zip(both.dropped, both.dropped_labels, strict=True))
        findings = [
            _coverage(inventories, rows, frames, self._sha, self._thumb if thumbs else None),
            _labels(frames, self._sha, rows, inventories, left_out=left_out),
            _twins(frames, self._sha, inventories, split=split),
            _lookalikes(frames, self._sha, self._thumb, inventories, over=None if thumbs else size),
            _sources(plans, inventories, frames),
            _floors(frames, task=plans[0].task),
            _groups(rows, frames, plans),
        ]
        return DataReport(
            version=MONITOR_VERSION,
            images=sum(len(inv.images) for inv in inventories),
            train_rows=len(frames.train),
            holdout_rows=len(frames.holdout),
            split=split,
            seconds=self._spent + (time.perf_counter() - started),
            findings=findings,
            dropped=[_rel(p, inventories) for p in both.dropped],
        )


def drop_twins(
    frames: LinkedFrames, hashes: Mapping[str, str | None]
) -> tuple[LinkedFrames, list[str], list[str]]:
    """The holdout without the images whose bytes sit in training, what came out, and
    their labels. For a user's split, at the pause, on request. A holdout that would
    be left empty is refused."""
    assert frames.holdout is not None
    seen = {d for p in frames.train["image"] if (d := hashes.get(str(p))) is not None}
    twin = frames.holdout["image"].map(lambda p: hashes.get(str(p)) in seen).to_numpy(dtype=bool)
    if twin.all():
        raise linking.LinkError(
            "every holdout image is a byte copy of a training image; dropping them would leave "
            "no holdout rows"
        )
    return (
        LinkedFrames(frames.train, frames.holdout.loc[~twin].reset_index(drop=True)),
        [str(p) for p in frames.holdout.loc[twin, "image"]],
        [str(v) for v in frames.holdout.loc[twin, "label"]],
    )


# ─── the checks ───────────────────────────────────────────────────────────


def _coverage(
    inventories: Sequence[Inventory],
    rows: Sequence[TableRows | None],
    frames: LinkedFrames,
    sha: Mapping[str, str | None],
    thumb: Mapping[str, tuple[int, float] | None] | None,
) -> Finding:
    """Both ways: images no row claims, rows that name no image, and files that
    cannot be read or decoded; the decode is known only when the thumbnail pass ran."""
    assert frames.holdout is not None
    unreadable = [
        str(p)
        for p in (*frames.train["image"], *frames.holdout["image"])
        if sha.get(str(p)) is None or (thumb is not None and thumb.get(str(p)) is None)
    ]
    orphans: list[str] = []
    unnamed: list[str] = []
    images = 0
    for inv, table in zip(inventories, rows, strict=True):
        images += len(inv.images)
        if table is None:
            continue
        claimed = {p for p in table.claimed if p is not None}
        orphans.extend(str(p) for p in inv.images if p not in claimed)
        unnamed.extend(
            k or "(blank)" for k, p in zip(table.keys, table.claimed, strict=True) if p is None
        )
    shown = [
        *(f"{_rel(p, inventories)} cannot be read" for p in unreadable),
        *(f"{_rel(p, inventories)} has no row" for p in orphans),
        *(f"row {k!r} names no image" for k in unnamed),
    ]
    if not shown:
        return Finding(check="coverage", severity="pass", summary="every image has a row and reads")
    share = (len(unreadable) + len(orphans)) / max(1, images)
    severity: Severity = "warn" if unreadable or share >= ORPHANS_WARN else "note"
    parts = []
    if unreadable:
        parts.append(f"{len(unreadable)} images cannot be read")
    if orphans:
        parts.append(f"{len(orphans)} images ({len(orphans) / max(1, images):.1%}) have no row")
    if unnamed:
        parts.append(f"{len(unnamed)} rows name no image")
    return Finding(
        check="coverage",
        severity=severity,
        count=len(unreadable) + len(orphans),
        share=share,
        summary=", ".join(parts),
        way_out="they are left out of the run; the list is in monitor.json",
        examples=shown[:EXAMPLES],
        details=shown[:DETAILS],
    )


def _labels(
    frames: LinkedFrames,
    sha: Mapping[str, str | None],
    rows: Sequence[TableRows | None],
    inventories: Sequence[Inventory],
    *,
    left_out: Sequence[tuple[str, str]] = (),
) -> Finding:
    """Duplicate and conflicting labels. On the training side: the same bytes under
    more than one label, or under one label twice; a holdout twin the split left out
    counts with the training copy it matched, so a copy filed under two labels is a
    conflict whichever side it fell on. In a table: rows that name one image
    together, which ``apply`` sets aside. Inside the holdout the same groups are
    told to the person and kept out of ``count``, so the brief says nothing a
    holdout row could be read from. Across the split, see twins."""
    assert frames.holdout is not None

    def groups(
        frame: pd.DataFrame, extra: Sequence[tuple[str, str]] = ()
    ) -> list[list[tuple[str, str]]]:
        by: dict[str, list[tuple[str, str]]] = {}
        pairs = [*zip(frame["image"], frame["label"], strict=True), *extra]
        for p, label in pairs:
            if (d := sha.get(str(p))) is not None:
                by.setdefault(d, []).append((str(p), str(label)))
        return [g for g in by.values() if len(g) > 1]

    conflicts = [g for g in groups(frames.train, left_out) if len({label for _, label in g}) > 1]
    copies = [g for g in groups(frames.train) if len({label for _, label in g}) == 1]
    hidden = [g for g in groups(frames.holdout) if len({label for _, label in g}) > 1]
    keyed: list[tuple[str, list[str]]] = []
    dup_rows = 0
    for table in rows:
        if table is None or table.label is None:
            continue
        together: dict[Path, list[str]] = {}
        for i, (named, kept) in enumerate(zip(table.claimed, table.resolved, strict=True)):
            if named is not None and kept is None:
                together.setdefault(named, []).append(str(table.label.iloc[i]))
        for image, labels in together.items():
            if len(set(labels)) > 1:
                keyed.append((str(image), labels))
            else:
                dup_rows += len(labels)
    shown = [
        *(", ".join(f"{_rel(p, inventories)}:{label}" for p, label in g) for g in conflicts),
        *(f"{_rel(p, inventories)} listed as {' and '.join(sorted(set(ls)))}" for p, ls in keyed),
        *(", ".join(f"holdout {_rel(p, inventories)}:{label}" for p, label in g) for g in hidden),
    ]
    count = sum(len(g) for g in conflicts) + sum(len(ls) for _, ls in keyed)
    if not shown and not copies and not dup_rows:
        return Finding(check="labels", severity="pass", summary="every image carries one label")
    if not shown:
        quiet = []
        if copies:
            quiet.append(
                f"{sum(len(g) for g in copies)} training images are byte copies under one label"
            )
        if dup_rows:
            quiet.append(f"{dup_rows} table rows list one image under one label")
        return Finding(
            check="labels",
            severity="note",
            summary=" and ".join(quiet),
            way_out="a copy counts twice in training; the rows are in monitor.json",
        )
    loud = []
    if conflicts:
        loud.append(
            f"{len(conflicts)} groups of byte-identical training images carry more than one label"
        )
    if keyed:
        loud.append(f"{len(keyed)} table keys name one image under two labels")
    if hidden:
        loud.append(
            f"{len(hidden)} groups of byte-identical holdout images carry more than one label"
        )
    return Finding(
        check="labels",
        severity="warn",
        count=count,
        summary="; ".join(loud),
        way_out="the labels are used as they are; fix the table or the folders, or say so at the pause",
        examples=shown[:EXAMPLES],
        details=shown[:DETAILS],
    )


def _twins(
    frames: LinkedFrames,
    sha: Mapping[str, str | None],
    inventories: Sequence[Inventory],
    *,
    split: Split,
) -> Finding:
    """The same bytes on both sides of the split. A model scores a twin from memory,
    so a holdout score can read higher than on clean data by at most this share. Our
    split leaves them out before this runs; under a user's split the person can say
    drop at the pause."""
    assert frames.holdout is not None
    train_of: dict[str, tuple[str, set[str]]] = {}
    for p, label in zip(frames.train["image"], frames.train["label"], strict=True):
        if (d := sha.get(str(p))) is not None:
            train_of.setdefault(d, (str(p), set()))[1].add(str(label))
    pairs = [
        (str(p), train_of[d][0], str(label) in train_of[d][1])
        for p, label in zip(frames.holdout["image"], frames.holdout["label"], strict=True)
        if (d := sha.get(str(p))) is not None and d in train_of
    ]
    if not pairs:
        return Finding(
            check="twins", severity="pass", summary="no image sits on both sides of the split"
        )
    share = len(pairs) / len(frames.holdout)
    same = sum(s for _, _, s in pairs)
    shown = [f"{_rel(h, inventories)} = {_rel(t, inventories)}" for h, t, _ in pairs]
    placed = (
        "pass --train and --holdout to place each pair yourself"
        if split == "ours"
        else "answer drop at the pause to take them out of your holdout, or keep them with yes"
    )
    return Finding(
        check="twins",
        severity="warn",
        count=len(pairs),
        share=share,
        summary=(
            f"{len(pairs)} holdout images ({share:.1%}) are byte-identical to a training "
            f"image, {same} under the same label"
        ),
        way_out=(
            f"a score on this holdout can read up to {share:.1%} higher than on clean data; "
            f"the pairs are in monitor.json; {placed}"
        ),
        examples=shown[:EXAMPLES],
        details=shown[:DETAILS],
    )


def _lookalikes(
    frames: LinkedFrames,
    sha: Mapping[str, str | None],
    thumb: Mapping[str, tuple[int, float] | None],
    inventories: Sequence[Inventory],
    *,
    over: int | None,
) -> Finding:
    """A holdout image whose 64-bit hash is a training image's or one bit off, both
    thumbnails with detail, not already byte twins: a re-encoded or resized copy,
    or a plain scene. The search is 65 dict lookups per holdout row; the cost is the
    decode, so over the byte budget the pass is skipped and says so. A pointer for a
    person, never a warn: the hash cannot tell a copy from a plain scene, and the
    spot check that looks at the pairs follows the target."""
    assert frames.holdout is not None
    if over is not None:
        return Finding(
            check="lookalikes",
            severity="note",
            summary=(
                f"skipped: {over / 1024**3:.1f} GB of images is over the "
                f"{LOOKALIKE_BUDGET // 1024**3} GB budget for the thumbnail pass"
            ),
            way_out="the byte checks ran; the pass can be run on a smaller folder",
        )
    train_of: dict[int, tuple[str, str]] = {}
    for p, label in zip(frames.train["image"], frames.train["label"], strict=True):
        if (t := thumb.get(str(p))) is not None and t[1] >= DETAIL_FLOOR:
            train_of.setdefault(t[0], (str(p), str(label)))
    train_digests = {d for p in frames.train["image"] if (d := sha.get(str(p))) is not None}
    pairs: list[tuple[str, str]] = []
    same = 0
    for p, label in zip(frames.holdout["image"], frames.holdout["label"], strict=True):
        t = thumb.get(str(p))
        if t is None or t[1] < DETAIL_FLOOR or sha.get(str(p)) in train_digests:
            continue
        h = t[0]
        hit = next(
            (train_of[k] for k in (h, *(h ^ (1 << i) for i in range(64))) if k in train_of), None
        )
        if hit is not None:
            pairs.append((str(p), hit[0]))
            same += hit[1] == str(label)
    if not pairs:
        return Finding(
            check="lookalikes",
            severity="pass",
            summary="no holdout image looks like a training image at a coarse hash",
        )
    share = len(pairs) / len(frames.holdout)
    shown = [f"{_rel(h, inventories)} ~ {_rel(t, inventories)}" for h, t in pairs]
    return Finding(
        check="lookalikes",
        severity="note",
        count=len(pairs),
        share=share,
        summary=(
            f"{len(pairs)} holdout images ({share:.1%}) look like a training image at a coarse "
            f"hash, {same} under the same label: a re-encoded copy or a plain scene"
        ),
        way_out="the pairs are in monitor.json",
        examples=shown[:EXAMPLES],
        details=shown[:DETAILS],
    )


def _folder_labels(inv: Inventory) -> dict[str, str]:
    """Each inventoried image's top-level folder name, when the root has at least
    two folders holding images directly."""
    roots = list(inv.split_pair) if inv.split_pair else [inv.root]
    out: dict[str, str] = {}
    for root in roots:
        by_folder: Counter[str] = Counter()
        for p in inv.images:
            folder = class_of(p, root)
            if folder is not None and p.parent.parent == root:
                by_folder[folder] += 1
        if len(by_folder) < 2:
            continue
        for p in inv.images:
            folder = class_of(p, root)
            if folder is not None and folder in by_folder and p.parent.parent == root:
                out[str(p)] = folder
    return out


def _sources(
    plans: Sequence[LinkPlan],
    inventories: Sequence[Inventory],
    frames: LinkedFrames,
) -> Finding:
    """Two label sources, when the folder has both: folder names beside a table plan,
    or a table column beside a class-folder plan. Agreement on at least half the
    images makes it a second source; a source that agrees on fewer is a batch layout
    or an unrelated column, and no finding."""
    assert frames.holdout is not None
    label_of = {
        str(p): str(label)
        for part in (frames.train, frames.holdout)
        for p, label in zip(part["image"], part["label"], strict=True)
    }
    best: tuple[str, int, int, list[str]] | None = None  # name, compared, disagree, examples
    for plan_, inv in zip(plans, inventories, strict=True):
        candidates: list[tuple[str, dict[str, str]]] = []
        if plan_.shape != "class_folders":
            folders = _folder_labels(inv)
            if folders:
                candidates.append(("the folder names", folders))
        else:
            for t in inv.tables:
                try:
                    rates = linking.join_rates(inv, t)
                except (OSError, ValueError):
                    continue
                if not rates:
                    continue
                key, (method, rate) = max(rates.items(), key=lambda kv: kv[1][1])
                if rate < linking.PAUSE:
                    continue
                read = linking.table_rows(inv, t, key, method)
                for column in read.frame.columns:
                    if str(column) == key or not pd.api.types.is_string_dtype(read.frame[column]):
                        continue
                    values = {
                        str(p): str(v)
                        for p, v in zip(read.claimed, read.frame[column], strict=True)
                        if p is not None and not pd.isna(v)
                    }
                    candidates.append((f"column {str(column)!r} of {t.name}", values))
        for name, values in candidates:
            compared = [(p, v) for p, v in values.items() if p in label_of]
            if not compared:
                continue
            agree = sum(1 for p, v in compared if v.casefold() == label_of[p].casefold())
            if agree / len(compared) < RIVAL_SHARE:
                continue
            disagree = [
                f"{_rel(p, inventories)}: {label_of[p]} here, {v} there"
                for p, v in compared
                if v.casefold() != label_of[p].casefold()
            ]
            if best is None or len(disagree) < best[2]:
                best = (name, len(compared), len(disagree), disagree)
    if best is None:
        return Finding(check="sources", severity="pass", summary="no second label source")
    name, n_compared, n_disagree, listed = best
    agrees, disagrees = (
        ("agrees", "disagrees") if name.startswith("column") else ("agree", "disagree")
    )
    if not n_disagree:
        return Finding(
            check="sources",
            severity="pass",
            summary=f"{name} {agrees} with the labels on all {n_compared} images",
        )
    return Finding(
        check="sources",
        severity="warn",
        count=n_disagree,
        share=n_disagree / n_compared,
        summary=f"{name} {disagrees} with the labels on {n_disagree} of {n_compared} images",
        way_out="the plan's labels are used; the disagreements are in monitor.json",
        examples=listed[:EXAMPLES],
        details=listed[:DETAILS],
    )


def _floors(frames: LinkedFrames, *, task: str) -> Finding:
    """A holdout class with no training images cannot be learned; a class more than
    IMBALANCE times smaller than the largest is worth knowing, and the metric chosen
    from the profile is what answers it."""
    assert frames.holdout is not None
    if task != "classification":
        return Finding(check="floors", severity="pass", summary="a number is predicted, no classes")
    train = Counter(str(v) for v in frames.train["label"])
    holdout = Counter(str(v) for v in frames.holdout["label"])
    absent = sorted(c for c in holdout if c not in train)
    if absent:
        return Finding(
            check="floors",
            severity="warn",
            count=len(absent),
            summary=f"{len(absent)} holdout classes have no training image",
            way_out="a model cannot learn them; move some of their images to training",
            examples=absent[:EXAMPLES],
            details=absent[:DETAILS],
        )
    unscored = sorted(c for c in train if c not in holdout)
    if unscored:
        return Finding(
            check="floors",
            severity="warn",
            count=len(unscored),
            summary=f"{len(unscored)} training classes have no holdout image",
            way_out=(
                "they cannot be scored; add images, or place them yourself with --train and "
                "--holdout"
            ),
            examples=unscored[:EXAMPLES],
            details=unscored[:DETAILS],
        )
    largest, smallest = train.most_common()[0], train.most_common()[-1]
    if largest[1] > IMBALANCE * smallest[1]:
        return Finding(
            check="floors",
            severity="note",
            count=1,
            summary=(
                f"class {largest[0]!r} has {largest[1]} training images and {smallest[0]!r} "
                f"has {smallest[1]}, more than {IMBALANCE} to 1"
            ),
            way_out="the metric is chosen for the balance; a model that guesses the big class looks better than it is",
        )
    return Finding(
        check="floors",
        severity="pass",
        summary=f"{len(train)} classes, every class on both sides, within {IMBALANCE} to 1",
    )


def _tokens(name: str) -> set[str]:
    out, word = set(), ""
    for ch in name:
        if ch.isalnum():
            word += ch
        elif word:
            out.add(word.casefold())
            word = ""
    if word:
        out.add(word.casefold())
    return out


def _groups(
    rows: Sequence[TableRows | None], frames: LinkedFrames, plans: Sequence[LinkPlan]
) -> Finding:
    """A column that groups the rows, by name or by shape, whose values sit on both
    sides of the split: the same subject, session or device in train and holdout,
    which a model can learn instead of the label. A question, not a proof. The
    plan's own columns, the key, the label, a one-hot block, the split, are never
    a group."""
    assert frames.holdout is not None
    train_paths = set(map(str, frames.train["image"]))
    holdout_paths = set(map(str, frames.holdout["image"]))
    best: tuple[str, float, int, list[str]] | None = None
    for table, plan_ in zip(rows, plans, strict=True):
        if table is None:
            continue
        own = {plan_.key_column, plan_.target_column, plan_.split_column, *plan_.onehot_columns}
        frame = table.frame
        n = len(frame)
        for column in frame.columns:
            name = str(column)
            if name in own:
                continue
            series = frame[column]
            if pd.api.types.is_float_dtype(series) and not bool(((series.dropna() % 1) == 0).all()):
                continue
            values = series.astype(str).where(series.notna(), None)
            distinct = int(series.dropna().nunique())
            if distinct < 3 or distinct == n:
                continue
            per_value = n / distinct
            named = bool(_tokens(name) & set(GROUP_NAMES))
            # By shape only for text codes: a number with a few dozen values is a
            # measurement far more often than an identity.
            shaped = (
                pd.api.types.is_string_dtype(series)
                and distinct >= GROUP_MIN_VALUES
                and GROUP_ROWS_PER_VALUE[0] <= per_value <= GROUP_ROWS_PER_VALUE[1]
            )
            if not (named or shaped):
                continue
            train_values: set[str] = set()
            holdout_of: list[tuple[str, str]] = []
            for p, v in zip(table.resolved, values, strict=True):
                if p is None or v is None:
                    continue
                if str(p) in train_paths:
                    train_values.add(v)
                elif str(p) in holdout_paths:
                    holdout_of.append((str(p), v))
            if not holdout_of:
                continue
            spanning = [(p, v) for p, v in holdout_of if v in train_values]
            share = len(spanning) / len(frames.holdout)
            if best is None or share > best[1]:
                shown = [
                    f"{Path(p).name} shares {name}={v} with a training image" for p, v in spanning
                ]
                best = (name, share, len(spanning), shown)
    if best is None:
        return Finding(check="groups", severity="pass", summary="no column groups the rows")
    name, share, count, shown = best
    if share < GROUP_WARN:
        return Finding(
            check="groups",
            severity="pass",
            summary=f"column {name!r} groups the rows and {share:.1%} of holdout rows share a value with train",
        )
    return Finding(
        check="groups",
        severity="warn",
        count=count,
        share=share,
        summary=(
            f"column {name!r} groups the rows and {share:.1%} of holdout rows share a value "
            "with a training row"
        ),
        way_out="if that is the same subject or session, split by it yourself with --train and --holdout",
        examples=shown[:EXAMPLES],
        details=shown[:DETAILS],
    )


def save(report: DataReport, root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / REPORT_JSON
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return path


def load(root: Path) -> DataReport | None:
    path = root / REPORT_JSON
    if not path.exists():
        return None
    try:
        return DataReport.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


__all__ = [
    "IMBALANCE",
    "LOOKALIKE_BUDGET",
    "REPORT_JSON",
    "Monitor",
    "drop_twins",
    "load",
    "save",
]
