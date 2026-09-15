"""Tests for the monitor: every check at every severity, the brief's bound, the budget,
the cache, and the round trip. Reports only; the one thing that changes data is the
twin removal, and that is asserted as such."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from PIL import Image, ImageOps

from iterate.adapters.data import monitor, workspace
from iterate.adapters.data.linking import apply, inventory, plan
from iterate.adapters.data.monitor import Monitor, drop_twins
from tests.unit.image_fixtures import CLASSES, class_tree, png

if TYPE_CHECKING:
    from iterate.schemas.monitor import DataReport

pytestmark = pytest.mark.unit

CHECKS = ("coverage", "labels", "twins", "lookalikes", "sources", "floors", "groups")


def _report(root: Path, checker: Monitor | None = None) -> tuple[DataReport, workspace.Sides]:
    checker = checker or Monitor()
    inv = inventory(root)
    p = plan(inv)
    frames = apply(p, inv)
    hashes = None if frames.holdout is not None else checker.hashes(map(str, frames.train["image"]))
    both = workspace.sides(p, frames, hashes=hashes)
    return checker.check([p], [inv], both), both


def _by_check(report: DataReport) -> dict[str, object]:
    return {f.check: f for f in report.findings}


def _user_split(root: Path, *, per_class: int = 6, holdout: int = 2) -> Path:
    class_tree(root / "train", per_class=per_class)
    class_tree(root / "test", per_class=holdout, seed=100)
    return root


def _table_root(
    root: Path,
    *,
    n: int = 40,
    labels: list[str] | None = None,
    extra: dict[str, list[object]] | None = None,
) -> Path:
    for i in range(n):
        png(root / "img" / f"{i:03d}.png", i)
    frame = pd.DataFrame(
        {"file": [f"{i:03d}.png" for i in range(n)], "label": labels or ["a", "b"] * (n // 2)}
    )
    for column, values in (extra or {}).items():
        frame[column] = values
    frame.to_csv(root / "labels.csv", index=False)
    return root


# ─── the clean case ──────────────────────────────────────────────────────────


def test_a_clean_class_tree_passes_every_check(tmp_path: Path) -> None:
    report, both = _report(class_tree(tmp_path / "pets", per_class=8))
    assert [f.check for f in report.findings] == list(CHECKS)
    assert all(f.severity == "pass" for f in report.findings)
    assert report.verdict == "pass"
    assert report.needs_a_look is False
    assert report.dropped == []
    lines = report.render().splitlines()
    assert lines[0].startswith("checks: 7 passed (24 images read in ")
    assert lines[1].startswith(
        "passed: every image has a row and reads; every image carries one label;"
    )
    assert report.brief() == (
        "Data checks before the run (host-run, deterministic): nothing found. "
        "Passed: coverage, labels, twins, lookalikes, sources, floors, groups."
    )
    assert (len(both.frames.train), len(both.frames.holdout)) == (19, 5)  # type: ignore[arg-type]


# ─── twins ───────────────────────────────────────────────────────────────────


def test_our_split_leaves_byte_copies_out_of_the_holdout(tmp_path: Path) -> None:
    """Whatever side the split puts a copy on, no holdout image shares bytes with a
    training image afterwards, and the report says what was left out."""
    root = class_tree(tmp_path / "pets", per_class=8)
    for i in range(6):
        (root / "cat" / f"cat_copy_{i}.png").write_bytes((root / "cat" / "cat_0.png").read_bytes())
    inv = inventory(root)
    p = plan(inv)
    frames = apply(p, inv)
    checker = Monitor()
    hashes = checker.hashes(map(str, frames.train["image"]))
    plain = workspace.sides(p, frames).frames
    both = workspace.sides(p, frames, hashes=hashes)
    assert plain.holdout is not None
    assert both.frames.holdout is not None
    train_sha = {hashes[str(q)] for q in both.frames.train["image"]}
    expected = [str(q) for q in plain.holdout["image"] if hashes[str(q)] in train_sha]
    assert both.dropped == expected
    assert expected  # the fixture put at least one copy across the split
    assert not any(hashes[str(q)] in train_sha for q in both.frames.holdout["image"])
    report = checker.check([p], [inv], both)
    assert _by_check(report)["twins"].severity == "pass"  # type: ignore[attr-defined]
    assert report.dropped == [Path(q).relative_to(root).as_posix() for q in expected]
    assert (
        f"dropped: {len(expected)} holdout images left out of the holdout because their bytes are in training"
        in report.render()
    )
    assert f"{len(expected)} holdout images left out as byte copies" in report.brief()


def test_a_users_holdout_twin_is_a_warn_that_drop_takes_out(tmp_path: Path) -> None:
    root = _user_split(tmp_path / "pets")
    (root / "test" / "cat" / "cat_99.png").write_bytes(
        (root / "train" / "cat" / "cat_0.png").read_bytes()
    )
    checker = Monitor()
    report, both = _report(root, checker)
    twins = _by_check(report)["twins"]
    assert twins.severity == "warn"  # type: ignore[attr-defined]
    assert twins.count == 1  # type: ignore[attr-defined]
    assert twins.share == pytest.approx(1 / 7)  # type: ignore[attr-defined]
    assert (
        "1 holdout images (14.3%) are byte-identical to a training image, 1 under the same label"
        in twins.summary
    )  # type: ignore[attr-defined]
    assert twins.examples == ["test/cat/cat_99.png = train/cat/cat_0.png"]  # type: ignore[attr-defined]
    assert "answer drop at the pause" in twins.way_out  # type: ignore[attr-defined]
    assert report.needs_a_look is True

    paths = map(str, (*both.frames.train["image"], *both.frames.holdout["image"]))  # type: ignore[union-attr]
    kept, gone = drop_twins(both.frames, checker.hashes(paths))
    assert gone == [str(root / "test" / "cat" / "cat_99.png")]
    assert kept.holdout is not None
    assert len(kept.holdout) == 6
    again = checker.check([plan(inventory(root))], [inventory(root)], workspace.Sides(kept, gone))
    assert _by_check(again)["twins"].severity == "pass"  # type: ignore[attr-defined]
    assert "dropped: 1 holdout images dropped from your holdout at your request" in again.render()


def test_a_twin_under_another_label_says_so(tmp_path: Path) -> None:
    root = _user_split(tmp_path / "pets")
    (root / "test" / "dog" / "dog_99.png").write_bytes(
        (root / "train" / "cat" / "cat_0.png").read_bytes()
    )
    report, _ = _report(root)
    assert "0 under the same label" in _by_check(report)["twins"].summary  # type: ignore[attr-defined]
    assert _by_check(report)["labels"].severity == "pass"  # type: ignore[attr-defined]


# ─── labels ──────────────────────────────────────────────────────────────────


def test_a_training_copy_under_two_labels_is_a_conflict(tmp_path: Path) -> None:
    root = _user_split(tmp_path / "pets")
    (root / "train" / "dog" / "dog_copy.png").write_bytes(
        (root / "train" / "cat" / "cat_0.png").read_bytes()
    )
    report, _ = _report(root)
    labels = _by_check(report)["labels"]
    assert labels.severity == "warn"  # type: ignore[attr-defined]
    assert labels.count == 2  # type: ignore[attr-defined]
    assert labels.examples == ["train/cat/cat_0.png:cat, train/dog/dog_copy.png:dog"]  # type: ignore[attr-defined]
    assert "2 training images carry more than one label (warn)" in report.brief()


def test_a_training_copy_under_one_label_is_a_note(tmp_path: Path) -> None:
    root = _user_split(tmp_path / "pets")
    (root / "train" / "cat" / "cat_copy.png").write_bytes(
        (root / "train" / "cat" / "cat_0.png").read_bytes()
    )
    report, _ = _report(root)
    labels = _by_check(report)["labels"]
    assert labels.severity == "note"  # type: ignore[attr-defined]
    assert "2 training images are byte copies under one label" in labels.summary  # type: ignore[attr-defined]


def test_a_conflict_inside_the_holdout_reaches_the_person_not_the_brief(tmp_path: Path) -> None:
    root = _user_split(tmp_path / "pets")
    (root / "test" / "dog" / "dog_copy.png").write_bytes(
        (root / "test" / "cat" / "cat_100.png").read_bytes()
    )
    report, _ = _report(root)
    labels = _by_check(report)["labels"]
    assert labels.severity == "warn"  # type: ignore[attr-defined]
    assert labels.count == 0  # type: ignore[attr-defined]
    assert "holdout test/cat/cat_100.png:cat" in labels.examples[0]  # type: ignore[attr-defined]
    assert "labels: see the report (warn)" in report.brief()
    assert "cat_100" not in report.brief()


def test_a_table_key_listed_twice_under_two_labels_is_a_conflict(tmp_path: Path) -> None:
    root = _table_root(tmp_path / "t", n=40)
    frame = pd.read_csv(root / "labels.csv")
    frame = pd.concat(
        [frame, pd.DataFrame({"file": ["003.png"], "label": ["z"]})], ignore_index=True
    )
    frame.to_csv(root / "labels.csv", index=False)
    report, _ = _report(root)
    labels = _by_check(report)["labels"]
    assert labels.severity == "warn"  # type: ignore[attr-defined]
    assert labels.examples == ["img/003.png listed as b and z"]  # type: ignore[attr-defined]
    coverage = _by_check(report)["coverage"]
    assert coverage.severity == "pass"  # type: ignore[attr-defined]


def test_a_table_key_listed_twice_under_one_label_is_a_note(tmp_path: Path) -> None:
    root = _table_root(tmp_path / "t", n=40)
    frame = pd.read_csv(root / "labels.csv")
    frame = pd.concat([frame, frame.iloc[[3]]], ignore_index=True)
    frame.to_csv(root / "labels.csv", index=False)
    report, _ = _report(root)
    labels = _by_check(report)["labels"]
    assert labels.severity == "note"  # type: ignore[attr-defined]
    assert "2 table rows list one image under one label" in labels.summary  # type: ignore[attr-defined]


# ─── lookalikes ──────────────────────────────────────────────────────────────


def _gradient(path: Path, *, size: int = 64, fmt: str = "PNG", quality: int = 95) -> Image.Image:
    a = np.tile(np.arange(size, dtype=np.uint8) * (255 // size), (size, 1))
    im = Image.fromarray(np.stack([a, a.T, a], axis=-1))
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, fmt, quality=quality) if fmt == "JPEG" else im.save(path, fmt)
    return im


def test_re_encoded_and_resized_copies_are_lookalikes_and_a_mirror_is_not(tmp_path: Path) -> None:
    root = _user_split(tmp_path / "pets")
    im = _gradient(root / "train" / "cat" / "grad.png")
    im.save(root / "test" / "cat" / "grad_q95.jpg", "JPEG", quality=95)
    im.resize((32, 32)).save(root / "test" / "cat" / "grad_half.png")
    ImageOps.mirror(im).save(root / "test" / "dog" / "grad_mirror.png")
    png(root / "test" / "emu" / "flat.png", 0, klass=9)  # a flat 12x8 tile never counts
    report, _ = _report(root)
    look = _by_check(report)["lookalikes"]
    assert look.severity == "note"  # type: ignore[attr-defined]
    assert sorted(look.details) == [  # type: ignore[attr-defined]
        "test/cat/grad_half.png ~ train/cat/grad.png",
        "test/cat/grad_q95.jpg ~ train/cat/grad.png",
    ]
    assert "2 under the same label" in look.summary  # type: ignore[attr-defined]
    assert _by_check(report)["twins"].severity == "pass"  # type: ignore[attr-defined]


def test_over_the_budget_the_thumbnail_pass_is_skipped_and_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(monitor, "LOOKALIKE_BUDGET", 64)
    root = _user_split(tmp_path / "pets")
    checker = Monitor()
    report, _ = _report(root, checker)
    look = _by_check(report)["lookalikes"]
    assert look.severity == "note"  # type: ignore[attr-defined]
    assert look.summary.startswith("skipped: ")  # type: ignore[attr-defined]
    assert not checker._thumb
    assert _by_check(report)["twins"].severity == "pass"  # type: ignore[attr-defined]


# ─── coverage ────────────────────────────────────────────────────────────────


def test_orphans_and_unnamed_rows_are_coverage_findings(tmp_path: Path) -> None:
    root = _table_root(tmp_path / "t", n=100)
    png(root / "img" / "stray.png", 500)
    frame = pd.read_csv(root / "labels.csv")
    frame = pd.concat(
        [frame, pd.DataFrame({"file": ["missing.png"], "label": ["a"]})], ignore_index=True
    )
    frame.to_csv(root / "labels.csv", index=False)
    report, _ = _report(root)
    coverage = _by_check(report)["coverage"]
    assert coverage.severity == "note"  # type: ignore[attr-defined]
    assert coverage.summary == "1 images (1.0%) have no row, 1 rows name no image"  # type: ignore[attr-defined]
    assert coverage.examples == ["img/stray.png has no row", "row 'missing.png' names no image"]  # type: ignore[attr-defined]
    for i in range(4):
        png(root / "img" / f"stray_{i}.png", 600 + i)
    report, _ = _report(root)
    assert _by_check(report)["coverage"].severity == "warn"  # type: ignore[attr-defined]


def test_an_image_that_cannot_be_decoded_is_a_coverage_warn(tmp_path: Path) -> None:
    root = _user_split(tmp_path / "pets")
    (root / "test" / "cat" / "broken.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 4)
    report, _ = _report(root)
    coverage = _by_check(report)["coverage"]
    assert coverage.severity == "warn"  # type: ignore[attr-defined]
    assert coverage.examples == ["test/cat/broken.png cannot be read"]  # type: ignore[attr-defined]
    assert _by_check(report)["lookalikes"].severity == "pass"  # type: ignore[attr-defined]


# ─── a second label source ───────────────────────────────────────────────────


def test_folder_names_that_disagree_with_the_table_are_a_warn(tmp_path: Path) -> None:
    root = tmp_path / "t"
    for k, c in enumerate(CLASSES):
        for i in range(10):
            png(root / c / f"{c}_{i}.png", i, klass=k)
    rows = [(f"{c}/{c}_{i}.png", c) for c in CLASSES for i in range(10)]
    rows[0] = (rows[0][0], "dog")
    rows[11] = (rows[11][0], "emu")
    pd.DataFrame({"image": [r[0] for r in rows], "label": [r[1] for r in rows]}).to_csv(
        root / "labels.csv", index=False
    )
    report, _ = _report(root)
    sources = _by_check(report)["sources"]
    assert sources.severity == "warn"  # type: ignore[attr-defined]
    assert sources.summary == "the folder names disagree with the labels on 2 of 30 images"  # type: ignore[attr-defined]
    assert sources.examples == [
        "cat/cat_0.png: dog here, cat there",
        "dog/dog_1.png: emu here, dog there",
    ]  # type: ignore[attr-defined]


def test_a_table_column_beside_class_folders_is_a_second_source(tmp_path: Path) -> None:
    root = class_tree(tmp_path / "pets", per_class=8)
    names = [f"{c}_{i}.png" for c in CLASSES for i in range(8)]
    kinds = [c for c in CLASSES for _ in range(8)]
    pd.DataFrame({"file": names, "kind": kinds, "note": ["x"] * 24}).to_csv(
        root / "meta.csv", index=False
    )
    report, _ = _report(root)
    sources = _by_check(report)["sources"]
    assert sources.severity == "pass"  # type: ignore[attr-defined]
    assert sources.summary == "column 'kind' of meta.csv agrees with the labels on all 24 images"  # type: ignore[attr-defined]
    kinds[0], kinds[9], kinds[17] = "dog", "emu", "cat"
    pd.DataFrame({"file": names, "kind": kinds, "note": ["x"] * 24}).to_csv(
        root / "meta.csv", index=False
    )
    report, _ = _report(root)
    assert _by_check(report)["sources"].severity == "warn"  # type: ignore[attr-defined]
    assert "disagrees with the labels on 3 of 24 images" in _by_check(report)["sources"].summary  # type: ignore[attr-defined]


def test_batch_folders_beside_a_table_are_no_second_source(tmp_path: Path) -> None:
    root = tmp_path / "t"
    for i in range(40):
        png(root / f"batch{i % 2}" / f"{i:03d}.png", i)
    pd.DataFrame(
        {"image": [f"batch{i % 2}/{i:03d}.png" for i in range(40)], "label": ["a", "b"] * 20}
    ).to_csv(root / "labels.csv", index=False)
    report, _ = _report(root)
    assert _by_check(report)["sources"].summary == "no second label source"  # type: ignore[attr-defined]


# ─── floors ──────────────────────────────────────────────────────────────────


def test_a_holdout_class_with_no_training_image_is_a_warn(tmp_path: Path) -> None:
    root = _user_split(tmp_path / "pets")
    png(root / "test" / "owl" / "owl_0.png", 7, klass=5)
    png(root / "test" / "owl" / "owl_1.png", 8, klass=5)
    report, _ = _report(root)
    floors = _by_check(report)["floors"]
    assert floors.severity == "warn"  # type: ignore[attr-defined]
    assert floors.examples == ["owl"]  # type: ignore[attr-defined]


def test_a_lopsided_class_is_a_note_and_a_number_is_no_class(tmp_path: Path) -> None:
    root = tmp_path / "t"
    labels = ["a"] * 80 + ["b"] * 3
    for i in range(83):
        png(root / "img" / f"{i:03d}.png", i)
    pd.DataFrame({"file": [f"{i:03d}.png" for i in range(83)], "label": labels}).to_csv(
        root / "labels.csv", index=False
    )
    report, _ = _report(root)
    floors = _by_check(report)["floors"]
    assert floors.severity == "note"  # type: ignore[attr-defined]
    assert "more than 20 to 1" in floors.summary  # type: ignore[attr-defined]
    scores = [1.0 + i * 0.37 for i in range(83)]
    pd.DataFrame({"file": [f"{i:03d}.png" for i in range(83)], "score": scores}).to_csv(
        root / "labels.csv", index=False
    )
    report, _ = _report(root)
    assert _by_check(report)["floors"].summary == "a number is predicted, no classes"  # type: ignore[attr-defined]


# ─── groups ──────────────────────────────────────────────────────────────────


def test_an_id_column_spanning_the_split_is_a_warn_and_a_split_by_it_passes(tmp_path: Path) -> None:
    patients = [f"P{i // 4}" for i in range(40)]
    root = _table_root(tmp_path / "t", n=40, extra={"patient_id": patients})
    report, _ = _report(root)
    groups = _by_check(report)["groups"]
    assert groups.severity == "warn"  # type: ignore[attr-defined]
    assert "column 'patient_id' groups the rows" in groups.summary  # type: ignore[attr-defined]
    assert "--train and --holdout" in groups.way_out  # type: ignore[attr-defined]

    by_patient = tmp_path / "split"
    for side, ids in (("train", range(0, 8)), ("test", range(8, 10))):
        rows = [(i, patients[i]) for i in range(40) if int(patients[i][1:]) in ids]
        for i, _ in rows:
            png(by_patient / side / "img" / f"{i:03d}.png", i)
        pd.DataFrame(
            {
                "file": [f"{i:03d}.png" for i, _ in rows],
                "label": ["a", "b"] * (len(rows) // 2),
                "patient_id": [pid for _, pid in rows],
            }
        ).to_csv(by_patient / side / "labels.csv", index=False)
    inv_a, inv_b = inventory(by_patient / "train"), inventory(by_patient / "test")
    plan_a, plan_b = plan(inv_a), plan(inv_b)
    frames = workspace.Sides(
        __import__("iterate.adapters.data.linking", fromlist=["LinkedFrames"]).LinkedFrames(
            apply(plan_a, inv_a).train, apply(plan_b, inv_b).train
        )
    )
    report = Monitor().check([plan_a, plan_b], [inv_a, inv_b], frames)
    assert _by_check(report)["groups"].severity == "pass"  # type: ignore[attr-defined]


def test_a_small_category_column_and_a_blank_cell_are_no_group(tmp_path: Path) -> None:
    regions: list[object] = ["north", "south", "east", "west"] * 10
    regions[3] = None
    root = _table_root(tmp_path / "t", n=40, extra={"region": regions})
    report, _ = _report(root)
    assert _by_check(report)["groups"].summary == "no column groups the rows"  # type: ignore[attr-defined]


# ─── the brief, the file, the cache ──────────────────────────────────────────


def test_the_brief_carries_no_file_name_and_no_label(tmp_path: Path) -> None:
    root = _user_split(tmp_path / "pets")
    (root / "test" / "cat" / "cat_99.png").write_bytes(
        (root / "train" / "cat" / "cat_0.png").read_bytes()
    )
    (root / "train" / "dog" / "dog_copy.png").write_bytes(
        (root / "train" / "cat" / "cat_1.png").read_bytes()
    )
    report, _ = _report(root)
    brief = report.brief()
    for word in ("cat_99", "cat_0", "dog_copy", "cat", "dog", "emu", ".png"):
        assert word not in brief
    assert "1 holdout images (14.3%) are byte-identical to a training image (warn)" in brief
    assert "2 training images carry more than one label (warn)" in brief
    assert "cat_99.png" in report.render()


def test_the_report_round_trips_and_a_broken_file_loads_as_none(tmp_path: Path) -> None:
    report, _ = _report(class_tree(tmp_path / "pets", per_class=8))
    path = monitor.save(report, tmp_path / "ws")
    assert path == tmp_path / "ws" / "monitor.json"
    assert monitor.load(tmp_path / "ws") == report
    assert json.loads(path.read_text(encoding="utf-8"))["version"] == "1"
    path.write_text("{oops", encoding="utf-8")
    assert monitor.load(tmp_path / "ws") is None
    assert monitor.load(tmp_path / "nowhere") is None


def test_a_second_check_reads_no_file_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = class_tree(tmp_path / "pets", per_class=8)
    checker = Monitor()
    _report(root, checker)
    reads: list[str] = []
    real = Path.read_bytes

    def counted(self: Path) -> bytes:
        reads.append(str(self))
        return real(self)

    monkeypatch.setattr(Path, "read_bytes", counted)
    _report(root, checker)
    assert not [r for r in reads if r.endswith(".png")]
