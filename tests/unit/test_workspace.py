"""Tests for the canonical data folder: layout, links, CSVs that load, idempotence."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pytest

from iterate.adapters.data import workspace
from iterate.adapters.data.images import detect_image_column, resolve_paths
from iterate.adapters.data.linking import LinkError, apply, inventory, plan
from iterate.adapters.data.tabular import load_split
from tests.unit.image_fixtures import CLASSES, class_tree, png

pytestmark = pytest.mark.unit


_png = png


def _class_tree(root: Path, *, per_class: int = 8, seed: int = 0) -> Path:
    return class_tree(root, per_class=per_class, seed=seed)


def test_the_layout_for_classification_split_here(tmp_path: Path) -> None:
    source = _class_tree(tmp_path / "pets")
    inv = inventory(source)
    p = plan(inv)
    ws = workspace.write(p, apply(p, inv), sources=[source], out=tmp_path / "out")

    assert ws.root.parent == tmp_path / "out"
    assert ws.root.name.startswith("pets-")
    assert sorted(q.name for q in ws.root.iterdir()) == [
        "holdout",
        "holdout.csv",
        "link.json",
        "raw_files",
        "train",
        "train.csv",
    ]
    # raw_files is a byte copy of the source tree
    assert sorted(q.name for q in (ws.root / "raw_files").iterdir()) == list(CLASSES)
    assert (ws.root / "raw_files" / "cat" / "cat_0.png").read_bytes() == (
        source / "cat" / "cat_0.png"
    ).read_bytes()
    # train and holdout hold class subfolders and link into raw_files
    assert sorted(q.name for q in (ws.root / "train").iterdir()) == list(CLASSES)
    linked = next((ws.root / "train" / "cat").iterdir())
    assert (
        linked.stat().st_nlink >= 2
        or linked.read_bytes() == (ws.root / "raw_files" / "cat" / linked.name).read_bytes()
    )
    train = pd.read_csv(ws.train_csv)
    holdout = pd.read_csv(ws.holdout_csv)
    assert list(train.columns) == ["image", "label"]
    assert len(train) + len(holdout) == 24
    assert len(holdout) == 5  # 20% of 24, stratified
    assert train["image"].str.startswith("train/").all()
    assert holdout["image"].str.startswith("holdout/").all()
    payload = json.loads(ws.plan_file.read_text())
    assert payload["plan"]["shape"] == "class_folders"
    assert payload["sources"] == [str(source.resolve())]


def test_the_csvs_load_through_the_day_one_seam(tmp_path: Path) -> None:
    source = _class_tree(tmp_path / "pets")
    inv = inventory(source)
    p = plan(inv)
    ws = workspace.write(p, apply(p, inv), sources=[source], out=tmp_path / "out")
    ds = load_split(ws.train_csv, ws.holdout_csv, target="label", task=p.task)
    column = detect_image_column(pd.read_csv(ws.train_csv), ["image"], ws.train_csv)
    assert column is not None
    ds = resolve_paths(ds, column)
    assert ds.user_split is True
    assert ds.task == "classification"
    assert all(v.startswith(str(ws.root)) for v in ds.train_features["image"])


def test_regression_lays_images_out_flat(tmp_path: Path) -> None:
    root = tmp_path / "faces"
    for i in range(30):
        _png(root / "photos" / f"p{i:03d}.jpg", i)
    pd.DataFrame(
        {
            "path": [f"photos/p{i:03d}.jpg" for i in range(30)],
            "score": [1.0 + i * 0.137 for i in range(30)],
        }
    ).to_csv(root / "scores.csv", index=False)
    inv = inventory(root)
    p = plan(inv)
    assert p.task == "regression"
    ws = workspace.write(p, apply(p, inv), sources=[root], out=tmp_path / "out")
    train_entries = list((ws.root / "train").iterdir())
    assert all(q.is_file() for q in train_entries)  # no class folders
    assert len(train_entries) == 24


def test_the_users_split_is_kept_and_both_folders_are_copied(tmp_path: Path) -> None:
    _class_tree(tmp_path / "data" / "train", per_class=5)
    _class_tree(tmp_path / "data" / "test", per_class=2, seed=100)
    inv = inventory(tmp_path / "data")
    p = plan(inv)
    assert p.split == "folders"
    ws = workspace.write(p, apply(p, inv), sources=[tmp_path / "data"], out=tmp_path / "out")
    assert len(pd.read_csv(ws.train_csv)) == 15
    assert len(pd.read_csv(ws.holdout_csv)) == 6
    assert (ws.root / "raw_files" / "test" / "cat").is_dir()


def test_colliding_file_names_get_a_suffix(tmp_path: Path) -> None:
    root = tmp_path / "dup"
    for c in ("a", "b"):
        for i in range(6):
            _png(root / c / "same.png" if i == 0 else root / c / f"{i}.png", i)
    # two different images both named same.png under different classes -> regression
    # layout is flat, so they would collide
    frames_root = tmp_path / "flat"
    for i in range(12):
        _png(
            frames_root / ("x" if i % 2 else "y") / "shot.png"
            if i < 2
            else frames_root / "more" / f"{i}.png",
            i,
        )
    pd.DataFrame(
        {
            "image": ["x/shot.png", "y/shot.png"] + [f"more/{i}.png" for i in range(2, 12)],
            "score": [0.5 + i * 0.31 for i in range(12)],
        }
    ).to_csv(frames_root / "t.csv", index=False)
    inv = inventory(frames_root)
    p = plan(inv)
    ws = workspace.write(p, apply(p, inv), sources=[frames_root], out=tmp_path / "out")
    train_names = sorted(q.name for q in (ws.root / "train").iterdir())
    holdout_names = sorted(q.name for q in (ws.root / "holdout").iterdir())
    assert len(train_names) == len(set(train_names))
    assert len(holdout_names) == len(set(holdout_names))
    assert len(train_names) + len(holdout_names) == 12
    assert sum(1 for n in train_names + holdout_names if n.startswith("shot")) == 2


def test_writing_twice_reuses_the_workspace(tmp_path: Path) -> None:
    source = _class_tree(tmp_path / "pets")
    inv = inventory(source)
    p = plan(inv)
    first = workspace.write(p, apply(p, inv), sources=[source], out=tmp_path / "out")
    stamp = first.train_csv.stat().st_mtime_ns
    second = workspace.write(p, apply(p, inv), sources=[source], out=tmp_path / "out")
    assert second.root == first.root
    assert second.train_csv.stat().st_mtime_ns == stamp
    assert len(list((tmp_path / "out").iterdir())) == 1


# ─── what the review found ─────────────────────────────────────────────────


def test_two_sources_with_the_same_basename_keep_their_own_bytes(tmp_path: Path) -> None:
    from iterate.adapters.data.linking import LinkedFrames

    a = _class_tree(tmp_path / "a" / "data", per_class=3)
    b = _class_tree(tmp_path / "b" / "data", per_class=2, seed=500)
    inv_a, inv_b = inventory(a), inventory(b)
    frames = LinkedFrames(apply(plan(inv_a), inv_a).train, apply(plan(inv_b), inv_b).train)
    p = plan(inv_a).model_copy(update={"split": "folders"})
    ws = workspace.write(p, frames, sources=[a, b], out=tmp_path / "out")
    assert sorted(q.name for q in (ws.root / "raw_files").iterdir()) == ["holdout", "train"]
    train_row = pd.read_csv(ws.train_csv).iloc[0]
    original = a / train_row["label"] / Path(train_row["image"]).name
    assert (ws.root / train_row["image"]).read_bytes() == original.read_bytes()
    holdout_row = pd.read_csv(ws.holdout_csv).iloc[0]
    original_h = b / holdout_row["label"] / Path(holdout_row["image"]).name
    assert (ws.root / holdout_row["image"]).read_bytes() == original_h.read_bytes()


def test_an_edited_label_lands_on_a_new_workspace(tmp_path: Path) -> None:
    root = tmp_path / "ids"
    for i in range(12):
        _png(root / "images" / f"{i}.jpg", i)
    pd.DataFrame({"id": [str(i) for i in range(12)], "label": ["a", "b"] * 6}).to_csv(
        root / "labels.csv", index=False
    )
    inv = inventory(root)
    first = workspace.write(plan(inv), apply(plan(inv), inv), sources=[root], out=tmp_path / "out")
    frame = pd.read_csv(root / "labels.csv")
    frame.loc[0, "label"] = "b"  # same length, different label
    frame.to_csv(root / "labels.csv", index=False)
    inv = inventory(root)
    second = workspace.write(plan(inv), apply(plan(inv), inv), sources=[root], out=tmp_path / "out")
    assert second.root != first.root


def test_labels_that_look_like_paths_stay_inside_the_workspace(tmp_path: Path) -> None:
    root = tmp_path / "odd"
    for i in range(12):
        _png(root / "images" / f"{i}.jpg", i)
    pd.DataFrame({"id": [str(i) for i in range(12)], "label": ["../escape", "a/b"] * 6}).to_csv(
        root / "labels.csv", index=False
    )
    inv = inventory(root)
    ws = workspace.write(plan(inv), apply(plan(inv), inv), sources=[root], out=tmp_path / "out")
    names = sorted(q.name for q in (ws.root / "train").iterdir())
    assert names == ["_._escape", "a_b"]
    assert not (tmp_path / "out" / "escape").exists()


def test_a_class_with_one_image_is_refused_before_anything_is_copied(tmp_path: Path) -> None:
    from iterate.adapters.data.linking import LinkError

    root = _class_tree(tmp_path / "pets", per_class=3)
    _png(root / "lonely" / "only.png", 99)
    inv = inventory(root)
    with pytest.raises(LinkError, match=r"single image cannot be split: \['lonely'\]"):
        workspace.write(plan(inv), apply(plan(inv), inv), sources=[root], out=tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_an_output_inside_the_data_folder_is_refused(tmp_path: Path) -> None:
    from iterate.adapters.data.linking import LinkError

    root = _class_tree(tmp_path / "pets")
    inv = inventory(root)
    with pytest.raises(LinkError, match="contain each other"):
        workspace.write(plan(inv), apply(plan(inv), inv), sources=[root], out=root / "out")


def test_case_only_collisions_get_a_suffix(tmp_path: Path) -> None:
    root = tmp_path / "case"
    _png(root / "x" / "Shot.png", 1)
    _png(root / "y" / "shot.png", 2)
    for i in range(10):
        _png(root / "more" / f"{i}.png", i + 10)
    pd.DataFrame(
        {
            "image": ["x/Shot.png", "y/shot.png"] + [f"more/{i}.png" for i in range(10)],
            "score": [0.5 + i * 0.31 for i in range(12)],
        }
    ).to_csv(root / "t.csv", index=False)
    inv = inventory(root)
    ws = workspace.write(plan(inv), apply(plan(inv), inv), sources=[root], out=tmp_path / "out")
    for folder in ("train", "holdout"):
        names = [q.name for q in (ws.root / folder).iterdir()]
        assert len({n.casefold() for n in names}) == len(names)
    total = len(list((ws.root / "train").iterdir())) + len(list((ws.root / "holdout").iterdir()))
    assert total == 12


def test_a_stale_file_in_a_slot_is_refused_not_reused(tmp_path: Path) -> None:
    from iterate.adapters.data.linking import LinkError

    root = _class_tree(tmp_path / "pets")
    inv = inventory(root)
    p = plan(inv)
    frames = apply(p, inv)
    name = workspace.workspace_name([root.resolve()], p, workspace.sides(p, frames).frames)
    slot = tmp_path / "out" / name / "train" / "cat"
    slot.mkdir(parents=True)
    _png(slot / "cat_0.png", 12345)  # a different image already sits where cat_0 goes
    with pytest.raises(LinkError, match="already holds a different file"):
        workspace.write(p, frames, sources=[root], out=tmp_path / "out")


# ─── the plans a person said yes to ───────────────────────────────────────


def _agent_plan(root: Path) -> tuple[Path, object]:
    for i in range(12):
        _png(root / "img" / f"{i}.jpg", i)
    pd.DataFrame(
        {"file": [f"{i}.jpg" for i in range(12)], "region": ["n", "s"] * 6, "code": ["x", "y"] * 6}
    ).to_csv(root / "meta.csv", index=False)
    from iterate.adapters.data.linking import plan_from_choice

    plan_ = plan_from_choice(
        inventory(root),
        table=root / "meta.csv",
        key_column="file",
        key_to_file="basename",
        target_column="code",
        source="agent",
    )
    return root, plan_


def test_an_accepted_plan_is_remembered_by_the_folders_contents(tmp_path: Path) -> None:
    root, plan_ = _agent_plan(tmp_path / "src")
    out = tmp_path / "out"
    assert workspace.recall_plan([root], out=out) is None
    path = workspace.remember_plan([plan_], sources=[root], out=out)  # type: ignore[list-item]
    assert path.parent == out / "plans"
    assert path.name.startswith("src-")
    assert workspace.recall_plan([root], out=out) == [plan_]

    frame = pd.read_csv(root / "meta.csv")
    frame["code"] = ["y", "x"] * 6  # same length, different labels
    frame.to_csv(root / "meta.csv", index=False)
    assert workspace.recall_plan([root], out=out) is None

    frame["code"] = ["x", "y"] * 6
    frame.to_csv(root / "meta.csv", index=False)
    assert workspace.recall_plan([root], out=out) == [plan_]
    workspace.forget_plan([root], out=out)
    assert workspace.recall_plan([root], out=out) is None
    workspace.forget_plan([root], out=out)  # nothing there is fine


def test_a_broken_memory_reads_as_nothing_remembered(tmp_path: Path) -> None:
    root, _ = _agent_plan(tmp_path / "src")
    out = tmp_path / "out"
    path = workspace.plan_path([root], out=out)
    path.parent.mkdir(parents=True)
    path.write_text("{not json", encoding="utf-8")
    assert workspace.recall_plan([root], out=out) is None
    path.write_text(json.dumps({"plans": [{"shape": "table_join"}]}), encoding="utf-8")
    assert workspace.recall_plan([root], out=out) is None


def test_two_folders_remember_two_plans_and_one_is_not_enough(tmp_path: Path) -> None:
    a, plan_a = _agent_plan(tmp_path / "train")
    b, plan_b = _agent_plan(tmp_path / "test")
    out = tmp_path / "out"
    workspace.remember_plan([plan_a, plan_b], sources=[a, b], out=out)  # type: ignore[list-item]
    assert workspace.recall_plan([a, b], out=out) == [plan_a, plan_b]
    assert workspace.recall_plan([b, a], out=out) is None
    workspace.remember_plan([plan_a], sources=[a, b], out=out)  # type: ignore[list-item]
    assert workspace.recall_plan([a, b], out=out) is None


def test_the_key_survives_a_loop_a_dangling_link_and_an_unreadable_table(tmp_path: Path) -> None:
    root, _ = _agent_plan(tmp_path / "src")
    before = workspace.plan_key([root])
    (root / "loop").symlink_to(root)
    (root / "gone.jpg").symlink_to(root / "nowhere.jpg")
    locked = root / "notes" / "private.csv"
    locked.parent.mkdir()
    locked.write_text("a,b\n1,2\n", encoding="utf-8")
    locked.chmod(0)
    try:
        after = workspace.plan_key([root])
        assert after == workspace.plan_key([root])
    finally:
        locked.chmod(0o644)
    assert after != before


def test_a_memory_too_deep_to_parse_reads_as_nothing_remembered(tmp_path: Path) -> None:
    root, _ = _agent_plan(tmp_path / "src")
    out = tmp_path / "out"
    path = workspace.plan_path([root], out=out)
    path.parent.mkdir(parents=True)
    path.write_text("[" * 200_000, encoding="utf-8")
    assert workspace.recall_plan([root], out=out) is None


# ─── the split before the pause ───────────────────────────────────────────


def test_sides_are_the_rows_write_lays_out(tmp_path: Path) -> None:
    source = _class_tree(tmp_path / "pets")
    inv = inventory(source)
    p = plan(inv)
    frames = apply(p, inv)
    both = workspace.sides(p, frames)
    assert both.frames.holdout is not None
    assert both.dropped == []
    ws = workspace.write(p, frames, sources=[source], out=tmp_path / "out", both=both)
    train = pd.read_csv(ws.train_csv)
    holdout = pd.read_csv(ws.holdout_csv)
    assert sorted(Path(q).name for q in train["image"]) == sorted(
        Path(str(q)).name for q in both.frames.train["image"]
    )
    assert sorted(Path(q).name for q in holdout["image"]) == sorted(
        Path(str(q)).name for q in both.frames.holdout["image"]
    )
    again = workspace.write(p, frames, sources=[source], out=tmp_path / "out")
    assert again.root == ws.root  # the same split made inside write lands on the same folder


def test_a_tree_too_small_for_both_sides_is_refused_before_any_copy(tmp_path: Path) -> None:
    from iterate.adapters.data.linking import LinkError

    source = _class_tree(tmp_path / "pets", per_class=3)
    inv = inventory(source)
    p = plan(inv)
    frames = apply(p, inv)
    with pytest.raises(LinkError, match="3 classes need at least 3 images on each side"):
        workspace.sides(p, frames)
    with pytest.raises(LinkError, match="on each side"):
        workspace.write(p, frames, sources=[source], out=tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_write_leaves_byte_copies_out_of_the_holdout_on_its_own(tmp_path: Path) -> None:
    source = _class_tree(tmp_path / "pets")
    for i in range(6):
        (source / "cat" / f"cat_copy_{i}.png").write_bytes(
            (source / "cat" / "cat_0.png").read_bytes()
        )
    inv = inventory(source)
    p = plan(inv)
    frames = apply(p, inv)
    ws = workspace.write(p, frames, sources=[source], out=tmp_path / "out")
    train = {
        Path(q).read_bytes() for q in pd.read_csv(ws.train_csv)["image"].map(lambda q: ws.root / q)
    }
    holdout = [
        Path(q).read_bytes()
        for q in pd.read_csv(ws.holdout_csv)["image"].map(lambda q: ws.root / q)
    ]
    assert not any(b in train for b in holdout)
    plain = workspace.sides(p, frames).frames
    assert plain.holdout is not None
    assert len(holdout) < len(plain.holdout)  # the split put a copy across, and it came out
    assert sorted(q.name for q in (ws.root / "raw_files" / "cat").iterdir())[:2] == [
        "cat_0.png",
        "cat_1.png",
    ]


def test_write_never_hashes_a_holdout_the_user_gave(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from iterate.adapters.data.linking import LinkedFrames

    a = _class_tree(tmp_path / "train", per_class=5)
    b = _class_tree(tmp_path / "test", per_class=2, seed=100)
    inv_a, inv_b = inventory(a), inventory(b)
    frames = LinkedFrames(apply(plan(inv_a), inv_a).train, apply(plan(inv_b), inv_b).train)

    def boom(paths: object) -> dict[str, str | None]:
        raise AssertionError("hashed a given holdout")

    monkeypatch.setattr(workspace, "file_hashes", boom)
    ws = workspace.write(plan(inv_a), frames, sources=[a, b], out=tmp_path / "out")
    assert len(pd.read_csv(ws.holdout_csv)) == 6


def test_the_memory_keeps_whether_the_twins_were_dropped(tmp_path: Path) -> None:
    root, plan_ = _agent_plan(tmp_path / "src")
    out = tmp_path / "out"
    assert workspace.recall_drop([root], out=out) is False
    workspace.remember_plan([plan_], sources=[root], out=out, drop_twins=True)  # type: ignore[list-item]
    assert workspace.recall_drop([root], out=out) is True
    assert workspace.recall_plan([root], out=out) == [plan_]
    workspace.remember_plan([plan_], sources=[root], out=out)  # type: ignore[list-item]
    assert workspace.recall_drop([root], out=out) is False


def test_write_refuses_a_link_out_made_after_the_inventory(tmp_path: Path) -> None:
    source = _class_tree(tmp_path / "pets")
    inv = inventory(source)
    p = plan(inv)
    frames = apply(p, inv)
    png(tmp_path / "elsewhere" / "secret.png", 99)
    (source / "cat" / "late.png").symlink_to(tmp_path / "elsewhere" / "secret.png")
    with pytest.raises(LinkError, match=r"1 link.* lead outside the folder you gave"):
        workspace.write(p, frames, sources=[source], out=tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_a_copy_that_fails_is_a_link_error_not_a_traceback(tmp_path: Path) -> None:
    source = _class_tree(tmp_path / "pets")
    locked = source / "notes.txt"
    locked.write_text("x", encoding="utf-8")
    locked.chmod(0)
    try:
        if os.access(locked, os.R_OK):
            pytest.skip("this user reads any file")
        inv = inventory(source)
        p = plan(inv)
        with pytest.raises(LinkError, match=r"1 file\(s\) could not be copied into .*notes\.txt"):
            workspace.write(p, apply(p, inv), sources=[source], out=tmp_path / "out")
    finally:
        locked.chmod(0o644)
