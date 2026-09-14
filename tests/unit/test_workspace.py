"""Tests for the canonical data folder: layout, links, CSVs that load, idempotence."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from iterate.adapters.data import workspace
from iterate.adapters.data.images import detect_image_column, resolve_paths
from iterate.adapters.data.linking import apply, inventory, plan
from iterate.adapters.data.tabular import load_split

pytestmark = pytest.mark.unit

CLASSES = ("cat", "dog", "emu")


def _png(path: Path, seed: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 8), (seed * 9 % 255, 60, 120)).save(path)


def _class_tree(root: Path, *, per_class: int = 8, seed: int = 0) -> Path:
    for c in CLASSES:
        for i in range(per_class):
            _png(root / c / f"{c}_{seed + i}.png", seed + i)
    return root


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
    name = workspace.workspace_name([root.resolve()], p, frames)
    slot = tmp_path / "out" / name / "train" / "cat"
    slot.mkdir(parents=True)
    _png(slot / "cat_0.png", 12345)  # a different image already sits where cat_0 goes
    with pytest.raises(LinkError, match="already holds a different file"):
        workspace.write(p, frames, sources=[root], out=tmp_path / "out")
