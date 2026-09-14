"""Tests for the rules ladder: every shape it links, and every refusal it names."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pytest
from PIL import Image

from iterate.adapters.data import linking
from iterate.adapters.data.linking import LinkError, apply, inventory, plan, render

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit

CLASSES = ("cat", "dog", "emu")


def _png(path: Path, seed: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 8), (seed * 9 % 255, 60, 120)).save(path)


def _class_tree(root: Path, *, per_class: int = 4, seed: int = 0) -> Path:
    for c in CLASSES:
        for i in range(per_class):
            _png(root / c / f"{c}_{seed + i}.png", seed + i)
    return root


def _table_and_images(
    root: Path,
    *,
    key: str,
    values: list[str],
    labels: list[str],
    files: list[str],
    sub: str = "images",
) -> Path:
    for i, f in enumerate(files):
        _png(root / sub / f, i)
    pd.DataFrame({key: values, "label": labels}).to_csv(root / "labels.csv", index=False)
    return root


# ─── inventory ─────────────────────────────────────────────────────────────


def test_wrapper_folders_collapse_and_hidden_entries_are_skipped(tmp_path: Path) -> None:
    root = tmp_path / "archive" / "archive"
    _class_tree(root)
    (root / "__MACOSX").mkdir()
    _png(root / "__MACOSX" / "._cat_0.png")
    (root / ".DS_Store").write_bytes(b"x")
    inv = inventory(tmp_path)
    assert inv.root == root
    assert inv.collapsed == ["archive", "archive"]
    assert len(inv.images) == 12


def test_a_train_and_test_pair_is_recognised(tmp_path: Path) -> None:
    _class_tree(tmp_path / "train")
    _class_tree(tmp_path / "Test", per_class=2, seed=100)
    inv = inventory(tmp_path)
    assert inv.split_pair is not None
    assert [p.name for p in inv.split_pair] == ["train", "Test"]


def test_a_file_path_is_not_a_folder(tmp_path: Path) -> None:
    (tmp_path / "x.csv").write_text("a,b\n")
    with pytest.raises(LinkError, match="not a folder"):
        inventory(tmp_path / "x.csv")


# ─── refusals ──────────────────────────────────────────────────────────────


def test_a_folder_of_containers_is_refused_by_name(tmp_path: Path) -> None:
    (tmp_path / "data.parquet").write_bytes(b"PAR1")
    (tmp_path / "more.h5").write_bytes(b"\x89HDF")
    with pytest.raises(LinkError, match=r"inside \.h5, \.parquet files"):
        plan(inventory(tmp_path))


def test_a_folder_without_images_is_refused(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("nothing here")
    with pytest.raises(LinkError, match="no image files"):
        plan(inventory(tmp_path))


def test_images_with_no_rule_are_refused_with_the_flags(tmp_path: Path) -> None:
    for i in range(6):
        _png(tmp_path / f"img{i}.png", i)
    with pytest.raises(LinkError, match="--labels <csv> with --key"):
        plan(inventory(tmp_path))


def test_a_single_class_folder_is_not_a_classification_dataset(tmp_path: Path) -> None:
    for i in range(6):
        _png(tmp_path / "only" / f"{i}.png", i)
    with pytest.raises(LinkError, match="no rule links them"):
        plan(inventory(tmp_path))


# ─── class folders ─────────────────────────────────────────────────────────


def test_class_folders_are_split_here(tmp_path: Path) -> None:
    inv = inventory(_class_tree(tmp_path))
    p = plan(inv)
    assert p.shape == "class_folders"
    assert p.split == "ours"
    assert p.coverage == 1.0
    frames = apply(p, inv)
    assert frames.holdout is None
    assert len(frames.train) == 12
    assert set(frames.train["label"]) == set(CLASSES)
    assert "labels: the folder names (3 classes)" in render(p, inv, frames)


def test_class_folders_under_train_and_test_keep_the_users_split(tmp_path: Path) -> None:
    _class_tree(tmp_path / "train", per_class=5)
    _class_tree(tmp_path / "test", per_class=2, seed=100)
    inv = inventory(tmp_path)
    p = plan(inv)
    assert p.split == "folders"
    frames = apply(p, inv)
    assert frames.holdout is not None
    assert (len(frames.train), len(frames.holdout)) == (15, 6)
    assert all("/test/" in v for v in frames.holdout["image"])


# ─── tables ────────────────────────────────────────────────────────────────


def test_bare_ids_join_on_the_stem(tmp_path: Path) -> None:
    root = _table_and_images(
        tmp_path,
        key="id",
        values=[str(1000 + i) for i in range(20)],
        labels=["a", "b"] * 10,
        files=[f"{1000 + i}.jpg" for i in range(20)],
    )
    inv = inventory(root)
    p = plan(inv)
    assert p.shape == "table_join"
    assert p.key_column == "id"
    assert p.key_to_file in ("stem", "stem_int")
    assert p.target_column == "label"
    assert p.task == "classification"
    frames = apply(p, inv)
    assert len(frames.train) == 20
    assert frames.train["image"].str.endswith(".jpg").all()


def test_a_path_column_relative_to_the_table_is_a_csv_of_paths(tmp_path: Path) -> None:
    root = _table_and_images(
        tmp_path,
        key="image",
        values=[f"images/{i}.png" for i in range(12)],
        labels=[str(1.5 + i * 0.37) for i in range(12)],
        files=[f"{i}.png" for i in range(12)],
    )
    inv = inventory(root)
    p = plan(inv)
    assert p.shape == "csv_of_paths"
    assert p.key_to_file == "path"
    assert p.task == "regression"
    assert "task: regression" in render(p, inv, apply(p, inv))


def test_onehot_columns_collapse_to_one_label(tmp_path: Path) -> None:
    for i in range(9):
        _png(tmp_path / "img" / f"leaf_{i:03d}.jpg", i)
    frame = pd.DataFrame(
        {
            "image_id": [f"leaf_{i:03d}" for i in range(9)],
            "healthy": [int(i % 3 == 0) for i in range(9)],
            "rust": [int(i % 3 == 1) for i in range(9)],
            "scab": [int(i % 3 == 2) for i in range(9)],
        }
    )
    frame.to_csv(tmp_path / "train.csv", index=False)
    inv = inventory(tmp_path)
    p = plan(inv)
    assert p.onehot_columns == ["healthy", "rust", "scab"]
    assert p.target_column is None
    frames = apply(p, inv)
    assert list(frames.train["label"][:3]) == ["healthy", "rust", "scab"]
    assert "(one-hot)" in render(p, inv, frames)


def test_a_split_column_is_the_users_split(tmp_path: Path) -> None:
    root = _table_and_images(
        tmp_path,
        key="file",
        values=[f"f{i}.png" for i in range(15)],
        labels=["x", "y", "z"] * 5,
        files=[f"f{i}.png" for i in range(15)],
        sub="all",
    )
    frame = pd.read_csv(root / "labels.csv")
    frame["split"] = ["test" if i % 5 == 0 else "train" for i in range(15)]
    frame.to_csv(root / "labels.csv", index=False)
    inv = inventory(root)
    p = plan(inv)
    assert p.split == "column"
    assert p.split_column == "split"
    frames = apply(p, inv)
    assert frames.holdout is not None
    assert (len(frames.train), len(frames.holdout)) == (12, 3)


def test_coverage_between_the_thresholds_is_a_plan_with_a_note(tmp_path: Path) -> None:
    root = _table_and_images(
        tmp_path,
        key="id",
        values=[str(i) for i in range(20)],
        labels=["a", "b"] * 10,
        files=[f"{i}.jpg" for i in range(19)],  # one row has no file: 95%
    )
    inv = inventory(root)
    p = plan(inv)
    assert linking.PAUSE <= p.coverage < linking.ACCEPT
    assert p.notes
    assert "dropped" in p.notes[0]
    assert len(apply(p, inv).train) == 19


def test_coverage_under_the_floor_is_no_plan(tmp_path: Path) -> None:
    root = _table_and_images(
        tmp_path,
        key="id",
        values=[str(i) for i in range(20)],
        labels=["a", "b"] * 10,
        files=[f"{i}.jpg" for i in range(10)],  # 50%
    )
    with pytest.raises(LinkError, match="looks like the label table"):
        plan(inventory(root))


def test_a_key_that_matches_two_files_does_not_resolve(tmp_path: Path) -> None:
    root = _table_and_images(
        tmp_path,
        key="name",
        values=[f"pic{i}" for i in range(10)],
        labels=["a", "b"] * 5,
        files=[f"pic{i}.png" for i in range(10)],
    )
    for i in range(10):
        _png(root / "copies" / f"pic{i}.png", i)  # same stems again
    with pytest.raises(LinkError, match="looks like the label table"):
        plan(inventory(root))


# ─── flags ─────────────────────────────────────────────────────────────────


def test_flags_choose_the_table_and_its_columns(tmp_path: Path) -> None:
    root = _table_and_images(
        tmp_path,
        key="photo",
        values=[f"{i}.png" for i in range(10)],
        labels=["a", "b"] * 5,
        files=[f"{i}.png" for i in range(10)],
    )
    frame = pd.read_csv(root / "labels.csv")
    frame["notes"] = "free text"
    frame = frame.rename(columns={"label": "verdict"})
    frame.to_csv(root / "labels.csv", index=False)
    inv = inventory(root)
    with pytest.raises(LinkError):
        plan(inv)  # two candidate targets, no name the rules know
    p = plan(inv, labels=root / "labels.csv", key="photo", target="verdict")
    assert p.source == "flags"
    assert p.target_column == "verdict"
    assert "linked by: flags" in render(p, inv, apply(p, inv))


def test_a_flag_naming_a_missing_column_is_refused(tmp_path: Path) -> None:
    root = _table_and_images(
        tmp_path, key="id", values=["1", "2"], labels=["a", "b"], files=["1.png", "2.png"]
    )
    with pytest.raises(LinkError, match="no column 'nope'"):
        plan(inventory(root), labels=root / "labels.csv", key="nope")


# ─── what the review found ─────────────────────────────────────────────────


def test_a_blank_cell_in_any_column_does_not_crash_the_ladder(tmp_path: Path) -> None:
    root = _table_and_images(
        tmp_path,
        key="id",
        values=[str(i) for i in range(20)],
        labels=["a", "b"] * 10,
        files=[f"{i}.jpg" for i in range(20)],
    )
    frame = pd.read_csv(root / "labels.csv")
    frame["notes"] = ["free text"] * 19 + [None]
    frame.to_csv(root / "labels.csv", index=False)
    p = plan(inventory(root))
    assert p.key_column == "id"
    assert p.coverage == 1.0


def test_a_blank_id_turns_the_column_float_and_still_links(tmp_path: Path) -> None:
    root = _table_and_images(
        tmp_path,
        key="id",
        values=[str(i) for i in range(20)],
        labels=["a", "b"] * 10,
        files=[f"{i}.jpg" for i in range(20)],
    )
    frame = pd.read_csv(root / "labels.csv")
    frame.loc[3, "id"] = None
    frame.to_csv(root / "labels.csv", index=False)
    p = plan(inventory(root))
    assert p.key_column == "id"
    assert p.coverage == pytest.approx(0.95)


def test_two_tables_that_fit_equally_well_are_a_tie(tmp_path: Path) -> None:
    root = _table_and_images(
        tmp_path,
        key="id",
        values=[str(i) for i in range(20)],
        labels=["a", "b"] * 10,
        files=[f"{i}.jpg" for i in range(20)],
    )
    pd.DataFrame({"id": [str(i) for i in range(20)], "label": ["cat"] * 20}).to_csv(
        root / "sample_submission.csv", index=False
    )
    with pytest.raises(LinkError, match="both link the images equally well"):
        plan(inventory(root))


def test_two_key_columns_that_disagree_are_refused(tmp_path: Path) -> None:
    root = _table_and_images(
        tmp_path,
        key="id",
        values=[str(i) for i in range(10)],
        labels=["a", "b"] * 5,
        files=[f"{i}.jpg" for i in range(10)],
    )
    frame = pd.read_csv(root / "labels.csv")
    frame["image"] = [f"images/{9 - i}.jpg" for i in range(10)]  # the same files, reversed
    frame.to_csv(root / "labels.csv", index=False)
    inv = inventory(root)
    with pytest.raises(LinkError, match="point at different files; pass --key"):
        plan(inv)
    assert plan(inv, labels=root / "labels.csv", key="id").key_column == "id"


def test_duplicate_keys_do_not_count_and_do_not_double_an_image(tmp_path: Path) -> None:
    root = _table_and_images(
        tmp_path,
        key="id",
        values=[str(i) for i in range(60)] + ["0", "1"],
        labels=["a", "b"] * 31,
        files=[f"{i}.jpg" for i in range(60)],
    )
    p = plan(inventory(root))
    assert p.coverage == pytest.approx(58 / 62)
    frames = apply(p, inventory(root))
    assert frames.train["image"].is_unique
    assert len(frames.train) == 58


def test_a_split_column_with_a_third_value_is_refused(tmp_path: Path) -> None:
    root = _table_and_images(
        tmp_path,
        key="file",
        values=[f"f{i}.png" for i in range(12)],
        labels=["x", "y", "z"] * 4,
        files=[f"f{i}.png" for i in range(12)],
        sub="all",
    )
    frame = pd.read_csv(root / "labels.csv")
    frame["split"] = ["train", "val", "test"] * 4
    frame.to_csv(root / "labels.csv", index=False)
    with pytest.raises(LinkError, match="more than one holdout name"):
        plan(inventory(root))
    frame["split"] = ["train", "maybe", "test"] * 4
    frame.to_csv(root / "labels.csv", index=False)
    with pytest.raises(LinkError, match=r"also holds \['maybe'\]"):
        plan(inventory(root))


def test_rows_without_a_label_are_dropped_and_noted(tmp_path: Path) -> None:
    root = _table_and_images(
        tmp_path,
        key="id",
        values=[str(i) for i in range(20)],
        labels=["a", "b"] * 10,
        files=[f"{i}.jpg" for i in range(20)],
    )
    frame = pd.read_csv(root / "labels.csv")
    frame.loc[[2, 5], "label"] = None
    frame.to_csv(root / "labels.csv", index=False)
    inv = inventory(root)
    p = plan(inv)
    assert p.coverage == pytest.approx(0.9)
    assert any("no label" in n for n in p.notes)
    assert len(apply(p, inv).train) == 18


def test_a_table_that_never_reaches_the_holdout_folder_is_refused(tmp_path: Path) -> None:
    for i in range(20):
        _png(tmp_path / "train" / f"{i}.jpg", i)
    for i in range(100, 110):
        _png(tmp_path / "test" / f"{i}.jpg", i)
    pd.DataFrame({"id": [str(i) for i in range(20)], "label": ["a", "b"] * 10}).to_csv(
        tmp_path / "train.csv", index=False
    )
    inv = inventory(tmp_path)
    p = plan(inv)
    assert p.split == "folders"
    with pytest.raises(LinkError, match="leaves no holdout rows"):
        apply(p, inv)


def test_a_third_split_folder_is_noted_not_dropped_silently(tmp_path: Path) -> None:
    _class_tree(tmp_path / "train", per_class=3)
    _class_tree(tmp_path / "valid", per_class=2, seed=50)
    _class_tree(tmp_path / "test", per_class=2, seed=100)
    inv = inventory(tmp_path)
    assert inv.ignored == ["valid"]
    p = plan(inv)
    assert any("ignoring folder(s) ['valid']" in n for n in p.notes)


def test_a_class_folder_with_folders_inside_is_refused(tmp_path: Path) -> None:
    _class_tree(tmp_path)
    _png(tmp_path / "cat" / "closeups" / "extra.png", 7)
    with pytest.raises(LinkError, match="a class folder with folders inside"):
        plan(inventory(tmp_path))


def test_hidden_files_inside_a_class_folder_never_become_rows(tmp_path: Path) -> None:
    _class_tree(tmp_path)
    _png(tmp_path / "cat" / "._ghost.png", 9)
    inv = inventory(tmp_path)
    frames = apply(plan(inv), inv)
    assert not frames.train["image"].str.contains("._ghost").any()
    assert len(frames.train) == 12


def test_a_sibling_folder_whose_name_extends_train_is_not_train(tmp_path: Path) -> None:
    _class_tree(tmp_path / "train", per_class=3)
    _class_tree(tmp_path / "test", per_class=2, seed=100)
    _class_tree(tmp_path / "train_extra", per_class=1, seed=200)
    inv = inventory(tmp_path)
    frames = apply(plan(inv), inv)
    assert not frames.train["image"].str.contains("train_extra").any()
    assert len(frames.train) == 9


def test_the_schema_refuses_a_plan_whose_parts_disagree() -> None:
    from pydantic import ValidationError

    from iterate.schemas.link import LinkPlan

    with pytest.raises(ValidationError, match="needs split_column"):
        LinkPlan(
            shape="table_join",
            labels_file="x.csv",
            key_column="id",
            key_to_file="stem",
            target_column="label",
            task="classification",
            split="column",
            coverage=1.0,
        )
    with pytest.raises(ValidationError, match="not both"):
        LinkPlan(
            shape="table_join",
            labels_file="x.csv",
            key_column="id",
            key_to_file="stem",
            target_column="label",
            onehot_columns=["a", "b"],
            task="classification",
            coverage=1.0,
        )


# ─── refusal kinds, and a plan from named picks ────────────────────────────


def _unnamed_target(root: Path) -> Path:
    for i in range(12):
        _png(root / "img" / f"{i}.jpg", i)
    pd.DataFrame(
        {
            "file": [f"{i}.jpg" for i in range(12)],
            "region": ["n", "s"] * 6,
            "code": ["x", "y", "z"] * 4,
        }
    ).to_csv(root / "meta.csv", index=False)
    return root


def test_a_refusal_says_whether_a_choice_would_settle_it(tmp_path: Path) -> None:
    with pytest.raises(LinkError) as no_rule_with_table:
        plan(inventory(_unnamed_target(tmp_path / "a")))
    assert no_rule_with_table.value.ambiguous is True

    for i in range(6):
        _png(tmp_path / "b" / f"img{i}.png", i)
    with pytest.raises(LinkError) as no_rule_no_table:
        plan(inventory(tmp_path / "b"))
    assert no_rule_no_table.value.ambiguous is False

    (tmp_path / "c" / "data.parquet").parent.mkdir()
    (tmp_path / "c" / "data.parquet").write_bytes(b"PAR1")
    with pytest.raises(LinkError) as containers:
        plan(inventory(tmp_path / "c"))
    assert containers.value.ambiguous is False

    _class_tree(tmp_path / "d")
    _png(tmp_path / "d" / "cat" / "closeups" / "extra.png", 7)
    with pytest.raises(LinkError) as nested_no_table:
        plan(inventory(tmp_path / "d"))
    assert nested_no_table.value.ambiguous is False
    pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(tmp_path / "d" / "notes.csv", index=False)
    with pytest.raises(LinkError) as nested_with_table:
        plan(inventory(tmp_path / "d"))
    assert nested_with_table.value.ambiguous is True


def test_a_stray_split_value_is_a_fault_in_the_data_not_a_choice(tmp_path: Path) -> None:
    root = _table_and_images(
        tmp_path,
        key="id",
        values=[str(i) for i in range(10)],
        labels=["a", "b"] * 5,
        files=[f"{i}.jpg" for i in range(10)],
    )
    frame = pd.read_csv(root / "labels.csv")
    frame["split"] = ["train"] * 6 + ["test"] * 3 + ["maybe"]
    frame.to_csv(root / "labels.csv", index=False)
    with pytest.raises(LinkError, match="also holds") as caught:
        plan(inventory(root))
    assert caught.value.ambiguous is False


def test_a_plan_from_named_picks_is_measured_like_a_rules_plan(tmp_path: Path) -> None:
    root = _table_and_images(
        tmp_path,
        key="id",
        values=[str(i) for i in range(20)],
        labels=["a", "b"] * 10,
        files=[f"{i}.jpg" for i in range(20)],
    )
    inv = inventory(root)
    by_rules = plan(inv)
    by_choice = linking.plan_from_choice(
        inv,
        table=root / "labels.csv",
        key_column="id",
        key_to_file="stem",
        target_column="label",
        source="agent",
    )
    assert by_choice.model_dump(exclude={"source"}) == by_rules.model_dump(exclude={"source"})
    assert by_choice.source == "agent"
    assert apply(by_choice, inv).train.equals(apply(by_rules, inv).train)


def test_named_picks_are_checked_against_the_folder(tmp_path: Path) -> None:
    inv = inventory(_unnamed_target(tmp_path))
    table = tmp_path / "meta.csv"
    with pytest.raises(LinkError, match="is not a table under"):
        linking.plan_from_choice(
            inv,
            table=tmp_path / "other.csv",
            key_column="file",
            key_to_file="basename",
            target_column="code",
        )
    with pytest.raises(LinkError, match="has no column 'label'"):
        linking.plan_from_choice(
            inv, table=table, key_column="file", key_to_file="basename", target_column="label"
        )
    with pytest.raises(LinkError, match="not a way to read a key"):
        linking.plan_from_choice(
            inv, table=table, key_column="file", key_to_file="magic", target_column="code"
        )
    with pytest.raises(LinkError, match="cannot be the split and the key"):
        linking.plan_from_choice(
            inv,
            table=table,
            key_column="file",
            key_to_file="basename",
            target_column="code",
            split_column="file",
        )
    with pytest.raises(LinkError, match=r"resolves 0.0% of rows.*under the 90% floor"):
        linking.plan_from_choice(
            inv, table=table, key_column="file", key_to_file="stem_int", target_column="code"
        )


def test_join_rates_name_the_best_reading_per_column(tmp_path: Path) -> None:
    inv = inventory(_unnamed_target(tmp_path))
    rates = linking.join_rates(inv, tmp_path / "meta.csv")
    assert rates == {"file": ("basename", 1.0)}


def test_the_split_column_found_by_name_may_not_be_the_target(tmp_path: Path) -> None:
    root = tmp_path / "d"
    for i in range(24):
        _png(root / "img" / f"{i:03d}.jpg", i)
    pd.DataFrame(
        {
            "file": [f"{i:03d}.jpg" for i in range(24)],
            "split": ["train"] * 18 + ["test"] * 6,
            "region": ["n", "s"] * 12,
        }
    ).to_csv(root / "meta.csv", index=False)
    inv = inventory(root)
    with pytest.raises(LinkError, match="'split' cannot be the split and the key or target"):
        linking.plan_from_choice(
            inv,
            table=root / "meta.csv",
            key_column="file",
            key_to_file="basename",
            target_column="split",
        )
    ok = linking.plan_from_choice(
        inv,
        table=root / "meta.csv",
        key_column="file",
        key_to_file="basename",
        target_column="region",
    )
    assert (ok.target_column, ok.split_column) == ("region", "split")


def test_the_floor_is_measured_not_assumed(tmp_path: Path) -> None:
    root = _unnamed_target(tmp_path)
    inv = inventory(root)
    for i in range(4):
        (root / "img" / f"{i}.jpg").unlink()
    inv = inventory(root)
    with pytest.raises(LinkError, match=r"66\.7% of rows.*under the 90% floor"):
        linking.plan_from_choice(
            inv,
            table=root / "meta.csv",
            key_column="file",
            key_to_file="basename",
            target_column="code",
        )
    for i in range(3):
        _png(root / "img" / f"{i}.jpg", i)
    inv = inventory(root)
    p = linking.plan_from_choice(
        inv,
        table=root / "meta.csv",
        key_column="file",
        key_to_file="basename",
        target_column="code",
    )
    assert p.coverage == pytest.approx(11 / 12)
    assert p.notes == ["91.7% of rows resolved to an image; the rest are dropped"]


def test_named_picks_refuse_a_one_row_table_and_a_decimal_key(tmp_path: Path) -> None:
    root = _unnamed_target(tmp_path)
    inv = inventory(root)
    frame = pd.read_csv(root / "meta.csv")
    frame["weight"] = [0.5 + i * 0.25 for i in range(12)]
    frame.to_csv(root / "meta.csv", index=False)
    with pytest.raises(LinkError, match="holds decimals"):
        linking.plan_from_choice(
            inv,
            table=root / "meta.csv",
            key_column="weight",
            key_to_file="stem",
            target_column="code",
        )
    frame.head(1).to_csv(root / "meta.csv", index=False)
    with pytest.raises(LinkError, match="fewer than two rows"):
        linking.plan_from_choice(
            inv,
            table=root / "meta.csv",
            key_column="file",
            key_to_file="basename",
            target_column="code",
        )
