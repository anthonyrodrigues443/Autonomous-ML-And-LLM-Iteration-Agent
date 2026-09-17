"""Tests for the image adapter: the CSV-of-paths and class-folder seams the vision
family rides on."""

from __future__ import annotations

import hashlib
import os
import shlex
import shutil
import subprocess
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import pytest
from PIL import Image

from iterate.adapters.data.images import (
    IMAGE_COLUMN,
    LABEL_COLUMN,
    ImageColumn,
    OutsideDataError,
    content_hash,
    detect_image_column,
    file_hashes,
    frame_from_folder,
    image_column,
    load_image_folder,
    load_image_split,
    materialise,
    profile_images,
    resolve_paths,
    split_folders,
)
from iterate.adapters.data.tabular import load_csv, load_split
from tests.unit.image_fixtures import png

if TYPE_CHECKING:
    from collections.abc import Sequence

pytestmark = pytest.mark.unit

CLASSES = ("cat", "dog", "bird")


def _png(path: Path, *, size: tuple[int, int] = (12, 8), mode: str = "RGB", seed: int = 0) -> None:
    colour = 90 + seed % 100 if mode == "L" else (seed * 7 % 255, 40, 200)
    Image.new(mode, size, colour).save(path)


def _make_csv_dataset(tmp_path: Path, *, per_class: int = 8) -> Path:
    """A flat images/ folder plus data.csv with RELATIVE paths, three classes.

    One image per class is portrait, one is grayscale, so the profile has something
    to count; every other image is a 12x8 RGB PNG.
    """
    images = tmp_path / "images"
    images.mkdir()
    rows = []
    index = 0
    for label in CLASSES:
        for k in range(per_class):
            _png(
                images / f"{index:05d}.png",
                size=(8, 12) if k == 0 else (12, 8),
                mode="L" if k == 1 else "RGB",
                seed=index,
            )
            rows.append({IMAGE_COLUMN: f"images/{index:05d}.png", LABEL_COLUMN: label})
            index += 1
    path = tmp_path / "data.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _make_class_tree(root: Path, *, per_class: int = 8, seed: int = 0) -> Path:
    """The standard layout: one subfolder per class."""
    index = seed
    for label in CLASSES:
        (root / label).mkdir(parents=True)
        for _ in range(per_class):
            _png(root / label / f"{label}_{index}.png", seed=index)
            index += 1
    return root


def _loaded(tmp_path: Path) -> tuple[Path, ImageColumn, object]:
    path = _make_csv_dataset(tmp_path)
    dataset = load_csv(path, target=LABEL_COLUMN)
    column = detect_image_column(pd.read_csv(path), [IMAGE_COLUMN], path)
    assert column is not None
    return path, column, resolve_paths(dataset, column)


# ─── detection ─────────────────────────────────────────────────────────────


def test_the_image_column_is_detected_from_relative_paths(tmp_path: Path) -> None:
    path = _make_csv_dataset(tmp_path)
    column = detect_image_column(pd.read_csv(path), [IMAGE_COLUMN], path)
    assert column == ImageColumn(column=IMAGE_COLUMN, root=tmp_path.resolve())


def test_a_second_feature_column_makes_it_tabular(tmp_path: Path) -> None:
    path = _make_csv_dataset(tmp_path)
    frame = pd.read_csv(path)
    frame["extra"] = 1
    assert detect_image_column(frame, [IMAGE_COLUMN, "extra"], path) is None


def test_a_string_column_that_is_not_paths_is_tabular(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    pd.DataFrame({"product": ["Dell XPS", "HP Envy"], "label": [0, 1]}).to_csv(path, index=False)
    assert detect_image_column(pd.read_csv(path), ["product"], path) is None


def test_paths_that_do_not_exist_are_not_an_image_column(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    pd.DataFrame({"image": ["images/a.jpg", "images/b.jpg"], "label": [0, 1]}).to_csv(
        path, index=False
    )
    assert detect_image_column(pd.read_csv(path), ["image"], path) is None


def test_resolve_makes_every_path_absolute_and_keeps_the_split(tmp_path: Path) -> None:
    path = _make_csv_dataset(tmp_path)
    dataset = load_csv(path, target=LABEL_COLUMN)
    column = detect_image_column(pd.read_csv(path), [IMAGE_COLUMN], path)
    assert column is not None
    resolved = resolve_paths(dataset, column)
    for value in (*resolved.train_features[IMAGE_COLUMN], *resolved.test_features[IMAGE_COLUMN]):
        assert value.startswith(str(tmp_path.resolve()))
    assert list(resolved.train_features.index) == list(dataset.train_features.index)


# ─── folders ───────────────────────────────────────────────────────────────


def test_a_class_folder_tree_is_split_here(tmp_path: Path) -> None:
    root = _make_class_tree(tmp_path / "pets")
    dataset = load_image_folder(root)
    assert dataset.user_split is False
    assert dataset.n_train + dataset.n_test == 24
    assert set(dataset.train_target) == set(CLASSES)
    assert dataset.features == [IMAGE_COLUMN]
    assert all(p.startswith(str(root.resolve())) for p in dataset.train_features[IMAGE_COLUMN])


def test_a_folder_with_train_and_test_inside_is_the_users_split(tmp_path: Path) -> None:
    root = tmp_path / "pets"
    _make_class_tree(root / "train", per_class=6)
    _make_class_tree(root / "test", per_class=2, seed=100)
    assert split_folders(root) is not None
    dataset = load_image_folder(root)
    assert dataset.user_split is True
    assert dataset.n_train == 18
    assert dataset.n_test == 6
    assert all("/test/" in p for p in dataset.test_features[IMAGE_COLUMN])


def test_two_folders_are_the_users_split(tmp_path: Path) -> None:
    train = _make_class_tree(tmp_path / "a", per_class=5)
    holdout = _make_class_tree(tmp_path / "b", per_class=3, seed=50)
    dataset = load_image_split(train, holdout)
    assert dataset.user_split is True
    assert (dataset.n_train, dataset.n_test) == (15, 9)


def test_a_folder_without_images_is_refused(tmp_path: Path) -> None:
    (tmp_path / "empty" / "cat").mkdir(parents=True)
    with pytest.raises(ValueError, match="no images"):
        frame_from_folder(tmp_path / "empty")


def test_hidden_files_and_non_images_are_skipped(tmp_path: Path) -> None:
    root = _make_class_tree(tmp_path / "pets", per_class=2)
    (root / "cat" / ".DS_Store").write_bytes(b"junk")
    (root / "cat" / "notes.txt").write_text("not an image")
    assert len(frame_from_folder(root)) == 6


# ─── bytes ─────────────────────────────────────────────────────────────────


def test_content_hash_changes_when_one_image_changes(tmp_path: Path) -> None:
    _, column, dataset = _loaded(tmp_path)
    paths = [*dataset.train_features[IMAGE_COLUMN], *dataset.test_features[IMAGE_COLUMN]]
    before = content_hash(dataset, column, file_hashes(paths))
    assert before == content_hash(dataset, column, file_hashes(paths))
    _png(tmp_path / "images" / "00000.png", seed=999)
    after = content_hash(dataset, column, file_hashes(paths))
    assert after != before
    assert dataset.data_hash not in (before, after)  # the csv hash alone cannot see bytes


def _contiguous(labels: list[str]) -> bool:
    """True when every class sits in one unbroken block."""
    blocks = 1 + sum(1 for a, b in pairwise(labels) if a != b)
    return blocks == len(set(labels))


def test_materialise_hides_the_class_in_name_order_and_time(tmp_path: Path) -> None:
    from pathlib import Path as _Path

    root = tmp_path / "pets"
    _make_class_tree(root / "train", per_class=6)
    _make_class_tree(root / "test", per_class=4, seed=100)
    dataset = load_image_folder(root)
    column = ImageColumn(column=IMAGE_COLUMN, root=root)
    paths = [*dataset.train_features[IMAGE_COLUMN], *dataset.test_features[IMAGE_COLUMN]]
    hashes = file_hashes(paths)

    opaque, where = materialise(dataset, column, hashes, tmp_path / "cache")

    seen = [*opaque.train_features[IMAGE_COLUMN], *opaque.test_features[IMAGE_COLUMN]]
    for source, value in zip(paths, seen, strict=True):
        copy = _Path(value)
        assert copy.parent == where
        assert copy.suffix == ""
        digest = hashes[source]
        assert digest is not None
        assert copy.name == digest[:16]
        assert not any(label in copy.name for label in CLASSES)
        assert copy.read_bytes() == _Path(source).read_bytes()
        assert copy.stat().st_mtime == 0
        assert copy.stat().st_nlink == 1  # a copy, never a link to the source
    assert not _contiguous(list(opaque.train_target))
    assert not _contiguous(list(opaque.test_target))
    assert opaque.data_hash == where.name
    files_before = sorted(where.iterdir())
    again, where_again = materialise(dataset, column, hashes, tmp_path / "cache")
    assert where_again == where
    assert sorted(where.iterdir()) == files_before
    assert list(again.test_features[IMAGE_COLUMN]) == list(opaque.test_features[IMAGE_COLUMN])


def test_materialise_gives_a_missing_file_an_opaque_path_too(tmp_path: Path) -> None:
    from pathlib import Path as _Path

    root = _make_class_tree(tmp_path / "pets")
    dataset = load_image_folder(root)
    victim = dataset.test_features[IMAGE_COLUMN].iloc[0]
    _Path(victim).unlink()
    paths = [*dataset.train_features[IMAGE_COLUMN], *dataset.test_features[IMAGE_COLUMN]]
    opaque, where = materialise(
        dataset, column := ImageColumn(IMAGE_COLUMN, root), file_hashes(paths), tmp_path / "cache"
    )
    del column
    replacement = _Path(opaque.test_features[IMAGE_COLUMN].iloc[0])
    assert replacement.parent == where
    assert not replacement.exists()
    assert not any(label in replacement.name for label in CLASSES)
    seen = [*opaque.train_features[IMAGE_COLUMN], *opaque.test_features[IMAGE_COLUMN]]
    assert not any(str(root) in value for value in seen)


@pytest.mark.parametrize(
    ("train", "holdout"), [("train", "val"), ("train", "holdout"), ("Train", "Test")]
)
def test_every_split_folder_pair_is_recognised(tmp_path: Path, train: str, holdout: str) -> None:
    root = tmp_path / "pets"
    _make_class_tree(root / train, per_class=3)
    _make_class_tree(root / holdout, per_class=2, seed=100)
    dataset = load_image_folder(root)
    assert dataset.user_split is True
    assert (dataset.n_train, dataset.n_test) == (9, 6)


def test_a_sibling_folder_next_to_the_split_is_ignored_with_a_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    root = tmp_path / "pets"
    _make_class_tree(root / "train", per_class=3)
    _make_class_tree(root / "test", per_class=2, seed=100)
    _make_class_tree(root / "extra", per_class=1, seed=200)
    with caplog.at_level("WARNING"):
        dataset = load_image_folder(root)
    assert (dataset.n_train, dataset.n_test) == (9, 6)
    assert "ignoring ['extra']" in caplog.text


def test_every_folder_row_carries_its_own_parents_label(tmp_path: Path) -> None:
    from pathlib import Path as _Path

    root = _make_class_tree(tmp_path / "pets")
    dataset = load_image_folder(root)
    for frame, target in (
        (dataset.train_features, dataset.train_target),
        (dataset.test_features, dataset.test_target),
    ):
        for path, label in zip(frame[IMAGE_COLUMN], target, strict=True):
            assert _Path(path).parent.name == label


def test_absolute_paths_in_the_csv_are_detected_and_kept(tmp_path: Path) -> None:
    path = _make_csv_dataset(tmp_path)
    frame = pd.read_csv(path)
    frame[IMAGE_COLUMN] = [str((tmp_path / v).resolve()) for v in frame[IMAGE_COLUMN]]
    absolute = tmp_path / "absolute.csv"
    frame.to_csv(absolute, index=False)
    column = detect_image_column(pd.read_csv(absolute), [IMAGE_COLUMN], absolute)
    assert column is not None
    dataset = resolve_paths(load_csv(absolute, target=LABEL_COLUMN), column)
    assert list(dataset.train_features[IMAGE_COLUMN]) == [
        str(v) for v in dataset.train_features[IMAGE_COLUMN]
    ]
    assert all(v.startswith(str(tmp_path.resolve())) for v in dataset.train_features[IMAGE_COLUMN])


# ─── profile ───────────────────────────────────────────────────────────────


def test_the_profile_reports_classes_sizes_and_modes(tmp_path: Path) -> None:
    _, column, dataset = _loaded(tmp_path)
    paths = [*dataset.train_features[IMAGE_COLUMN], *dataset.test_features[IMAGE_COLUMN]]
    profile = profile_images(dataset, column, file_hashes(paths))
    assert profile.classes == 3
    assert profile.widths[0] == 8
    assert profile.widths[2] == 12
    assert profile.heights[2] == 12
    expected_modes = {}
    for p in dataset.train_features[IMAGE_COLUMN]:
        with Image.open(p) as im:
            expected_modes[im.mode] = expected_modes.get(im.mode, 0) + 1
    assert {m: round(share * dataset.n_train) for m, share in profile.modes} == expected_modes
    assert profile.portrait_share == pytest.approx(
        sum(
            1
            for p in dataset.train_features[IMAGE_COLUMN]
            if Image.open(p).height > Image.open(p).width
        )
        / dataset.n_train
    )
    assert dict(profile.class_balance).keys() == set(CLASSES)
    assert dict(profile.formats) == {"PNG": 1.0}
    assert profile.unreadable == 0
    assert profile.shared_across_split == 0
    text = profile.render()
    assert "Every row is one image" in text
    assert "Classes: 3" in text
    assert "PNG 100%" in text
    assert "cardinality" not in text


def test_unreadable_and_shared_images_are_counted(tmp_path: Path) -> None:
    _, column, dataset = _loaded(tmp_path)
    from pathlib import Path as _Path

    broken = _Path(dataset.train_features[IMAGE_COLUMN].iloc[0])
    broken.write_bytes(b"not an image")
    shared_source = _Path(dataset.test_features[IMAGE_COLUMN].iloc[0])
    shared_target = _Path(dataset.train_features[IMAGE_COLUMN].iloc[-1])
    shared_target.write_bytes(shared_source.read_bytes())
    missing = _Path(dataset.train_features[IMAGE_COLUMN].iloc[1])
    missing.unlink()

    paths = [*dataset.train_features[IMAGE_COLUMN], *dataset.test_features[IMAGE_COLUMN]]
    profile = profile_images(dataset, column, file_hashes(paths))
    assert profile.shared_across_split == 1
    assert profile.unreadable == 2


def test_a_numeric_score_profiles_as_regression_not_classes(tmp_path: Path) -> None:
    # Enough rows that the training split carries more than twenty distinct values,
    # the same rule that reads a numeric tabular target as regression.
    path = _make_csv_dataset(tmp_path, per_class=12)
    frame = pd.read_csv(path)
    frame[LABEL_COLUMN] = [1.0 + (i * 0.137) % 4.0 for i in range(len(frame))]
    frame.to_csv(path, index=False)
    dataset = load_csv(path, target=LABEL_COLUMN)
    column = detect_image_column(pd.read_csv(path), [IMAGE_COLUMN], path)
    assert column is not None
    dataset = resolve_paths(dataset, column)
    paths = [*dataset.train_features[IMAGE_COLUMN], *dataset.test_features[IMAGE_COLUMN]]
    profile = profile_images(dataset, column, file_hashes(paths))
    assert profile.classes == 0
    assert profile.class_balance == ()
    assert profile.target_spread is not None
    mean, std, low, high = profile.target_spread
    assert 1.0 <= low <= mean <= high < 5.0
    assert std > 0
    text = profile.render()
    assert "regression" in text
    assert "Spread: mean=" in text
    assert "Class balance" not in text


def test_the_image_profile_renders_facts_only_when_given() -> None:
    from iterate.adapters.data.images import ImageProfile

    base = {
        "n_train": 10,
        "n_test": 3,
        "column": "image",
        "classes": 2,
        "class_balance": (("a", 0.5), ("b", 0.5)),
        "target_spread": None,
        "widths": (12, 12, 12),
        "heights": (8, 8, 8),
        "portrait_share": 0.0,
        "modes": (("RGB", 1.0),),
        "formats": (("PNG", 1.0),),
        "unreadable": 0,
        "shared_across_split": 0,
    }
    plain = ImageProfile(**base).render()  # type: ignore[arg-type]
    assert "Data checks" not in plain
    told = ImageProfile(**base, facts=("2 holdout images left out as byte copies.",)).render()  # type: ignore[arg-type]
    assert told == plain + "\nData checks: 2 holdout images left out as byte copies."


# ─── prepare ──────────────────────────────────────────────────────────────


def _tiny_csv(root: Path, *, per_class: int = 8) -> Path:
    rows = []
    for k, c in enumerate(CLASSES):
        for i in range(per_class):
            path = root / "images" / f"{c}_{i}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            _png(path, seed=k * 30 + i)
            rows.append({"image": f"images/{path.name}", "label": c})
    csv = root / "data.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return csv


def test_prepare_copies_under_byte_names_outside_the_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from iterate.adapters.data.images import image_cache_dir, prepare_images

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert image_cache_dir() == tmp_path / "xdg" / "iterate" / "images"
    csv = _tiny_csv(tmp_path / "data")
    prepared = prepare_images(load_csv(csv, target="label"), csv)
    assert prepared.cache_dir.parent == tmp_path / "xdg" / "iterate" / "images"
    paths = [
        Path(str(p))
        for f in (prepared.dataset.train_features, prepared.dataset.test_features)
        for p in f["image"]
    ]
    assert {p.parent for p in paths} == {prepared.cache_dir}
    assert all(len(p.name) == 16 and "." not in p.name for p in paths)
    assert prepared.dataset.data_hash == prepared.cache_dir.name


def test_prepare_leaves_twins_out_of_a_split_it_made_and_not_out_of_a_users(
    tmp_path: Path,
) -> None:
    from iterate.adapters.data.images import prepare_images

    csv = _tiny_csv(tmp_path / "data")
    frame = pd.read_csv(csv)
    source = csv.parent / "images" / "cat_0.png"
    for i in range(6):
        copy = csv.parent / "images" / f"cat_copy_{i}.png"
        copy.write_bytes(source.read_bytes())
        frame.loc[len(frame)] = {"image": f"images/{copy.name}", "label": "cat"}
    frame.to_csv(csv, index=False)

    loaded = load_csv(csv, target="label")
    column = detect_image_column(loaded.train_features, loaded.features, csv)
    assert column is not None
    absolute = resolve_paths(loaded, column)
    hashes = file_hashes(
        [str(p) for f in (absolute.train_features, absolute.test_features) for p in f["image"]]
    )
    in_train = {hashes[str(p)] for p in absolute.train_features["image"]}
    expected = [str(p) for p in absolute.test_features["image"] if hashes[str(p)] in in_train]
    assert expected, "the fixture put no copy across the split"

    prepared = prepare_images(loaded, csv, into=tmp_path / "cache")
    assert list(prepared.dropped) == expected
    assert prepared.dataset.n_test == loaded.n_test - len(expected)
    assert f"{len(expected)} holdout images left out as byte copies of training images." in (
        prepared.dataset.facts
    )

    train = csv.parent / "train.csv"
    holdout = csv.parent / "holdout.csv"
    rows = pd.read_csv(csv)
    rows["image"] = rows["image"].map(lambda v: str(csv.parent / v))
    rows.iloc[:20].to_csv(train, index=False)
    rows.iloc[20:].to_csv(holdout, index=False)
    theirs = prepare_images(load_split(train, holdout, target="label"), train, into=tmp_path / "c2")
    assert theirs.dropped == ()
    assert theirs.dataset.n_test == len(rows) - 20


def test_prepare_refuses_a_holdout_made_only_of_copies(tmp_path: Path) -> None:
    from iterate.adapters.data.images import prepare_images

    root = tmp_path / "data"
    rows = []
    for k, c in enumerate(CLASSES):
        first = root / "images" / f"{c}_0.png"
        first.parent.mkdir(parents=True, exist_ok=True)
        _png(first, seed=k * 30)
        for i in range(8):
            path = root / "images" / f"{c}_{i}.png"
            if i:
                path.write_bytes(first.read_bytes())
            rows.append({"image": f"images/{path.name}", "label": c})
    pd.DataFrame(rows).to_csv(root / "data.csv", index=False)
    with pytest.raises(ValueError, match="leaves no holdout rows"):
        prepare_images(load_csv(root / "data.csv", target="label"), root / "data.csv", into=root)


def test_prepare_adds_the_monitor_brief_and_facts_after_the_ones_there(tmp_path: Path) -> None:
    from dataclasses import replace

    from iterate.adapters.data import monitor
    from iterate.adapters.data.images import prepare_images
    from iterate.schemas.monitor import DataReport

    csv = _tiny_csv(tmp_path / "data")
    report = DataReport(
        version="1",
        images=24,
        train_rows=19,
        holdout_rows=5,
        split="ours",
        seconds=0.0,
        findings=[],
    )
    monitor.save(report, csv.parent)
    loaded = replace(load_csv(csv, target="label"), facts=("first.",))
    prepared = prepare_images(loaded, csv, into=tmp_path / "cache", facts=("last.",))
    assert prepared.dataset.facts == ("first.", report.brief(), "last.")
    assert prepared.profile.render().splitlines()[-1].startswith("Data checks: first. Data checks")


@pytest.mark.parametrize(
    ("width", "height", "size"), [(64, 64, 64), (100, 120, 96), (500, 375, 160), (20, 20, 32)]
)
def test_the_default_size_follows_the_median_short_side(width: int, height: int, size: int) -> None:
    from iterate.adapters.data.images import ImageProfile, default_image_size

    profile = ImageProfile(
        n_train=10,
        n_test=3,
        column="image",
        classes=2,
        class_balance=(("a", 0.5), ("b", 0.5)),
        target_spread=None,
        widths=(width, width, width),
        heights=(height, height, height),
        portrait_share=0.0,
        modes=(("RGB", 1.0),),
        formats=(("PNG", 1.0),),
        unreadable=0,
        shared_across_split=0,
    )
    assert default_image_size(profile) == size


def test_the_profile_keeps_the_median_short_side_and_the_default_size_uses_it(
    tmp_path: Path,
) -> None:
    from iterate.adapters.data.images import ImageProfile, default_image_size, prepare_images

    csv = _tiny_csv(tmp_path / "data")
    prepared = prepare_images(load_csv(csv, target="label"), csv, into=tmp_path / "cache")
    assert prepared.profile.short_sides[1] == 8
    mixed = ImageProfile(
        n_train=4,
        n_test=1,
        column="image",
        classes=2,
        class_balance=(("a", 0.5), ("b", 0.5)),
        target_spread=None,
        widths=(100, 200, 300),
        heights=(100, 200, 300),
        portrait_share=0.5,
        modes=(("RGB", 1.0),),
        formats=(("PNG", 1.0),),
        unreadable=0,
        shared_across_split=0,
        short_sides=(100, 100, 100),
    )
    assert default_image_size(mixed) == 96


def test_a_relative_cache_home_is_ignored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from iterate.adapters.data.images import image_cache_dir

    monkeypatch.setenv("XDG_CACHE_HOME", "relative/cache")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert image_cache_dir() == tmp_path / "home" / ".cache" / "iterate" / "images"


def test_prepare_refuses_a_csv_without_an_image_column(tmp_path: Path) -> None:
    from iterate.adapters.data.images import prepare_images

    csv = tmp_path / "t.csv"
    pd.DataFrame({"a": range(10), "label": ["x", "y"] * 5}).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="does not hold image paths"):
        prepare_images(load_csv(csv, target="label"), csv, into=tmp_path / "cache")


def test_a_path_named_on_both_sides_is_a_twin_even_when_the_file_is_missing(
    tmp_path: Path,
) -> None:
    from dataclasses import replace

    from iterate.adapters.data.images import _without_twins

    csv = _tiny_csv(tmp_path / "data")
    loaded = load_csv(csv, target="label")
    column = detect_image_column(loaded.train_features, loaded.features, csv)
    assert column is not None
    resolved = resolve_paths(loaded, column)
    gone = str(csv.parent / "images" / "gone.png")
    train = resolved.train_features.copy()
    holdout = resolved.test_features.copy()
    train.iloc[0, 0] = gone
    holdout.iloc[0, 0] = gone
    dataset = replace(resolved, train_features=train, test_features=holdout)
    hashes = file_hashes([str(p) for f in (train, holdout) for p in f["image"]])
    kept, dropped = _without_twins(dataset, column, hashes)
    assert dropped == [gone]
    assert kept.n_test == resolved.n_test - 1


# ─── only the data given ──────────────────────────────────────────────────


def _rows_csv(
    folder: Path, extra: Sequence[str] = (), *, n: int = 12, name: str = "data.csv"
) -> Path:
    rows = []
    for i in range(n):
        png(folder / "images" / f"{i:03d}.png", i, klass=i % 2)
        rows.append({"image": f"images/{i:03d}.png", "label": "ab"[i % 2]})
    rows += [{"image": v, "label": "a"} for v in extra]
    pd.DataFrame(rows).to_csv(folder / name, index=False)
    return folder / name


@pytest.mark.parametrize(
    ("shape", "says"),
    [
        ("absolute", "put those images under that folder"),
        ("dotdot", "put those images under that folder"),
        ("link", "images/leak.png is a link to"),
        ("dotdot_after_link", "sub is a link to"),
    ],
)
def test_a_row_outside_the_csv_folder_is_refused_before_any_copy(
    tmp_path: Path, shape: str, says: str
) -> None:
    from iterate.adapters.data.images import prepare_images

    secret = tmp_path / "elsewhere" / "images" / "secret.png"
    png(secret, 99, klass=1)
    folder = tmp_path / "data"
    (folder / "images").mkdir(parents=True)
    value = {
        "absolute": str(secret),
        "dotdot": "../elsewhere/images/secret.png",
        "link": "images/leak.png",
        "dotdot_after_link": "sub/../images/secret.png",
    }[shape]
    if shape == "link":
        (folder / "images" / "leak.png").symlink_to(secret)
    if shape == "dotdot_after_link":
        (tmp_path / "elsewhere" / "deep").mkdir()
        (folder / "sub").symlink_to(tmp_path / "elsewhere" / "deep")
    csv = _rows_csv(folder, [value])
    with pytest.raises(
        OutsideDataError, match=r"1 image path.* lead outside .*the folder this"
    ) as no:
        prepare_images(load_csv(csv, target="label"), csv, into=tmp_path / "cache")
    assert says in str(no.value)
    assert not (tmp_path / "cache").exists()


def test_a_linked_images_folder_is_named_and_the_copy_command_pastes(tmp_path: Path) -> None:
    big = tmp_path / "big disk" / "eurosat"
    _rows_csv(big)
    project = tmp_path / "My Drive" / "project"
    project.mkdir(parents=True)
    (project / "images").symlink_to(big / "images")
    shutil.copyfile(big / "data.csv", project / "data.csv")
    with pytest.raises(OutsideDataError) as no:
        image_column(load_csv(project / "data.csv", target="label"))
    message = str(no.value)
    assert "12 image path(s) lead outside" in message
    assert f"images is a link to {(big / 'images').resolve()}" in message

    command = no.value.command
    assert command is not None
    assert message.endswith(f"and pass the copy: {command}")
    copy = f"{project.resolve()}-copy"
    assert shlex.split(command)[-2:] == [str(project.resolve()), copy]
    subprocess.run(command, shell=True, check=True)
    assert not Path(copy, "images").is_symlink()
    assert image_column(load_csv(Path(copy, "data.csv"), target="label")) is not None


def test_absolute_rows_through_a_link_out_get_advice_that_works_and_no_copy_command(
    tmp_path: Path,
) -> None:
    big = tmp_path / "big disk" / "eurosat"
    _rows_csv(big)
    project = tmp_path / "My Drive" / "project"
    project.mkdir(parents=True)
    (project / "images").symlink_to(big / "images")
    frame = pd.read_csv(big / "data.csv")
    relative = frame.copy()
    frame["image"] = [str(project / v) for v in frame["image"]]
    frame.to_csv(project / "data.csv", index=False)
    with pytest.raises(OutsideDataError) as no:
        image_column(load_csv(project / "data.csv", target="label"))
    message = str(no.value)
    assert f"images is a link to {(big / 'images').resolve()}" in message
    assert no.value.command is None
    assert "cp " not in message
    assert "put the CSV beside the images it names" in message
    assert f"write its image paths relative to {project.resolve()}" in message

    shutil.copyfile(project / "data.csv", big / "beside.csv")
    assert image_column(load_csv(big / "beside.csv", target="label")) is not None

    relative.to_csv(project / "data.csv", index=False)
    with pytest.raises(OutsideDataError) as again:
        image_column(load_csv(project / "data.csv", target="label"))
    assert again.value.command is not None
    subprocess.run(again.value.command, shell=True, check=True)
    copy = Path(f"{project.resolve()}-copy", "data.csv")
    assert image_column(load_csv(copy, target="label")) is not None


@pytest.mark.parametrize(
    ("rows", "link"),
    [
        (["../x.png", "images/1.png"], "images"),
        (["sub/../../x.png"], None),
        (["../other/1.png"], None),
    ],
)
def test_a_refusal_names_the_first_link_out_along_any_refused_row(
    tmp_path: Path, rows: list[str], link: str | None
) -> None:
    from iterate.adapters.data.images import refuse_outside

    base = tmp_path.resolve()
    project = base / "project"
    (project / "real").mkdir(parents=True)
    png(base / "x.png", 1)
    png(base / "pool" / "1.png", 2)
    (project / "sub").symlink_to(project / "real")
    (project / "images").symlink_to(base / "pool")
    (base / "other").symlink_to(base / "pool")
    with pytest.raises(OutsideDataError) as no:
        refuse_outside(rows, project / "data.csv")
    message = str(no.value)
    if link is None:
        assert "is a link to" not in message
        assert no.value.command is None
    else:
        assert f": {link} is a link to {base / 'pool'}." in message
        assert no.value.command is not None


def test_absolute_rows_spelled_through_a_linked_folder_are_inside(tmp_path: Path) -> None:
    from iterate.adapters.data.images import prepare_images

    real = tmp_path / "real"
    (tmp_path / "link").symlink_to(real, target_is_directory=True)
    csv = _rows_csv(real)
    frame = pd.read_csv(csv)
    frame["image"] = [str(tmp_path / "link" / v) for v in frame["image"]]
    frame.to_csv(csv, index=False)
    shown = tmp_path / "link" / "data.csv"
    prepared = prepare_images(load_csv(shown, target="label"), shown, into=tmp_path / "cache")
    assert prepared.profile.unreadable == 0


def test_a_folder_spelled_in_another_case_is_inside_on_a_case_folding_disk(
    tmp_path: Path,
) -> None:
    csv = _rows_csv(tmp_path / "data")
    upper = tmp_path / "DATA"
    if not upper.exists():
        pytest.skip("this disk tells case apart")
    frame = pd.read_csv(csv)
    frame["image"] = [str(tmp_path / "data" / v) for v in frame["image"]]
    frame.to_csv(csv, index=False)
    assert image_column(load_csv(upper / "data.csv", target="label")) is not None


def test_a_csv_that_is_a_link_reads_from_the_folder_it_leads_to(tmp_path: Path) -> None:
    from iterate.adapters.data.images import prepare_images

    csv = _rows_csv(tmp_path / "real")
    shown = tmp_path / "shown" / "data.csv"
    shown.parent.mkdir()
    shown.symlink_to(csv)
    prepared = prepare_images(load_csv(shown, target="label"), shown, into=tmp_path / "cache")
    assert prepared.profile.unreadable == 0


def test_the_frame_holds_the_path_that_was_checked(tmp_path: Path) -> None:
    folder = tmp_path / "data"
    csv = _rows_csv(folder)
    (folder / "images" / "again.png").symlink_to(folder / "images" / "000.png")
    frame = pd.read_csv(csv)
    frame.loc[len(frame)] = {"image": "images/./again.png", "label": "a"}
    frame.to_csv(csv, index=False)
    loaded = load_csv(csv, target="label")
    column = image_column(loaded)
    assert column is not None
    resolved = resolve_paths(loaded, column)
    paths = {*resolved.train_features["image"], *resolved.test_features["image"]}
    assert str(folder.resolve() / "images" / "again.png") not in paths
    assert not any(os.path.islink(p) for p in paths)


def test_a_row_swapped_for_a_link_out_after_the_check_is_refused_at_the_write(
    tmp_path: Path,
) -> None:
    secret = tmp_path / "elsewhere" / "secret.png"
    png(secret, 99)
    csv = _rows_csv(tmp_path / "data")
    loaded = load_csv(csv, target="label")
    column = image_column(loaded)
    assert column is not None
    (csv.parent / "images" / "003.png").unlink()
    (csv.parent / "images" / "003.png").symlink_to(secret)
    with pytest.raises(OutsideDataError, match="1 image path"):
        resolve_paths(loaded, column)


def test_each_side_of_a_users_split_resolves_against_its_own_csv(tmp_path: Path) -> None:
    from iterate.adapters.data.images import prepare_images

    train = _rows_csv(tmp_path / "train", name="train.csv")
    holdout = _rows_csv(tmp_path / "holdout", n=6, name="holdout.csv")
    for p in (tmp_path / "holdout" / "images").iterdir():
        png(p, 100 + int(p.stem), klass=int(p.stem) % 2)
    prepared = prepare_images(
        load_split(train, holdout, target="label"), train, into=tmp_path / "cache"
    )
    assert prepared.profile.shared_across_split == 0
    holdout_bytes = {
        hashlib.sha256(p.read_bytes()).hexdigest()[:16]
        for p in (tmp_path / "holdout" / "images").iterdir()
    }
    assert {Path(p).name for p in prepared.dataset.test_features["image"]} == holdout_bytes


def test_a_holdout_csv_naming_the_training_folder_is_refused(tmp_path: Path) -> None:
    train = _rows_csv(tmp_path / "train", name="train.csv")
    (tmp_path / "holdout").mkdir()
    pd.DataFrame(
        {"image": ["../train/images/000.png", "../train/images/001.png"], "label": ["a", "b"]}
    ).to_csv(tmp_path / "holdout" / "h.csv", index=False)
    with pytest.raises(OutsideDataError, match=r"h\.csv: 2 image path"):
        image_column(load_split(train, tmp_path / "holdout" / "h.csv", target="label"))


def test_a_holdout_csv_written_beside_the_training_csv_is_refused_with_the_fix(
    tmp_path: Path,
) -> None:
    train = _rows_csv(tmp_path, name="train.csv")
    for i in range(8):
        png(tmp_path / "test_images" / f"{i}.png", 50 + i, klass=i % 2)
    holdout = tmp_path / "test" / "holdout.csv"
    holdout.parent.mkdir()
    pd.DataFrame(
        {"image": [f"test_images/{i}.png" for i in range(8)], "label": ["a", "b"] * 4}
    ).to_csv(holdout, index=False)
    with pytest.raises(OutsideDataError) as no:
        image_column(load_split(train, holdout, target="label"))
    message = str(no.value)
    assert f"{holdout.resolve()}: 8 of its first 8 image paths do not exist" in message
    assert f"in {holdout.parent.resolve()}" in message
    assert f"they exist beside {train.resolve()}" in message

    pd.DataFrame(
        {"image": [f"../test_images/{i}.png" for i in range(8)], "label": ["a", "b"] * 4}
    ).to_csv(holdout, index=False)
    with pytest.raises(OutsideDataError, match="lead outside"):
        image_column(load_split(train, holdout, target="label"))

    png(tmp_path / "test" / "test_images" / "0.png", 50)
    pd.DataFrame({"image": ["test_images/0.png", "gone.png"], "label": ["a", "b"]}).to_csv(
        holdout, index=False
    )
    with pytest.raises(OutsideDataError) as missing:
        image_column(load_split(train, holdout, target="label"))
    assert "1 of its first 2 image paths do not exist" in str(missing.value)
    assert "they exist beside" not in str(missing.value)


def test_a_dataset_no_loader_made_needs_its_csv(tmp_path: Path) -> None:
    from dataclasses import replace

    csv = _rows_csv(tmp_path / "data")
    loaded = replace(load_csv(csv, target="label"), sources=None)
    with pytest.raises(ValueError, match="needs the CSV it came from"):
        image_column(loaded)
    assert image_column(loaded, csv) is not None


def test_a_monitor_report_that_is_a_link_is_not_read(tmp_path: Path) -> None:
    from iterate.adapters.data import monitor
    from iterate.adapters.data.images import prepare_images
    from iterate.schemas.monitor import DataReport

    csv = _tiny_csv(tmp_path / "data")
    report = DataReport(
        version="1",
        images=24,
        train_rows=19,
        holdout_rows=5,
        split="ours",
        seconds=0.0,
        findings=[],
    )
    elsewhere = monitor.save(report, tmp_path / "elsewhere")
    beside = csv.parent / monitor.REPORT_JSON
    shutil.copyfile(elsewhere, beside)
    read = prepare_images(load_csv(csv, target="label"), csv, into=tmp_path / "cache")
    assert read.dataset.facts == (report.brief(),)

    beside.unlink()
    beside.symlink_to(elsewhere)
    linked = prepare_images(load_csv(csv, target="label"), csv, into=tmp_path / "cache")
    assert linked.dataset.facts == ()
