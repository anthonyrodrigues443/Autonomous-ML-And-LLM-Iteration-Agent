"""Tests for the image adapter: the CSV-of-paths and class-folder seams the vision
family rides on."""

from __future__ import annotations

from itertools import pairwise
from typing import TYPE_CHECKING

import pandas as pd
import pytest
from PIL import Image

from iterate.adapters.data.images import (
    IMAGE_COLUMN,
    LABEL_COLUMN,
    ImageColumn,
    content_hash,
    detect_image_column,
    file_hashes,
    frame_from_folder,
    load_image_folder,
    load_image_split,
    materialise,
    profile_images,
    resolve_paths,
    split_folders,
)
from iterate.adapters.data.tabular import load_csv

if TYPE_CHECKING:
    from pathlib import Path

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
