"""Tests for the tabular data adapter — deterministic, stratified, leakage-safe split."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pytest

from iterate.adapters.data.tabular import TabularDataset, load_csv

if TYPE_CHECKING:
    from pathlib import Path


def _make_csv(tmp_path: Path, *, n: int = 100, positives: int = 20) -> Path:
    """An imbalanced binary-classification CSV: `positives` ones out of `n` rows."""
    labels = [1] * positives + [0] * (n - positives)
    frame = pd.DataFrame(
        {
            "f1": range(n),
            "f2": [i * 0.5 for i in range(n)],
            "churn": labels,
        }
    )
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "data.csv"
    frame.to_csv(path, index=False)
    return path


def test_split_sizes_and_metadata(tmp_path: Path) -> None:
    ds = load_csv(_make_csv(tmp_path), target="churn", test_size=0.2, seed=42)
    assert isinstance(ds, TabularDataset)
    assert ds.n_train == 80
    assert ds.n_test == 20
    assert ds.target == "churn"
    assert ds.features == ["f1", "f2"]
    assert "churn" not in ds.train_features.columns  # target excluded from features


def test_split_is_deterministic_for_same_seed(tmp_path: Path) -> None:
    path = _make_csv(tmp_path)
    a = load_csv(path, target="churn", seed=42)
    b = load_csv(path, target="churn", seed=42)
    assert list(a.train_features.index) == list(b.train_features.index)
    assert list(a.test_features.index) == list(b.test_features.index)


def test_different_seed_gives_a_different_split(tmp_path: Path) -> None:
    path = _make_csv(tmp_path)
    a = load_csv(path, target="churn", seed=42)
    b = load_csv(path, target="churn", seed=7)
    assert list(a.test_features.index) != list(b.test_features.index)


def test_train_and_holdout_are_disjoint(tmp_path: Path) -> None:
    ds = load_csv(_make_csv(tmp_path), target="churn")
    assert set(ds.train_features.index).isdisjoint(set(ds.test_features.index))


def test_stratification_preserves_class_balance(tmp_path: Path) -> None:
    # 20% positives overall → 20% in both train and holdout when stratified.
    ds = load_csv(_make_csv(tmp_path, n=100, positives=20), target="churn", test_size=0.2)
    assert ds.test_target.mean() == pytest.approx(0.20)
    assert ds.train_target.mean() == pytest.approx(0.20)


def test_missing_target_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="target column 'nope' not in"):
        load_csv(_make_csv(tmp_path), target="nope")


def test_data_hash_is_stable_and_content_sensitive(tmp_path: Path) -> None:
    path = _make_csv(tmp_path)
    h1 = load_csv(path, target="churn").data_hash
    h2 = load_csv(path, target="churn").data_hash
    assert h1 == h2  # same data → same hash

    other = _make_csv(tmp_path / "sub", n=100, positives=30)  # different contents
    h3 = load_csv(other, target="churn").data_hash
    assert h3 != h1


def test_continuous_target_does_not_crash(tmp_path: Path) -> None:
    # A regression-style target (many unique floats) → stratify ignored, no error.
    frame = pd.DataFrame({"f1": range(50), "price": [i * 1.37 for i in range(50)]})
    path = tmp_path / "reg.csv"
    frame.to_csv(path, index=False)
    ds = load_csv(path, target="price", test_size=0.2)
    assert ds.n_train == 40
    assert ds.n_test == 10


def test_a_non_utf8_csv_loads_instead_of_raising(tmp_path: Path) -> None:
    """Found by a real dataset: `pd.read_csv` assumes UTF-8 and raises
    UnicodeDecodeError on anything else. Any European export with a euro sign or an
    accented name is latin-1, so the user's first contact with the tool would be a
    decoding traceback on a file that opens fine in every spreadsheet."""
    path = tmp_path / "latin1.csv"
    path.write_bytes(("a,b,label\n" + "".join(
        f"{i},caf\xe9{i},{i % 2}\n" for i in range(40)
    )).encode("latin-1"))
    with pytest.raises(UnicodeDecodeError):
        path.read_text(encoding="utf-8")  # the file really is not UTF-8
    ds = load_csv(path, target="label")
    assert ds.n_train + ds.n_test == 40
    assert "b" in ds.features


def test_an_integer_regression_target_loads_instead_of_stratifying(tmp_path: Path) -> None:
    """Found by the diamonds dataset: an integer target was always read as a class
    label, so 11,602 distinct prices became 11,602 "classes" and the stratified
    split raised before the run could start. Prices, counts and years are all
    integers, so this is the common case, not an edge one."""
    path = tmp_path / "prices.csv"
    frame = pd.DataFrame({"carat": range(200), "depth": range(200),
                          "price": [300 + i * 7 for i in range(200)]})
    frame.to_csv(path, index=False)
    ds = load_csv(path, target="price")           # would raise before the fix
    assert ds.n_train + ds.n_test == 200


def test_a_few_valued_integer_target_is_still_classification(tmp_path: Path) -> None:
    """The fix must not reclassify anything that already worked: 0/1 labels and
    small multiclass targets stay classification and stay stratified."""
    from iterate.adapters.data.tabular import looks_like_classification

    for values in ([0, 1] * 50, [0, 1, 2, 3] * 25, [True, False] * 50):
        assert looks_like_classification(pd.Series(values))
    assert not looks_like_classification(pd.Series(range(500)))


def test_a_smaller_holdout_keeps_the_same_records_every_time(tmp_path: Path) -> None:
    """Every candidate must be scored on IDENTICAL records or the comparison stops
    being paired, which is the only reason a 100-record slice is trustworthy for
    ranking at all."""
    from iterate.adapters.data.tabular import with_smaller_holdout

    frame = pd.DataFrame({"text": [f"row {i}" for i in range(200)], "y": ["a", "b"] * 100})
    path = tmp_path / "d.csv"
    frame.to_csv(path, index=False)
    dataset = load_csv(path, target="y")

    first = with_smaller_holdout(dataset, 20)
    second = with_smaller_holdout(dataset, 20)

    assert first.n_test == 20
    assert first.n_train == dataset.n_train  # training data is untouched
    assert list(first.test_features["text"]) == list(second.test_features["text"])
    assert list(first.test_target) == list(dataset.test_target[:20])


def test_asking_for_more_holdout_than_exists_is_a_no_op(tmp_path: Path) -> None:
    from iterate.adapters.data.tabular import with_smaller_holdout

    frame = pd.DataFrame({"x": list(range(100)), "y": ["a", "b"] * 50})
    path = tmp_path / "d.csv"
    frame.to_csv(path, index=False)
    dataset = load_csv(path, target="y")

    assert with_smaller_holdout(dataset, 10_000).n_test == dataset.n_test
    assert with_smaller_holdout(dataset, 0).n_test == dataset.n_test


# ─── the user's own split ────────────────────────────────────────────────


def _make_split(tmp_path: Path, *, n_train: int = 80, n_test: int = 20) -> tuple[Path, Path]:
    """Two CSVs the user split themselves, same columns, target in the middle."""
    tmp_path.mkdir(parents=True, exist_ok=True)

    def frame(n: int, offset: int) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "f1": range(offset, offset + n),
                "churn": [i % 2 for i in range(n)],
                "f2": [i * 0.5 for i in range(n)],
            }
        )

    train, holdout = tmp_path / "train.csv", tmp_path / "holdout.csv"
    frame(n_train, 0).to_csv(train, index=False)
    frame(n_test, 1000).to_csv(holdout, index=False)
    return train, holdout


def test_load_split_keeps_the_users_rows_exactly(tmp_path: Path) -> None:
    from iterate.adapters.data.tabular import load_split

    train, holdout = _make_split(tmp_path)
    ds = load_split(train, holdout, target="churn")
    assert ds.user_split is True
    assert (ds.n_train, ds.n_test) == (80, 20)
    assert ds.features == ["f1", "f2"]
    # Membership is exactly the user's; only the row order moves (fixed seed).
    assert sorted(ds.train_features["f1"]) == list(range(80))
    assert sorted(ds.test_features["f1"]) == list(range(1000, 1020))
    assert list(ds.test_features["f1"]) != list(range(1000, 1020))
    # Pairing survives the shuffle: the label was built as (f1 - offset) % 2.
    for f, y in zip(ds.test_features["f1"], ds.test_target, strict=True):
        assert int(y) == (int(f) - 1000) % 2
    for f, y in zip(ds.train_features["f1"], ds.train_target, strict=True):
        assert int(y) == int(f) % 2
    assert ds.test_size == pytest.approx(0.2)


def test_load_split_holdout_index_is_disjoint_from_train(tmp_path: Path) -> None:
    from iterate.adapters.data.tabular import load_split

    ds = load_split(*_make_split(tmp_path), target="churn")
    assert set(ds.train_features.index).isdisjoint(set(ds.test_features.index))


def test_load_split_reorders_holdout_columns_to_the_training_layout(tmp_path: Path) -> None:
    from iterate.adapters.data.tabular import load_split

    train, holdout = _make_split(tmp_path)
    shuffled = pd.read_csv(holdout)[["f2", "churn", "f1"]]
    shuffled.to_csv(holdout, index=False)
    ds = load_split(train, holdout, target="churn")
    assert list(ds.test_features.columns) == ["f1", "f2"]


def test_load_split_rejects_a_column_mismatch(tmp_path: Path) -> None:
    from iterate.adapters.data.tabular import load_split

    train, holdout = _make_split(tmp_path)
    frame = pd.read_csv(holdout).drop(columns=["f2"])
    frame["f3"] = 1
    frame.to_csv(holdout, index=False)
    with pytest.raises(ValueError, match=r"missing \['f2'\] and has extra \['f3'\]"):
        load_split(train, holdout, target="churn")


def test_load_split_rejects_a_missing_target_in_either_file(tmp_path: Path) -> None:
    from iterate.adapters.data.tabular import load_split

    train, holdout = _make_split(tmp_path)
    pd.read_csv(holdout).drop(columns=["churn"]).to_csv(holdout, index=False)
    with pytest.raises(ValueError, match="not in the holdout columns"):
        load_split(train, holdout, target="churn")
    train, holdout = _make_split(tmp_path / "again")
    pd.read_csv(train).drop(columns=["churn"]).to_csv(train, index=False)
    with pytest.raises(ValueError, match="not in the train columns"):
        load_split(train, holdout, target="churn")


def test_load_split_rejects_an_empty_label(tmp_path: Path) -> None:
    from iterate.adapters.data.tabular import load_split

    train, holdout = _make_split(tmp_path)
    frame = pd.read_csv(holdout)
    frame.loc[2, "churn"] = None
    frame.to_csv(holdout, index=False)
    with pytest.raises(ValueError, match="1 empty 'churn' value"):
        load_split(train, holdout, target="churn")


def test_load_split_takes_a_regression_target_without_a_class_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    from iterate.adapters.data.tabular import load_split

    train, holdout = _make_split(tmp_path)
    for path in (train, holdout):
        frame = pd.read_csv(path)
        frame["churn"] = frame["f1"] * 1.37 + 0.5
        frame.to_csv(path, index=False)
    with caplog.at_level("WARNING"):
        ds = load_split(train, holdout, target="churn")
    assert ds.user_split is True
    assert "absent from train" not in caplog.text


def test_a_label_sorted_user_holdout_still_gives_a_mixed_loop_slice(tmp_path: Path) -> None:
    from iterate.adapters.data.tabular import load_split, with_smaller_holdout

    train, holdout = _make_split(tmp_path)
    frame = pd.read_csv(holdout)
    frame["churn"] = [0] * 10 + [1] * 10
    frame.to_csv(holdout, index=False)
    ds = load_split(train, holdout, target="churn")
    sliced = with_smaller_holdout(ds, 8)
    assert set(sliced.test_target) == {0, 1}
    assert list(sliced.test_features.index) == list(ds.test_features.index[:8])


def test_load_split_warns_when_the_holdout_has_an_unseen_class(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    from iterate.adapters.data.tabular import load_split

    train, holdout = _make_split(tmp_path)
    frame = pd.read_csv(holdout)
    frame.loc[0, "churn"] = 7
    frame.to_csv(holdout, index=False)
    with caplog.at_level("WARNING"):
        ds = load_split(train, holdout, target="churn")
    assert ds.n_test == 20
    assert "absent from train" in caplog.text


def test_load_split_hash_covers_both_files(tmp_path: Path) -> None:
    from iterate.adapters.data.tabular import load_split

    train, holdout = _make_split(tmp_path)
    before = load_split(train, holdout, target="churn").data_hash
    frame = pd.read_csv(holdout)
    frame.loc[3, "f2"] = 99.0
    frame.to_csv(holdout, index=False)
    after_holdout = load_split(train, holdout, target="churn").data_hash
    assert after_holdout != before
    frame = pd.read_csv(train)
    frame.loc[3, "f2"] = 99.0
    frame.to_csv(train, index=False)
    assert load_split(train, holdout, target="churn").data_hash not in (before, after_holdout)
