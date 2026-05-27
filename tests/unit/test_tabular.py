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
