"""Tests for the tabular ModelTarget — baseline + run, leakage-safe, metric panel."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pytest

from iterate.adapters.data.tabular import load_csv
from iterate.schemas.experiment import Candidate, ExperimentResult
from iterate.targets.base import BenchmarkTarget
from iterate.targets.model import ModelTarget

if TYPE_CHECKING:
    from pathlib import Path


def _classification_csv(tmp_path: Path) -> Path:
    # Learnable signal (target depends on `num`) + a categorical column to exercise preprocessing.
    n = 120
    frame = pd.DataFrame(
        {
            "num": [i % 10 for i in range(n)],
            "cat": (["a", "b", "c"] * (n // 3 + 1))[:n],
            "churn": [1 if (i % 10) >= 6 else 0 for i in range(n)],
        }
    )
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "clf.csv"
    frame.to_csv(path, index=False)
    return path


def _regression_csv(tmp_path: Path) -> Path:
    n = 120
    frame = pd.DataFrame({"x": range(n), "price": [i * 2.0 + 3 for i in range(n)]})
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "reg.csv"
    frame.to_csv(path, index=False)
    return path


def test_model_target_satisfies_the_protocol(tmp_path: Path) -> None:
    ds = load_csv(_classification_csv(tmp_path), target="churn")
    assert isinstance(ModelTarget(ds, metric="f1"), BenchmarkTarget)


def test_baseline_returns_a_metrics_panel(tmp_path: Path) -> None:
    ds = load_csv(_classification_csv(tmp_path), target="churn")
    result = ModelTarget(ds, metric="f1").baseline()
    assert isinstance(result, ExperimentResult)
    assert result.metrics is not None
    assert result.metrics.primary == "f1"
    assert result.metrics.direction == "maximize"
    assert set(result.metrics.values) == {"accuracy", "f1", "precision", "recall"}
    assert result.metrics.n_samples == ds.n_test


def test_baseline_handles_categorical_features(tmp_path: Path) -> None:
    # The string 'cat' column must be preprocessed (not crash).
    ds = load_csv(_classification_csv(tmp_path), target="churn")
    result = ModelTarget(ds, metric="accuracy").baseline()
    assert result.metrics is not None
    assert 0.0 <= result.metrics.primary_value <= 1.0


def test_run_applies_candidate_hyperparams(tmp_path: Path) -> None:
    ds = load_csv(_classification_csv(tmp_path), target="churn")
    candidate = Candidate(
        description="shallower trees",
        changes={"max_depth": 2},
        rationale="reduce overfitting",
    )
    result = ModelTarget(ds, metric="f1").run(candidate)
    assert isinstance(result, ExperimentResult)
    assert result.metrics is not None
    assert result.experiment_id == candidate.id


def test_regression_metric_direction_and_panel(tmp_path: Path) -> None:
    ds = load_csv(_regression_csv(tmp_path), target="price")
    result = ModelTarget(ds, metric="rmse").baseline()
    assert result.metrics is not None
    assert result.metrics.direction == "minimize"
    assert set(result.metrics.values) == {"rmse", "mae", "mse", "r2"}


def test_unknown_metric_raises(tmp_path: Path) -> None:
    ds = load_csv(_classification_csv(tmp_path), target="churn")
    with pytest.raises(ValueError, match="unknown metric"):
        ModelTarget(ds, metric="bleu")
