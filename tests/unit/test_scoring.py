"""The single-ruler contract: what `core.scoring` computes and when it refuses."""

from __future__ import annotations

import math

import numpy as np
import pytest

from iterate.core.scoring import (
    CLASSIFICATION_METRICS,
    LABEL_METRICS,
    PROBA_METRICS,
    REGRESSION_METRICS,
    direction,
    requires_proba,
    score,
    task_for_metric,
)

BINARY_TRUE = [0, 1, 0, 1, 1, 0]
BINARY_PRED = [0, 1, 0, 1, 0, 0]
BINARY_PROBA = [0.1, 0.9, 0.2, 0.8, 0.4, 0.3]

MULTI_TRUE = [0, 1, 2, 1, 0, 2]
MULTI_PRED = [0, 1, 2, 1, 0, 1]
MULTI_PROBA = [
    [0.8, 0.1, 0.1],
    [0.1, 0.8, 0.1],
    [0.1, 0.1, 0.8],
    [0.2, 0.7, 0.1],
    [0.6, 0.3, 0.1],
    [0.2, 0.5, 0.3],
]


def test_proba_metrics_are_classification_metrics() -> None:
    assert PROBA_METRICS <= CLASSIFICATION_METRICS
    assert LABEL_METRICS <= CLASSIFICATION_METRICS
    for metric in PROBA_METRICS:
        assert task_for_metric(metric) == "classification"


def test_log_loss_and_brier_minimize_roc_auc_maximizes() -> None:
    assert direction("log_loss") == "minimize"
    assert direction("brier") == "minimize"
    assert direction("roc_auc") == "maximize"


def test_requires_proba_only_for_the_probability_panel() -> None:
    assert all(requires_proba(m) for m in PROBA_METRICS)
    assert not any(requires_proba(m) for m in LABEL_METRICS | REGRESSION_METRICS)


def test_label_panel_unchanged_when_no_probabilities_given() -> None:
    values = score("classification", BINARY_TRUE, BINARY_PRED)
    assert set(values) == LABEL_METRICS


def test_binary_probabilities_add_the_full_panel() -> None:
    values = score("classification", BINARY_TRUE, BINARY_PRED, y_proba=BINARY_PROBA)
    assert set(values) == CLASSIFICATION_METRICS
    # Label metrics are still computed, so history stays comparable across
    # iterations that did and didn't emit probabilities.
    labels_only = score("classification", BINARY_TRUE, BINARY_PRED)
    for name, value in labels_only.items():
        assert values[name] == value


def test_binary_accepts_a_two_column_probability_matrix() -> None:
    matrix = [[1 - p, p] for p in BINARY_PROBA]
    from_matrix = score("classification", BINARY_TRUE, BINARY_PRED, y_proba=matrix)
    from_column = score("classification", BINARY_TRUE, BINARY_PRED, y_proba=BINARY_PROBA)
    assert from_matrix == pytest.approx(from_column)


def test_multiclass_panel_omits_brier() -> None:
    values = score("classification", MULTI_TRUE, MULTI_PRED, y_proba=MULTI_PROBA)
    assert "roc_auc" in values
    assert "log_loss" in values
    # Multiclass Brier landed after the scikit-learn>=1.5 floor this package ships.
    assert "brier" not in values


def test_confidently_wrong_probabilities_stay_finite() -> None:
    """`Metrics` rejects non-finite values, so an unclipped inf here would sink
    an otherwise-valid experiment. sklearn clips log-loss; assert it stays true."""
    values = score("classification", [0, 1], [0, 1], y_proba=[1.0, 0.0])
    assert math.isfinite(values["log_loss"])
    assert values["log_loss"] > 0


def test_regression_ignores_probabilities() -> None:
    y_true = [1.0, 2.0, 3.0, 4.0]
    y_pred = [1.1, 1.9, 3.2, 3.8]
    assert score("regression", y_true, y_pred, y_proba=[0.5] * 4) == score(
        "regression", y_true, y_pred
    )


def test_regression_panel_is_the_four_known_metrics() -> None:
    values = score("regression", [1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert set(values) == REGRESSION_METRICS
    assert values["rmse"] == pytest.approx(math.sqrt(values["mse"]))


@pytest.mark.parametrize(
    "proba",
    [
        [[0.1, 0.2, 0.7]] * 6,  # three columns for a binary target
        np.zeros((6, 3, 2)).tolist(),  # 3-D
    ],
)
def test_malformed_binary_probabilities_raise(proba: object) -> None:
    with pytest.raises(ValueError, match="probabilities"):
        score("classification", BINARY_TRUE, BINARY_PRED, y_proba=proba)


def test_multiclass_probabilities_need_one_column_per_class() -> None:
    with pytest.raises(ValueError, match="one column per class"):
        score("classification", MULTI_TRUE, MULTI_PRED, y_proba=[[0.5, 0.5]] * 6)
    with pytest.raises(ValueError, match="one column per class"):
        score("classification", MULTI_TRUE, MULTI_PRED, y_proba=[0.5] * 6)


def test_binary_string_labels_score_without_probabilities() -> None:
    """Regression: the label panel used to inherit sklearn's pos_label=1 default,
    so any binary string target (the common Yes/No churn encoding) raised."""
    values = score("classification", ["no", "yes", "no", "yes"], ["no", "yes", "no", "no"])
    assert set(values) == LABEL_METRICS
    assert values["recall"] == pytest.approx(0.5)


def test_string_labels_score_against_the_greater_class_as_positive() -> None:
    y_true = ["no", "yes", "no", "yes"]
    y_pred = ["no", "yes", "no", "no"]
    values = score("classification", y_true, y_pred, y_proba=[0.1, 0.9, 0.2, 0.4])
    assert values["roc_auc"] == pytest.approx(1.0)
    assert math.isfinite(values["brier"])


def test_unknown_metric_names_the_known_ones() -> None:
    with pytest.raises(ValueError, match="roc_auc"):
        task_for_metric("nonsense")
