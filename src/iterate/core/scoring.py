"""Metric scoring — the single ruler every execution path is judged by.

Both the spec path (`ModelTarget`) and the code-gen path (`core.codegen`) score
through these functions, so "improvement" stays an apples-to-apples comparison
no matter how a candidate was run. Keep all metric computation here; nothing
should reimplement it.

Deliberate v0.1 limit: a fixed 8-metric panel, scored on predicted labels (no
probability-based metrics like ROC-AUC / log-loss yet). Tracked with its removal
target (v0.4) in LIMITATIONS.md.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Literal

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

if TYPE_CHECKING:
    from iterate.adapters.models.registry import Task

CLASSIFICATION_METRICS = frozenset({"accuracy", "f1", "precision", "recall"})
REGRESSION_METRICS = frozenset({"rmse", "mae", "mse", "r2"})
_MINIMIZE = frozenset({"rmse", "mae", "mse"})


def task_for_metric(metric: str) -> Task:
    if metric in REGRESSION_METRICS:
        return "regression"
    if metric in CLASSIFICATION_METRICS:
        return "classification"
    known = sorted(CLASSIFICATION_METRICS | REGRESSION_METRICS)
    raise ValueError(f"unknown metric {metric!r}; expected one of {known}")


def direction(metric: str) -> Literal["maximize", "minimize"]:
    return "minimize" if metric in _MINIMIZE else "maximize"


def score(task: Task, y_true: Any, y_pred: Any) -> dict[str, float]:
    if task == "regression":
        mse = float(mean_squared_error(y_true, y_pred))
        return {
            "rmse": math.sqrt(mse),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "mse": mse,
            "r2": float(r2_score(y_true, y_pred)),
        }
    average = "binary" if len(set(y_true)) <= 2 else "macro"
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
    }


__all__ = [
    "CLASSIFICATION_METRICS",
    "REGRESSION_METRICS",
    "direction",
    "score",
    "task_for_metric",
]
