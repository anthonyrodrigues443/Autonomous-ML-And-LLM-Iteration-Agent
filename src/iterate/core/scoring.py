"""Metric scoring — the single ruler every execution path is judged by.

Both the spec path (`ModelTarget`) and the code-gen path (`core.codegen`) score
through these functions, so "improvement" stays an apples-to-apples comparison
no matter how a candidate was run. Keep all metric computation here; nothing
should reimplement it.

Probability metrics (v0.4) are a bonus panel: pass `y_proba` and the classification
panel gains ROC-AUC, log-loss and (binary only) Brier alongside the label metrics,
which are always computed so a run's history stays comparable across iterations
that did and didn't emit probabilities. `y_proba` shape follows the task — one
positive-class probability per row for binary, one row per sample by one column
per class for multiclass.

Policy lives in the callers, not here: this module raises on malformed
probabilities rather than deciding whether that should sink an experiment. Whether
a bad probability file is fatal depends on `requires_proba(primary_metric)`, which
only the caller knows.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

if TYPE_CHECKING:
    from iterate.adapters.models.registry import Task

PROBA_METRICS = frozenset({"roc_auc", "log_loss", "brier"})
LABEL_METRICS = frozenset({"accuracy", "f1", "precision", "recall"})
CLASSIFICATION_METRICS = LABEL_METRICS | PROBA_METRICS
REGRESSION_METRICS = frozenset({"rmse", "mae", "mse", "r2"})
_MINIMIZE = frozenset({"rmse", "mae", "mse", "log_loss", "brier"})


def task_for_metric(metric: str) -> Task:
    if metric in REGRESSION_METRICS:
        return "regression"
    if metric in CLASSIFICATION_METRICS:
        return "classification"
    known = sorted(CLASSIFICATION_METRICS | REGRESSION_METRICS)
    raise ValueError(f"unknown metric {metric!r}; expected one of {known}")


def direction(metric: str) -> Literal["maximize", "minimize"]:
    return "minimize" if metric in _MINIMIZE else "maximize"


def requires_proba(metric: str) -> bool:
    """Whether this metric can only be computed from predicted probabilities.

    The coder's `probabilities.csv` is optional in general but mandatory when the
    run's primary metric answers True here.
    """
    return metric in PROBA_METRICS


def score(task: Task, y_true: Any, y_pred: Any, *, y_proba: Any = None) -> dict[str, float]:
    if task == "regression":
        mse = float(mean_squared_error(y_true, y_pred))
        return {
            "rmse": math.sqrt(mse),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "mse": mse,
            "r2": float(r2_score(y_true, y_pred)),
        }
    # The positive class is the greater label, stated explicitly rather than left
    # to sklearn's pos_label=1 default: a string target ("yes"/"no") has no label
    # 1, so the default raises. Naming it here also keeps the label panel and the
    # probability panel pointed at the SAME positive class — f1 and brier
    # disagreeing about which class is positive would make the panel incoherent.
    classes = np.unique(np.asarray(y_true))
    binary = len(classes) <= 2
    average = "binary" if binary else "macro"
    positive: dict[str, Any] = {"pos_label": classes[-1]} if binary else {}
    values = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0, **positive)),
        "precision": float(
            precision_score(y_true, y_pred, average=average, zero_division=0, **positive)
        ),
        "recall": float(
            recall_score(y_true, y_pred, average=average, zero_division=0, **positive)
        ),
    }
    if y_proba is not None:
        values.update(_proba_scores(y_true, y_proba, classes))
    return values


def _proba_scores(y_true: Any, y_proba: Any, classes: Any) -> dict[str, float]:
    """The probability panel. Raises ValueError on a shape that can't be scored.

    sklearn clips log-loss internally (>=1.5), so a confidently-wrong probability
    yields a large finite number rather than `inf` — which matters because
    `Metrics` rejects non-finite values outright.
    """
    proba = np.asarray(y_proba, dtype=float)
    if proba.ndim > 2:
        raise ValueError(f"probabilities must be 1-D or 2-D, got {proba.ndim}-D")

    if len(classes) <= 2:
        if proba.ndim == 2:
            if proba.shape[1] != 2:
                raise ValueError(
                    f"binary probabilities need 1 or 2 columns, got {proba.shape[1]}"
                )
            proba = proba[:, 1]
        positive = classes[-1]
        return {
            "roc_auc": float(roc_auc_score(y_true, proba)),
            "log_loss": float(log_loss(y_true, proba, labels=classes)),
            # Binary only: multiclass Brier landed after the scikit-learn>=1.5 floor.
            "brier": float(brier_score_loss(y_true, proba, pos_label=positive)),
        }

    if proba.ndim != 2 or proba.shape[1] != len(classes):
        raise ValueError(
            f"multiclass probabilities need one column per class "
            f"({len(classes)}), got shape {proba.shape}"
        )
    return {
        "roc_auc": float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro")),
        "log_loss": float(log_loss(y_true, proba, labels=classes)),
    }


__all__ = [
    "CLASSIFICATION_METRICS",
    "LABEL_METRICS",
    "PROBA_METRICS",
    "REGRESSION_METRICS",
    "direction",
    "requires_proba",
    "score",
    "task_for_metric",
]
