"""Metric scoring — the single ruler every execution path is judged by.

Both the spec path (`ModelTarget`) and the code-gen path (`core.codegen`) score
through these functions, so "improvement" stays an apples-to-apples comparison
no matter how a candidate was run. Keep all metric computation here; nothing
should reimplement it.

Every metric is one row in `REGISTRY`, carrying its task, its direction, whether
it needs probabilities, and how to compute it. Adding a metric is that row and
nothing else — the exported frozensets, `direction()`, `task_for_metric()` and
`requires_proba()` are all derived from it. That single source of truth is the
point: the panel, the direction table and the CLI's own copy of the metric names
used to be three independent lists, and they drifted (the CLI called every
classification metric a maximize metric, which would have inverted the whole loop
the moment log-loss became selectable).

Probability metrics are a bonus panel: pass `y_proba` and the classification panel
gains ROC-AUC, PR-AUC, log-loss and (binary only) Brier alongside the label
metrics, which are always computed so a run's history stays comparable across
iterations that did and didn't emit probabilities. `y_proba` shape follows the
task — one positive-class probability per row for binary, one row per sample by
one column per class for multiclass.

Policy lives in the callers, not here: this module raises on malformed
probabilities rather than deciding whether that should sink an experiment. Whether
a bad probability file is fatal depends on `requires_proba(primary_metric)`, which
only the caller knows.
"""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
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
from sklearn.preprocessing import label_binarize

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from iterate.adapters.models.registry import Task

Direction = Literal["maximize", "minimize"]

# Averaging applies to f1 / precision / recall only, which is exactly the
# limitation LIMITATIONS.md tracks. ROC-AUC keeps macro averaging on multiclass:
# sklearn's supported set for `multi_class="ovr"` is narrower and varies by
# version, and this package's floor is scikit-learn>=1.5.
AVERAGES = ("binary", "micro", "macro", "weighted")


@dataclass(frozen=True)
class Inputs:
    """Everything a metric needs, resolved once per `score()` call."""

    y_true: Any
    y_pred: Any
    y_proba: Any
    classes: Any
    average: str

    @property
    def binary(self) -> bool:
        return len(self.classes) <= 2

    @property
    def positive(self) -> Any:
        """The positive class: the greater label, stated explicitly rather than
        left to sklearn's `pos_label=1` default, which raises on a string target
        ("yes"/"no" has no label 1). Naming it here also keeps the label panel and
        the probability panel pointed at the SAME class — f1 and Brier disagreeing
        about which class is positive would make the panel incoherent."""
        return self.classes[-1]

    @property
    def label_kwargs(self) -> dict[str, Any]:
        if self.average == "binary":
            return {"average": "binary", "pos_label": self.positive, "zero_division": 0}
        return {"average": self.average, "zero_division": 0}

    def positive_column(self) -> Any:
        """Binary probabilities as a 1-D positive-class column."""
        proba = np.asarray(self.y_proba, dtype=float)
        if proba.ndim > 2:
            raise ValueError(f"probabilities must be 1-D or 2-D, got {proba.ndim}-D")
        if proba.ndim == 2:
            if proba.shape[1] != 2:
                raise ValueError(
                    f"binary probabilities need 1 or 2 columns, got {proba.shape[1]}"
                )
            return proba[:, 1]
        return proba

    def class_matrix(self) -> Any:
        """Multiclass probabilities as an (n_samples, n_classes) matrix."""
        proba = np.asarray(self.y_proba, dtype=float)
        if proba.ndim != 2 or proba.shape[1] != len(self.classes):
            raise ValueError(
                f"multiclass probabilities need one column per class "
                f"({len(self.classes)}), got shape {proba.shape}"
            )
        return proba


@dataclass(frozen=True)
class MetricSpec:
    task: Task
    direction: Direction
    needs_proba: bool
    compute: Callable[[Inputs], float]
    # Binary-only metrics are skipped on a multiclass target rather than raising:
    # multiclass Brier landed after the scikit-learn>=1.5 floor this package ships.
    binary_only: bool = False


def _roc_auc(i: Inputs) -> float:
    if i.binary:
        return float(roc_auc_score(i.y_true, i.positive_column()))
    return float(roc_auc_score(i.y_true, i.class_matrix(), multi_class="ovr", average="macro"))


def _average_precision(i: Inputs) -> float:
    if i.binary:
        return float(
            average_precision_score(i.y_true, i.positive_column(), pos_label=i.positive)
        )
    # Binarized explicitly rather than handed a multiclass y_true: sklearn only
    # grew multiclass support for this metric well after the >=1.5 floor, and it
    # raises "multiclass format is not supported" below that.
    binarized = label_binarize(i.y_true, classes=list(i.classes))
    return float(average_precision_score(binarized, i.class_matrix(), average="macro"))


def _log_loss(i: Inputs) -> float:
    proba = i.positive_column() if i.binary else i.class_matrix()
    return float(log_loss(i.y_true, proba, labels=i.classes))


def _brier(i: Inputs) -> float:
    return float(brier_score_loss(i.y_true, i.positive_column(), pos_label=i.positive))


def _mse(i: Inputs) -> float:
    return float(mean_squared_error(i.y_true, i.y_pred))


PANEL: dict[str, MetricSpec] = {
    "accuracy": MetricSpec(
        "classification", "maximize", False, lambda i: float(accuracy_score(i.y_true, i.y_pred))
    ),
    "f1": MetricSpec(
        "classification",
        "maximize",
        False,
        lambda i: float(f1_score(i.y_true, i.y_pred, **i.label_kwargs)),
    ),
    "precision": MetricSpec(
        "classification",
        "maximize",
        False,
        lambda i: float(precision_score(i.y_true, i.y_pred, **i.label_kwargs)),
    ),
    "recall": MetricSpec(
        "classification",
        "maximize",
        False,
        lambda i: float(recall_score(i.y_true, i.y_pred, **i.label_kwargs)),
    ),
    "roc_auc": MetricSpec("classification", "maximize", True, _roc_auc),
    "average_precision": MetricSpec("classification", "maximize", True, _average_precision),
    "log_loss": MetricSpec("classification", "minimize", True, _log_loss),
    "brier": MetricSpec("classification", "minimize", True, _brier, binary_only=True),
    "rmse": MetricSpec("regression", "minimize", False, lambda i: math.sqrt(_mse(i))),
    "mae": MetricSpec(
        "regression", "minimize", False, lambda i: float(mean_absolute_error(i.y_true, i.y_pred))
    ),
    "mse": MetricSpec("regression", "minimize", False, _mse),
    "r2": MetricSpec(
        "regression", "maximize", False, lambda i: float(r2_score(i.y_true, i.y_pred))
    ),
}

# ─── the wider vocabulary, derived from sklearn rather than hand-written ─────
#
# PANEL above is what every run computes: a small, comparable, always-present set.
# REGISTRY below is what a run may SELECT as its primary, and it is derived from
# sklearn's own scorer registry so the vocabulary is not a list somebody has to
# remember to extend. The Researcher can propose Matthews correlation or balanced
# accuracy and the harness already knows both.
#
# Crucially, DIRECTION is derived too, never guessed. sklearn's scorers are all
# "higher is better" by construction, with loss metrics carrying a -1 sign (the
# `neg_` naming convention). That sign is the direction, straight from the library
# that defines the metric. An LLM proposing a metric therefore still cannot invert
# the loop, because it never supplies the direction — which is the whole reason
# direction stayed out of the model's hands in the first place.
_CLASSIFICATION_MODULES = frozenset({"_classification", "_ranking", "_scorer"})
_REGRESSION_MODULES = frozenset({"_regression"})


def _sklearn_compute(
    func: Callable[..., Any], kwargs: dict[str, Any], *, needs_proba: bool
) -> Callable[[Inputs], float]:
    # Checked once at derivation, not per call. Note that most binary scorers do
    # NOT carry pos_label in their kwargs at all (jaccard, f1, precision, recall
    # ship only average="binary"), so sklearn's own pos_label=1 default applies —
    # the exact default that made string targets unscorable. Reading the signature
    # is what catches those; inspecting kwargs alone would miss every one of them.
    accepts_pos_label = "pos_label" in inspect.signature(func).parameters

    def compute(i: Inputs) -> float:
        call = dict(kwargs)
        if accepts_pos_label and i.binary and call.get("average", "binary") in (None, "binary"):
            call["pos_label"] = i.positive
        if needs_proba:
            proba = i.positive_column() if i.binary else i.class_matrix()
            return float(func(i.y_true, proba, **call))
        return float(func(i.y_true, i.y_pred, **call))

    return compute


def _derive_from_sklearn() -> dict[str, MetricSpec]:
    """Every sklearn scorer that applies to a supervised tabular target.

    Reads private scorer attributes (`_sign`, `_score_func`, `_kwargs`,
    `_response_method`). They have been stable for years, but they are private, so
    every step is guarded and a failure degrades to the curated PANEL rather than
    breaking scoring. `test_scoring` asserts the derivation still works on the
    installed version, so a sklearn upgrade that changes this fails CI instead of a
    user's run.
    """
    try:
        from sklearn.metrics import get_scorer, get_scorer_names

        names = list(get_scorer_names())
    except Exception:  # pragma: no cover - sklearn API moved
        return {}

    out: dict[str, MetricSpec] = {}
    for name in names:
        try:
            scorer = get_scorer(name)
            func = scorer._score_func
            sign = int(scorer._sign)
            kwargs = dict(scorer._kwargs)
            response = scorer._response_method
        except Exception:  # pragma: no cover - one bad scorer must not sink the rest
            continue
        module = func.__module__.rsplit(".", 1)[-1]
        if module in _CLASSIFICATION_MODULES:
            task: Task = "classification"
        elif module in _REGRESSION_MODULES:
            task = "regression"
        else:
            # sklearn also registers clustering scores (rand, v-measure, mutual
            # info). They take two label assignments, not a prediction against a
            # target, so they are meaningless here and offering them would invite
            # the agent to pick one.
            continue
        needs_proba = response != "predict"
        out[name] = MetricSpec(
            task,
            "maximize" if sign > 0 else "minimize",
            needs_proba,
            _sklearn_compute(func, kwargs, needs_proba=needs_proba),
        )
    return out


# Curated entries win: PANEL's computes are the ones with their own tests, their
# own shape validation, and the binary/multiclass handling this project needs.
REGISTRY: dict[str, MetricSpec] = {**_derive_from_sklearn(), **PANEL}

CLASSIFICATION_METRICS = frozenset(
    name for name, spec in REGISTRY.items() if spec.task == "classification"
)
REGRESSION_METRICS = frozenset(
    name for name, spec in REGISTRY.items() if spec.task == "regression"
)
PROBA_METRICS = frozenset(name for name, spec in REGISTRY.items() if spec.needs_proba)
# Panel-scoped, unlike the three above: those answer "may this be selected", this
# answers "what does score() always return for a classification run".
LABEL_METRICS = frozenset(
    name for name, spec in PANEL.items() if spec.task == "classification" and not spec.needs_proba
)
PANEL_METRICS = frozenset(PANEL)


def task_for_metric(metric: str) -> Task:
    try:
        return REGISTRY[metric].task
    except KeyError:
        raise ValueError(
            f"unknown metric {metric!r}; expected one of {sorted(REGISTRY)}"
        ) from None


def direction(metric: str) -> Direction:
    try:
        return REGISTRY[metric].direction
    except KeyError:
        raise ValueError(
            f"unknown metric {metric!r}; expected one of {sorted(REGISTRY)}"
        ) from None


def requires_proba(metric: str) -> bool:
    """Whether this metric can only be computed from predicted probabilities.

    The coder's `probabilities.csv` is optional in general but mandatory when the
    run's primary metric answers True here.
    """
    return metric in PROBA_METRICS


def resolve_average(average: str | None, *, binary: bool) -> str:
    """The averaging strategy for f1 / precision / recall.

    None means auto, which preserves the pre-v0.4 behaviour exactly: binary on a
    two-class target, macro otherwise. An explicit choice is honoured, except that
    "binary" on a multiclass target is a contradiction and raises.
    """
    if average is None:
        return "binary" if binary else "macro"
    if average not in AVERAGES:
        raise ValueError(f"unknown average {average!r}; expected one of {list(AVERAGES)}")
    if average == "binary" and not binary:
        raise ValueError("average='binary' needs a two-class target")
    return average


def score(
    task: Task,
    y_true: Any,
    y_pred: Any,
    *,
    y_proba: Any = None,
    average: str | None = None,
    include: Sequence[str] = (),
) -> dict[str, float]:
    classes = np.unique(np.asarray(y_true))
    inputs = Inputs(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        classes=classes,
        average=resolve_average(average, binary=len(classes) <= 2),
    )
    values: dict[str, float] = {}
    for name, spec in PANEL.items():
        if spec.task != task:
            continue
        if spec.needs_proba and y_proba is None:
            continue
        if spec.binary_only and not inputs.binary:
            continue
        values[name] = spec.compute(inputs)
    # The panel stays small so history is comparable and cheap across every run.
    # A primary metric from the wider vocabulary is computed ON TOP, on request —
    # scoring all 49 every iteration would be slow and would bury the comparison.
    for name in include:
        if name in values:
            continue
        extra = REGISTRY.get(name)
        if extra is None or extra.task != task:
            continue
        if extra.needs_proba and y_proba is None:
            continue
        values[name] = extra.compute(inputs)
    return values


__all__ = [
    "AVERAGES",
    "CLASSIFICATION_METRICS",
    "LABEL_METRICS",
    "PANEL",
    "PANEL_METRICS",
    "PROBA_METRICS",
    "REGISTRY",
    "REGRESSION_METRICS",
    "Inputs",
    "MetricSpec",
    "direction",
    "requires_proba",
    "resolve_average",
    "score",
    "task_for_metric",
]
