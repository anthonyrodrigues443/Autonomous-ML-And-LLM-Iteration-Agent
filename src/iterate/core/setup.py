"""Resolving how a run will be measured, when the user did not say.

An explicit `--metric` always wins. Otherwise the Researcher proposes a metric and
a starting model, both validated against the registry and the target column, and
anything that fails validation falls back to the deterministic default. The result
is FIXED for the run.
"""

from __future__ import annotations

import functools
import math
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from iterate.core.scoring import (
    CLASSIFICATION_METRICS,
    PROBA_METRICS,
    REGISTRY,
    REGRESSION_METRICS,
    Inputs,
    resolve_average,
    task_for_metric,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from iterate.adapters.data.tabular import TabularDataset
    from iterate.core.researcher import Setup

# What v0.3 did, kept as the floor every failure path lands on.
_BINARY_DEFAULT = "f1"
_MULTICLASS_DEFAULT = "f1_macro"
_REGRESSION_DEFAULT = "rmse"
_CLASSIFIER = "sklearn.ensemble.HistGradientBoostingClassifier"
_REGRESSOR = "sklearn.ensemble.HistGradientBoostingRegressor"


@dataclass(frozen=True)
class RunSetup:
    """The frozen answer to "how is this run measured, and from what"."""

    metric: str
    starting_model: str
    why: str = ""
    chosen_by_agent: bool = False

    def render(self) -> str:
        """The line printed before the first experiment. A tool that silently picks
        your evaluation metric is worse than one that asks, so when the agent chose
        it says so and says why."""
        if not self.chosen_by_agent:
            return ""
        line = f"metric: {self.metric}"
        if self.why:
            line += f" — {self.why}"
        return line


def target_task(dataset: TabularDataset) -> str:
    """classification or regression, as the loader decided it. One definition only."""
    return dataset.task


def _n_classes(target: Any) -> int:
    try:
        return int(target.nunique())
    except Exception:
        return len(set(target))


def default_setup(dataset: TabularDataset) -> RunSetup:
    """What v0.3 would have run. Every failure path lands here."""
    task = target_task(dataset)
    if task == "regression":
        return RunSetup(metric=_REGRESSION_DEFAULT, starting_model=_REGRESSOR)
    binary = _n_classes(dataset.train_target) <= 2
    return RunSetup(
        metric=_BINARY_DEFAULT if binary else _MULTICLASS_DEFAULT,
        starting_model=_CLASSIFIER,
    )


def validate(metric: str, dataset: TabularDataset) -> str | None:
    """Why this metric cannot be used for this dataset, or None if it can."""
    if metric not in REGISTRY:
        return f"{metric!r} is not a known metric"
    wanted = task_for_metric(metric)
    actual = target_task(dataset)
    if wanted != actual:
        return f"{metric!r} is a {wanted} metric but the target looks like {actual}"
    return None


# The labels scikit-learn 1.8.0 lets each of these score at all.
_LABEL_RANGE: dict[str, Callable[[float], bool]] = {
    "mean_gamma_deviance": lambda low: low > 0,
    "mean_poisson_deviance": lambda low: low >= 0,
    "mean_squared_log_error": lambda low: low > -1,
    "root_mean_squared_log_error": lambda low: low > -1,
}
# One name per ruler: these score the same number as mae, mse and rmse.
_LONG_NAMES = frozenset({"mean_absolute_error", "mean_squared_error", "root_mean_squared_error"})


@functools.cache
def _scores_classes(binary: bool) -> frozenset[str]:
    """The class metrics that score a 2- or 3-class fixture through their own compute.
    The answer depends on two classes against more, and on nothing else in the data."""
    k = 2 if binary else 3
    y = np.repeat(np.arange(k), 3)
    proba = np.random.default_rng(0).dirichlet(np.ones(k), len(y))
    inputs = Inputs(
        y_true=y,
        y_pred=proba.argmax(1),
        y_proba=proba,
        classes=np.arange(k),
        average=resolve_average(None, binary=binary),
    )
    kept = set()
    for name in CLASSIFICATION_METRICS:
        spec = REGISTRY[name]
        if spec.binary_only and not binary:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                value = spec.compute(inputs)
        except (ValueError, TypeError):
            continue
        if np.isfinite(value):
            kept.add(name)
    return frozenset(kept)


def _classes(target: Any) -> set[Any]:
    return set(target.dropna())


def _lowest_label(dataset: TabularDataset) -> float | None:
    """The smallest label on either side, or None when the labels are not numbers."""
    both = pd.concat([dataset.train_target, dataset.test_target])
    numbers = pd.to_numeric(both, errors="coerce").to_numpy(dtype=float)
    if not numbers.size or not bool(np.isfinite(numbers).any()):
        return None
    low = float(np.nanmin(numbers))
    return low if math.isfinite(low) else None


def offered_for_images(dataset: TabularDataset) -> list[str]:
    """What the image setup pass may choose from: the metrics that score these labels on
    every experiment the run can produce."""
    if dataset.task == "regression":
        # A regressor can predict outside these ranges, so the agent is never offered them.
        return sorted(REGRESSION_METRICS - set(_LABEL_RANGE) - _LONG_NAMES)
    train, holdout = _classes(dataset.train_target), _classes(dataset.test_target)
    names = set(_scores_classes(len(train) <= 2))
    if len(train) <= 2:
        # On two classes a top-3 score is 1.000 for a perfect model and for an always-wrong one.
        names.discard("top_k_accuracy")
    if not train <= holdout:
        names -= PROBA_METRICS
    return sorted(names)


def refuse_labels(
    dataset: TabularDataset, *, metric: str | None, average: str | None
) -> str | None:
    """Why these labels cannot be scored as asked, or None. Every check is on the labels
    themselves, so it answers before an image is copied or a model is built."""
    if dataset.task == "regression":
        low = _lowest_label(dataset)
        inside = _LABEL_RANGE.get(metric or "")
        if low is None or inside is None or inside(low):
            return None
        return f"--metric {metric} cannot score labels as low as {low:g}"
    train, holdout = _classes(dataset.train_target), _classes(dataset.test_target)
    if len(holdout) <= 2 < len(train):
        return (
            f"the holdout holds {len(holdout)} of the {len(train)} training classes, so every "
            "metric would score it as two classes; add holdout images of the missing classes"
        )
    try:
        resolve_average(average, binary=len(train) <= 2)
    except ValueError as exc:
        return str(exc)
    if metric is not None and metric not in _scores_classes(len(train) <= 2):
        return (
            f"--metric {metric} cannot score a {len(train)}-class label; choose one of "
            + ", ".join(offered_for_images(dataset))
        )
    if metric in PROBA_METRICS and not train <= holdout:
        return (
            f"--metric {metric} needs a probability for every class, and the holdout has no "
            f"image of {len(train - holdout)} training class(es); choose a metric that reads "
            "labels, such as accuracy"
        )
    return None


def resolve(
    dataset: TabularDataset,
    *,
    explicit: str | None = None,
    proposed: Setup | None = None,
    offered: Sequence[str] | None = None,
) -> RunSetup:
    """The single place a run's ruler is decided. Never raises."""
    fallback = default_setup(dataset)

    if explicit:
        # The user's choice is not second-guessed. It was already validated at the
        # CLI boundary, where a bad name gets a proper error rather than a silent
        # substitution.
        return RunSetup(metric=explicit, starting_model=fallback.starting_model)

    if proposed is None or not proposed.metric:
        return fallback

    reason = validate(proposed.metric, dataset)
    # `offered` is the image lane's closed list. Tables pass nothing and keep the
    # proposals they always took.
    if reason is None and offered is not None and proposed.metric not in offered:
        reason = f"{proposed.metric!r} cannot score these labels on every experiment"
    if reason is not None:
        # Rejected proposals are logged by the caller and the run continues on the
        # default — an agent that picks badly costs nothing.
        return RunSetup(
            metric=fallback.metric,
            starting_model=fallback.starting_model,
            why=f"agent proposed {proposed.metric!r} but {reason}; using the default",
            chosen_by_agent=False,
        )

    return RunSetup(
        metric=proposed.metric,
        starting_model=_valid_model(proposed.starting_model, fallback.starting_model),
        why=proposed.why,
        chosen_by_agent=True,
    )


def _valid_model(proposed: str, fallback: str) -> str:
    """A starting model is a weaker claim than a metric — the agent rewrites the
    model every iteration anyway, so a bad one costs one experiment rather than the
    whole run's ruler. Accepted if it looks like an estimator name at all."""
    name = (proposed or "").strip()
    if not name or not name.replace(".", "").replace("_", "").isalnum():
        return fallback
    return name


__all__ = [
    "RunSetup",
    "default_setup",
    "offered_for_images",
    "refuse_labels",
    "resolve",
    "target_task",
    "validate",
]
