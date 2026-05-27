"""Tabular `ModelTarget` — the first concrete `BenchmarkTarget`.

Wraps a `TabularDataset` + a metric + an estimator in a leakage-safe sklearn
`Pipeline` (preprocessing fit on train only), trains, and scores on the sealed
holdout. `baseline()` measures the starting model; `run(candidate)` applies the
candidate's hyperparameter changes.

Scoped for Day 3: one estimator family (`HistGradientBoosting`) + hyperparameter
changes. Model-family switching and a richer candidate→model mapping arrive with
the model adapters (Day 4); robust error handling + execution venue come with the
executor (Day 5).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Literal

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from threadpoolctl import threadpool_limits

from iterate.schemas.experiment import ExperimentResult, Metrics

if TYPE_CHECKING:
    from iterate.adapters.data.tabular import TabularDataset
    from iterate.schemas.experiment import Candidate

Task = Literal["classification", "regression"]

_CLASSIFICATION_METRICS = {"accuracy", "f1", "precision", "recall"}
_REGRESSION_METRICS = {"rmse", "mae", "mse", "r2"}
_MINIMIZE = {"rmse", "mae", "mse"}


def _task_for_metric(metric: str) -> Task:
    if metric in _REGRESSION_METRICS:
        return "regression"
    if metric in _CLASSIFICATION_METRICS:
        return "classification"
    known = sorted(_CLASSIFICATION_METRICS | _REGRESSION_METRICS)
    raise ValueError(f"unknown metric {metric!r}; expected one of {known}")


def _direction(metric: str) -> Literal["maximize", "minimize"]:
    return "minimize" if metric in _MINIMIZE else "maximize"


def _score(task: Task, y_true: Any, y_pred: Any) -> dict[str, float]:
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


def _make_estimator(task: Task, params: dict[str, Any]) -> Any:
    if task == "classification":
        return HistGradientBoostingClassifier(**params)
    return HistGradientBoostingRegressor(**params)


class ModelTarget:
    """A tabular ML target: leakage-safe preprocessing + estimator, trained and scored."""

    def __init__(
        self,
        dataset: TabularDataset,
        *,
        metric: str,
        name: str = "tabular-model",
        max_threads: int = 1,
    ) -> None:
        self.name = name
        self._dataset = dataset
        self._metric = metric
        self._task: Task = _task_for_metric(metric)
        self._max_threads = max_threads

    def baseline(self) -> ExperimentResult:
        return self._evaluate({}, experiment_id="baseline")

    def run(self, candidate: Candidate) -> ExperimentResult:
        # Day 3: candidate.changes are estimator hyperparameters.
        return self._evaluate(candidate.changes, experiment_id=candidate.id)

    def _evaluate(self, params: dict[str, Any], *, experiment_id: str) -> ExperimentResult:
        estimator = _make_estimator(self._task, {"random_state": self._dataset.seed, **params})
        pipeline = Pipeline([("preprocess", self._preprocessor()), ("model", estimator)])
        # Cap OpenMP/BLAS threads: on small tabular data the thread-pool overhead
        # dwarfs the actual work (~200x slower here at 10 cores), so 1 thread wins.
        # Revisit for large datasets / DL, where parallelism actually pays off.
        with threadpool_limits(limits=self._max_threads):
            pipeline.fit(self._dataset.train_features, self._dataset.train_target)
            predictions = pipeline.predict(self._dataset.test_features)
        metrics = Metrics(
            values=_score(self._task, self._dataset.test_target, predictions),
            primary=self._metric,
            direction=_direction(self._metric),
            n_samples=self._dataset.n_test,
        )
        return ExperimentResult(experiment_id=experiment_id, metrics=metrics)

    def _preprocessor(self) -> Any:
        numeric = self._dataset.train_features.select_dtypes(include="number").columns.tolist()
        categorical = [col for col in self._dataset.features if col not in numeric]
        transformers: list[Any] = []
        if numeric:
            transformers.append(("num", SimpleImputer(strategy="median"), numeric))
        if categorical:
            encode = Pipeline(
                [
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]
            )
            transformers.append(("cat", encode, categorical))
        return ColumnTransformer(transformers, remainder="drop")


__all__ = ["ModelTarget"]
