"""Tabular `ModelTarget` — the first concrete `BenchmarkTarget`.

Wraps a `TabularDataset` + a metric + an estimator in a leakage-safe sklearn
`Pipeline` (preprocessing fit on train only), trains, and scores on the sealed
holdout. `baseline()` measures the default model; `run(candidate)` builds whatever
model the candidate names (any allow-listed estimator) with its params.

The estimator itself comes from the model factory (`adapters.models.registry`):
a candidate's `changes` is a `{"model", "params"}` spec, so model-family switching
and hyperparameters flow through the same path. Failure handling and the execution
venue live in the executor (`adapters.compute.local`), which catches a bad
candidate rather than letting it raise; arbitrary or uninstalled models come with
the sandboxed code-gen path (v0.2).
"""

from __future__ import annotations

import math
import os
import sys
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Literal

import joblib
from sklearn.compose import ColumnTransformer
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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from threadpoolctl import threadpool_limits

from iterate.adapters.models.registry import Task, build_estimator
from iterate.schemas.experiment import ExperimentResult, Metrics

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from iterate.adapters.data.tabular import TabularDataset
    from iterate.schemas.experiment import Candidate

_CLASSIFICATION_METRICS = {"accuracy", "f1", "precision", "recall"}
_REGRESSION_METRICS = {"rmse", "mae", "mse", "r2"}
_MINIMIZE = {"rmse", "mae", "mse"}

# Param names that mean "I need an eval set for early stopping." XGBoost uses
# ``early_stopping_rounds`` (plural); LightGBM uses ``early_stopping_round``
# (singular) and aliases bare ``early_stopping=<int>``. sklearn's HistGB also has
# ``early_stopping`` but as a bool/"auto" (internal CV — doesn't need eval_set),
# so we exclude the bool case below.
_EARLY_STOPPING_PARAM_NAMES = frozenset(
    {"early_stopping_rounds", "early_stopping_round", "early_stopping"}
)


_lightgbm_silenced = False


def _silence_lightgbm() -> None:
    """Route LightGBM's logger to a null sink (once). Its C++ logs bypass C stdio
    buffering, so the fd redirect alone doesn't catch them; this stops them at the
    source. No-op if LightGBM isn't installed."""
    global _lightgbm_silenced
    if _lightgbm_silenced:
        return
    _lightgbm_silenced = True
    try:
        import logging

        import lightgbm as lgb

        sink = logging.getLogger("iterate.lightgbm")
        sink.addHandler(logging.NullHandler())
        sink.setLevel(logging.CRITICAL)
        lgb.register_logger(sink)
    except Exception:  # pragma: no cover - lightgbm optional / API drift
        pass


def _flush_c_stdio() -> None:
    """Flush libc's stdio buffers so buffered native output goes to the *current*
    fd 1/2 (devnull, mid-suppression) rather than leaking after we restore them.

    LightGBM/XGBoost write through buffered C `stdout`; without this flush the
    buffer drains onto the real terminal once the fds are restored.
    """
    try:
        import ctypes

        libc = ctypes.CDLL(None)
        libc.fflush(None)  # NULL flushes all open streams
    except (OSError, AttributeError):  # pragma: no cover - platform-dependent
        pass


@contextmanager
def _silence_native_stdio() -> Iterator[None]:
    """Redirect C-level stdout+stderr to devnull for the duration of the block.

    XGBoost and especially LightGBM print training chatter (`[LightGBM] [Info] …`,
    row-wise/bin notices, per-round eval) from native code that Python-level
    `verbose`/`verbosity` flags don't fully suppress — it's written straight to file
    descriptors 1/2. We redirect those fds during the fit so the loop's own output
    stays clean. Python exceptions still propagate (they're not stdout text), so
    nothing actionable is lost. Only wraps fit, not the agent's logging.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved_out, saved_err = os.dup(1), os.dup(2)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        _flush_c_stdio()  # drain buffered native output to devnull before restoring
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(devnull)
        os.close(saved_out)
        os.close(saved_err)


def _wants_eval_set(params: dict[str, Any]) -> bool:
    """True iff the candidate's params describe XGBoost/LightGBM-style early stopping.

    Excludes sklearn's ``early_stopping=True|"auto"|False`` (HistGB does its own
    internal CV; passing it `eval_set` would just error at fit time).
    """
    for key in _EARLY_STOPPING_PARAM_NAMES & set(params):
        value = params[key]
        if key in {"early_stopping_rounds", "early_stopping_round"}:
            return True
        # ``early_stopping`` as a positive int → LightGBM rounds count.
        if isinstance(value, bool):
            continue  # bool first, since bool is a subclass of int
        if isinstance(value, int):
            return True
    return False


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
        # Empty spec -> the factory's default model for the task, no params.
        return self._evaluate({}, experiment_id="baseline")

    def run(self, candidate: Candidate) -> ExperimentResult:
        # candidate.changes is a {"model", "params"} spec: which estimator + its params.
        return self._evaluate(candidate.changes, experiment_id=candidate.id)

    def _evaluate(self, spec: dict[str, Any], *, experiment_id: str) -> ExperimentResult:
        with threadpool_limits(limits=self._max_threads):
            pipeline = self._build_and_fit(spec)
            # predict can also emit native chatter / a benign sklearn feature-names
            # warning — keep it quiet too so the loop's output stays clean.
            with _silence_native_stdio():
                predictions = pipeline.predict(self._dataset.test_features)
        metrics = Metrics(
            values=_score(self._task, self._dataset.test_target, predictions),
            primary=self._metric,
            direction=_direction(self._metric),
            n_samples=self._dataset.n_test,
        )
        return ExperimentResult(experiment_id=experiment_id, metrics=metrics)

    def save_model(self, spec: dict[str, Any], path: Path) -> Path:
        """Fit ``spec`` on the training data and persist the fitted pipeline to ``path``.

        Same fit the score was measured from (train-only, same seed) → the saved
        artifact is exactly the model that achieved the reported holdout score.
        Returns the path written. The caller (CLI) uses this to hand the user the
        winning model so they don't have to reconstruct it themselves.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with threadpool_limits(limits=self._max_threads):
            pipeline = self._build_and_fit(spec)
        joblib.dump(pipeline, path)
        return path

    def _build_and_fit(self, spec: dict[str, Any]) -> Pipeline:
        """Build the preprocess+estimator pipeline and fit it on the training split."""
        estimator = build_estimator(self._task, spec, seed=self._dataset.seed)
        pipeline = Pipeline([("preprocess", self._preprocessor()), ("model", estimator)])
        params = spec.get("params") or {}
        _silence_lightgbm()
        with _silence_native_stdio():
            if _wants_eval_set(params):
                # The candidate asked for early stopping → carve a 90/10 fit/eval
                # slice off the training set; the sealed holdout stays untouched.
                fit_params = self._fit_with_internal_eval_set(estimator, params, pipeline)
                pipeline.fit(*fit_params["xy"], **fit_params["fit_kwargs"])
            else:
                pipeline.fit(self._dataset.train_features, self._dataset.train_target)
        return pipeline

    def _fit_with_internal_eval_set(
        self, estimator: Any, params: dict[str, Any], pipeline: Pipeline
    ) -> dict[str, Any]:
        """Split train into 90% fit / 10% eval and prepare an ``eval_set`` fit_param.

        The preprocess step is fit on the 90% (no leakage from eval into the
        transforms) and the 10% eval features are transformed through it so the
        estimator sees an apples-to-apples eval_set. LightGBM additionally needs
        ``eval_metric`` for early stopping; we supply a sensible default if the
        candidate didn't.
        """
        stratify = self._dataset.train_target if self._task == "classification" else None
        x_fit, x_eval, y_fit, y_eval = train_test_split(
            self._dataset.train_features,
            self._dataset.train_target,
            test_size=0.1,
            random_state=self._dataset.seed,
            stratify=stratify,
        )
        preproc = pipeline.named_steps["preprocess"]
        preproc.fit(x_fit)
        x_eval_t = preproc.transform(x_eval)

        fit_kwargs: dict[str, Any] = {"model__eval_set": [(x_eval_t, y_eval)]}
        # LightGBM raises if eval_metric isn't set. Supply a task-appropriate
        # default unless the candidate already specified one in params.
        if type(estimator).__name__.startswith("LGBM") and "eval_metric" not in params:
            fit_kwargs["model__eval_metric"] = (
                "binary_logloss" if self._task == "classification" else "rmse"
            )
        return {"xy": (x_fit, y_fit), "fit_kwargs": fit_kwargs}

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
