"""Measuring the ceiling: the best score a brute-force sweep can reach, no LLM.

Without a ceiling, "no candidate beat the baseline" is unreadable. With one, the
same run reads as either "captured 0 of 20 points available" or "correctly found
the nothing that was there". The v0.4 certification established the method by
hand over an afternoon; this is the method as a command.

Two rules make the number trustworthy:

*The ceiling is measured through the product's own machinery.* Same `load_csv`
call the CLI makes, same seed, same sealed holdout, same `ModelTarget` pipeline,
same `core.scoring` ruler. A ceiling measured by a separate script with its own
preprocessing would be comparing the agent against a different game.

*A ceiling is a lower bound and is labelled as one.* It is the best of a fixed
list of reasonable models, not a proof of the maximum. The agent going past it is
a real result rather than an error — laptop price reached 110% of its hand-swept
ceiling — so nothing here clamps and the stored `method` says how it was reached.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from evals.store import Ceiling
from iterate.adapters.data.tabular import load_csv
from iterate.core.scoring import direction as metric_direction
from iterate.core.scoring import task_for_metric
from iterate.schemas.experiment import Candidate
from iterate.targets.model import ModelTarget

if TYPE_CHECKING:
    from collections.abc import Callable

    from evals.corpus import Dataset

# Bumped when the spec list below changes, so a stored ceiling says which sweep
# produced it and an old one can be spotted and re-measured.
METHOD = "brute_force_sweep_v1"

_FOREST_PARAMS: dict[str, Any] = {"n_estimators": 500, "n_jobs": -1}
_BOOSTER_PARAMS: dict[str, Any] = {"n_estimators": 800, "learning_rate": 0.03, "n_jobs": -1}

_CLASSIFICATION_SPECS: list[dict[str, Any]] = [
    {},  # the factory default — the same starting point the agent is given
    {"model": "sklearn.linear_model.LogisticRegression", "params": {"max_iter": 2000}},
    {"model": "sklearn.ensemble.RandomForestClassifier", "params": _FOREST_PARAMS},
    {"model": "sklearn.ensemble.ExtraTreesClassifier", "params": _FOREST_PARAMS},
    {"model": "sklearn.ensemble.HistGradientBoostingClassifier", "params": {}},
    {
        "model": "sklearn.ensemble.HistGradientBoostingClassifier",
        "params": {"learning_rate": 0.05, "max_iter": 500, "max_leaf_nodes": 63},
    },
    {
        "model": "sklearn.ensemble.HistGradientBoostingClassifier",
        "params": {
            "learning_rate": 0.03,
            "max_iter": 800,
            "min_samples_leaf": 5,
            "l2_regularization": 1.0,
        },
    },
    {"model": "lightgbm.LGBMClassifier", "params": {**_BOOSTER_PARAMS, "num_leaves": 63}},
    {"model": "xgboost.XGBClassifier", "params": {**_BOOSTER_PARAMS, "max_depth": 6}},
]

_REGRESSION_SPECS: list[dict[str, Any]] = [
    {},
    {"model": "sklearn.linear_model.Ridge", "params": {}},
    {"model": "sklearn.ensemble.RandomForestRegressor", "params": _FOREST_PARAMS},
    {"model": "sklearn.ensemble.ExtraTreesRegressor", "params": _FOREST_PARAMS},
    {"model": "sklearn.ensemble.HistGradientBoostingRegressor", "params": {}},
    {
        "model": "sklearn.ensemble.HistGradientBoostingRegressor",
        "params": {"learning_rate": 0.05, "max_iter": 500, "max_leaf_nodes": 63},
    },
    {
        "model": "sklearn.ensemble.HistGradientBoostingRegressor",
        "params": {
            "learning_rate": 0.03,
            "max_iter": 800,
            "min_samples_leaf": 5,
            "l2_regularization": 1.0,
        },
    },
    {"model": "lightgbm.LGBMRegressor", "params": {**_BOOSTER_PARAMS, "num_leaves": 63}},
    {"model": "xgboost.XGBRegressor", "params": {**_BOOSTER_PARAMS, "max_depth": 6}},
]


@dataclass(frozen=True)
class SpecResult:
    """One model in the sweep."""

    label: str
    score: float | None
    seconds: float
    error: str = ""


def specs_for(metric: str) -> list[dict[str, Any]]:
    return (
        _CLASSIFICATION_SPECS if task_for_metric(metric) == "classification" else _REGRESSION_SPECS
    )


def _label(spec: dict[str, Any]) -> str:
    model = str(spec.get("model") or "factory-default").rsplit(".", 1)[-1]
    params = spec.get("params") or {}
    if not isinstance(params, dict) or not params:
        return model
    tuned = {k: v for k, v in params.items() if k not in ("n_jobs",)}
    if not tuned:
        return model
    return f"{model}({', '.join(f'{k}={v}' for k, v in sorted(tuned.items()))})"


def _better(candidate: float, incumbent: float, direction: str) -> bool:
    return candidate > incumbent if direction == "maximize" else candidate < incumbent


def sweep(
    dataset: Dataset,
    *,
    threads: int = 4,
    on_progress: Callable[[SpecResult], None] | None = None,
) -> tuple[Ceiling, list[SpecResult]]:
    """Run every spec for this dataset's metric and return the best as its ceiling.

    Thread count changes how long this takes, not what it produces: every estimator
    in the list is deterministic given its seed regardless of how many threads it
    runs on, so the ceiling is reproducible across machines.
    """
    loaded = load_csv(dataset.path, target=dataset.target)
    target = ModelTarget(loaded, metric=dataset.metric, name=dataset.name, max_threads=threads)
    direction = metric_direction(dataset.metric)

    baseline_value: float | None = None
    try:
        baseline_result = target.baseline()
        if baseline_result.metrics is not None:
            baseline_value = baseline_result.metrics.primary_value
    except Exception as exc:  # a dataset the default model cannot even fit
        if on_progress:
            on_progress(SpecResult(label="baseline", score=None, seconds=0.0, error=str(exc)))

    results: list[SpecResult] = []
    best: float | None = None

    for spec in specs_for(dataset.metric):
        label = _label(spec)
        started = time.monotonic()
        try:
            outcome = target.run(
                Candidate(
                    description=label,
                    # `changes` must be non-empty and the factory-default spec is
                    # legitimately empty, so it travels as a note instead. An absent
                    # "model" key is what selects the factory default downstream.
                    changes=spec or {"note": "factory-default"},
                    rationale="ceiling sweep",
                )
            )
            value = outcome.metrics.primary_value if outcome.metrics is not None else None
            error = outcome.error or ""
        except Exception as exc:
            # An uninstalled optional library (lightgbm, xgboost) or a param the
            # estimator rejects. Skipped, recorded, never fatal: the ceiling is the
            # best of what could actually run on this machine, and the detail says
            # what could not.
            value, error = None, f"{type(exc).__name__}: {exc}"

        result = SpecResult(
            label=label, score=value, seconds=time.monotonic() - started, error=error
        )
        results.append(result)
        if on_progress:
            on_progress(result)
        if value is not None and (best is None or _better(value, best, direction)):
            best = value

    if best is None:
        raise RuntimeError(f"{dataset.name}: no model in the sweep produced a score")

    detail = json.dumps(
        [
            {"model": r.label, "score": r.score, "seconds": round(r.seconds, 1), "error": r.error}
            for r in results
        ]
    )

    ceiling = Ceiling(
        dataset=dataset.name,
        dataset_hash=dataset.content_hash(),
        metric=dataset.metric,
        ceiling=best,
        direction=direction,
        baseline=baseline_value,
        method=f"{METHOD} ({sum(1 for r in results if r.score is not None)} models)",
        measured_at=datetime.now(UTC).isoformat(),
        detail=detail,
    )
    return ceiling, results


__all__ = ["METHOD", "SpecResult", "specs_for", "sweep"]
