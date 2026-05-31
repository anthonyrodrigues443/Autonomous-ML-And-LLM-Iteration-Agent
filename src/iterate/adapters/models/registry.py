"""Model factory — build any estimator from an installed, allow-listed ML library.

A candidate names a model by import path + params; we instantiate it. This is NOT a
hand-curated list — any estimator in scikit-learn, XGBoost, or LightGBM is fair game,
chosen by the Proposer's research. Models from *uninstalled* libraries or custom code
need the sandboxed code-gen path (v0.2), not this factory.

Spec shape (from `Candidate.changes`):

    {"model": "lightgbm.LGBMClassifier", "params": {"num_leaves": 64}}

`model` is optional (defaults to HistGradientBoosting for the task) and `params` is
optional. The import path must live under an allow-listed library, and `random_state`
is injected (for determinism) only when the estimator accepts it.
"""

from __future__ import annotations

import importlib
import inspect
from typing import Any, Literal

Task = Literal["classification", "regression"]

_ALLOWED_PREFIXES = ("sklearn.", "xgboost.", "lightgbm.")
_DEFAULT_MODEL: dict[str, str] = {
    "classification": "sklearn.ensemble.HistGradientBoostingClassifier",
    "regression": "sklearn.ensemble.HistGradientBoostingRegressor",
}

# Quiet-by-default constructor params for the noisy boosting libraries. XGBoost
# and LightGBM otherwise print training chatter (info banners, per-round eval) that
# buries the loop's own output. We inject these only when the candidate didn't set
# them — the agent's explicit choice always wins. Keyed by the estimator class
# name's prefix; only applied when the class actually accepts the param.
_QUIET_DEFAULTS: dict[str, dict[str, Any]] = {
    "XGB": {"verbosity": 0},
    "LGBM": {"verbose": -1},
}


def build_estimator(task: Task, spec: dict[str, Any], *, seed: int) -> Any:
    """Build an sklearn-compatible estimator from a ``{"model", "params"}`` spec."""
    path = spec.get("model") or _DEFAULT_MODEL[task]
    if not isinstance(path, str):
        raise TypeError(f"'model' must be a string import path, got {type(path).__name__}")
    params = dict(spec.get("params") or {})
    cls = _resolve(path)
    accepted = inspect.signature(cls).parameters
    if "random_state" in accepted and "random_state" not in params:
        params["random_state"] = seed
    for prefix, defaults in _QUIET_DEFAULTS.items():
        if cls.__name__.startswith(prefix):
            for key, value in defaults.items():
                if key in accepted and key not in params:
                    params[key] = value
    return cls(**params)


def _resolve(path: str) -> Any:
    if not path.startswith(_ALLOWED_PREFIXES):
        raise ValueError(
            f"model {path!r} is not in an allowed library {_ALLOWED_PREFIXES}; "
            "models from uninstalled or custom libraries need the sandboxed code-gen path (v0.2)"
        )
    module_path, _, class_name = path.rpartition(".")
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise ValueError(f"could not import {module_path!r} for model {path!r}") from exc
    cls = getattr(module, class_name, None)
    if not isinstance(cls, type):
        raise ValueError(f"{path!r} does not resolve to a class")
    return cls


__all__ = ["Task", "build_estimator"]
