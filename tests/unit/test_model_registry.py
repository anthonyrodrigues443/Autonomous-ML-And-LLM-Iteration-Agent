"""Tests for the model factory — dynamic, allow-listed estimator construction."""

from __future__ import annotations

import pytest
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
)
from sklearn.linear_model import LinearRegression

from iterate.adapters.models.registry import build_estimator


def test_empty_spec_defaults_per_task() -> None:
    assert isinstance(build_estimator("classification", {}, seed=0), HistGradientBoostingClassifier)
    assert isinstance(build_estimator("regression", {}, seed=0), HistGradientBoostingRegressor)


def test_named_model_from_allowed_library() -> None:
    est = build_estimator(
        "classification",
        {"model": "sklearn.ensemble.RandomForestClassifier", "params": {"n_estimators": 7}},
        seed=0,
    )
    assert isinstance(est, RandomForestClassifier)
    assert est.n_estimators == 7


def test_xgboost_and_lightgbm_build() -> None:
    # The factory must support all three advertised libraries. This only *builds*
    # (no fit), so it stays instant even where LightGBM's macOS wheel is slow to train.
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    xgb = build_estimator("classification", {"model": "xgboost.XGBClassifier"}, seed=0)
    lgbm = build_estimator(
        "classification",
        {"model": "lightgbm.LGBMClassifier", "params": {"n_estimators": 2}},
        seed=0,
    )
    assert isinstance(xgb, XGBClassifier)
    assert isinstance(lgbm, LGBMClassifier)
    assert lgbm.n_estimators == 2


def test_random_state_injected_when_supported() -> None:
    est = build_estimator(
        "classification", {"model": "sklearn.ensemble.RandomForestClassifier"}, seed=42
    )
    assert est.random_state == 42


def test_random_state_skipped_when_unsupported() -> None:
    # LinearRegression has no random_state — building it must not blow up.
    est = build_estimator("regression", {"model": "sklearn.linear_model.LinearRegression"}, seed=42)
    assert isinstance(est, LinearRegression)


def test_explicit_random_state_not_overridden() -> None:
    est = build_estimator(
        "classification",
        {"model": "sklearn.ensemble.RandomForestClassifier", "params": {"random_state": 7}},
        seed=42,
    )
    assert est.random_state == 7


def test_disallowed_library_rejected() -> None:
    with pytest.raises(ValueError, match="not in an allowed library"):
        build_estimator("classification", {"model": "os.system"}, seed=0)


def test_path_resolving_to_non_class_rejected() -> None:
    with pytest.raises(ValueError, match="does not resolve to a class"):
        build_estimator("classification", {"model": "sklearn.ensemble.nonexistent_thing"}, seed=0)


def test_non_string_model_rejected() -> None:
    with pytest.raises(TypeError, match="must be a string import path"):
        build_estimator("classification", {"model": 123}, seed=0)
