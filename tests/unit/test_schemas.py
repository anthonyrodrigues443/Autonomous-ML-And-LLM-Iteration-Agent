"""Unit tests for the core domain schemas.

Each test proves a validator fires (raising ``ValidationError``) or that a
happy-path construction succeeds. The point is to lock the contracts the rest
of the loop depends on.
"""
from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from iterate.schemas.experiment import (
    Candidate,
    Experiment,
    ExperimentResult,
    FailureCase,
    Metrics,
)

pytestmark = pytest.mark.unit


def make_metrics() -> Metrics:
    return Metrics(values={"pr_auc": 0.83, "f1": 0.71}, primary="pr_auc", direction="maximize")


def make_candidate() -> Candidate:
    return Candidate(
        description="tune scale_pos_weight for class imbalance",
        changes={"scale_pos_weight": 8},
        rationale="baseline ignores the 7:1 imbalance; weighting should lift recall",
    )


# ─── Metrics ───────────────────────────────────────────────────


def test_metrics_happy_path_exposes_primary_value() -> None:
    m = make_metrics()
    assert m.primary_value == 0.83
    assert m.n_samples is None


def test_metrics_primary_must_be_a_key_in_values() -> None:
    with pytest.raises(ValidationError):
        Metrics(values={"f1": 0.7}, primary="pr_auc", direction="maximize")


def test_metrics_rejects_empty_values() -> None:
    with pytest.raises(ValidationError):
        Metrics(values={}, primary="f1", direction="maximize")


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_metrics_rejects_non_finite_values(bad: float) -> None:
    with pytest.raises(ValidationError):
        Metrics(values={"f1": bad}, primary="f1", direction="maximize")


def test_metrics_forbids_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        Metrics(value={"f1": 0.7}, primary="f1", direction="maximize")  # type: ignore[call-arg]


def test_metrics_rejects_invalid_direction() -> None:
    with pytest.raises(ValidationError):
        Metrics(values={"f1": 0.7}, primary="f1", direction="bigger")  # type: ignore[arg-type]


# ─── Candidate ─────────────────────────────────────────────────


def test_candidate_happy_path_defaults() -> None:
    c = make_candidate()
    assert c.source == "proposer"
    assert c.citations == []
    assert c.created_at.tzinfo is not None


def test_candidate_rejects_empty_changes() -> None:
    with pytest.raises(ValidationError):
        Candidate(description="noop", changes={}, rationale="changes nothing")


def test_candidate_ids_are_unique() -> None:
    assert make_candidate().id != make_candidate().id


# ─── FailureCase ───────────────────────────────────────────────


def test_failure_case_minimal() -> None:
    fc = FailureCase(input_data={"text": "ship it tomorrow"})
    assert fc.expected is None
    assert fc.metadata == {}


# ─── ExperimentResult ──────────────────────────────────────────


def test_result_success_requires_metrics() -> None:
    with pytest.raises(ValidationError):
        ExperimentResult(experiment_id="exp1")


def test_result_failure_may_omit_metrics() -> None:
    r = ExperimentResult(experiment_id="exp1", error="sandbox OOM")
    assert r.succeeded is False
    assert r.metrics is None


def test_result_success_path() -> None:
    r = ExperimentResult(experiment_id="exp1", metrics=make_metrics())
    assert r.succeeded is True
    assert r.failure_cases == []


# ─── Experiment ────────────────────────────────────────────────


def test_experiment_defaults_to_pending() -> None:
    e = Experiment(candidate=make_candidate(), target="churn-baseline", hypothesis="lift PR-AUC")
    assert e.status == "pending"
    assert e.result is None


def test_experiment_completed_requires_result() -> None:
    with pytest.raises(ValidationError):
        Experiment(
            candidate=make_candidate(),
            target="churn-baseline",
            hypothesis="lift PR-AUC",
            status="completed",
        )


def test_experiment_completed_with_result_is_valid() -> None:
    e = Experiment(
        candidate=make_candidate(),
        target="churn-baseline",
        hypothesis="lift PR-AUC",
        status="completed",
        result=ExperimentResult(experiment_id="exp1", metrics=make_metrics()),
    )
    assert e.result is not None
    assert e.result.succeeded is True


def test_experiment_ids_are_unique() -> None:
    a = Experiment(candidate=make_candidate(), target="t", hypothesis="h")
    b = Experiment(candidate=make_candidate(), target="t", hypothesis="h")
    assert a.id != b.id


def test_experiment_round_trips_through_json() -> None:
    original = Experiment(
        candidate=make_candidate(),
        target="churn-baseline",
        hypothesis="lift PR-AUC",
        status="completed",
        result=ExperimentResult(
            experiment_id="exp1",
            metrics=make_metrics(),
            failure_cases=[FailureCase(input_data={"id": 1}, expected=1, predicted=0)],
        ),
    )
    restored = Experiment.model_validate_json(original.model_dump_json())
    assert restored == original
    assert restored.result is not None
    assert restored.result.metrics is not None
    assert restored.result.metrics.primary_value == 0.83
