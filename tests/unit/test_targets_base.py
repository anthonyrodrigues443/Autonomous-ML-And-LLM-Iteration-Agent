"""Tests for the BenchmarkTarget protocol.

A target only has to match the shape (structural typing), so a tiny fake that
implements `name`, `baseline()`, and `run()` should satisfy it; one missing a
method should not.
"""

from __future__ import annotations

from iterate.schemas.experiment import Candidate, ExperimentResult, Metrics
from iterate.targets.base import BenchmarkTarget


def _metrics(score: float) -> Metrics:
    return Metrics(values={"f1": score}, primary="f1", direction="maximize")


class _FakeTarget:
    name = "fake"

    def baseline(self) -> ExperimentResult:
        return ExperimentResult(experiment_id="baseline", metrics=_metrics(0.50))

    def run(self, candidate: Candidate) -> ExperimentResult:
        return ExperimentResult(experiment_id=candidate.id, metrics=_metrics(0.60))


def test_fake_target_satisfies_the_protocol() -> None:
    assert isinstance(_FakeTarget(), BenchmarkTarget)


def test_baseline_returns_an_experiment_result() -> None:
    result = _FakeTarget().baseline()
    assert isinstance(result, ExperimentResult)
    assert result.metrics is not None
    assert result.metrics.primary_value == 0.50


def test_run_returns_an_experiment_result() -> None:
    candidate = Candidate(description="try X", changes={"max_depth": 6}, rationale="why")
    result = _FakeTarget().run(candidate)
    assert isinstance(result, ExperimentResult)
    assert result.metrics is not None
    assert result.metrics.primary_value == 0.60


def test_target_missing_run_is_not_a_benchmark_target() -> None:
    class _NoRun:
        name = "broken"

        def baseline(self) -> ExperimentResult:  # pragma: no cover - never called
            return ExperimentResult(experiment_id="x", metrics=_metrics(0.1))

    assert not isinstance(_NoRun(), BenchmarkTarget)
