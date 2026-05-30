"""Tests for the Orchestrator — the agentic loop driver, with fakes only.

Live end-to-end (real LLM + real ModelTarget) is deliberately deferred to Day 6;
here we keep the loop logic deterministic and fast.
"""

from __future__ import annotations

from typing import Any

import pytest

from iterate.adapters.compute.local import LocalExecutor
from iterate.core.memory import InMemoryMemory
from iterate.core.orchestrator import Orchestrator, RunResult
from iterate.core.proposer import ProposerError
from iterate.core.terminator import Composite, LoopState, MaxIterations, Patience, Terminator
from iterate.schemas.experiment import Candidate, ExperimentResult, Metrics


class _FakeTarget:
    """A `BenchmarkTarget` that returns canned baseline + per-model results."""

    def __init__(
        self,
        *,
        baseline_score: float | None = 0.70,
        results: dict[str, float | None] | None = None,
        direction: str = "maximize",
        name: str = "fake-target",
    ) -> None:
        self.name = name
        self._baseline_score = baseline_score
        self._results = results or {}
        self._direction = direction

    def _result(self, experiment_id: str, score: float | None) -> ExperimentResult:
        if score is None:
            return ExperimentResult(experiment_id=experiment_id, error="canned failure")
        return ExperimentResult(
            experiment_id=experiment_id,
            metrics=Metrics(
                values={"f1": score},
                primary="f1",
                direction=self._direction,
                n_samples=100,
            ),
        )

    def baseline(self) -> ExperimentResult:
        return self._result("baseline", self._baseline_score)

    def run(self, candidate: Candidate) -> ExperimentResult:
        model = candidate.changes.get("model", "")
        return self._result(candidate.id, self._results.get(model))


class _FakeProposer:
    """A Proposer that emits a preset Candidate sequence; records calls."""

    def __init__(
        self,
        candidates: list[Candidate] | None = None,
        *,
        always_error: bool = False,
    ) -> None:
        self._candidates = list(candidates or [])
        self._always_error = always_error
        self.calls: list[dict[str, Any]] = []

    def propose(
        self,
        *,
        data_summary: str,
        baseline: ExperimentResult,
        current_model: str,
        history: list[Any],
    ) -> Candidate:
        n = len(self.calls) + 1
        self.calls.append(
            {
                "data_summary": data_summary,
                "baseline": baseline,
                "current_model": current_model,
                "history": list(history),
            }
        )
        if self._always_error:
            raise ProposerError(f"fake error at call {n}")
        if not self._candidates:
            raise ProposerError("no more canned candidates")
        return self._candidates.pop(0)


def _cand(model: str) -> Candidate:
    return Candidate(description=f"try {model}", changes={"model": model}, rationale="test")


def _orch(
    target: _FakeTarget,
    proposer: _FakeProposer,
    terminator: Terminator,
    memory: InMemoryMemory | None = None,
) -> Orchestrator:
    return Orchestrator(
        target,  # type: ignore[arg-type]
        proposer,  # type: ignore[arg-type]
        LocalExecutor(),
        terminator,
        memory or InMemoryMemory(),
        data_summary="x",
        baseline_model="base.Model",
    )


def test_runs_through_iterations_and_returns_best() -> None:
    target = _FakeTarget(baseline_score=0.70, results={"a.A": 0.72, "b.B": 0.75, "c.C": 0.73})
    proposer = _FakeProposer([_cand("a.A"), _cand("b.B"), _cand("c.C")])
    res = _orch(target, proposer, MaxIterations(3)).run()
    assert isinstance(res, RunResult)
    assert res.stopped_because == "max_iterations"
    assert len(res.history) == 3
    assert res.best is not None
    assert res.best.candidate.changes["model"] == "b.B"
    assert res.best.result is not None
    assert res.best.result.metrics is not None
    assert res.best.result.metrics.primary_value == pytest.approx(0.75)


def test_stops_at_patience_after_no_improvement() -> None:
    target = _FakeTarget(baseline_score=0.70, results={"a.A": 0.65, "b.B": 0.65})
    proposer = _FakeProposer([_cand("a.A"), _cand("b.B")])
    res = _orch(target, proposer, Composite(MaxIterations(10), Patience(2))).run()
    assert res.stopped_because == "patience"
    assert len(res.history) == 2
    assert res.best is None


def test_proposer_error_counts_toward_patience_no_history_entry() -> None:
    target = _FakeTarget()
    proposer = _FakeProposer(always_error=True)
    res = _orch(target, proposer, Composite(MaxIterations(10), Patience(2))).run()
    assert res.stopped_because == "patience"
    assert len(res.history) == 0
    assert res.best is None
    assert len(proposer.calls) == 2


def test_best_is_none_when_all_proposals_fail() -> None:
    target = _FakeTarget(baseline_score=0.70, results={"a.A": None, "b.B": None})
    proposer = _FakeProposer([_cand("a.A"), _cand("b.B")])
    res = _orch(target, proposer, MaxIterations(2)).run()
    assert res.stopped_because == "max_iterations"
    assert len(res.history) == 2
    assert all(exp.status == "failed" for exp in res.history)
    assert res.best is None


def test_current_model_updates_to_best() -> None:
    target = _FakeTarget(baseline_score=0.70, results={"a.A": 0.72, "b.B": 0.71})
    proposer = _FakeProposer([_cand("a.A"), _cand("b.B")])
    _orch(target, proposer, MaxIterations(2)).run()
    assert proposer.calls[0]["current_model"] == "base.Model"
    assert proposer.calls[1]["current_model"] == "a.A"


def test_minimize_direction_picks_lower_score() -> None:
    target = _FakeTarget(
        baseline_score=0.5,
        results={"a.A": 0.4, "b.B": 0.3, "c.C": 0.45},
        direction="minimize",
    )
    proposer = _FakeProposer([_cand("a.A"), _cand("b.B"), _cand("c.C")])
    res = _orch(target, proposer, MaxIterations(3)).run()
    assert res.best is not None
    assert res.best.candidate.changes["model"] == "b.B"
    assert res.best.result is not None
    assert res.best.result.metrics is not None
    assert res.best.result.metrics.primary_value == pytest.approx(0.3)


def test_history_grows_for_each_proposer_call() -> None:
    target = _FakeTarget(results={"a.A": 0.72, "b.B": 0.71})
    proposer = _FakeProposer([_cand("a.A"), _cand("b.B")])
    _orch(target, proposer, MaxIterations(2)).run()
    assert len(proposer.calls[0]["history"]) == 0
    assert len(proposer.calls[1]["history"]) == 1


def test_baseline_failure_returns_no_iterations() -> None:
    target = _FakeTarget(baseline_score=None)
    proposer = _FakeProposer([_cand("a.A")])
    res = _orch(target, proposer, MaxIterations(10)).run()
    assert res.stopped_because == "baseline_failed"
    assert res.history == []
    assert res.best is None
    assert len(proposer.calls) == 0


def test_orchestrator_propagates_terminator_reason() -> None:
    """The Orchestrator returns whatever stop reason the Terminator gave."""

    class _CustomReason:
        def update_and_check(self, state: LoopState) -> str | None:
            return "custom-stop"

    target = _FakeTarget(results={"a.A": 0.72})
    proposer = _FakeProposer([_cand("a.A")])
    res = _orch(target, proposer, _CustomReason()).run()
    assert res.stopped_because == "custom-stop"


def test_proposer_error_records_structured_failure_in_memory() -> None:
    target = _FakeTarget()
    proposer = _FakeProposer(always_error=True)
    memory = InMemoryMemory()
    _orch(target, proposer, Composite(MaxIterations(10), Patience(2)), memory).run()

    failures = memory.proposer_failures(target.name)
    assert len(failures) == 2
    assert failures[0].iteration == 1
    assert failures[0].current_model == "base.Model"
    assert "fake error" in failures[0].error
    assert failures[1].iteration == 2


def test_cross_run_history_is_seen_by_next_runs_proposer() -> None:
    target = _FakeTarget(baseline_score=0.70, results={"a.A": 0.72, "b.B": 0.73})
    memory = InMemoryMemory()

    proposer_1 = _FakeProposer([_cand("a.A")])
    _orch(target, proposer_1, MaxIterations(1), memory).run()

    proposer_2 = _FakeProposer([_cand("b.B")])
    _orch(target, proposer_2, MaxIterations(1), memory).run()

    # Second run's first propose call sees the first run's experiment in history.
    second_run_history = proposer_2.calls[0]["history"]
    assert len(second_run_history) == 1
    assert second_run_history[0].candidate.changes["model"] == "a.A"
