"""Tests for the supervised agent loop (Supervisor + Coder), with fakes."""

from __future__ import annotations

from iterate.core.agent_loop import _winning_code, run_supervised
from iterate.core.coder import Cell, CodingResult
from iterate.core.memory import InMemoryMemory
from iterate.core.supervisor import SupervisorDecision
from iterate.core.terminator import MaxIterations
from iterate.schemas.experiment import Candidate, Experiment, ExperimentResult, Metrics


def _result(score: float) -> ExperimentResult:
    return ExperimentResult(
        experiment_id="x",
        metrics=Metrics(values={"f1": score}, primary="f1", direction="maximize", n_samples=100),
    )


class _FakeTarget:
    name = "tabular-model"

    def baseline(self) -> ExperimentResult:
        return _result(0.50)  # the bar to beat

    def run(self, candidate: object) -> ExperimentResult:  # pragma: no cover - unused
        raise NotImplementedError


class _FakeSupervisor:
    def __init__(self, decisions: list[SupervisorDecision]) -> None:
        self._decisions = list(decisions)
        self.seen_history_lens: list[int] = []

    def decide(self, *, data_summary: str, baseline: object, history: list) -> SupervisorDecision:
        self.seen_history_lens.append(len(history))
        return self._decisions.pop(0)


class _FakeCoder:
    def __init__(self, result: ExperimentResult) -> None:
        self._result = result

    def run(
        self, *, dataset: object, brief: str, experiment_id: str,
        starting_code: str | None = None, starting_score: float | None = None,
    ) -> CodingResult:
        cells = [
            Cell("# preamble", "loaded", "", None, "preamble"),
            Cell("model.fit(); to_csv('predictions.csv')", "ok", "", None, "agent"),
        ]
        return CodingResult(result=self._result, cells=cells)


def _loop(supervisor: object, coders: list[_FakeCoder], terminator: object):
    it = iter(coders)
    return run_supervised(
        target=_FakeTarget(),  # type: ignore[arg-type]
        dataset=object(),  # type: ignore[arg-type]
        supervisor=supervisor,  # type: ignore[arg-type]
        make_coder=lambda: next(it),  # type: ignore[arg-type,return-value]
        terminator=terminator,  # type: ignore[arg-type]
        memory=InMemoryMemory(),
        data_summary="d",
    )


def test_loop_runs_experiments_and_tracks_best() -> None:
    sup = _FakeSupervisor(
        [SupervisorDecision(False, "a", "try a"), SupervisorDecision(False, "b", "try b")]
    )
    result = _loop(sup, [_FakeCoder(_result(0.60)), _FakeCoder(_result(0.55))], MaxIterations(2))
    assert result.stopped_because == "max_iterations"
    assert len(result.history) == 2
    # each experiment reaches the supervisor exactly once (memory only, no double-count)
    assert sup.seen_history_lens == [0, 1]
    assert result.best is not None
    assert result.best.result.metrics.primary_value == 0.60  # the better of the two, beats baseline
    # the session cells are stored on the candidate for the notebook
    assert result.best.candidate.changes["cells"][0]["source"] == "preamble"


class _ExplodingCoder:
    def run(self, **kwargs: object) -> CodingResult:
        raise TimeoutError("backend timed out after retries")


def test_a_crashing_coder_fails_the_iteration_not_the_run() -> None:
    sup = _FakeSupervisor(
        [SupervisorDecision(False, "a", "try a"), SupervisorDecision(False, "b", "try b")]
    )
    result = _loop(
        sup, [_ExplodingCoder(), _FakeCoder(_result(0.60))], MaxIterations(2)  # type: ignore[list-item]
    )
    # iteration 1 exploded mid-experiment; the loop survived and iteration 2 scored
    assert result.stopped_because == "max_iterations"
    assert result.best is not None
    assert result.best.result.metrics.primary_value == 0.60


def test_supervisor_stop_ends_the_loop_immediately() -> None:
    sup = _FakeSupervisor([SupervisorDecision(True, "", "")])
    result = _loop(sup, [], MaxIterations(5))
    assert result.stopped_because == "supervisor"
    assert result.history == []
    assert result.best is None


def _exp_with_cells(cells: list[Cell]) -> Experiment:
    return Experiment(
        candidate=Candidate(
            description="d",
            changes={"cells": [c.__dict__ for c in cells]},
            rationale="r",
        ),
        target="t",
        hypothesis="h",
        status="completed",
        result=_result(0.6),
    )


def test_winning_code_concatenates_successful_staged_cells() -> None:
    # a staged session: prepare -> (errored attempt) -> model -> submit.
    # carry-forward keeps the successful cells in order and drops the errored one.
    best = _exp_with_cells(
        [
            Cell("# preamble", "", "", None, "preamble"),
            Cell("X_tr = prepare(X_train)", "shape (100, 8)", "", None, "agent"),
            Cell("model.fit(broken)", "", "", "NameError: broken", "agent"),
            Cell("print(validation_f1)", "0.61", "", None, "agent"),
            Cell("write_predictions()", "wrote 100", "", None, "agent"),
        ]
    )
    code = _winning_code(best)
    assert code is not None
    assert "X_tr = prepare(X_train)" in code
    assert "write_predictions()" in code
    assert "model.fit(broken)" not in code  # the errored cell is dropped
    assert "# preamble" not in code  # only agent cells carry forward
    # order preserved: prepare before submit
    assert code.index("X_tr = prepare") < code.index("write_predictions")


def test_winning_code_is_none_without_a_best() -> None:
    assert _winning_code(None) is None
