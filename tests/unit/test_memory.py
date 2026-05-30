"""Tests for the Memory protocol + InMemoryMemory + SqliteMemory.

The shared tests are parameterized over both backends — they must behave
identically. Sqlite-specific tests verify on-disk persistence + roundtrip.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from iterate.core.memory import InMemoryMemory, Memory, ProposerFailure, SqliteMemory
from iterate.schemas.experiment import Candidate, Experiment, ExperimentResult, Metrics

if TYPE_CHECKING:
    from pathlib import Path


def _baseline(score: float = 0.70) -> ExperimentResult:
    return ExperimentResult(
        experiment_id="baseline",
        metrics=Metrics(values={"f1": score}, primary="f1", direction="maximize", n_samples=100),
    )


def _experiment(model: str, score: float | None = None) -> Experiment:
    candidate = Candidate(description=f"try {model}", changes={"model": model}, rationale="r")
    if score is None:
        return Experiment(
            candidate=candidate,
            target="t",
            hypothesis="x",
            status="failed",
            result=ExperimentResult(experiment_id="x", error="boom"),
        )
    result = ExperimentResult(
        experiment_id="x",
        metrics=Metrics(values={"f1": score}, primary="f1", direction="maximize", n_samples=100),
    )
    return Experiment(
        candidate=candidate,
        target="t",
        hypothesis="x",
        status="completed",
        result=result,
    )


@pytest.fixture(params=["in-memory", "sqlite"])
def memory(request: pytest.FixtureRequest, tmp_path: Path) -> Memory:
    if request.param == "in-memory":
        return InMemoryMemory()
    return SqliteMemory(tmp_path / "memory.db")


# ─── Shared behavior (both backends) ─────────────────────────────────────


def test_implementations_satisfy_the_memory_protocol(tmp_path: Path) -> None:
    assert isinstance(InMemoryMemory(), Memory)
    assert isinstance(SqliteMemory(tmp_path / "memory.db"), Memory)


def test_records_and_returns_experiments_in_order(memory: Memory) -> None:
    run_id = memory.start_run("t", _baseline())
    memory.record(run_id, _experiment("a.A", 0.72))
    memory.record(run_id, _experiment("b.B", 0.75))

    history = memory.history("t")

    assert len(history) == 2
    assert history[0].candidate.changes["model"] == "a.A"
    assert history[1].candidate.changes["model"] == "b.B"


def test_history_scoped_by_target_name(memory: Memory) -> None:
    run_a = memory.start_run("target-a", _baseline())
    run_b = memory.start_run("target-b", _baseline())
    memory.record(run_a, _experiment("a.A", 0.7))
    memory.record(run_b, _experiment("b.B", 0.8))

    assert [e.candidate.changes["model"] for e in memory.history("target-a")] == ["a.A"]
    assert [e.candidate.changes["model"] for e in memory.history("target-b")] == ["b.B"]


def test_proposer_failures_kept_separate_from_history(memory: Memory) -> None:
    run_id = memory.start_run("t", _baseline())
    memory.record_proposer_failure(run_id, iteration=1, current_model="m", error="boom")

    assert memory.history("t") == []
    failures = memory.proposer_failures("t")
    assert len(failures) == 1
    assert failures[0].iteration == 1
    assert failures[0].current_model == "m"
    assert failures[0].error == "boom"
    assert isinstance(failures[0], ProposerFailure)


def test_history_accumulates_across_runs_on_same_target(memory: Memory) -> None:
    run_1 = memory.start_run("t", _baseline())
    memory.record(run_1, _experiment("a.A", 0.7))
    memory.finish_run(run_1, "max_iterations")

    run_2 = memory.start_run("t", _baseline())
    memory.record(run_2, _experiment("b.B", 0.8))

    history = memory.history("t")
    assert [e.candidate.changes["model"] for e in history] == ["a.A", "b.B"]


def test_finish_run_is_callable(memory: Memory) -> None:
    run_id = memory.start_run("t", _baseline())
    memory.finish_run(run_id, "max_iterations")  # no exception


def test_unknown_run_id_raises(memory: Memory) -> None:
    with pytest.raises(ValueError, match="unknown run_id"):
        memory.record("does-not-exist", _experiment("a.A", 0.7))


# ─── Sqlite-specific ─────────────────────────────────────────────────────


def test_sqlite_persists_across_instances(tmp_path: Path) -> None:
    path = tmp_path / "memory.db"

    first = SqliteMemory(path)
    run_id = first.start_run("t", _baseline())
    first.record(run_id, _experiment("xgboost.XGBClassifier", 0.78))
    first.finish_run(run_id, "max_iterations")
    first.close()

    second = SqliteMemory(path)
    history = second.history("t")
    assert len(history) == 1
    assert history[0].candidate.changes["model"] == "xgboost.XGBClassifier"


def test_sqlite_creates_parent_dir_and_schema_on_first_use(tmp_path: Path) -> None:
    nested = tmp_path / "sub" / "deeper" / "memory.db"
    memory = SqliteMemory(nested)

    assert nested.exists()
    run_id = memory.start_run("t", _baseline())  # would fail without schema
    memory.record(run_id, _experiment("a.A", 0.7))
    assert len(memory.history("t")) == 1


def test_sqlite_roundtrip_preserves_experiment_shape(tmp_path: Path) -> None:
    memory = SqliteMemory(tmp_path / "memory.db")
    run_id = memory.start_run("t", _baseline())
    original = _experiment("xgboost.XGBClassifier", 0.78)
    memory.record(run_id, original)

    loaded = memory.history("t")[0]
    assert loaded.id == original.id
    assert loaded.candidate.changes == original.candidate.changes
    assert loaded.candidate.rationale == original.candidate.rationale
    assert loaded.result is not None
    assert loaded.result.metrics is not None
    assert loaded.result.metrics.primary_value == pytest.approx(0.78)
    assert loaded.status == "completed"
