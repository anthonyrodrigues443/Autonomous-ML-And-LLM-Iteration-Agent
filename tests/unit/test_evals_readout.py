"""Reading a run out of a memory database written by any version."""

from __future__ import annotations

import json
import sqlite3
from typing import TYPE_CHECKING, Any

import pytest

from evals.readout import UnreadableRunError, read

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit

_SCHEMA = """
CREATE TABLE runs (
    id TEXT PRIMARY KEY, target_name TEXT NOT NULL, baseline_json TEXT NOT NULL,
    started_at TEXT NOT NULL, finished_at TEXT, stopped_because TEXT
);
CREATE TABLE experiments (
    id TEXT PRIMARY KEY, run_id TEXT NOT NULL, target_name TEXT NOT NULL,
    experiment_json TEXT NOT NULL, created_at TEXT NOT NULL
);
"""


def _metrics(value: float, *, primary: str = "f1", direction: str = "maximize") -> dict[str, Any]:
    return {"values": {primary: value}, "primary": primary, "direction": direction}


def _write_db(
    path: Path,
    *,
    baseline: dict[str, Any] | None,
    experiments: list[dict[str, Any]],
    stopped_because: str = "patience",
) -> Path:
    conn = sqlite3.connect(path)
    conn.executescript(_SCHEMA)
    conn.execute(
        "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?)",
        (
            "run-1",
            "tabular-model",
            json.dumps({"experiment_id": "baseline", "metrics": baseline}),
            "2026-08-08T10:00:00+00:00",
            "2026-08-08T11:00:00+00:00",
            stopped_because,
        ),
    )
    for index, payload in enumerate(experiments):
        conn.execute(
            "INSERT INTO experiments VALUES (?, ?, ?, ?, ?)",
            (
                f"exp-{index}",
                "run-1",
                "tabular-model",
                json.dumps(payload),
                f"2026-08-08T10:{index:02d}:00+00:00",
            ),
        )
    conn.commit()
    conn.close()
    return path


def _experiment(
    score: float | None,
    *,
    iteration: int = 1,
    rejected: bool = False,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    changes: dict[str, Any] = {"model": "sklearn.ensemble.RandomForestClassifier"}
    if rejected:
        changes["critic_rejected"] = "fitted the scaler on the holdout"
    payload: dict[str, Any] = {
        "id": f"exp-{iteration}",
        "iteration": iteration,
        "status": "completed" if score is not None else "failed",
        "candidate": {"description": "try a forest", "changes": changes},
        "result": (
            {"experiment_id": f"exp-{iteration}", "metrics": _metrics(score)}
            if score is not None
            else {"experiment_id": f"exp-{iteration}", "error": "kernel died"}
        ),
    }
    payload.update(extra or {})
    return payload


def test_reads_baseline_and_best(tmp_path: Path) -> None:
    db = _write_db(
        tmp_path / "memory.db",
        baseline=_metrics(0.60),
        experiments=[_experiment(0.62, iteration=1), _experiment(0.67, iteration=2)],
    )

    (run,) = read(db)

    assert run.baseline == pytest.approx(0.60)
    assert run.best == pytest.approx(0.67)
    assert run.metric == "f1"
    assert run.direction == "maximize"
    assert run.n_experiments == 2
    assert run.stopped_because == "patience"


def test_a_database_written_by_an_older_version_still_reads(tmp_path: Path) -> None:
    """The whole reason this module avoids the pydantic models.

    An experiment carrying fields the current schema does not know about — and
    missing ones it does — must still be readable, because that is exactly what a
    v0.1 database looks like to v0.5 code. Parsing through `extra="forbid"` models
    would raise here and make cross-version comparison impossible.
    """
    ancient = {
        "id": "exp-1",
        "iteration": 1,
        "status": "completed",
        "candidate": {"description": "old", "changes": {"model": "x"}},
        "result": {"experiment_id": "exp-1", "metrics": _metrics(0.71)},
        "a_field_removed_in_v0_2": "still here",
        "another_one": {"nested": True},
    }
    db = _write_db(tmp_path / "memory.db", baseline=_metrics(0.60), experiments=[ancient])

    (run,) = read(db)

    assert run.best == pytest.approx(0.71)


def test_a_rejected_score_cannot_become_the_best(tmp_path: Path) -> None:
    """A leak the Critic caught is not a result. Banking it in the measurement tool
    would reintroduce the exact bug the Critic exists to prevent."""
    db = _write_db(
        tmp_path / "memory.db",
        baseline=_metrics(0.60),
        experiments=[
            _experiment(0.81, iteration=1, rejected=True),
            _experiment(0.64, iteration=2),
        ],
    )

    (run,) = read(db)

    assert run.best == pytest.approx(0.64)
    assert run.n_rejected == 1


def test_best_is_none_when_nothing_beat_the_baseline(tmp_path: Path) -> None:
    db = _write_db(
        tmp_path / "memory.db",
        baseline=_metrics(0.60),
        experiments=[_experiment(0.55, iteration=1), _experiment(0.58, iteration=2)],
    )

    (run,) = read(db)

    assert run.best is None


def test_a_minimising_metric_takes_the_lowest_score(tmp_path: Path) -> None:
    db = _write_db(
        tmp_path / "memory.db",
        baseline=_metrics(400.0, primary="rmse", direction="minimize"),
        experiments=[
            {
                "id": "exp-1",
                "iteration": 1,
                "status": "completed",
                "candidate": {"description": "ridge", "changes": {"model": "x"}},
                "result": {
                    "experiment_id": "exp-1",
                    "metrics": _metrics(380.0, primary="rmse", direction="minimize"),
                },
            },
            {
                "id": "exp-2",
                "iteration": 2,
                "status": "completed",
                "candidate": {"description": "boosted", "changes": {"model": "y"}},
                "result": {
                    "experiment_id": "exp-2",
                    "metrics": _metrics(321.56, primary="rmse", direction="minimize"),
                },
            },
        ],
    )

    (run,) = read(db)

    assert run.direction == "minimize"
    assert run.best == pytest.approx(321.56)


def test_failed_experiments_are_counted_not_scored(tmp_path: Path) -> None:
    db = _write_db(
        tmp_path / "memory.db",
        baseline=_metrics(0.60),
        experiments=[_experiment(None, iteration=1), _experiment(0.63, iteration=2)],
    )

    (run,) = read(db)

    assert run.n_failed == 1
    assert run.best == pytest.approx(0.63)


def test_a_run_that_died_before_its_baseline_reads_as_no_baseline(tmp_path: Path) -> None:
    db = _write_db(tmp_path / "memory.db", baseline=None, experiments=[])

    (run,) = read(db)

    assert run.baseline is None
    assert run.best is None


def test_a_database_that_is_not_a_memory_db_raises(tmp_path: Path) -> None:
    path = tmp_path / "other.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE something_else (id TEXT)")
    conn.commit()
    conn.close()

    with pytest.raises(UnreadableRunError):
        read(path)
