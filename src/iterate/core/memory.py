"""Persistent memory for the agentic loop.

The Orchestrator records each experiment + each Proposer failure through a
`Memory` instance. `history(target_name)` returns every Experiment ever recorded
for that target — across all runs — which is what the Proposer reads to know what
has already been tried. The point: the agent's memory survives `iterate run`
exiting, so the next run picks up where the last left off.

Two implementations ship:
- `InMemoryMemory` — dict-backed, ephemeral, used by tests and any caller that
  doesn't want persistence.
- `SqliteMemory(path)` — backed by a single sqlite file on disk (default
  `.iterate/memory.db`). No server, no auth — file permissions control access.

Both implement the same `Memory` protocol; nothing upstream knows which it got.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable
from uuid import uuid4

from iterate.schemas.experiment import Experiment

if TYPE_CHECKING:
    from iterate.schemas.experiment import ExperimentResult


def _now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class ProposerFailure:
    """A structured record of a `ProposerError` — separate from Experiment history."""

    run_id: str
    target_name: str
    iteration: int
    current_model: str
    error: str
    created_at: datetime


@runtime_checkable
class Memory(Protocol):
    """The contract every memory backend obeys."""

    def start_run(self, target_name: str, baseline: ExperimentResult) -> str:
        """Begin a new run; return its id."""
        ...

    def record(self, run_id: str, experiment: Experiment) -> None:
        """Record one experiment (succeeded or failed at execution)."""
        ...

    def record_proposer_failure(
        self, run_id: str, iteration: int, current_model: str, error: str
    ) -> None:
        """Record a Proposer error (no Candidate; not in `history()`)."""
        ...

    def history(self, target_name: str) -> list[Experiment]:
        """All experiments recorded for this target, oldest first, across runs."""
        ...

    def proposer_failures(self, target_name: str) -> list[ProposerFailure]:
        """All proposer failures recorded for this target, oldest first, across runs."""
        ...

    def finish_run(self, run_id: str, stopped_because: str) -> None:
        """Mark a run as finished."""
        ...


# ──────────────────────────────────────────────────────────────────────────────
# InMemoryMemory — dict-backed, ephemeral
# ──────────────────────────────────────────────────────────────────────────────


class InMemoryMemory:
    """Dict-backed Memory. Loses state on process exit; ideal for tests."""

    def __init__(self) -> None:
        self._runs: dict[str, dict[str, Any]] = {}
        self._experiments: list[tuple[str, str, Experiment]] = []
        self._proposer_failures: list[ProposerFailure] = []

    def start_run(self, target_name: str, baseline: ExperimentResult) -> str:
        run_id = uuid4().hex
        self._runs[run_id] = {
            "target_name": target_name,
            "baseline": baseline,
            "started_at": _now(),
            "finished_at": None,
            "stopped_because": None,
        }
        return run_id

    def record(self, run_id: str, experiment: Experiment) -> None:
        target_name = self._target_name_for(run_id)
        self._experiments.append((run_id, target_name, experiment))

    def record_proposer_failure(
        self, run_id: str, iteration: int, current_model: str, error: str
    ) -> None:
        target_name = self._target_name_for(run_id)
        self._proposer_failures.append(
            ProposerFailure(
                run_id=run_id,
                target_name=target_name,
                iteration=iteration,
                current_model=current_model,
                error=error,
                created_at=_now(),
            )
        )

    def history(self, target_name: str) -> list[Experiment]:
        return [exp for (_, tname, exp) in self._experiments if tname == target_name]

    def proposer_failures(self, target_name: str) -> list[ProposerFailure]:
        return [f for f in self._proposer_failures if f.target_name == target_name]

    def finish_run(self, run_id: str, stopped_because: str) -> None:
        record = self._runs.get(run_id)
        if record is None:
            raise ValueError(f"unknown run_id: {run_id}")
        record["finished_at"] = _now()
        record["stopped_because"] = stopped_because

    def _target_name_for(self, run_id: str) -> str:
        record = self._runs.get(run_id)
        if record is None:
            raise ValueError(f"unknown run_id: {run_id}")
        target_name: str = record["target_name"]
        return target_name


# ──────────────────────────────────────────────────────────────────────────────
# SqliteMemory — persistent, single file on disk
# ──────────────────────────────────────────────────────────────────────────────


_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id              TEXT PRIMARY KEY,
    target_name     TEXT NOT NULL,
    baseline_json   TEXT NOT NULL,
    started_at      TEXT NOT NULL,
    finished_at     TEXT,
    stopped_because TEXT
);
CREATE TABLE IF NOT EXISTS experiments (
    id              TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL,
    target_name     TEXT NOT NULL,
    experiment_json TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(id)
);
CREATE INDEX IF NOT EXISTS idx_experiments_target ON experiments(target_name, created_at);
CREATE TABLE IF NOT EXISTS proposer_failures (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT NOT NULL,
    target_name     TEXT NOT NULL,
    iteration       INTEGER NOT NULL,
    current_model   TEXT NOT NULL,
    error           TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(id)
);
CREATE INDEX IF NOT EXISTS idx_proposer_failures_target
    ON proposer_failures(target_name, created_at);
"""


class SqliteMemory:
    """File-backed Memory (a single sqlite file). Persists across processes."""

    def __init__(self, db_path: Path | str) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def start_run(self, target_name: str, baseline: ExperimentResult) -> str:
        run_id = uuid4().hex
        self._conn.execute(
            "INSERT INTO runs (id, target_name, baseline_json, started_at) VALUES (?, ?, ?, ?)",
            (run_id, target_name, baseline.model_dump_json(), _now().isoformat()),
        )
        self._conn.commit()
        return run_id

    def record(self, run_id: str, experiment: Experiment) -> None:
        target_name = self._target_name_for(run_id)
        self._conn.execute(
            "INSERT INTO experiments (id, run_id, target_name, experiment_json, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                experiment.id,
                run_id,
                target_name,
                experiment.model_dump_json(),
                _now().isoformat(),
            ),
        )
        self._conn.commit()

    def record_proposer_failure(
        self, run_id: str, iteration: int, current_model: str, error: str
    ) -> None:
        target_name = self._target_name_for(run_id)
        self._conn.execute(
            "INSERT INTO proposer_failures "
            "(run_id, target_name, iteration, current_model, error, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (run_id, target_name, iteration, current_model, error, _now().isoformat()),
        )
        self._conn.commit()

    def history(self, target_name: str) -> list[Experiment]:
        rows = self._conn.execute(
            "SELECT experiment_json FROM experiments WHERE target_name = ? ORDER BY created_at",
            (target_name,),
        ).fetchall()
        return [Experiment.model_validate_json(row["experiment_json"]) for row in rows]

    def proposer_failures(self, target_name: str) -> list[ProposerFailure]:
        rows = self._conn.execute(
            "SELECT run_id, target_name, iteration, current_model, error, created_at "
            "FROM proposer_failures WHERE target_name = ? ORDER BY created_at",
            (target_name,),
        ).fetchall()
        return [
            ProposerFailure(
                run_id=row["run_id"],
                target_name=row["target_name"],
                iteration=row["iteration"],
                current_model=row["current_model"],
                error=row["error"],
                created_at=_parse_iso(row["created_at"]),
            )
            for row in rows
        ]

    def finish_run(self, run_id: str, stopped_because: str) -> None:
        cursor = self._conn.execute(
            "UPDATE runs SET stopped_because = ?, finished_at = ? WHERE id = ?",
            (stopped_because, _now().isoformat(), run_id),
        )
        self._conn.commit()
        if cursor.rowcount == 0:
            raise ValueError(f"unknown run_id: {run_id}")

    def close(self) -> None:
        self._conn.close()

    def _target_name_for(self, run_id: str) -> str:
        row = self._conn.execute("SELECT target_name FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            raise ValueError(f"unknown run_id: {run_id}")
        target_name: str = row["target_name"]
        return target_name


def _parse_iso(value: str) -> datetime:
    return datetime.fromisoformat(value)


__all__ = [
    "InMemoryMemory",
    "Memory",
    "ProposerFailure",
    "SqliteMemory",
]
