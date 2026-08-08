"""The eval store — the accumulating record of every cell ever measured.

This is what makes the harness a system rather than a script. A full sweep is
hours, and the released versions are frozen: v0.2's number on churn can never
change. Recomputing the grid every time a version lands would make the harness too
expensive to actually use, which is how measurement tools die. So results
accumulate here and a sweep fills only what is missing.

Three tables, and the keys matter more than the columns:

* `sweeps` — one row per invocation, holding the conditions and the harness's own
  git sha, so a result can be traced to the code that produced it.
* `cells` — one row per (version, dataset, dataset hash, repeat, conditions).
  That tuple is the identity, and it is unique. Change the dataset bytes or the
  budget and you get a NEW cell rather than an overwrite, because the old number
  is still a true measurement of something else.
* `ceilings` — keyed by dataset AND its hash, for the same reason. A ceiling
  measured on last month's export does not bound this month's.

Separate from `.iterate/memory.db` on purpose. That one is a run's own working
memory, written by whichever version produced it; this one is the harness's
long-lived ledger and is only ever written by the harness.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Self
from uuid import uuid4

if TYPE_CHECKING:
    from pathlib import Path

    from evals.config import Conditions

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sweeps (
    id                     TEXT PRIMARY KEY,
    started_at             TEXT NOT NULL,
    finished_at            TEXT,
    conditions_json        TEXT NOT NULL,
    conditions_fingerprint TEXT NOT NULL,
    harness_sha            TEXT NOT NULL DEFAULT '',
    note                   TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS cells (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    sweep_id               TEXT NOT NULL,
    version                TEXT NOT NULL,
    dataset                TEXT NOT NULL,
    dataset_hash           TEXT NOT NULL,
    repeat                 INTEGER NOT NULL,
    conditions_fingerprint TEXT NOT NULL,
    started_at             TEXT NOT NULL,
    finished_at            TEXT NOT NULL DEFAULT '',
    status                 TEXT NOT NULL,
    metric                 TEXT NOT NULL DEFAULT '',
    direction              TEXT NOT NULL DEFAULT '',
    baseline               REAL,
    best                   REAL,
    n_experiments          INTEGER NOT NULL DEFAULT 0,
    n_failed               INTEGER NOT NULL DEFAULT 0,
    n_rejected             INTEGER NOT NULL DEFAULT 0,
    stopped_because        TEXT NOT NULL DEFAULT '',
    duration_seconds       REAL,
    memory_db              TEXT NOT NULL DEFAULT '',
    log_path               TEXT NOT NULL DEFAULT '',
    argv                   TEXT NOT NULL DEFAULT '',
    error                  TEXT NOT NULL DEFAULT ''
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_cell_identity
    ON cells(version, dataset, dataset_hash, repeat, conditions_fingerprint);
CREATE TABLE IF NOT EXISTS ceilings (
    dataset      TEXT NOT NULL,
    dataset_hash TEXT NOT NULL,
    metric       TEXT NOT NULL,
    ceiling      REAL NOT NULL,
    direction    TEXT NOT NULL,
    baseline     REAL,
    method       TEXT NOT NULL,
    measured_at  TEXT NOT NULL,
    detail       TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (dataset, dataset_hash, metric)
);
"""

# A cell's outcome. `ok` alone means the numbers are usable; everything else is a
# recorded failure that a later sweep is free to retry.
STATUS_OK = "ok"
STATUS_RUN_FAILED = "run_failed"
STATUS_TIMEOUT = "timeout"
STATUS_UNREADABLE = "unreadable"


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class CellRecord:
    """One measured (version, dataset, repeat) cell."""

    version: str
    dataset: str
    dataset_hash: str
    repeat: int
    conditions_fingerprint: str
    status: str
    started_at: str = ""
    finished_at: str = ""
    metric: str = ""
    direction: str = ""
    baseline: float | None = None
    best: float | None = None
    n_experiments: int = 0
    n_failed: int = 0
    n_rejected: int = 0
    stopped_because: str = ""
    duration_seconds: float | None = None
    memory_db: str = ""
    log_path: str = ""
    argv: str = ""
    error: str = ""

    @property
    def identity(self) -> tuple[str, str, str, int]:
        return (self.version, self.dataset, self.dataset_hash, self.repeat)


@dataclass(frozen=True)
class Ceiling:
    """The best score a no-LLM brute-force sweep reached on one dataset."""

    dataset: str
    dataset_hash: str
    metric: str
    ceiling: float
    direction: str
    baseline: float | None
    method: str
    measured_at: str
    detail: str = ""


class Store:
    """Thin sqlite wrapper. Every write commits; a sweep that dies halfway keeps
    the cells it already finished, which is the whole point of resuming."""

    def __init__(self, path: Path | str) -> None:
        self._conn = sqlite3.connect(path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._conn.close()

    # ─── sweeps ────────────────────────────────────────────────────────────
    def start_sweep(self, conditions: Conditions, *, harness_sha: str = "", note: str = "") -> str:
        sweep_id = uuid4().hex
        self._conn.execute(
            "INSERT INTO sweeps (id, started_at, conditions_json, conditions_fingerprint, "
            "harness_sha, note) VALUES (?, ?, ?, ?, ?, ?)",
            (
                sweep_id,
                _now(),
                conditions.as_json(),
                conditions.fingerprint(),
                harness_sha,
                note,
            ),
        )
        self._conn.commit()
        return sweep_id

    def finish_sweep(self, sweep_id: str) -> None:
        self._conn.execute("UPDATE sweeps SET finished_at = ? WHERE id = ?", (_now(), sweep_id))
        self._conn.commit()

    # ─── cells ─────────────────────────────────────────────────────────────
    def record_cell(self, sweep_id: str, cell: CellRecord) -> None:
        """Write a cell, replacing any earlier attempt at the same identity.

        Replace rather than ignore, because the earlier row at this identity is
        normally a failed attempt being retried. A successful cell is never
        re-attempted unless the caller passes --force, so this cannot quietly
        overwrite a good measurement with a worse one.
        """
        self._conn.execute(
            "INSERT OR REPLACE INTO cells (sweep_id, version, dataset, dataset_hash, repeat, "
            "conditions_fingerprint, started_at, finished_at, status, metric, direction, "
            "baseline, best, n_experiments, n_failed, n_rejected, stopped_because, "
            "duration_seconds, memory_db, log_path, argv, error) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                sweep_id,
                cell.version,
                cell.dataset,
                cell.dataset_hash,
                cell.repeat,
                cell.conditions_fingerprint,
                cell.started_at or _now(),
                cell.finished_at,
                cell.status,
                cell.metric,
                cell.direction,
                cell.baseline,
                cell.best,
                cell.n_experiments,
                cell.n_failed,
                cell.n_rejected,
                cell.stopped_because,
                cell.duration_seconds,
                cell.memory_db,
                cell.log_path,
                cell.argv,
                cell.error,
            ),
        )
        self._conn.commit()

    def completed(self, conditions_fingerprint: str) -> set[tuple[str, str, str, int]]:
        """Identities that already have a usable result under these conditions.

        Only `ok` cells count. A cell that timed out or crashed is left open so the
        next sweep picks it up, which is the difference between resuming and giving
        up on whatever failed once.
        """
        rows = self._conn.execute(
            "SELECT version, dataset, dataset_hash, repeat FROM cells "
            "WHERE conditions_fingerprint = ? AND status = ?",
            (conditions_fingerprint, STATUS_OK),
        ).fetchall()
        return {(r["version"], r["dataset"], r["dataset_hash"], int(r["repeat"])) for r in rows}

    def cells(self, conditions_fingerprint: str | None = None) -> list[CellRecord]:
        sql = "SELECT * FROM cells"
        params: tuple[str, ...] = ()
        if conditions_fingerprint is not None:
            sql += " WHERE conditions_fingerprint = ?"
            params = (conditions_fingerprint,)
        sql += " ORDER BY dataset, version, repeat"
        return [_cell_from_row(row) for row in self._conn.execute(sql, params).fetchall()]

    # ─── ceilings ──────────────────────────────────────────────────────────
    def put_ceiling(self, ceiling: Ceiling, *, replace: bool = False) -> Ceiling:
        """Store a ceiling, keeping the better of the old and new value.

        A ceiling is a LOWER BOUND on what is achievable, so the right value for a
        dataset is the best anyone has ever reached on it, not the best the most
        recent sweep happened to find. Two measurements of the same data proved this
        the day the harness was written: the automated sweep found a far better
        ceiling than the hand sweep on laptop price, and a slightly worse one on
        churn. Overwriting would have thrown away real knowledge in the second case
        and quietly lowered the bar the agent is judged against.

        `replace` forces the new value in, for the case where an old ceiling is not
        merely worse but wrong.
        """
        existing = self.get_ceiling(ceiling.dataset, ceiling.dataset_hash, ceiling.metric)
        if existing is not None and not replace:
            better = (
                ceiling.ceiling > existing.ceiling
                if ceiling.direction == "maximize"
                else ceiling.ceiling < existing.ceiling
            )
            if not better:
                return existing

        self._write_ceiling(ceiling)
        return ceiling

    def _write_ceiling(self, ceiling: Ceiling) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO ceilings (dataset, dataset_hash, metric, ceiling, direction, "
            "baseline, method, measured_at, detail) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                ceiling.dataset,
                ceiling.dataset_hash,
                ceiling.metric,
                ceiling.ceiling,
                ceiling.direction,
                ceiling.baseline,
                ceiling.method,
                ceiling.measured_at,
                ceiling.detail,
            ),
        )
        self._conn.commit()

    def get_ceiling(self, dataset: str, dataset_hash: str, metric: str) -> Ceiling | None:
        row = self._conn.execute(
            "SELECT * FROM ceilings WHERE dataset = ? AND dataset_hash = ? AND metric = ?",
            (dataset, dataset_hash, metric),
        ).fetchone()
        return _ceiling_from_row(row) if row is not None else None

    def ceilings(self) -> list[Ceiling]:
        rows = self._conn.execute("SELECT * FROM ceilings ORDER BY dataset").fetchall()
        return [_ceiling_from_row(row) for row in rows]


def _cell_from_row(row: sqlite3.Row) -> CellRecord:
    return CellRecord(
        version=row["version"],
        dataset=row["dataset"],
        dataset_hash=row["dataset_hash"],
        repeat=int(row["repeat"]),
        conditions_fingerprint=row["conditions_fingerprint"],
        status=row["status"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        metric=row["metric"],
        direction=row["direction"],
        baseline=row["baseline"],
        best=row["best"],
        n_experiments=int(row["n_experiments"]),
        n_failed=int(row["n_failed"]),
        n_rejected=int(row["n_rejected"]),
        stopped_because=row["stopped_because"],
        duration_seconds=row["duration_seconds"],
        memory_db=row["memory_db"],
        log_path=row["log_path"],
        argv=row["argv"],
        error=row["error"],
    )


def _ceiling_from_row(row: sqlite3.Row) -> Ceiling:
    return Ceiling(
        dataset=row["dataset"],
        dataset_hash=row["dataset_hash"],
        metric=row["metric"],
        ceiling=float(row["ceiling"]),
        direction=row["direction"],
        baseline=row["baseline"],
        method=row["method"],
        measured_at=row["measured_at"],
        detail=row["detail"],
    )


__all__ = [
    "STATUS_OK",
    "STATUS_RUN_FAILED",
    "STATUS_TIMEOUT",
    "STATUS_UNREADABLE",
    "Ceiling",
    "CellRecord",
    "Store",
]
