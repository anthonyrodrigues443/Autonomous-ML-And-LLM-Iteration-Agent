"""Reading a finished run out of its memory database.

Deliberately reads sqlite with raw SQL and `json.loads`, never through
`iterate.schemas.experiment`. That looks like a missed reuse opportunity and is the
single decision that makes this harness able to do its job.

The point of the harness is comparing v0.1 against v0.5. Those versions wrote their
databases with their own `Experiment` models, and today's models are
`extra="forbid"` pydantic — so a database written by an older version fails
validation against the current schema the moment a field was added or renamed
(`Candidate.citations` landed in v0.4, `Experiment.digest` before it). Parsing
through the current models would mean the tool could only ever read runs produced
by the version it shipped with, which is exactly the comparison it exists to make.

So this module treats a memory database as an untrusted foreign format: introspect
which columns exist, take only what is needed, and let anything missing be missing.
Every field it reads has been stable since v0.1 (`runs.baseline_json`,
`experiments.experiment_json`, and the `metrics.values` / `metrics.primary` pair
inside them); anything newer is read defensively.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

# Set on `candidate.changes` by the Critic when an experiment is proven to have
# cheated (v0.4+). Absent in every earlier database, which is correct: there was no
# Critic, so nothing was ever rejected.
_REJECTED_KEY = "critic_rejected"


@dataclass(frozen=True)
class ExperimentRow:
    """One experiment as the database recorded it, flattened."""

    iteration: int
    status: str
    score: float | None
    rejected: bool
    error: str | None

    @property
    def counts_as_best(self) -> bool:
        """Rejected scores are excluded from best-of.

        A leaked score that the Critic caught is not a result. Banking it here would
        reintroduce, in the measurement tool, exactly the bug the Critic was built to
        prevent in the loop.
        """
        return self.score is not None and not self.rejected


@dataclass(frozen=True)
class RunReadout:
    """One run, reduced to the numbers the harness compares."""

    run_id: str
    target_name: str
    metric: str
    direction: str
    baseline: float | None
    best: float | None
    stopped_because: str
    experiments: list[ExperimentRow] = field(default_factory=list)

    @property
    def n_experiments(self) -> int:
        return len(self.experiments)

    @property
    def n_failed(self) -> int:
        return sum(1 for e in self.experiments if e.score is None)

    @property
    def n_rejected(self) -> int:
        return sum(1 for e in self.experiments if e.rejected)


class UnreadableRunError(Exception):
    """The database exists but is not shaped like any iterate memory db."""


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    try:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    except sqlite3.DatabaseError as exc:  # not a sqlite file at all
        raise UnreadableRunError(f"cannot introspect {table}: {exc}") from exc
    return {str(row[1]) for row in rows}


def _primary_score(metrics: dict[str, Any] | None) -> float | None:
    """The primary metric's value, or None if this result has no usable score."""
    if not metrics:
        return None
    values = metrics.get("values")
    primary = metrics.get("primary")
    if not isinstance(values, dict) or not isinstance(primary, str):
        return None
    value = values.get(primary)
    return float(value) if isinstance(value, int | float) else None


def _metric_identity(metrics: dict[str, Any] | None) -> tuple[str, str]:
    """(name, direction) as the run itself recorded them.

    Taken from the database rather than looked up in today's metric registry, for
    the same reason as everything else here: an old run's ruler is whatever that
    version used, and a registry lookup would silently re-rule history.
    """
    if not metrics:
        return ("", "")
    primary = metrics.get("primary")
    direction = metrics.get("direction")
    return (
        primary if isinstance(primary, str) else "",
        direction if isinstance(direction, str) else "",
    )


def _experiment_row(payload: dict[str, Any]) -> ExperimentRow:
    result = payload.get("result") or {}
    metrics = result.get("metrics") if isinstance(result, dict) else None
    changes = (payload.get("candidate") or {}).get("changes") or {}
    return ExperimentRow(
        iteration=int(payload.get("iteration") or 0),
        status=str(payload.get("status") or ""),
        score=_primary_score(metrics if isinstance(metrics, dict) else None),
        rejected=bool(changes.get(_REJECTED_KEY)) if isinstance(changes, dict) else False,
        error=(result.get("error") if isinstance(result, dict) else None),
    )


def _best(rows: list[ExperimentRow], baseline: float | None, direction: str) -> float | None:
    """The best banked score, or None when nothing beat the baseline.

    Returning None rather than the baseline keeps "the agent produced nothing" a
    distinct fact downstream, which is the same distinction the whole harness is
    built to preserve.
    """
    scores = [e.score for e in rows if e.counts_as_best and e.score is not None]
    if not scores:
        return None
    best = max(scores) if direction == "maximize" else min(scores)
    if baseline is None:
        return best
    improved = best > baseline if direction == "maximize" else best < baseline
    return best if improved else None


def read(db_path: Path | str) -> list[RunReadout]:
    """Every run in one memory database, oldest first.

    The harness gives each cell its own database, so this is normally a single run;
    it returns a list anyway so a developer can point it at their own working
    `.iterate/memory.db` without surprises.
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        return _read(conn)
    finally:
        conn.close()


def _read(conn: sqlite3.Connection) -> list[RunReadout]:
    run_columns = _columns(conn, "runs")
    if not {"id", "baseline_json"} <= run_columns:
        raise UnreadableRunError("runs table has no id/baseline_json — not an iterate memory db")

    wanted = [
        c for c in ("id", "target_name", "baseline_json", "stopped_because") if c in run_columns
    ]
    selected = set(wanted)
    order = "started_at" if "started_at" in run_columns else "rowid"
    rows = conn.execute(f"SELECT {', '.join(wanted)} FROM runs ORDER BY {order}").fetchall()

    readouts: list[RunReadout] = []
    for row in rows:
        baseline_payload = json.loads(row["baseline_json"])
        metrics = baseline_payload.get("metrics") if isinstance(baseline_payload, dict) else None
        metric, direction = _metric_identity(metrics if isinstance(metrics, dict) else None)
        baseline = _primary_score(metrics if isinstance(metrics, dict) else None)

        experiments = [
            _experiment_row(json.loads(exp["experiment_json"]))
            for exp in conn.execute(
                "SELECT experiment_json FROM experiments WHERE run_id = ? ORDER BY created_at",
                (row["id"],),
            ).fetchall()
        ]

        readouts.append(
            RunReadout(
                run_id=str(row["id"]),
                target_name=str(row["target_name"]) if "target_name" in selected else "",
                metric=metric,
                direction=direction,
                baseline=baseline,
                best=_best(experiments, baseline, direction),
                stopped_because=(
                    str(row["stopped_because"] or "") if "stopped_because" in selected else ""
                ),
                experiments=experiments,
            )
        )
    return readouts


__all__ = ["ExperimentRow", "RunReadout", "UnreadableRunError", "read"]
