"""Core domain contracts for the iteration loop.

These five models are the typed objects that flow between the agent's
components: the Proposer emits a ``Candidate``; the Orchestrator wraps it in an
``Experiment``; the Executor runs it and produces an ``ExperimentResult`` that
nests ``Metrics`` and a sample of ``FailureCase`` records. Objects are composed
(nested) rather than referenced by id so each ``Experiment`` is a self-contained,
auditable snapshot. Every model still carries an ``id`` (or back-reference) so the
sqlite Memory store can normalize and retrieve them later.
"""

import math
from datetime import UTC, datetime
from typing import Any, Literal, Self
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _new_id() -> str:
    return uuid4().hex


def _utc_now() -> datetime:
    return datetime.now(UTC)


class Metrics(BaseModel):
    """Quality measurements for one experiment.

    The metric names are task-specific and chosen by the agent (``values``);
    ``primary`` and ``direction`` define which one decides "better" and how.
    """

    model_config = ConfigDict(extra="forbid")

    values: dict[str, float]
    primary: str
    direction: Literal["maximize", "minimize"]
    n_samples: int | None = None

    @field_validator("values")
    @classmethod
    def _values_finite_and_nonempty(cls, v: dict[str, float]) -> dict[str, float]:
        if not v:
            raise ValueError("values must contain at least one metric")
        for name, value in v.items():
            if not math.isfinite(value):
                raise ValueError(f"metric {name!r} is not finite: {value}")
        return v

    @model_validator(mode="after")
    def _primary_in_values(self) -> Self:
        if self.primary not in self.values:
            raise ValueError(
                f"primary {self.primary!r} not in values keys {list(self.values)}"
            )
        return self

    @property
    def primary_value(self) -> float:
        return self.values[self.primary]


class FailureCase(BaseModel):
    """A single instance where the target produced a wrong or poor output."""

    model_config = ConfigDict(extra="forbid")

    input_data: Any
    expected: Any = None
    predicted: Any = None
    error_type: str | None = None
    score: float | None = None
    explanation: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Candidate(BaseModel):
    """A proposed change to try, emitted by the Proposer."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=_new_id)
    description: str
    changes: dict[str, Any]
    rationale: str
    source: Literal["proposer", "researcher", "human", "memory"] = "proposer"
    citations: list[str] = Field(default_factory=list)
    expected_improvement: float | None = None
    created_at: datetime = Field(default_factory=_utc_now)

    @field_validator("changes")
    @classmethod
    def _changes_nonempty(cls, v: dict[str, Any]) -> dict[str, Any]:
        if not v:
            raise ValueError("changes must describe at least one modification")
        return v


class ExperimentResult(BaseModel):
    """The outcome of running an experiment, produced by the Executor.

    ``metrics`` is present on a successful run and ``None`` when execution
    crashed (in which case ``error`` carries the reason).
    """

    model_config = ConfigDict(extra="forbid")

    experiment_id: str
    metrics: Metrics | None = None
    failure_cases: list[FailureCase] = Field(default_factory=list)
    artifacts: dict[str, str] = Field(default_factory=dict)
    logs: str | None = None
    duration_seconds: float | None = None
    cost_usd: float | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=_utc_now)

    @model_validator(mode="after")
    def _success_has_metrics(self) -> Self:
        if self.error is None and self.metrics is None:
            raise ValueError("a successful result (error is None) must include metrics")
        return self

    @property
    def succeeded(self) -> bool:
        return self.error is None


class ExperimentDigest(BaseModel):
    """A compact, structured summary of ONE finished experiment notebook, produced
    by the Summarizer so the Supervisor can reason over many experiments without
    ever holding the raw notebooks (which would bloat context and induce
    hallucination by mid-run). The digest is the unit of cross-notebook knowledge
    transfer: what was tried, what the data showed, what helped or hurt, and the
    score it reached."""

    model_config = ConfigDict(extra="forbid")

    techniques: list[str] = Field(default_factory=list)  # preprocessing + encoders + model + params
    data_insights: list[str] = Field(default_factory=list)  # facts learned about the data this run
    what_helped: list[str] = Field(default_factory=list)  # technique -> positive score effect
    what_hurt: list[str] = Field(default_factory=list)  # technique -> negative / no effect
    score: float | None = None  # the experiment's holdout score (None if it failed)
    val_trail: str = ""  # within-session validation scores in order, e.g. "0.55 -> 0.58"
    takeaway: str = ""  # one line: what this suggests trying next


class Experiment(BaseModel):
    """A committed run that wraps a Candidate against a target.

    Carries the run through its lifecycle: ``status`` advances from ``pending``
    to a terminal state and ``result`` is filled in once execution completes.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=_new_id)
    candidate: Candidate
    target: str
    hypothesis: str
    status: Literal["pending", "running", "completed", "failed", "aborted"] = "pending"
    iteration: int = 0
    result: ExperimentResult | None = None
    digest: ExperimentDigest | None = None  # Summarizer's compact summary, for the next Supervisor
    parent_id: str | None = None
    created_at: datetime = Field(default_factory=_utc_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None

    @model_validator(mode="after")
    def _completed_has_result(self) -> Self:
        if self.status == "completed" and self.result is None:
            raise ValueError("a completed experiment must have a result")
        return self


__all__ = [
    "Candidate",
    "Experiment",
    "ExperimentDigest",
    "ExperimentResult",
    "FailureCase",
    "Metrics",
]
