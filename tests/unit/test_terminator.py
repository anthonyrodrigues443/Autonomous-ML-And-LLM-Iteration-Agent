"""Tests for the Terminator protocol + concretes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from iterate.core.terminator import (
    Composite,
    Deadline,
    LoopState,
    MaxIterations,
    Patience,
    Plateau,
    Terminator,
    default_terminator,
)
from iterate.schemas.experiment import Candidate, Experiment, ExperimentResult, Metrics

if TYPE_CHECKING:
    from iterate.core.terminator import AttemptOutcome


def _state(
    *,
    iteration: int = 1,
    outcome: AttemptOutcome = "no_improvement",
    elapsed_seconds: float = 0.0,
    last_experiment: Experiment | None = None,
) -> LoopState:
    baseline = ExperimentResult(
        experiment_id="baseline",
        metrics=Metrics(values={"f1": 0.70}, primary="f1", direction="maximize", n_samples=100),
    )
    return LoopState(
        iteration=iteration,
        baseline=baseline,
        best=None,
        last_experiment=last_experiment,
        last_attempt_outcome=outcome,
        elapsed_seconds=elapsed_seconds,
    )


def _experiment(score: float) -> Experiment:
    candidate = Candidate(description="x", changes={"model": "m"}, rationale="r")
    result = ExperimentResult(
        experiment_id="x",
        metrics=Metrics(values={"f1": score}, primary="f1", direction="maximize", n_samples=100),
    )
    return Experiment(
        candidate=candidate, target="t", hypothesis="x", status="completed", result=result
    )


# ─── MaxIterations ─────────────────────────────────────────────────────────


def test_max_iterations_fires_at_exactly_n() -> None:
    t = MaxIterations(3)
    assert t.update_and_check(_state(iteration=1)) is None
    assert t.update_and_check(_state(iteration=2)) is None
    assert t.update_and_check(_state(iteration=3)) == "max_iterations"


def test_max_iterations_rejects_zero() -> None:
    with pytest.raises(ValueError, match="n must"):
        MaxIterations(0)


# ─── Patience ──────────────────────────────────────────────────────────────


def test_patience_counts_no_improvement_and_proposer_error() -> None:
    t = Patience(2)
    assert t.update_and_check(_state(iteration=1, outcome="no_improvement")) is None
    assert t.update_and_check(_state(iteration=2, outcome="proposer_error")) == "patience"


def test_patience_resets_on_improvement() -> None:
    t = Patience(2)
    t.update_and_check(_state(iteration=1, outcome="no_improvement"))
    t.update_and_check(_state(iteration=2, outcome="improved"))
    assert t.update_and_check(_state(iteration=3, outcome="no_improvement")) is None
    assert t.update_and_check(_state(iteration=4, outcome="no_improvement")) == "patience"


def test_patience_rejects_zero() -> None:
    with pytest.raises(ValueError, match="k must"):
        Patience(0)


# ─── Deadline ──────────────────────────────────────────────────────────────


def test_deadline_fires_when_budget_reached() -> None:
    t = Deadline(seconds=1.0)
    assert t.update_and_check(_state(elapsed_seconds=0.5)) is None
    assert t.update_and_check(_state(elapsed_seconds=1.0)) == "deadline"


def test_deadline_rejects_non_positive() -> None:
    with pytest.raises(ValueError, match="seconds must"):
        Deadline(0)


# ─── Plateau ───────────────────────────────────────────────────────────────


def test_plateau_needs_window_filled_then_fires_on_tight_spread() -> None:
    t = Plateau(window=3, epsilon=0.01)
    assert (
        t.update_and_check(_state(outcome="improved", last_experiment=_experiment(0.700))) is None
    )
    assert (
        t.update_and_check(_state(outcome="no_improvement", last_experiment=_experiment(0.701)))
        is None
    )
    # 3rd succeeded score fills the window; spread 0.002 < epsilon 0.01 → plateau.
    assert (
        t.update_and_check(_state(outcome="no_improvement", last_experiment=_experiment(0.702)))
        == "plateau"
    )


def test_plateau_does_not_fire_with_wide_spread() -> None:
    t = Plateau(window=2, epsilon=0.01)
    t.update_and_check(_state(outcome="improved", last_experiment=_experiment(0.70)))
    assert t.update_and_check(_state(outcome="improved", last_experiment=_experiment(0.80))) is None


def test_plateau_ignores_proposer_error_and_failed_results() -> None:
    t = Plateau(window=2, epsilon=0.01)
    t.update_and_check(_state(outcome="improved", last_experiment=_experiment(0.70)))
    # Proposer error — last_experiment is None; should be ignored, window not advanced.
    assert t.update_and_check(_state(outcome="proposer_error", last_experiment=None)) is None
    # A succeeded score of 0.705 (window now has [0.70, 0.705], spread 0.005 < 0.01).
    assert (
        t.update_and_check(_state(outcome="improved", last_experiment=_experiment(0.705)))
        == "plateau"
    )


def test_plateau_rejects_invalid_args() -> None:
    with pytest.raises(ValueError, match="window must"):
        Plateau(window=1, epsilon=0.01)
    with pytest.raises(ValueError, match="epsilon must"):
        Plateau(window=2, epsilon=-0.1)


# ─── Composite ─────────────────────────────────────────────────────────────


def test_composite_returns_first_fired_reason() -> None:
    c = Composite(MaxIterations(1), Patience(5))
    # MaxIterations fires at iteration 1; Patience hasn't accumulated enough.
    assert c.update_and_check(_state(iteration=1, outcome="no_improvement")) == "max_iterations"


class _Spy:
    def __init__(self) -> None:
        self.calls = 0

    def update_and_check(self, state: LoopState) -> str | None:
        self.calls += 1
        return None


def test_composite_calls_all_children_even_after_early_fire() -> None:
    spy = _Spy()
    c = Composite(MaxIterations(1), spy)
    c.update_and_check(_state(iteration=1))
    assert spy.calls == 1  # the spy was still called even though MaxIterations fired


def test_composite_rejects_empty() -> None:
    with pytest.raises(ValueError, match="at least one"):
        Composite()


# ─── Factory + Protocol conformance ────────────────────────────────────────


def test_default_terminator_composes_max_iter_and_patience() -> None:
    t = default_terminator(max_iterations=5, patience=2)
    t.update_and_check(_state(iteration=1, outcome="no_improvement"))
    assert t.update_and_check(_state(iteration=2, outcome="no_improvement")) == "patience"


def test_default_terminator_optionally_includes_deadline() -> None:
    t = default_terminator(max_iterations=10, patience=10, deadline_seconds=1.0)
    assert t.update_and_check(_state(iteration=1, elapsed_seconds=1.5)) == "deadline"


def test_all_concretes_satisfy_the_protocol() -> None:
    assert isinstance(MaxIterations(1), Terminator)
    assert isinstance(Patience(1), Terminator)
    assert isinstance(Deadline(1.0), Terminator)
    assert isinstance(Plateau(2, 0.01), Terminator)
    assert isinstance(Composite(MaxIterations(1)), Terminator)
