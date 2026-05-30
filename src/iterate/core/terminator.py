"""Stop conditions for the agentic loop.

The Orchestrator delegates "should I stop?" to a `Terminator` each iteration. A
terminator is stateful (some — patience, plateau — need to track history), so
`update_and_check` is one method: feed the current `LoopState`, return a stop
reason string or `None` to continue. One method removes a class of ordering bugs
(notify-then-check vs check-then-notify).

Concretes shipped Day 3:
- `MaxIterations(n)` — hard iteration cap.
- `Patience(k)` — k consecutive non-improvements (proposer_error counts).
- `Deadline(seconds)` — wall-clock budget.
- `Plateau(window, epsilon)` — spread across the last `window` succeeded scores < ε.
- `Composite(*terminators)` — calls all (each accumulates state correctly),
  returns the first non-None reason.

`default_terminator()` gives a sane Composite for typical use.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from iterate.schemas.experiment import Experiment, ExperimentResult


AttemptOutcome = Literal["improved", "no_improvement", "proposer_error"]


@dataclass(frozen=True)
class LoopState:
    """The slice of the Orchestrator's state that terminators read each iteration."""

    iteration: int
    baseline: ExperimentResult
    best: Experiment | None
    last_experiment: Experiment | None
    last_attempt_outcome: AttemptOutcome
    elapsed_seconds: float


@runtime_checkable
class Terminator(Protocol):
    """The contract every stop condition obeys."""

    def update_and_check(self, state: LoopState) -> str | None:
        """Update internal state with the latest iteration; return a stop reason or None."""
        ...


class MaxIterations:
    """Stop after exactly `n` attempted iterations."""

    def __init__(self, n: int) -> None:
        if n < 1:
            raise ValueError("n must be >= 1")
        self._n = n

    def update_and_check(self, state: LoopState) -> str | None:
        return "max_iterations" if state.iteration >= self._n else None


class Patience:
    """Stop after `k` consecutive non-improvements (proposer_error counts)."""

    def __init__(self, k: int) -> None:
        if k < 1:
            raise ValueError("k must be >= 1")
        self._k = k
        self._streak = 0

    def update_and_check(self, state: LoopState) -> str | None:
        if state.last_attempt_outcome == "improved":
            self._streak = 0
        else:
            self._streak += 1
        return "patience" if self._streak >= self._k else None


class Deadline:
    """Stop when the wall-clock budget is exhausted."""

    def __init__(self, seconds: float) -> None:
        if seconds <= 0:
            raise ValueError("seconds must be > 0")
        self._budget = seconds

    def update_and_check(self, state: LoopState) -> str | None:
        return "deadline" if state.elapsed_seconds >= self._budget else None


class Plateau:
    """Stop when the spread across the last `window` succeeded scores is below `epsilon`.

    Direction-agnostic by design: a tight cluster of recent scores means we've stopped
    moving the metric in *either* direction, which is what plateau really means.
    Proposer errors and execution failures don't contribute to the window.
    """

    def __init__(self, window: int, epsilon: float) -> None:
        if window < 2:
            raise ValueError("window must be >= 2")
        if epsilon < 0:
            raise ValueError("epsilon must be >= 0")
        self._window = window
        self._epsilon = epsilon
        self._scores: deque[float] = deque(maxlen=window)

    def update_and_check(self, state: LoopState) -> str | None:
        exp = state.last_experiment
        if exp is None or exp.result is None:
            return None
        result = exp.result
        if not result.succeeded or result.metrics is None:
            return None
        self._scores.append(result.metrics.primary_value)
        if len(self._scores) < self._window:
            return None
        return "plateau" if (max(self._scores) - min(self._scores)) < self._epsilon else None


class Composite:
    """Combine terminators: call all (each accumulates state), return the first reason."""

    def __init__(self, *terminators: Terminator) -> None:
        if not terminators:
            raise ValueError("Composite requires at least one terminator")
        self._terminators = list(terminators)

    def update_and_check(self, state: LoopState) -> str | None:
        # Call each so internal state stays correct; return the first non-None reason.
        reasons = [t.update_and_check(state) for t in self._terminators]
        for reason in reasons:
            if reason is not None:
                return reason
        return None


def default_terminator(
    *,
    max_iterations: int = 10,
    patience: int = 3,
    deadline_seconds: float | None = None,
) -> Terminator:
    """A sane default Composite for typical use (always bounded by max_iterations)."""
    parts: list[Terminator] = [MaxIterations(max_iterations), Patience(patience)]
    if deadline_seconds is not None:
        parts.append(Deadline(deadline_seconds))
    return Composite(*parts)


__all__ = [
    "AttemptOutcome",
    "Composite",
    "Deadline",
    "LoopState",
    "MaxIterations",
    "Patience",
    "Plateau",
    "Terminator",
    "default_terminator",
]
