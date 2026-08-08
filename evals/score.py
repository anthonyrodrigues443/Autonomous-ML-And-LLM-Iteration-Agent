"""Headroom-normalised scoring — turning a raw score into a readable verdict.

A run that reports "no candidate beat the baseline" is unreadable on its own. It
means the agent failed on a dataset with room, or it means the dataset had no room
and the agent correctly found none. The v0.4 certification could not tell those
apart until every dataset got a brute-force ceiling measured by hand, with no LLM
involved. This module is that method as code: a run scores as the fraction of
AVAILABLE gain it captured, so 0.0 on a dataset with 20% headroom and 0.0 on a
dataset with none stop looking like the same result.

Two properties worth stating because they are easy to get wrong:

*Capture is not clamped to 100%.* Ceilings from a hand sweep are LOWER bounds on
what is achievable, not maxima, so beating one is a real and interesting outcome
rather than an error to be squashed — the laptop-price run captured 110% by
beating a hand-engineered baseline with a technique from a 2022 paper. Clamping
would have hidden the single best result this project has produced.

*Zero headroom is a distinct state, not a zero.* On a dataset already at its
ceiling the correct behaviour is to capture nothing, and rendering that as `0.0`
in a column of percentages reads as failure. It gets its own status.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from statistics import median

# Floating-point slack when asking "did this run move at all". Scores here are
# metric values (f1, rmse), not currency, so an absolute epsilon on a relative
# comparison is fine and keeps the rule readable.
_EPSILON = 1e-9


class Status(StrEnum):
    """Why a capture fraction is, or is not, a number."""

    CAPTURED = "captured"  # headroom existed; `fraction` is meaningful
    NO_HEADROOM = "no_headroom"  # ceiling == baseline; capturing nothing is correct
    UNMEASURED = "unmeasured"  # no ceiling recorded for this dataset yet
    BAD_CEILING = "bad_ceiling"  # ceiling is worse than the baseline: a corpus error
    NO_BASELINE = "no_baseline"  # the run never established a baseline (it died early)


@dataclass(frozen=True)
class Capture:
    """What one run achieved, expressed against what was available to achieve."""

    status: Status
    gain: float  # improvement over the baseline, always signed so positive is better
    headroom: float | None  # improvement available, same sign convention
    fraction: float | None  # gain / headroom, None unless status is CAPTURED
    baseline: float | None
    best: float | None
    ceiling: float | None
    direction: str

    @property
    def improved(self) -> bool:
        return self.gain > _EPSILON

    @property
    def beat_the_ceiling(self) -> bool:
        """The run went past a ceiling that was supposed to bound it.

        Interesting either way: on a hand-swept ceiling it means the sweep was too
        short, and on a leaderboard ceiling it means something worth publishing.
        """
        return self.headroom is not None and self.gain > self.headroom + _EPSILON

    def render(self) -> str:
        """One cell of the results table."""
        if self.status is Status.CAPTURED and self.fraction is not None:
            return f"{self.fraction * 100:.0f}%"
        return _first_render(self.status, self.gain, 1 if self.improved else 0)


def _signed(better: float, worse: float, direction: str) -> float:
    """Improvement of `better` over `worse`, positive when it is genuinely better.

    The whole module works in this one convention so that maximise and minimise
    metrics share every downstream comparison. Getting this backwards is the same
    class of bug as the metric-direction one from sprint 2, where a minimising
    metric would have banked the worst result as best.
    """
    return (better - worse) if direction == "maximize" else (worse - better)


def capture(
    *,
    baseline: float | None,
    best: float | None,
    ceiling: float | None,
    direction: str,
) -> Capture:
    """Score one run against the gain that was available to it.

    `best` is None when no candidate beat the baseline, which is a gain of zero
    rather than missing data — the run happened and produced nothing.
    """
    if direction not in ("maximize", "minimize"):
        raise ValueError(f"direction must be maximize or minimize, got {direction!r}")

    if baseline is None:
        return Capture(
            status=Status.NO_BASELINE,
            gain=0.0,
            headroom=None,
            fraction=None,
            baseline=None,
            best=best,
            ceiling=ceiling,
            direction=direction,
        )

    gain = _signed(best, baseline, direction) if best is not None else 0.0

    if ceiling is None:
        return Capture(
            status=Status.UNMEASURED,
            gain=gain,
            headroom=None,
            fraction=None,
            baseline=baseline,
            best=best,
            ceiling=None,
            direction=direction,
        )

    headroom = _signed(ceiling, baseline, direction)

    if headroom < -_EPSILON:
        return Capture(
            status=Status.BAD_CEILING,
            gain=gain,
            headroom=headroom,
            fraction=None,
            baseline=baseline,
            best=best,
            ceiling=ceiling,
            direction=direction,
        )

    if headroom <= _EPSILON:
        return Capture(
            status=Status.NO_HEADROOM,
            gain=gain,
            headroom=0.0,
            fraction=None,
            baseline=baseline,
            best=best,
            ceiling=ceiling,
            direction=direction,
        )

    return Capture(
        status=Status.CAPTURED,
        gain=gain,
        headroom=headroom,
        fraction=gain / headroom,
        baseline=baseline,
        best=best,
        ceiling=ceiling,
        direction=direction,
    )


@dataclass(frozen=True)
class Aggregate:
    """Several repeats of the same (version, dataset) cell, summarised.

    A single run of an LLM-driven loop says almost nothing: the same version on the
    same data varies run to run. The median is what goes in the table and the spread
    is what says whether the median means anything.
    """

    n: int
    status: Status
    median_fraction: float | None
    min_fraction: float | None
    max_fraction: float | None
    median_gain: float
    n_improved: int

    def render(self) -> str:
        if self.status is not Status.CAPTURED or self.median_fraction is None:
            return _first_render(self.status, self.median_gain, self.n_improved)
        cell = f"{self.median_fraction * 100:.0f}%"
        if self.n > 1 and self.min_fraction is not None and self.max_fraction is not None:
            cell += f" ({self.min_fraction * 100:.0f}-{self.max_fraction * 100:.0f})"
        return cell


def _first_render(status: Status, gain: float, n_improved: int) -> str:
    if status is Status.NO_HEADROOM:
        return "none avail." if n_improved == 0 else "above ceiling"
    if status is Status.UNMEASURED:
        return f"{gain:+.4f} (no ceiling)"
    if status is Status.BAD_CEILING:
        return "corpus error"
    return "no baseline"


def aggregate(captures: list[Capture]) -> Aggregate:
    """Summarise repeats of one cell. Raises on an empty list rather than inventing a
    zero, because "the cell did not run" and "the cell captured nothing" are the two
    things this whole module exists to keep apart."""
    if not captures:
        raise ValueError("cannot aggregate zero runs")

    # The status is a property of the dataset, not of the run, so any disagreement
    # means the corpus changed mid-sweep and the cell is not comparable.
    statuses = {c.status for c in captures}
    status = captures[0].status if len(statuses) == 1 else Status.BAD_CEILING

    gains = [c.gain for c in captures]
    fractions = [c.fraction for c in captures if c.fraction is not None]

    return Aggregate(
        n=len(captures),
        status=status,
        median_fraction=median(fractions) if fractions else None,
        min_fraction=min(fractions) if fractions else None,
        max_fraction=max(fractions) if fractions else None,
        median_gain=median(gains),
        n_improved=sum(1 for c in captures if c.improved),
    )


__all__ = ["Aggregate", "Capture", "Status", "aggregate", "capture"]
