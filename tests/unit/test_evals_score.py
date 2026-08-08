"""Headroom-normalised scoring."""

from __future__ import annotations

import pytest

from evals.score import Status, aggregate, capture

pytestmark = pytest.mark.unit


def test_maximise_captures_half_the_available_gain() -> None:
    result = capture(baseline=0.60, best=0.65, ceiling=0.70, direction="maximize")

    assert result.status is Status.CAPTURED
    assert result.fraction == pytest.approx(0.5)
    assert result.improved


def test_minimise_reproduces_the_recorded_laptop_price_result() -> None:
    """The one certification number with unambiguous arithmetic behind it.

    rmse 411.89 baseline, a hand-swept ceiling 82.34 below it, and a run that
    reached 321.56 was recorded as capturing 110%. If the sign convention for
    minimising metrics ever flips, this is the test that says so.
    """
    result = capture(baseline=411.89, best=321.56, ceiling=329.55, direction="minimize")

    assert result.status is Status.CAPTURED
    assert result.gain == pytest.approx(90.33)
    assert result.headroom == pytest.approx(82.34)
    assert result.fraction == pytest.approx(1.0970, abs=1e-4)
    assert result.beat_the_ceiling


def test_capture_is_not_clamped_to_one() -> None:
    result = capture(baseline=0.5, best=0.9, ceiling=0.6, direction="maximize")

    assert result.fraction == pytest.approx(4.0)
    assert result.render() == "400%"


def test_no_headroom_is_not_a_zero() -> None:
    """A dataset already at its ceiling. Capturing nothing is the correct outcome
    and must not render as a failing 0%."""
    result = capture(baseline=1.0, best=None, ceiling=1.0, direction="maximize")

    assert result.status is Status.NO_HEADROOM
    assert result.fraction is None
    assert result.render() == "none avail."


def test_no_headroom_but_the_run_improved_anyway() -> None:
    result = capture(baseline=1.0, best=1.2, ceiling=1.0, direction="maximize")

    assert result.status is Status.NO_HEADROOM
    assert result.render() == "above ceiling"


def test_missing_ceiling_reports_the_raw_gain() -> None:
    result = capture(baseline=0.60, best=0.64, ceiling=None, direction="maximize")

    assert result.status is Status.UNMEASURED
    assert result.gain == pytest.approx(0.04)
    assert "no ceiling" in result.render()


def test_a_ceiling_below_the_baseline_is_a_corpus_error() -> None:
    result = capture(baseline=0.70, best=0.72, ceiling=0.60, direction="maximize")

    assert result.status is Status.BAD_CEILING
    assert result.render() == "corpus error"


def test_a_run_with_no_baseline_is_not_a_zero_capture() -> None:
    result = capture(baseline=None, best=None, ceiling=0.7, direction="maximize")

    assert result.status is Status.NO_BASELINE
    assert result.render() == "no baseline"


def test_no_candidate_beat_the_baseline_is_a_gain_of_zero() -> None:
    result = capture(baseline=0.60, best=None, ceiling=0.70, direction="maximize")

    assert result.status is Status.CAPTURED
    assert result.gain == 0.0
    assert result.fraction == pytest.approx(0.0)
    assert not result.improved


def test_an_unknown_direction_raises_rather_than_guessing() -> None:
    with pytest.raises(ValueError, match="direction"):
        capture(baseline=0.6, best=0.7, ceiling=0.8, direction="higher")


def test_aggregate_reports_the_median_with_its_spread() -> None:
    captures = [
        capture(baseline=0.60, best=best, ceiling=0.70, direction="maximize")
        for best in (0.62, 0.65, 0.69)
    ]

    summary = aggregate(captures)

    assert summary.n == 3
    assert summary.median_fraction == pytest.approx(0.5)
    assert summary.n_improved == 3
    assert summary.render() == "50% (20-90)"


def test_aggregate_of_one_run_omits_the_spread() -> None:
    summary = aggregate([capture(baseline=0.6, best=0.65, ceiling=0.7, direction="maximize")])

    assert summary.render() == "50%"


def test_aggregate_refuses_an_empty_list() -> None:
    """ "The cell did not run" and "the cell captured nothing" are the two things
    this module exists to keep apart, so an empty list cannot become a zero."""
    with pytest.raises(ValueError, match="zero runs"):
        aggregate([])


def test_repeats_measured_against_different_ceilings_are_flagged() -> None:
    summary = aggregate(
        [
            capture(baseline=0.6, best=0.65, ceiling=0.7, direction="maximize"),
            capture(baseline=0.6, best=0.65, ceiling=None, direction="maximize"),
        ]
    )

    assert summary.status is Status.BAD_CEILING
