"""The accumulating eval store: identity, resume, and ceiling keying."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from evals.config import Conditions
from evals.store import STATUS_OK, STATUS_TIMEOUT, Ceiling, CellRecord, Store

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit

CONDITIONS = Conditions(
    model="gemma4:12b",
    backend="ollama",
    max_iterations=10,
    patience=3,
    repeats=3,
    timeout_minutes=90,
    sweep_threads=4,
)


def _cell(**overrides: object) -> CellRecord:
    base = CellRecord(
        version="0.4.0",
        dataset="churn",
        dataset_hash="abc123",
        repeat=1,
        conditions_fingerprint=CONDITIONS.fingerprint(),
        status=STATUS_OK,
        baseline=0.60,
        best=0.64,
        metric="f1",
        direction="maximize",
    )
    return replace(base, **overrides)  # type: ignore[arg-type]


def test_a_recorded_cell_comes_back(tmp_path: Path) -> None:
    with Store(tmp_path / "results.db") as store:
        sweep = store.start_sweep(CONDITIONS, harness_sha="deadbee")
        store.record_cell(sweep, _cell())

        (stored,) = store.cells()

    assert stored.dataset == "churn"
    assert stored.best == pytest.approx(0.64)


def test_only_successful_cells_count_as_done(tmp_path: Path) -> None:
    """A cell that timed out is left open so the next sweep retries it. That is the
    difference between resuming and giving up on whatever failed once."""
    with Store(tmp_path / "results.db") as store:
        sweep = store.start_sweep(CONDITIONS)
        store.record_cell(sweep, _cell(repeat=1))
        store.record_cell(sweep, _cell(repeat=2, status=STATUS_TIMEOUT))

        done = store.completed(CONDITIONS.fingerprint())

    assert done == {("0.4.0", "churn", "abc123", 1)}


def test_retrying_a_failed_cell_replaces_it(tmp_path: Path) -> None:
    with Store(tmp_path / "results.db") as store:
        sweep = store.start_sweep(CONDITIONS)
        store.record_cell(sweep, _cell(status=STATUS_TIMEOUT, best=None))
        store.record_cell(sweep, _cell(status=STATUS_OK, best=0.71))

        cells = store.cells()

    assert len(cells) == 1
    assert cells[0].best == pytest.approx(0.71)


def test_changed_dataset_bytes_make_a_new_cell_not_an_overwrite(tmp_path: Path) -> None:
    """The old number is still a true measurement, just of different data."""
    with Store(tmp_path / "results.db") as store:
        sweep = store.start_sweep(CONDITIONS)
        store.record_cell(sweep, _cell(dataset_hash="abc123", best=0.64))
        store.record_cell(sweep, _cell(dataset_hash="def456", best=0.70))

        assert len(store.cells()) == 2


def test_changed_conditions_do_not_count_as_done(tmp_path: Path) -> None:
    stricter = replace(CONDITIONS, max_iterations=20)

    with Store(tmp_path / "results.db") as store:
        sweep = store.start_sweep(CONDITIONS)
        store.record_cell(sweep, _cell())

        assert store.completed(CONDITIONS.fingerprint())
        assert not store.completed(stricter.fingerprint())


def test_a_condition_change_changes_the_fingerprint() -> None:
    assert CONDITIONS.fingerprint() != replace(CONDITIONS, model="qwen3:14b").fingerprint()
    assert CONDITIONS.fingerprint() == replace(CONDITIONS).fingerprint()


def _ceiling_record(
    value: float,
    *,
    method: str = "brute_force_sweep_v1",
    metric: str = "f1",
    direction: str = "maximize",
) -> Ceiling:
    return Ceiling(
        dataset="churn",
        dataset_hash="abc123",
        metric=metric,
        ceiling=value,
        direction=direction,
        baseline=0.60,
        method=method,
        measured_at="2026-08-08T12:00:00+00:00",
    )


def test_a_ceiling_is_keyed_to_the_bytes_it_was_measured_on(tmp_path: Path) -> None:
    with Store(tmp_path / "results.db") as store:
        store.put_ceiling(_ceiling_record(0.70))

        assert store.get_ceiling("churn", "abc123", "f1") is not None
        assert store.get_ceiling("churn", "def456", "f1") is None
        assert store.get_ceiling("churn", "abc123", "roc_auc") is None


def test_a_worse_re_measurement_does_not_lower_the_ceiling(tmp_path: Path) -> None:
    """A ceiling is a lower bound on what is achievable, so it is the best anyone has
    ever reached, not whatever the latest sweep found. Proved on day one: the
    automated sweep beat the hand sweep on laptop price and lost to it on churn."""
    base = _ceiling_record(0.70, method="hand_sweep")

    with Store(tmp_path / "results.db") as store:
        store.put_ceiling(base)
        kept = store.put_ceiling(replace(base, ceiling=0.66, method="brute_force_sweep_v1"))

        stored = store.get_ceiling("churn", "abc123", "f1")

    assert kept.ceiling == pytest.approx(0.70)
    assert stored is not None
    assert stored.ceiling == pytest.approx(0.70)
    assert stored.method == "hand_sweep"


def test_a_worse_re_measurement_on_a_minimising_metric_is_also_kept_out(tmp_path: Path) -> None:
    base = _ceiling_record(
        248.85, method="brute_force_sweep_v1", metric="rmse", direction="minimize"
    )

    with Store(tmp_path / "results.db") as store:
        store.put_ceiling(base)
        store.put_ceiling(replace(base, ceiling=329.55, method="hand_sweep"))

        stored = store.get_ceiling("churn", "abc123", "rmse")

    assert stored is not None
    assert stored.ceiling == pytest.approx(248.85)


def test_force_overrides_a_ceiling_that_is_wrong_rather_than_worse(tmp_path: Path) -> None:
    base = _ceiling_record(0.99, method="measured_with_a_leak")

    with Store(tmp_path / "results.db") as store:
        store.put_ceiling(base)
        store.put_ceiling(replace(base, ceiling=0.70, method="corrected"), replace=True)

        stored = store.get_ceiling("churn", "abc123", "f1")

    assert stored is not None
    assert stored.ceiling == pytest.approx(0.70)


def test_a_better_re_measurement_raises_the_ceiling(tmp_path: Path) -> None:
    base = _ceiling_record(0.70, method="brute_force_sweep_v1")

    with Store(tmp_path / "results.db") as store:
        store.put_ceiling(base)
        store.put_ceiling(replace(base, ceiling=0.74, method="brute_force_sweep_v2"))

        stored = store.get_ceiling("churn", "abc123", "f1")

    assert stored is not None
    assert stored.ceiling == pytest.approx(0.74)
    assert stored.method == "brute_force_sweep_v2"
