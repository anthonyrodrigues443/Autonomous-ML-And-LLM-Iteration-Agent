"""The report refuses to draw comparisons that are not comparisons."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from evals.config import Conditions, EvalConfig
from evals.report import build
from evals.store import STATUS_OK, STATUS_TIMEOUT, Ceiling, CellRecord, Store

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit

CONDITIONS = Conditions(
    model="gemma4:12b",
    backend="ollama",
    max_iterations=10,
    patience=3,
    repeats=1,
    timeout_minutes=90,
    sweep_threads=4,
)
CONFIG = EvalConfig(versions=["0.4.0", "dev"], conditions=CONDITIONS)


def _cell(
    version: str, *, best: float | None, data_hash: str = "abc123", **kw: object
) -> CellRecord:
    base = CellRecord(
        version=version,
        dataset="churn",
        dataset_hash=data_hash,
        repeat=1,
        conditions_fingerprint=CONDITIONS.fingerprint(),
        status=STATUS_OK,
        metric="f1",
        direction="maximize",
        baseline=0.60,
        best=best,
    )
    return replace(base, **kw)  # type: ignore[arg-type]


def _ceiling(data_hash: str = "abc123") -> Ceiling:
    return Ceiling(
        dataset="churn",
        dataset_hash=data_hash,
        metric="f1",
        ceiling=0.70,
        direction="maximize",
        baseline=0.60,
        method="brute_force_sweep_v1",
        measured_at="2026-08-08T00:00:00+00:00",
    )


def test_an_empty_store_says_so_instead_of_drawing_a_table(tmp_path: Path) -> None:
    with Store(tmp_path / "results.db") as store:
        markdown = build(store, CONFIG)

    assert "No cells recorded" in markdown


def test_the_table_reports_captured_headroom(tmp_path: Path) -> None:
    with Store(tmp_path / "results.db") as store:
        sweep = store.start_sweep(CONDITIONS)
        store.put_ceiling(_ceiling())
        store.record_cell(sweep, _cell("0.4.0", best=None))
        store.record_cell(sweep, _cell("dev", best=0.65))

        markdown = build(store, CONFIG)

    assert "| **dev** | 50% |" in markdown
    assert "| **0.4.0** | 0% |" in markdown


def test_a_dataset_with_no_ceiling_shows_the_raw_gain(tmp_path: Path) -> None:
    with Store(tmp_path / "results.db") as store:
        sweep = store.start_sweep(CONDITIONS)
        store.record_cell(sweep, _cell("dev", best=0.64))

        markdown = build(store, CONFIG)

    assert "no ceiling" in markdown


def test_changed_dataset_bytes_are_called_out(tmp_path: Path) -> None:
    """Silently averaging results measured on two different files is the failure
    this warning exists to prevent."""
    with Store(tmp_path / "results.db") as store:
        sweep = store.start_sweep(CONDITIONS)
        store.record_cell(sweep, _cell("dev", best=0.65, data_hash="abc123"))
        store.record_cell(sweep, _cell("0.4.0", best=0.63, data_hash="def456"))

        markdown = build(store, CONFIG)

    assert "dataset bytes changed" in markdown


def test_cells_from_other_conditions_are_excluded_and_flagged(tmp_path: Path) -> None:
    other = replace(CONDITIONS, max_iterations=25)

    with Store(tmp_path / "results.db") as store:
        sweep = store.start_sweep(CONDITIONS)
        store.put_ceiling(_ceiling())
        store.record_cell(sweep, _cell("dev", best=0.65))
        store.record_cell(
            sweep,
            _cell("0.4.0", best=0.69, conditions_fingerprint=other.fingerprint()),
        )

        markdown = build(store, CONFIG)

    assert "different sets of conditions" in markdown
    assert "| **0.4.0** | - |" in markdown


def test_a_failed_cell_reports_its_status_not_a_score(tmp_path: Path) -> None:
    with Store(tmp_path / "results.db") as store:
        sweep = store.start_sweep(CONDITIONS)
        store.put_ceiling(_ceiling())
        store.record_cell(sweep, _cell("dev", best=None, status=STATUS_TIMEOUT))

        markdown = build(store, CONFIG)

    assert "timeout x1" in markdown
