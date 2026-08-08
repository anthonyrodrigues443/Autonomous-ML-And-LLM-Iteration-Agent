"""The results table, built only from what the store already holds.

Reading from the store rather than from a live sweep means the table can be
rebuilt at any time, on any machine with the database, without running anything.
It also means the report can never quietly report a number that was not recorded.

The report's real job is refusing to draw a comparison that is not one. Two things
break comparability and both get surfaced at the top rather than buried: cells
measured under different conditions, and cells measured on different bytes of the
same-named dataset.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from evals.score import aggregate, capture
from evals.store import STATUS_OK

if TYPE_CHECKING:
    from evals.config import EvalConfig
    from evals.store import CellRecord, Store

_MISSING = "-"


def _direction_for(cell: CellRecord) -> str:
    """The run's own recorded direction, falling back to today's registry.

    A failed cell never recorded one, and the fallback keeps such a row renderable;
    a successful cell always carries the direction its own version used, which is
    the one its numbers were banked under.
    """
    if cell.direction in ("maximize", "minimize"):
        return cell.direction
    try:
        from iterate.core.scoring import direction

        return direction(cell.metric)
    except Exception:
        return "maximize"


def _cell_text(store: Store, cells: list[CellRecord]) -> str:
    usable = [c for c in cells if c.status == STATUS_OK]
    if not usable:
        if not cells:
            return _MISSING
        return f"{cells[0].status} x{len(cells)}"

    captures = []
    for cell in usable:
        ceiling = store.get_ceiling(cell.dataset, cell.dataset_hash, cell.metric)
        captures.append(
            capture(
                baseline=cell.baseline,
                best=cell.best,
                ceiling=ceiling.ceiling if ceiling else None,
                direction=_direction_for(cell),
            )
        )
    return aggregate(captures).render()


def _table(rows: list[str], header: list[str]) -> list[str]:
    return [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
        *rows,
    ]


def _comparability_warnings(cells: list[CellRecord]) -> list[str]:
    warnings: list[str] = []

    fingerprints = {c.conditions_fingerprint for c in cells}
    if len(fingerprints) > 1:
        warnings.append(
            f"**{len(fingerprints)} different sets of conditions** are present in the store "
            "(model, budget or repeats changed between sweeps). Only cells sharing a "
            "fingerprint are comparable; this table shows the configured one."
        )

    by_dataset: dict[str, set[str]] = defaultdict(set)
    for cell in cells:
        by_dataset[cell.dataset].add(cell.dataset_hash)
    drifted = sorted(name for name, hashes in by_dataset.items() if len(hashes) > 1)
    if drifted:
        warnings.append(
            f"**dataset bytes changed** for: {', '.join(drifted)}. Results measured on "
            "different content are shown separately, never averaged."
        )

    return warnings


def build(store: Store, config: EvalConfig) -> str:
    """The full markdown report for the configured conditions."""
    fingerprint = config.conditions.fingerprint()
    everything = store.cells()
    cells = [c for c in everything if c.conditions_fingerprint == fingerprint]

    grouped: dict[tuple[str, str], list[CellRecord]] = defaultdict(list)
    datasets: list[str] = []
    for cell in cells:
        grouped[(cell.version, cell.dataset)].append(cell)
        if cell.dataset not in datasets:
            datasets.append(cell.dataset)
    datasets.sort()

    conditions = config.conditions
    lines = [
        "# iterate eval results",
        "",
        "Internal. Not a product feature. Regenerate with `make eval-report`.",
        "",
        f"- generated: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}",
        f"- model: `{conditions.model}` on `{conditions.backend}`",
        f"- budget: {conditions.max_iterations} iterations, patience {conditions.patience}, "
        f"{conditions.repeats} repeats per cell",
        f"- conditions fingerprint: `{fingerprint}`",
        "",
    ]

    for warning in _comparability_warnings(everything):
        lines += [f"> {warning}", ""]

    lines += [
        "## Captured headroom",
        "",
        "Each cell is the median across repeats of the fraction of AVAILABLE gain the "
        "agent captured, with the spread in brackets. Not clamped: above 100% means the "
        "run beat the brute-force ceiling, which is a real outcome, not an error.",
        "",
    ]
    if cells:
        rows = [
            "| "
            + " | ".join(
                [f"**{version}**"]
                + [_cell_text(store, grouped.get((version, name), [])) for name in datasets]
            )
            + " |"
            for version in config.versions
        ]
        lines += _table(rows, ["version", *datasets])
    else:
        lines.append("No cells recorded under these conditions yet. Run `make eval`.")
    lines.append("")

    # Always rendered, even with no cells: ceilings are measured separately and are
    # the more expensive half to produce, so a report that hid them would look like
    # the work had not been done.
    lines += ["## Ceilings", ""]
    ceilings = store.ceilings()
    if ceilings:
        ceiling_rows = [
            "| "
            + " | ".join(
                [
                    c.dataset,
                    c.metric,
                    f"`{c.dataset_hash}`",
                    f"{c.baseline:.4f}" if c.baseline is not None else _MISSING,
                    f"{c.ceiling:.4f}",
                    c.method,
                    c.measured_at[:10],
                ]
            )
            + " |"
            for c in ceilings
        ]
        lines += _table(
            ceiling_rows,
            ["dataset", "metric", "data hash", "baseline", "ceiling", "method", "measured"],
        )
    else:
        lines.append("None measured yet. Run `make eval-ceilings`.")
    lines.append("")

    if not cells:
        return "\n".join(lines)

    lines += ["## Cell detail", ""]
    detail_rows = []
    for version in config.versions:
        for name in datasets:
            group = grouped.get((version, name), [])
            if not group:
                continue
            ok = [c for c in group if c.status == STATUS_OK]
            bests = [f"{c.best:.4f}" if c.best is not None else "none" for c in ok]
            detail_rows.append(
                "| "
                + " | ".join(
                    [
                        version,
                        name,
                        f"{len(ok)}/{len(group)}",
                        ", ".join(bests) or _MISSING,
                        str(sum(c.n_experiments for c in ok)),
                        str(sum(c.n_failed for c in ok)),
                        str(sum(c.n_rejected for c in ok)),
                        _duration(ok),
                    ]
                )
                + " |"
            )
    lines += _table(
        detail_rows,
        [
            "version",
            "dataset",
            "ok/run",
            "best per repeat",
            "experiments",
            "failed",
            "rejected",
            "median min",
        ],
    )
    lines.append("")

    return "\n".join(lines)


def _duration(cells: list[CellRecord]) -> str:
    times = [c.duration_seconds for c in cells if c.duration_seconds is not None]
    if not times:
        return _MISSING
    return f"{sorted(times)[len(times) // 2] / 60:.0f}"


__all__ = ["build"]
