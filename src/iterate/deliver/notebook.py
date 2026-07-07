"""Render a finished experiment as a runnable Jupyter notebook.

The human-facing deliverable: each notebook reproduces ONE experiment top to
bottom — load the data with the exact same split, run the approach, recompute the
score. Faithfulness is the point (reproduce the reported number, not a lookalike),
so the cells load + score through `iterate`'s own helpers rather than a hand-rolled
split that might drift.

Backend capture (the full record of every experiment) is Memory; this just renders
it. A run can emit the winner only (`best`) or one notebook per experiment (`all`).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook, new_output

from iterate.core.codegen import is_code_candidate

if TYPE_CHECKING:
    from pathlib import Path

    from nbformat import NotebookNode

    from iterate.schemas.experiment import Candidate, Experiment


def build_notebook(
    experiment: Experiment,
    *,
    data_path: str,
    target: str,
    metric: str,
    baseline_score: float | None = None,
    is_best: bool = False,
    leaderboard: list[Experiment] | None = None,
) -> NotebookNode:
    """Render one experiment into a self-reproducing notebook.

    ``leaderboard`` (the full run history) adds a "what was tried" table at the top
    — used for the winner notebook so one file tells the whole story.
    """
    candidate = experiment.candidate
    nb = new_notebook()
    nb.cells = [
        new_markdown_cell(_header_md(experiment, target, metric, baseline_score, is_best=is_best)),
    ]
    if leaderboard:
        nb.cells.append(new_markdown_cell(_leaderboard_md(leaderboard, metric, baseline_score)))
    nb.cells.append(new_code_cell(_load_cell(data_path, target)))
    if is_code_candidate(candidate.changes):
        nb.cells.append(new_code_cell(str(candidate.changes["code"]).strip()))
        nb.cells.append(new_code_cell(_score_code_cell(metric)))
    else:
        nb.cells.append(new_code_cell(_spec_cell(candidate, metric)))
    nb.cells.append(new_markdown_cell(_rationale_md(candidate)))
    return nb


def build_session_notebook(
    cells: list[Any],
    *,
    title: str,
    metric: str,
    score: float | None = None,
    baseline_score: float | None = None,
    hypothesis: str | None = None,
    findings: Any = None,
    honesty_note: str | None = None,
) -> NotebookNode:
    """Render a cell-by-cell coding session as a runnable notebook.

    Each `Cell` (code + captured stdout/error) becomes a code cell, with a short
    markdown note showing what it printed or the error — so the notebook *is* the
    session: the data inspection, the feature engineering, the dead ends, in order.
    When a cell carries the model's `thinking` (think mode), it is rendered as a
    markdown reasoning block right above the code it produced — the model's own
    "why" narrating the session, and the raw material for debugging the prompt.

    ``hypothesis`` (the supervisor's brief, which opens with the run's so-far
    knowledge) renders as a Hypothesis cell up top; ``findings`` (the Summarizer's
    digest, dict or `ExperimentDigest`) renders as a Findings cell at the end — so
    every notebook reads as hypothesis -> staged work -> findings, and the
    knowledge each notebook inherits and hands on is visible in the artifact.
    """
    nb = new_notebook()
    head = [f"# {title}", ""]
    if score is not None:
        line = f"**{metric} = {score:.4f}**"
        if baseline_score is not None:
            line += f"  ({score - baseline_score:+.4f} vs baseline {baseline_score:.4f})"
        head.append(line)
    if honesty_note:
        # The score alone can mislead (a duplicate submission still shows a
        # positive delta vs baseline) — the machine verdict rides the header.
        head.append(f"\n**Note:** _{honesty_note}_")
    head.append("\n_cell-by-cell session — outputs are the actual execution results_")
    nb.cells = [new_markdown_cell("\n".join(head))]
    if hypothesis and hypothesis.strip():
        nb.cells.append(
            new_markdown_cell(
                "## Hypothesis\n\n_The supervisor's brief for this experiment (the run's"
                " findings so far, then the move under test):_\n\n> "
                + hypothesis.strip().replace("\n", "\n> ")
            )
        )
    for count, cell in enumerate(cells, start=1):
        thinking = _cell_get(cell, "thinking").strip()
        if thinking:
            nb.cells.append(new_markdown_cell(_thinking_markdown(thinking)))
        if _cell_get(cell, "source") == "fallback":
            # Honesty marker: this submission was banked by the harness, not the coder.
            nb.cells.append(
                new_markdown_cell(
                    "## Harness fallback\n\n_The session ended without a valid submission,"
                    " so the harness banked this floor submission (not written by the"
                    " coding agent this iteration)._"
                )
            )
        elif _cell_get(cell, "error"):
            # A dead end in the R&D journey — annotated so a reader of the deliverable
            # sees it was part of the process, not an unnoticed failure.
            nb.cells.append(
                new_markdown_cell("_(dead end — this cell errored; kept for the record)_")
            )
        node = new_code_cell(_cell_get(cell, "code").strip())
        node.execution_count = count
        node.outputs = _cell_outputs(cell)
        nb.cells.append(node)
    findings_md = _findings_markdown(findings)
    if findings_md:
        nb.cells.append(new_markdown_cell(findings_md))
    return nb


def _findings_markdown(digest: Any) -> str | None:
    """The Summarizer's digest as a Findings cell (dict or `ExperimentDigest`).
    Returns None when there is nothing worth rendering."""
    if digest is None:
        return None

    def get(key: str) -> Any:
        return digest.get(key) if isinstance(digest, dict) else getattr(digest, key, None)

    parts = ["## Findings"]
    for label, key in (
        ("What helped", "what_helped"),
        ("What hurt or did nothing", "what_hurt"),
        ("Data insights", "data_insights"),
    ):
        items = get(key) or []
        if items:
            parts.append(f"**{label}**\n" + "\n".join(f"- {item}" for item in items))
    trail = get("val_trail")
    if trail:
        parts.append(f"**Validation trail:** {trail}")
    takeaway = get("takeaway")
    if takeaway:
        parts.append(f"**Takeaway:** {takeaway}")
    return "\n\n".join(parts) if len(parts) > 1 else None


def _thinking_markdown(thinking: str) -> str:
    """The model's reasoning as a quoted markdown block (full text — it exists to
    debug what the prompt made the model think, so nothing is trimmed)."""
    quoted = "\n".join(f"> {line}" if line.strip() else ">" for line in thinking.splitlines())
    return "**Model reasoning**\n\n" + quoted


def _cell_outputs(cell: Any) -> list[Any]:
    """Build the code cell's outputs: the real captured outputs if present, else
    synthesize from the captured stdout/error strings (older cells / fallback)."""
    captured = cell.get("outputs") if isinstance(cell, dict) else getattr(cell, "outputs", None)
    if captured:
        return [_to_nb_output(o) for o in captured]
    error = _cell_get(cell, "error")
    if error:
        return [new_output("error", ename="Error", evalue=_error_oneline(error),
                           traceback=error.splitlines())]
    stdout = _cell_get(cell, "stdout")
    if stdout.strip():
        return [new_output("stream", name="stdout", text=stdout)]
    return []


def _to_nb_output(captured: dict[str, Any]) -> Any:
    """Convert a captured output dict (from a kernel) to an nbformat output."""
    kind = captured.get("type", "stream")
    rest = {k: v for k, v in captured.items() if k != "type"}
    return new_output(kind, **rest)


def _cell_get(cell: Any, key: str) -> str:
    """Read a string cell field whether it's a `Cell` object or a dict."""
    value = cell.get(key) if isinstance(cell, dict) else getattr(cell, key, None)
    return value or ""


def save_notebook(node: NotebookNode, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        nbformat.write(node, handle)
    return path


def slug(text: str, *, limit: int = 40) -> str:
    """A filesystem-safe slug from a free-text description."""
    cleaned = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return (cleaned[:limit].rstrip("-")) or "experiment"


def _header_md(
    experiment: Experiment,
    target: str,
    metric: str,
    baseline_score: float | None,
    *,
    is_best: bool,
) -> str:
    result = experiment.result
    title = experiment.candidate.description.strip() or "experiment"
    lines = [f"# {'best: ' if is_best else ''}{title}", ""]
    if result is not None and result.succeeded and result.metrics is not None:
        score = result.metrics.primary_value
        line = f"**{metric} = {score:.4f}**"
        if baseline_score is not None:
            line += f"  ({score - baseline_score:+.4f} vs baseline {baseline_score:.4f})"
        lines.append(line)
    elif result is not None and result.error:
        lines.append(f"**FAILED:** {_error_oneline(result.error)}")
    lines.append("")
    lines.append(f"_iteration {experiment.iteration} · target column `{target}`_")
    return "\n".join(lines)


def _error_oneline(error: str) -> str:
    """The most informative single line of an error — the last non-empty line,
    which for a traceback is the actual exception (not 'code script failed:')."""
    lines = [ln.strip() for ln in error.strip().splitlines() if ln.strip()]
    return (lines[-1] if lines else "failed")[:80]


def _leaderboard_md(
    history: list[Experiment], metric: str, baseline_score: float | None
) -> str:
    """A compact "what was tried" table: every experiment, its score, pass/fail."""
    rows = ["## What was tried", "", f"| iter | approach | {metric} | result |", "|---|---|---|---|"]
    if baseline_score is not None:
        rows.append(f"| base | baseline | {baseline_score:.4f} | — |")
    for exp in history:
        approach = exp.candidate.description.strip().replace("|", "/")[:60] or "?"
        result = exp.result
        if result is not None and result.succeeded and result.metrics is not None:
            rows.append(f"| {exp.iteration} | {approach} | {result.metrics.primary_value:.4f} | ✓ |")
        else:
            why = _error_oneline(result.error) if result and result.error else "failed"
            rows.append(f"| {exp.iteration} | {approach} | — | ✗ {why.replace('|', '/')} |")
    return "\n".join(rows)


def _load_cell(data_path: str, target: str) -> str:
    return (
        "# Load the data with the exact split iterate measured on (same seed/stratify).\n"
        "from iterate.adapters.data.tabular import load_csv\n\n"
        f"ds = load_csv({data_path!r}, target={target!r})\n"
        "X_train, y_train = ds.train_features, ds.train_target\n"
        "X_holdout, y_holdout = ds.test_features, ds.test_target"
    )


def _score_code_cell(metric: str) -> str:
    return (
        "# Run the approach and score it on the sealed holdout, the same ruler iterate used.\n"
        "from iterate.core.scoring import score, task_for_metric\n\n"
        "predictions = train_and_predict(X_train, y_train, X_holdout)\n"
        f"print(score(task_for_metric({metric!r}), y_holdout, list(predictions)))"
    )


def _spec_cell(candidate: Candidate, metric: str) -> str:
    return (
        "# Spec-path candidate: rebuild + score through the same pipeline iterate used.\n"
        "from iterate.adapters.data.tabular import load_csv  # noqa: F811 (ds already loaded)\n"
        "from iterate.schemas.experiment import Candidate\n"
        "from iterate.targets.model import ModelTarget\n\n"
        f"target = ModelTarget(ds, metric={metric!r})\n"
        f"candidate = Candidate(description={candidate.description!r}, "
        f"changes={candidate.changes!r}, rationale={candidate.rationale!r})\n"
        "result = target.run(candidate)\n"
        "print(result.metrics.values if result.metrics else result.error)"
    )


def _rationale_md(candidate: Candidate) -> str:
    return f"### Rationale\n\n{candidate.rationale.strip() or '(none given)'}"


__all__ = ["build_notebook", "build_session_notebook", "save_notebook", "slug"]
