"""CLI entry point.

`iterate version` / `iterate config` / `iterate run`. The `run` command wires the
agent end-to-end: load data → build target → build LLM client → reconstruct
baseline from `--source` if given → loop via the Orchestrator → render a summary.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from iterate import __version__
from iterate.adapters.compute.local import LocalExecutor
from iterate.adapters.data.tabular import load_csv
from iterate.config import get_settings
from iterate.core.memory import SqliteMemory
from iterate.core.orchestrator import Orchestrator
from iterate.core.proposer import Proposer, summarize_dataset
from iterate.core.reconstructor import Reconstructor
from iterate.core.terminator import default_terminator
from iterate.llm.factory import build_client
from iterate.targets.model import ModelTarget

if TYPE_CHECKING:
    from iterate.core.memory import Memory
    from iterate.core.orchestrator import RunResult
    from iterate.schemas.experiment import Candidate, Experiment

app = typer.Typer(
    name="iterate",
    help=(
        "Autonomous research-aware iteration agent for ML models and LLM prompts. "
        "See https://github.com/anthonyrodrigues443/iterate"
    ),
    no_args_is_help=True,
)

console = Console()

# Metric → task. Matches the private logic in iterate.targets.model.
_CLASSIFICATION_METRICS = {"accuracy", "f1", "precision", "recall"}
_REGRESSION_METRICS = {"rmse", "mae", "mse", "r2"}


@app.callback()
def _root() -> None:
    """Root callback — its presence keeps subcommands as real subcommands."""


@app.command()
def version() -> None:
    """Print the iterate version."""
    typer.echo(f"iterate {__version__}")


@app.command()
def config() -> None:
    """Show the resolved configuration (the backend api-key is masked)."""
    settings = get_settings()
    typer.echo(f"model:        {settings.iterate_model}")
    typer.echo(f"backend_url:  {settings.iterate_backend_url}")
    typer.echo(f"api_key:      {_mask(settings.iterate_backend_api_key)}")
    typer.echo(f"timeout:      {settings.iterate_backend_timeout}s")
    typer.echo(f"ollama_host:  {settings.ollama_host}")
    typer.echo(f"memory_db:    {settings.iterate_memory_db}")


@app.command()
def run(
    data: Path = typer.Option(
        ..., "--data", help="Path to the CSV dataset.", exists=True, dir_okay=False
    ),
    target: str = typer.Option(..., "--target", help="Name of the target column."),
    metric: str = typer.Option(
        ..., "--metric", help="Primary metric: f1 | accuracy | rmse | mae | r2 | …"
    ),
    baseline: float | None = typer.Option(
        None, "--baseline", help="Your reported baseline score (sanity check; requires --source)."
    ),
    source: Path | None = typer.Option(
        None,
        "--source",
        help="md / txt / .py / .ipynb describing the user's approach (read as TEXT, never executed).",
        exists=True,
        dir_okay=False,
    ),
    backend: str = typer.Option(
        "ollama", "--backend", help="ollama | openai-compatible (or aliases: groq, together, …)"
    ),
    model: str | None = typer.Option(None, "--model", help="Override the backend's default model."),
    base_url: str | None = typer.Option(
        None, "--base-url", help="Override the backend's base URL."
    ),
    api_key: str | None = typer.Option(None, "--api-key", help="Override the backend's API key."),
    max_iterations: int = typer.Option(
        10, "--max-iterations", min=1, help="Hard cap on experiments."
    ),
    patience: int = typer.Option(
        3, "--patience", min=1, help="Stop after N consecutive non-improvements."
    ),
    until: str | None = typer.Option(
        None, "--until", help='Wall-clock deadline, e.g. "30m" or "2h".'
    ),
    fresh: bool = typer.Option(
        False,
        "--fresh",
        help="Archive the existing memory db and start a new chapter with the factory default baseline.",
    ),
    memory_path: Path | None = typer.Option(None, "--memory", help="Override the memory db path."),
) -> None:
    """Run the agent on a tabular dataset."""
    # ─── Validate ──────────────────────────────────────────────────────────
    if baseline is not None and source is None:
        raise typer.BadParameter("--baseline requires --source")
    metric = metric.lower()
    if metric not in _CLASSIFICATION_METRICS and metric not in _REGRESSION_METRICS:
        raise typer.BadParameter(
            f"unknown metric {metric!r}; expected one of "
            f"{sorted(_CLASSIFICATION_METRICS | _REGRESSION_METRICS)}"
        )

    settings = get_settings()
    resolved_memory_path = memory_path or Path(settings.iterate_memory_db)

    # ─── New chapter? Archive the existing db. ─────────────────────────────
    # Any of --fresh, --source, --baseline+--source means "new chapter."
    starting_new_chapter = fresh or source is not None
    if starting_new_chapter:
        archived = _archive_memory_db(resolved_memory_path)
        if archived is not None:
            console.print(
                f"[dim]memory: archived [/dim]{resolved_memory_path}[dim] → "
                f"[/dim]{archived.name}[dim]; starting fresh[/dim]"
            )

    # ─── Cloud backend? API key required. ──────────────────────────────────
    if backend != "ollama":
        resolved_key = api_key or _resolved_api_key_from_env(settings, backend)
        if not resolved_key:
            raise typer.BadParameter(
                f"backend {backend!r} requires --api-key or a corresponding env var "
                f"(ITERATE_BACKEND_API_KEY / OPENAI_API_KEY / GROQ_API_KEY / …)"
            )
        api_key = resolved_key

    # ─── Configure logging so per-iteration messages stream live. ──────────
    _configure_logging()

    # ─── Load data + build target ──────────────────────────────────────────
    dataset = load_csv(data, target=target)
    model_target = ModelTarget(dataset, metric=metric)
    data_summary = summarize_dataset(dataset)
    direction = "minimize" if metric in _REGRESSION_METRICS else "maximize"

    # ─── LLM client + memory ───────────────────────────────────────────────
    client = build_client(backend, model=model, base_url=base_url, api_key=api_key)
    memory: Memory = SqliteMemory(resolved_memory_path)

    # ─── Baseline selection (the precedence we locked) ─────────────────────
    baseline_candidate: Candidate | None
    baseline_model: str

    if source is not None:
        source_text = _read_source(source)
        baseline_candidate = Reconstructor(client).reconstruct(
            data_summary=data_summary,
            source_text=source_text,
            metric=metric,
            direction=direction,
        )
        baseline_model = str(baseline_candidate.changes["model"])
        console.print(
            f"[bold]baseline from source[/bold] ({source.name}): {baseline_candidate.description}"
        )
    elif not fresh:
        prior = _prior_best(memory, model_target.name, direction)
        if prior is not None and prior.result and prior.result.metrics:
            baseline_candidate = prior.candidate
            baseline_model = str(prior.candidate.changes["model"])
            console.print(
                f"[bold]baseline from memory[/bold]: {prior.candidate.description} "
                f"({metric}={prior.result.metrics.primary_value:.4f}); re-measuring"
            )
        else:
            baseline_candidate = None
            baseline_model = _default_baseline_model(metric)
    else:
        baseline_candidate = None
        baseline_model = _default_baseline_model(metric)

    # ─── Terminator + Orchestrator ─────────────────────────────────────────
    proposer = Proposer(client)
    deadline_seconds = _parse_duration(until) if until is not None else None
    terminator = default_terminator(
        max_iterations=max_iterations, patience=patience, deadline_seconds=deadline_seconds
    )

    orchestrator = Orchestrator(
        model_target,
        proposer,
        LocalExecutor(),
        terminator,
        memory,
        data_summary=data_summary,
        baseline_model=baseline_model,
        baseline_candidate=baseline_candidate,
    )

    console.print(
        f"\n[dim]Running on {model_target.name}; target={target!r}, metric={metric}[/dim]\n"
    )
    result = orchestrator.run()

    # ─── Sanity check on user-reported baseline ────────────────────────────
    if baseline is not None and result.baseline.metrics is not None:
        _check_baseline_divergence(
            reported=baseline, measured=result.baseline.metrics.primary_value
        )

    # ─── Summary ───────────────────────────────────────────────────────────
    _render_summary(result, metric)


# ── helpers ──────────────────────────────────────────────────────────────


def _mask(secret: str) -> str:
    if len(secret) <= 4:
        return "****"
    return f"{secret[:2]}…{secret[-2:]}"


def _configure_logging() -> None:
    """Stream orchestrator INFO logs to the console via rich, once."""
    root = logging.getLogger()
    if root.handlers:
        return  # something already configured (likely a test)
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, show_path=False, markup=False)],
    )


def _read_source(path: Path) -> str:
    """Read a source document as text; for `.ipynb`, walk cells."""
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() != ".ipynb":
        return raw
    try:
        nb = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"could not parse {path.name} as a notebook: {exc}") from exc
    parts: list[str] = []
    for cell in nb.get("cells", []):
        src = cell.get("source", "")
        text = "".join(src) if isinstance(src, list) else str(src)
        if not text.strip():
            continue
        if cell.get("cell_type") == "code":
            parts.append(f"```python\n{text}\n```")
        else:
            parts.append(text)
    return "\n\n".join(parts)


def _parse_duration(text: str) -> float:
    """Parse '30s' / '15m' / '2h' / '1h30m' to seconds."""
    text = text.strip().lower()
    total = 0.0
    number = ""
    for ch in text:
        if ch.isdigit() or ch == ".":
            number += ch
            continue
        if not number:
            raise typer.BadParameter(f"invalid duration {text!r}")
        value = float(number)
        if ch == "h":
            total += value * 3600
        elif ch == "m":
            total += value * 60
        elif ch == "s":
            total += value
        else:
            raise typer.BadParameter(f"invalid duration unit {ch!r} in {text!r}")
        number = ""
    if number:  # trailing bare number — assume seconds
        total += float(number)
    if total <= 0:
        raise typer.BadParameter(f"duration must be > 0 ({text!r} parsed as 0)")
    return total


def _archive_memory_db(path: Path) -> Path | None:
    """Rename an existing memory db to a timestamped `.bak`. Returns the new path."""
    if not path.exists():
        return None
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    archived = path.with_name(f"{path.stem}.{stamp}.bak{path.suffix}")
    path.rename(archived)
    return archived


def _resolved_api_key_from_env(settings: object, backend: str) -> str | None:
    """Pull an api key from settings env-overrides for a cloud backend.

    The user might have set ITERATE_BACKEND_API_KEY (generic), or a vendor-specific
    one like OPENAI_API_KEY / GROQ_API_KEY.
    """
    candidates = {
        "openai-compatible": ("openai_api_key", "iterate_backend_api_key"),
        "openai": ("openai_api_key", "iterate_backend_api_key"),
        "groq": ("groq_api_key", "iterate_backend_api_key"),
        "together": ("together_api_key", "iterate_backend_api_key"),
        "deepseek": ("deepseek_api_key", "iterate_backend_api_key"),
        "vllm": ("iterate_backend_api_key",),
    }
    for attr in candidates.get(backend, ()):
        value = getattr(settings, attr, None)
        if value and value != "ollama":  # "ollama" is the placeholder default
            return str(value)
    return None


def _default_baseline_model(metric: str) -> str:
    """Factory default per task. Mirrors the private mapping in adapters.models.registry."""
    if metric in _CLASSIFICATION_METRICS:
        return "sklearn.ensemble.HistGradientBoostingClassifier"
    return "sklearn.ensemble.HistGradientBoostingRegressor"


def _prior_best(memory: Memory, target_name: str, direction: str) -> Experiment | None:
    """Return the best succeeded experiment for this target across memory, or None."""
    history = memory.history(target_name)
    succeeded = [e for e in history if e.result and e.result.succeeded and e.result.metrics]
    if not succeeded:
        return None
    def _score(experiment: Experiment) -> float:
        assert experiment.result is not None
        assert experiment.result.metrics is not None
        return experiment.result.metrics.primary_value

    return min(succeeded, key=_score) if direction == "minimize" else max(succeeded, key=_score)


def _check_baseline_divergence(
    *, reported: float, measured: float, threshold: float = 0.10
) -> None:
    if reported == 0:
        return
    divergence = abs(measured - reported) / abs(reported)
    if divergence > threshold:
        console.print(
            f"\n[yellow]warning:[/yellow] your reported baseline {reported:.4f} "
            f"differs from our re-measurement {measured:.4f} ({divergence:.1%} divergence). "
            f"Likely a feature/eval mismatch."
        )


def _render_summary(result: RunResult, metric: str) -> None:
    baseline_score = (
        result.baseline.metrics.primary_value if result.baseline.metrics is not None else None
    )
    best_id = result.best.id if result.best is not None else None

    table = Table(title="Run summary", show_lines=False)
    table.add_column("iter", justify="right")
    table.add_column("model")
    table.add_column(metric, justify="right")
    table.add_column("Δ vs baseline", justify="right")

    if baseline_score is not None:
        table.add_row("base", "baseline", f"{baseline_score:.4f}", "—")

    for exp in result.history:
        model_name = str(exp.candidate.changes.get("model", "?"))
        if exp.id == best_id:
            model_name += "  [bold green]← best[/bold green]"
        if exp.result is None or exp.result.metrics is None:
            err = exp.result.error if exp.result else "no result"
            table.add_row(str(exp.iteration), model_name, "[red]FAILED[/red]", str(err)[:40])
            continue
        score = exp.result.metrics.primary_value
        delta = (score - baseline_score) if baseline_score is not None else 0.0
        arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "—")
        table.add_row(str(exp.iteration), model_name, f"{score:.4f}", f"{arrow} {delta:+.4f}")

    console.print()
    console.print(table)
    console.print(f"\n[bold]stopped:[/bold] {result.stopped_because}")
    if result.best is not None and result.best.result and result.best.result.metrics:
        improvement = (
            result.best.result.metrics.primary_value - baseline_score
            if baseline_score is not None
            else 0.0
        )
        console.print(
            f"[bold]best:[/bold] {result.best.candidate.description} "
            f"({metric}={result.best.result.metrics.primary_value:.4f}, "
            f"{improvement:+.4f} vs baseline)"
        )
    else:
        console.print("[dim]no candidate beat the baseline.[/dim]")


if __name__ == "__main__":
    app()
