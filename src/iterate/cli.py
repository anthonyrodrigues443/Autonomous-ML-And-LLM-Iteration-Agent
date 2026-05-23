"""CLI entry point.

Stub for Week 1. Real commands (`init`, `run`, `history`, `why-failed`, `best`,
`revisit`) ship as the framework lands.
"""
from __future__ import annotations

import typer

from iterate import __version__

app = typer.Typer(
    name="iterate",
    help=(
        "Autonomous research-aware iteration agent for ML models and LLM prompts. "
        "See https://github.com/anthonyrodrigues443/iterate"
    ),
    no_args_is_help=True,
)


@app.command()
def version() -> None:
    """Print the iterate version."""
    typer.echo(f"iterate {__version__}")


if __name__ == "__main__":
    app()
