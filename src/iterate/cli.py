"""CLI entry point.

A working command skeleton for Week 1. Real commands (`init`, `run`, `history`,
`why-failed`, `best`, `revisit`) ship as the framework lands.
"""

from __future__ import annotations

import typer

from iterate import __version__
from iterate.config import get_settings

app = typer.Typer(
    name="iterate",
    help=(
        "Autonomous research-aware iteration agent for ML models and LLM prompts. "
        "See https://github.com/anthonyrodrigues443/iterate"
    ),
    no_args_is_help=True,
)


@app.callback()
def _root() -> None:
    """Root callback — its presence keeps `version`/`config` as real subcommands.

    A Typer app with a single command otherwise promotes that command to the top
    level, which breaks `iterate <command>`.
    """


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


def _mask(secret: str) -> str:
    """Show only the first/last two chars of a secret."""
    if len(secret) <= 4:
        return "****"
    return f"{secret[:2]}…{secret[-2:]}"


if __name__ == "__main__":
    app()
