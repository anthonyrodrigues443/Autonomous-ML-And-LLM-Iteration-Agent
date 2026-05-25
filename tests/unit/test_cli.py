"""Tests for the CLI scaffold — guards against the single-command collapse bug."""

from __future__ import annotations

from typer.testing import CliRunner

from iterate import __version__
from iterate.cli import app

runner = CliRunner()


def test_help_lists_both_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "version" in result.output
    assert "config" in result.output


def test_version_runs_as_a_subcommand() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_config_shows_fields_and_masks_the_key() -> None:
    result = runner.invoke(app, ["config"])
    assert result.exit_code == 0
    for label in ("model:", "backend_url:", "api_key:", "timeout:"):
        assert label in result.output
    assert "…" in result.output or "****" in result.output  # key is masked


def test_no_args_shows_help() -> None:
    result = runner.invoke(app, [])
    assert "Usage" in result.output
