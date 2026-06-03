"""Tests for the CLI scaffold — guards against the single-command collapse bug."""

from __future__ import annotations

from typing import TYPE_CHECKING

from typer.testing import CliRunner

from iterate import __version__, userconfig
from iterate.cli import app

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

runner = CliRunner()


def test_help_lists_both_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "version" in result.output
    assert "config" in result.output
    assert "setup" in result.output


def test_run_help_shows_code_and_compute_flags() -> None:
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    assert "--code" in result.output
    assert "--spec" in result.output
    assert "--compute" in result.output
    assert "--install" in result.output


def test_setup_saves_local_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    # backend=ollama, model=blank, compute=local, install=no
    result = runner.invoke(app, ["setup"], input="ollama\n\nlocal\nn\n")
    assert result.exit_code == 0, result.output
    cfg = userconfig.load_user_config()
    assert cfg == {"backend": "ollama", "compute": "local", "install": False}


def test_setup_saves_e2b_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    # backend=openai, model=gpt-4o, api_key=sk-x, compute=e2b, e2b key=e2b-x
    result = runner.invoke(app, ["setup"], input="openai\ngpt-4o\nsk-x\ne2b\ne2b-x\n")
    assert result.exit_code == 0, result.output
    cfg = userconfig.load_user_config()
    assert cfg["backend"] == "openai"
    assert cfg["model"] == "gpt-4o"
    assert cfg["api_key"] == "sk-x"
    assert cfg["compute"] == "e2b"
    assert cfg["e2b_api_key"] == "e2b-x"


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
