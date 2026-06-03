"""Tests for the persisted user config (~/.config/iterate/config.toml)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from iterate import userconfig

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_config_path_honors_xdg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert userconfig.config_path() == tmp_path / "iterate" / "config.toml"


def test_save_then_load_round_trips(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert not userconfig.exists()
    userconfig.save_user_config(
        {"backend": "groq", "model": "llama-3.3-70b", "compute": "e2b", "install": True}
    )
    assert userconfig.exists()
    loaded = userconfig.load_user_config()
    assert loaded == {
        "backend": "groq",
        "model": "llama-3.3-70b",
        "compute": "e2b",
        "install": True,
    }


def test_save_drops_unknown_and_empty(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    userconfig.save_user_config(
        {"backend": "ollama", "nonsense": "x", "model": "", "install": False}
    )
    loaded = userconfig.load_user_config()
    assert "nonsense" not in loaded
    assert "model" not in loaded  # empty string not written
    assert loaded == {"backend": "ollama", "install": False}


def test_load_missing_is_empty(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert userconfig.load_user_config() == {}
