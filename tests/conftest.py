"""Shared pytest fixtures.

Populated as the framework lands. v1 conftest stays intentionally light.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def _isolate_user_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Point user config and the user cache at throwaway dirs so tests never read the
    developer's real ~/.config/iterate/config.toml (which would leak a saved backend/key
    into the run) or a refreshed price list under ~/.cache/iterate (which would change
    what a priced winner prints). Tests that need their own still override the variables."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg-config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg-cache"))


@pytest.fixture(autouse=True)
def _no_key_is_checked_over_the_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """A prompt run asks its provider for the model list to check the key. No test may
    reach a provider; one that tests the check passes its own lister."""
    monkeypatch.setattr("iterate.llm.factory._list_models", lambda provider, timeout, model: None)
    monkeypatch.setattr("iterate.llm.factory._claude_serves", lambda provider, timeout, model: None)


@pytest.fixture(autouse=True)
def _no_wait_before_a_record_is_asked_again(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("iterate.core.prompt_runtime._SECOND_ASK_WAIT", 0.0)


@pytest.fixture(autouse=True)
def _no_anthropic_variable_reaches_a_test(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Anthropic SDK reads these by itself, and a Claude Code shell sets
    ANTHROPIC_BASE_URL. A test that needs one sets it."""
    for name in list(os.environ):
        if name.upper().startswith("ANTHROPIC_"):
            monkeypatch.delenv(name, raising=False)


@pytest.fixture
def repo_root() -> Path:
    """Path to the repository root (where pyproject.toml lives)."""
    return Path(__file__).parent.parent


@pytest.fixture
def env_has_anthropic_key() -> bool:
    """True if ANTHROPIC_API_KEY is set (skip live-API tests otherwise)."""
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


@pytest.fixture(autouse=True, scope="session")
def _torch_stays_out_of_the_test_process() -> Iterator[None]:
    """torch and lightgbm cannot share a process on macOS, so every torch test runs in a
    child. Checked when the session ends, so a file collected late cannot load it here
    unnoticed."""
    yield
    assert "torch" not in sys.modules, "a test loaded torch into the pytest process"
