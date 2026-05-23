"""Shared pytest fixtures.

Populated as the framework lands. v1 conftest stays intentionally light.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture
def repo_root() -> Path:
    """Path to the repository root (where pyproject.toml lives)."""
    return Path(__file__).parent.parent


@pytest.fixture
def env_has_anthropic_key() -> bool:
    """True if ANTHROPIC_API_KEY is set (skip live-API tests otherwise)."""
    return bool(os.environ.get("ANTHROPIC_API_KEY"))
