"""Tests for the ComputeBackend protocol + its implementations."""

from __future__ import annotations

import pytest

from iterate.adapters.compute.base import ComputeBackend
from iterate.adapters.compute.local import LocalExecutor
from iterate.adapters.compute.sandbox import SandboxExecutor


def test_local_executor_satisfies_the_protocol() -> None:
    assert isinstance(LocalExecutor(), ComputeBackend)


def test_sandbox_executor_satisfies_the_protocol() -> None:
    # The stub still conforms to the shape (it just raises when called).
    assert isinstance(SandboxExecutor(), ComputeBackend)


def test_sandbox_executor_is_a_stub_for_now() -> None:
    with pytest.raises(NotImplementedError, match="lands in"):
        SandboxExecutor().execute(target=object(), candidate=None)  # type: ignore[arg-type]
