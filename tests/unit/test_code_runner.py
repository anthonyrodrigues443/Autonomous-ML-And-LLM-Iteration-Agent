"""Tests for the code runners.

LocalCodeRunner is tested for real (offline subprocess). E2BCodeRunner is tested
with an injected fake sandbox so it needs neither the e2b extra nor a key; a live
e2b test lives in the integration suite.
"""

from __future__ import annotations

from typing import Any

import pytest

from iterate.adapters.compute.runner import (
    CodeRunner,
    E2BCodeRunner,
    LocalCodeRunner,
    RunResult,
)

# ─── LocalCodeRunner (real subprocess, offline) ──────────────────────────


def test_local_runner_round_trips_inputs_and_outputs() -> None:
    script = (
        "data = open('in.txt').read()\n"
        "open('out.txt', 'w').write(data.upper())\n"
        "print('done')\n"
    )
    result = LocalCodeRunner().run(
        script,
        inputs={"in.txt": b"hello"},
        outputs=["out.txt"],
        timeout=30,
    )
    assert result.succeeded
    assert result.exit_code == 0
    assert "done" in result.stdout
    assert result.outputs["out.txt"] == b"HELLO"


def test_local_runner_captures_timeout() -> None:
    result = LocalCodeRunner().run(
        "import time; time.sleep(5)",
        inputs={},
        outputs=[],
        timeout=0.5,
    )
    assert result.timed_out
    assert not result.succeeded


def test_local_runner_captures_nonzero_exit_and_stderr() -> None:
    result = LocalCodeRunner().run(
        "import sys; sys.stderr.write('boom'); sys.exit(3)",
        inputs={},
        outputs=[],
        timeout=30,
    )
    assert result.exit_code == 3
    assert not result.succeeded
    assert "boom" in result.stderr


def test_local_runner_missing_output_is_absent_not_an_error() -> None:
    result = LocalCodeRunner().run("print('hi')", inputs={}, outputs=["never.txt"], timeout=30)
    assert result.succeeded
    assert "never.txt" not in result.outputs


# ─── E2BCodeRunner (fake sandbox injected) ───────────────────────────────


class _FakeLogs:
    def __init__(self, stdout: list[str], stderr: list[str]) -> None:
        self.stdout = stdout
        self.stderr = stderr


class _FakeExecution:
    def __init__(self, stdout: list[str], stderr: list[str], error: Any = None) -> None:
        self.logs = _FakeLogs(stdout, stderr)
        self.error = error


class _FakeFiles:
    def __init__(self, store: dict[str, bytes]) -> None:
        self._store = store
        self.written: dict[str, bytes] = {}

    def write(self, path: str, content: bytes) -> None:
        self.written[path] = content

    def read(self, path: str, format: str = "bytes") -> bytes:
        if path not in self._store:
            raise FileNotFoundError(path)
        return self._store[path]


class _FakeSandbox:
    def __init__(self, *, outputs: dict[str, bytes], execution: _FakeExecution) -> None:
        self.files = _FakeFiles(outputs)
        self._execution = execution
        self.killed = False
        self.ran: str | None = None

    def run_code(self, script: str, timeout: float | None = None) -> _FakeExecution:
        self.ran = script
        return self._execution

    def kill(self) -> None:
        self.killed = True


def test_e2b_runner_uploads_runs_reads_and_tears_down() -> None:
    sandbox = _FakeSandbox(
        outputs={"/home/user/preds.csv": b"0,1,0"},
        execution=_FakeExecution(stdout=["training\n", "done\n"], stderr=[]),
    )
    runner = E2BCodeRunner(sandbox_factory=lambda _timeout: sandbox)

    result = runner.run(
        "print('done')",
        inputs={"train.csv": b"x,y"},
        outputs=["preds.csv"],
        timeout=60,
    )

    assert result.succeeded
    assert result.stdout == "training\ndone\n"
    assert result.outputs["preds.csv"] == b"0,1,0"
    assert sandbox.files.written == {"/home/user/train.csv": b"x,y"}
    assert sandbox.ran == "print('done')"
    assert sandbox.killed  # torn down even on the happy path


def test_e2b_runner_reports_execution_error() -> None:
    class _Err:
        name = "ValueError"
        value = "bad input"

    sandbox = _FakeSandbox(
        outputs={},
        execution=_FakeExecution(stdout=[], stderr=[], error=_Err()),
    )
    runner = E2BCodeRunner(sandbox_factory=lambda _timeout: sandbox)
    result = runner.run("raise ValueError('bad input')", inputs={}, outputs=[], timeout=60)

    assert not result.succeeded
    assert result.exit_code == 1
    assert "ValueError: bad input" in result.stderr
    assert sandbox.killed


def test_e2b_runner_tears_down_even_when_run_raises() -> None:
    class _Boom:
        files = _FakeFiles({})

        def run_code(self, script: str, timeout: float | None = None) -> Any:
            raise RuntimeError("sandbox blew up")

        def __init__(self) -> None:
            self.killed = False

        def kill(self) -> None:
            self.killed = True

    boom = _Boom()
    runner = E2BCodeRunner(sandbox_factory=lambda _timeout: boom)
    with pytest.raises(RuntimeError, match="blew up"):
        runner.run("x", inputs={}, outputs=[], timeout=60)
    assert boom.killed


# ─── Protocol conformance ────────────────────────────────────────────────


def test_both_runners_satisfy_the_protocol() -> None:
    assert isinstance(LocalCodeRunner(), CodeRunner)
    assert isinstance(E2BCodeRunner(), CodeRunner)


def test_run_result_succeeded_logic() -> None:
    assert RunResult(stdout="", stderr="", exit_code=0).succeeded
    assert not RunResult(stdout="", stderr="", exit_code=1).succeeded
    assert not RunResult(stdout="", stderr="", exit_code=0, timed_out=True).succeeded
