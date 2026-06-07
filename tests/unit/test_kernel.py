"""Tests for the StatefulKernel seam.

`LocalKernel` boots a real IPython kernel (offline, no key) — these prove state
persists across cells, errors are captured not raised, outputs are read back, and
timeouts are caught. `E2BKernel` is unit-tested with a fake sandbox; its live test
is opt-in.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from iterate.adapters.compute.kernel import CellResult, E2BKernel, LocalKernel, StatefulKernel

if TYPE_CHECKING:
    from collections.abc import Iterator


# ─── LocalKernel (real kernel, offline) ──────────────────────────────────


@pytest.fixture
def kernel() -> Iterator[LocalKernel]:
    k = LocalKernel()
    k.start({"data.txt": b"hello"})
    try:
        yield k
    finally:
        k.close()


def test_local_kernel_state_persists_across_cells(kernel: LocalKernel) -> None:
    assert kernel.run_cell("x = 21", timeout=30).ok
    result = kernel.run_cell("print(x * 2)", timeout=30)  # x survives into the next cell
    assert result.ok, result.error
    assert result.stdout.strip() == "42"


def test_local_kernel_captures_error_and_survives(kernel: LocalKernel) -> None:
    result = kernel.run_cell("raise ValueError('boom')", timeout=30)
    assert not result.ok
    assert result.error is not None
    assert "boom" in result.error
    # the kernel is still usable after a cell error (feedback, not fatal)
    assert kernel.run_cell("print('alive')", timeout=30).stdout.strip() == "alive"


def test_local_kernel_reads_inputs_and_outputs(kernel: LocalKernel) -> None:
    # the input file written at start() is visible to the cell...
    read = kernel.run_cell("print(open('data.txt').read())", timeout=30)
    assert read.stdout.strip() == "hello"
    # ...and a file the cell writes is read back via read_output
    assert kernel.run_cell("open('out.txt','w').write('done')", timeout=30).ok
    assert kernel.read_output("out.txt") == b"done"
    assert kernel.read_output("missing.txt") is None


def test_local_kernel_captures_structured_outputs(kernel: LocalKernel) -> None:
    result = kernel.run_cell("print('hi')\n6 * 7", timeout=30)  # a stream + an execute_result
    assert result.ok, result.error
    kinds = [o["type"] for o in result.outputs]
    assert "stream" in kinds
    assert "execute_result" in kinds
    res = next(o for o in result.outputs if o["type"] == "execute_result")
    assert res["data"]["text/plain"] == "42"


def test_local_kernel_captures_error_output(kernel: LocalKernel) -> None:
    result = kernel.run_cell("undefined_name", timeout=30)
    assert not result.ok
    err = next(o for o in result.outputs if o["type"] == "error")
    assert err["ename"] == "NameError"


def test_namespace_summary_lists_defined_variables(kernel: LocalKernel) -> None:
    kernel.run_cell("import pandas as pd\ndf = pd.DataFrame({'a': [1, 2, 3]})\nk = 7", timeout=30)
    ns = kernel.namespace_summary()
    assert "df DataFrame (3, 1)" in ns  # the agent can see its own variables + shapes
    assert "k = 7" in ns


def test_local_kernel_catches_timeout(kernel: LocalKernel) -> None:
    result = kernel.run_cell("import time; time.sleep(10)", timeout=1.0)
    assert result.timed_out
    assert not result.ok


# ─── install fallback (uv venvs ship without pip) ────────────────────────


class _Proc:
    def __init__(self, returncode: int, stderr: str = "") -> None:
        self.returncode = returncode
        self.stderr = stderr


def test_install_falls_back_to_uv_when_venv_has_no_pip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: Any) -> _Proc:
        calls.append(list(cmd))
        if "pip" in cmd and cmd[0] != "uv":
            return _Proc(1, "/x/python3: No module named pip")
        return _Proc(0)

    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/local/bin/uv")
    assert LocalKernel().install(["catboost"]) == ""
    assert calls[1][:3] == ["uv", "pip", "install"]  # the fallback actually fired
    assert "catboost" in calls[1]


def test_install_bootstraps_ensurepip_when_uv_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []
    pip_attempts = 0

    def fake_run(cmd: list[str], **kwargs: Any) -> _Proc:
        nonlocal pip_attempts
        calls.append(list(cmd))
        if "ensurepip" in cmd:
            return _Proc(0)
        pip_attempts += 1
        return _Proc(1, "No module named pip") if pip_attempts == 1 else _Proc(0)

    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setattr("shutil.which", lambda name: None)
    assert LocalKernel().install(["xgboost"]) == ""
    assert any("ensurepip" in c for c in calls)


def test_install_returns_the_error_log_on_real_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # a genuine pip failure (package doesn't exist) must surface, not retry forever
    monkeypatch.setattr(
        "subprocess.run", lambda cmd, **kw: _Proc(1, "ERROR: no matching distribution")
    )
    log = LocalKernel().install(["no-such-package"])
    assert "no matching distribution" in log


# ─── E2BKernel (fake sandbox) ─────────────────────────────────────────────


class _FakeLogs:
    def __init__(self, stdout: list[str], stderr: list[str]) -> None:
        self.stdout = stdout
        self.stderr = stderr


class _FakeExecution:
    def __init__(self, stdout: list[str], error: Any = None) -> None:
        self.logs = _FakeLogs(stdout, [])
        self.error = error


class _FakeSandbox:
    def __init__(self, store: dict[str, bytes]) -> None:
        self._store = store
        self.written: dict[str, bytes] = {}
        self.ran: list[str] = []
        self.killed = False

    class _Files:
        def __init__(self, outer: _FakeSandbox) -> None:
            self._outer = outer

        def write(self, path: str, content: bytes) -> None:
            self._outer.written[path] = content

        def read(self, path: str, format: str = "bytes") -> bytes:
            if path not in self._outer._store:
                raise FileNotFoundError(path)
            return self._outer._store[path]

    @property
    def files(self) -> _FakeSandbox._Files:
        return _FakeSandbox._Files(self)

    def run_code(self, code: str, timeout: float | None = None) -> _FakeExecution:
        self.ran.append(code)
        return _FakeExecution(stdout=["ok\n"])

    def kill(self) -> None:
        self.killed = True


def test_e2b_kernel_reuses_one_sandbox_across_cells() -> None:
    sandbox = _FakeSandbox(store={"/home/user/preds.csv": b"0\n1\n"})
    k = E2BKernel(sandbox_factory=lambda: sandbox)
    k.start({"train.csv": b"x,y"})
    k.run_cell("a = 1", timeout=30)
    k.run_cell("b = 2", timeout=30)
    assert sandbox.written == {"/home/user/train.csv": b"x,y"}
    assert sandbox.ran == ["a = 1", "b = 2"]  # same sandbox, state persists
    assert k.read_output("preds.csv") == b"0\n1\n"
    k.close()
    assert sandbox.killed


def test_both_kernels_satisfy_the_protocol() -> None:
    assert isinstance(LocalKernel(), StatefulKernel)
    assert isinstance(E2BKernel(), StatefulKernel)


def test_cell_result_ok_logic() -> None:
    assert CellResult("", "").ok
    assert not CellResult("", "", error="boom").ok
    assert not CellResult("", "", timed_out=True).ok
