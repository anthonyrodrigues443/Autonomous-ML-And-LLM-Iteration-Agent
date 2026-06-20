"""Stateful kernels — the substrate for cell-by-cell execution.

A `StatefulKernel` keeps a live namespace across cells: the coding agent runs one
cell, sees its real output, then runs the next *in the same session* (variables,
imports, fitted objects all persist). This is what lets the agent inspect the data
and build on what it sees, instead of writing a whole pipeline blind.

Two venues, one protocol:
- `LocalKernel` — a real IPython kernel via `jupyter_client`, on this machine. No
  isolation (generated code runs with the user's permissions); offline, the free
  default driver's venue.
- `E2BKernel` — one ephemeral e2b sandbox reused across cells (its `run_code` keeps
  kernel state), isolated.

Both: `start(inputs)` writes the input files (train/holdout features/meta) and boots
the kernel; `run_cell` executes a cell and returns its streams + any error (never
raises on a failing cell — a traceback is captured and fed back); `read_output`
reads a named file the session produced (e.g. predictions.csv); `close` tears down.

**Sealed-holdout invariant:** the holdout *labels* are never written into the
kernel's working dir — only holdout features cross in. Scoring stays host-side.
"""

from __future__ import annotations

import contextlib
import queue
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Callable

_ANSI = re.compile(r"\x1b\[[0-9;]*m")

# Introspects the live namespace so the agent can see what it has defined (and not
# re-import / re-derive / mis-name). Defensive per-variable; skips modules/functions.
_NS_SNIPPET = (
    "for _k in sorted(k for k in list(globals()) if not k.startswith('_')):\n"
    "    try:\n"
    "        import pandas as _pd\n"
    "        _v = globals()[_k]\n"
    "        if isinstance(_v, _pd.DataFrame): print(_k, 'DataFrame', tuple(_v.shape))\n"
    "        elif isinstance(_v, _pd.Series): print(_k, 'Series len', len(_v))\n"
    "        elif hasattr(_v, 'shape'): print(_k, type(_v).__name__, tuple(getattr(_v, 'shape')))\n"
    "        elif hasattr(_v, 'predict'): print(_k, type(_v).__name__, '(fitted model)')\n"
    "        elif type(_v).__module__ == 'builtins' and not callable(_v):\n"
    "            print(_k, '=', repr(_v)[:40])\n"
    "    except Exception: pass\n"
)


def _run_install(cmd: list[str]) -> str:
    """Run one install command; "" on success, else the error log (never raises)."""
    import subprocess

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=False)
    except subprocess.TimeoutExpired:
        return f"install timed out: {' '.join(cmd)}"
    except FileNotFoundError:
        return f"command not found: {cmd[0]}"
    return "" if proc.returncode == 0 else (proc.stderr.strip() or f"exit code {proc.returncode}")


def _strip_ansi(text: str) -> str:
    """IPython tracebacks come colour-coded; strip the escapes for clean feedback."""
    return _ANSI.sub("", text)


@dataclass(frozen=True)
class CellResult:
    """The outcome of running one cell: its streams, any error, and the structured
    outputs (stream / execute_result / display_data / error) as nbformat-ready dicts,
    so the session can be rendered as a genuinely *executed* notebook."""

    stdout: str
    stderr: str
    error: str | None = None  # the exception/traceback if the cell raised
    timed_out: bool = False
    outputs: list[dict[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.error is None and not self.timed_out


@runtime_checkable
class StatefulKernel(Protocol):
    """A live kernel whose namespace persists across `run_cell` calls."""

    def start(self, inputs: dict[str, bytes]) -> None:
        """Boot the kernel with ``inputs`` present in its working directory."""
        ...

    def run_cell(self, code: str, *, timeout: float) -> CellResult:
        """Execute one cell. MUST capture a failing cell (error/timeout) in the
        `CellResult`, never raise — a traceback is feedback for the next cell."""
        ...

    def install(self, packages: list[str]) -> str:
        """Install packages into the kernel's environment (install-on-demand for a
        missing import). Returns "" on success, else an error log. Best-effort."""
        ...

    def namespace_summary(self) -> str:
        """A compact listing of the user-defined variables currently live (names +
        shapes/types), so the agent builds on what exists instead of re-deriving or
        mis-naming it. Empty string if unavailable."""
        ...

    def read_output(self, name: str) -> bytes | None:
        """Read a file the session wrote (e.g. predictions.csv), or None if absent."""
        ...

    def close(self) -> None:
        """Tear the kernel down and clean up."""
        ...


class LocalKernel:
    """A real IPython kernel on this machine (no isolation; offline)."""

    def __init__(self) -> None:
        self._km: Any = None
        self._kc: Any = None
        self._tmp: tempfile.TemporaryDirectory[str] | None = None
        self._workdir: Path | None = None

    def start(self, inputs: dict[str, bytes]) -> None:
        from jupyter_client.manager import start_new_kernel

        self._tmp = tempfile.TemporaryDirectory(prefix="iterate-kernel-")
        self._workdir = Path(self._tmp.name)
        for name, content in inputs.items():
            (self._workdir / name).write_bytes(content)
        self._km, self._kc = start_new_kernel(cwd=str(self._workdir))

    def run_cell(self, code: str, *, timeout: float) -> CellResult:
        if self._kc is None or self._km is None:
            raise RuntimeError("kernel not started")
        msg_id = self._kc.execute(code)
        out: list[str] = []
        err: list[str] = []
        error: str | None = None
        outputs: list[dict[str, Any]] = []
        while True:
            try:
                msg = self._kc.get_iopub_msg(timeout=timeout)
            except queue.Empty:
                self._km.interrupt_kernel()
                return CellResult("".join(out), "".join(err), timed_out=True, outputs=outputs)
            if msg.get("parent_header", {}).get("msg_id") != msg_id:
                continue  # a message from an earlier cell; ignore
            mtype = msg["msg_type"]
            content = msg["content"]
            if mtype == "stream":
                (out if content.get("name") == "stdout" else err).append(content.get("text", ""))
                outputs.append(
                    {"type": "stream", "name": content.get("name", "stdout"),
                     "text": content.get("text", "")}
                )
            elif mtype == "execute_result":
                outputs.append(
                    {"type": "execute_result", "data": content.get("data", {}),
                     "metadata": content.get("metadata", {}),
                     "execution_count": content.get("execution_count")}
                )
            elif mtype == "display_data":
                outputs.append(
                    {"type": "display_data", "data": content.get("data", {}),
                     "metadata": content.get("metadata", {})}
                )
            elif mtype == "error":
                tb = "\n".join(content.get("traceback", []))
                error = _strip_ansi(tb) or f"{content.get('ename')}: {content.get('evalue')}"
                outputs.append(
                    {"type": "error", "ename": content.get("ename", ""),
                     "evalue": content.get("evalue", ""),
                     "traceback": content.get("traceback", [])}
                )
            elif mtype == "status" and content.get("execution_state") == "idle":
                break
        return CellResult("".join(out), "".join(err), error=error, outputs=outputs)

    def install(self, packages: list[str]) -> str:
        import shutil
        import sys

        if not packages:
            return ""
        log = _run_install([sys.executable, "-m", "pip", "install", "--quiet", *packages])
        if not log or "No module named pip" not in log:
            return log
        # uv-managed venvs ship without pip — fall back to uv targeting the kernel's
        # interpreter, else bootstrap pip via ensurepip and retry.
        if shutil.which("uv"):
            return _run_install(
                ["uv", "pip", "install", "--quiet", "--python", sys.executable, *packages]
            )
        _run_install([sys.executable, "-m", "ensurepip", "--upgrade"])
        return _run_install([sys.executable, "-m", "pip", "install", "--quiet", *packages])

    def namespace_summary(self) -> str:
        if self._kc is None:
            return ""
        result = self.run_cell(_NS_SNIPPET, timeout=15.0)
        return "" if result.error else result.stdout.strip()

    def read_output(self, name: str) -> bytes | None:
        if self._workdir is None:
            return None
        path = self._workdir / name
        return path.read_bytes() if path.exists() else None

    def close(self) -> None:
        if self._kc is not None:
            self._kc.stop_channels()
        if self._km is not None:
            self._km.shutdown_kernel(now=True)
        if self._tmp is not None:
            self._tmp.cleanup()
        self._km = self._kc = self._tmp = self._workdir = None


class E2BKernel:
    """One ephemeral e2b sandbox reused across cells (isolated; needs E2B_API_KEY).

    The sandbox's `run_code` keeps Jupyter-kernel state across calls, so the same
    sandbox *is* the session. `sandbox_factory` is injectable for tests.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        work_dir: str = "/home/user",
        lease_seconds: float = 900.0,
        sandbox_factory: Callable[[], Any] | None = None,
    ) -> None:
        self._api_key = api_key
        self._work_dir = work_dir
        # e2b sandboxes default to a 300s lifetime; a cell-by-cell session (LLM
        # latency between cells is NOT sandbox-execution time) easily outlives that
        # and the sandbox would die mid-session. We renew a sliding lease on every
        # cell instead — alive while the session works, auto-reaped ~lease_seconds
        # after the last activity (so a crash leaves at most one lease of orphan
        # cost, not hours). 900s comfortably exceeds any single cell + think-gap and
        # stays under the 3600s Hobby-plan cap.
        self._lease_seconds = lease_seconds
        self._sandbox_factory = sandbox_factory
        self._sandbox: Any = None

    def start(self, inputs: dict[str, bytes]) -> None:
        self._sandbox = self._make_sandbox()
        self._renew_lease()  # bump off the 300s default before any think-gap
        for name, content in inputs.items():
            self._sandbox.files.write(f"{self._work_dir}/{name}", content)

    def _renew_lease(self) -> None:
        """Slide the sandbox's expiry to now + lease_seconds. Best-effort: a dead
        sandbox surfaces on the next run_code, not here, so never fail a cell over
        a keepalive hiccup."""
        if self._sandbox is None:
            return
        setter = getattr(self._sandbox, "set_timeout", None)
        if callable(setter):
            with contextlib.suppress(Exception):
                setter(int(self._lease_seconds))

    def run_cell(self, code: str, *, timeout: float) -> CellResult:
        if self._sandbox is None:
            raise RuntimeError("kernel not started")
        self._renew_lease()
        execution = self._sandbox.run_code(code, timeout=timeout)
        stdout = "".join(execution.logs.stdout)
        stderr = "".join(execution.logs.stderr)
        outputs: list[dict[str, Any]] = []
        if stdout:
            outputs.append({"type": "stream", "name": "stdout", "text": stdout})
        if stderr:
            outputs.append({"type": "stream", "name": "stderr", "text": stderr})
        for result in getattr(execution, "results", None) or []:
            data: dict[str, str] = {}
            for attr, mime in (("text", "text/plain"), ("html", "text/html"), ("png", "image/png")):
                value = getattr(result, attr, None)
                if value:
                    data[mime] = value
            if data:
                outputs.append({"type": "display_data", "data": data, "metadata": {}})
        err = getattr(execution, "error", None)
        error = None
        if err is not None:
            traceback = getattr(err, "traceback", None) or [str(getattr(err, "value", err))]
            error = _strip_ansi("\n".join(traceback)) or f"{getattr(err, 'name', 'Error')}"
            outputs.append(
                {"type": "error", "ename": getattr(err, "name", "Error"),
                 "evalue": str(getattr(err, "value", "")), "traceback": traceback}
            )
        return CellResult(stdout, stderr, error=error, outputs=outputs)

    def install(self, packages: list[str]) -> str:
        if self._sandbox is None or not packages:
            return ""
        self._renew_lease()  # a cold wheel install can be slow; don't let the lease lapse
        execution = self._sandbox.run_code(f"!pip install -q {' '.join(packages)}")
        return "".join(execution.logs.stderr)

    def namespace_summary(self) -> str:
        if self._sandbox is None:
            return ""
        result = self.run_cell(_NS_SNIPPET, timeout=15.0)
        return "" if result.error else result.stdout.strip()

    def read_output(self, name: str) -> bytes | None:
        if self._sandbox is None:
            return None
        try:
            data = self._sandbox.files.read(f"{self._work_dir}/{name}", format="bytes")
        except Exception:
            return None
        return data if isinstance(data, bytes) else str(data).encode()

    def close(self) -> None:
        if self._sandbox is not None:
            self._sandbox.kill()
            self._sandbox = None

    def _make_sandbox(self) -> Any:
        if self._sandbox_factory is not None:
            return self._sandbox_factory()
        try:
            from e2b_code_interpreter import Sandbox
        except ImportError as exc:  # pragma: no cover - e2b ships in core; defensive only
            raise RuntimeError("e2b_code_interpreter failed to import; reinstall iterate-ai") from exc
        return Sandbox(api_key=self._api_key)


__all__ = ["CellResult", "E2BKernel", "LocalKernel", "StatefulKernel"]
