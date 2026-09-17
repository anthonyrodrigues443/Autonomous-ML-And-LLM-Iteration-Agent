"""Stateful kernels: a live namespace that persists across cells.

`LocalKernel` is an IPython kernel on this machine, confined to its own folder where
the platform can enforce it (see `confine`); `E2BKernel` is one sandbox reused across
cells. `run_cell` never raises on a failing cell; the traceback is returned. Holdout
labels are never written into the kernel's working directory, so scoring stays
host-side.
"""

from __future__ import annotations

import contextlib
import json
import os
import queue
import re
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from iterate.adapters.compute import confine

if TYPE_CHECKING:
    from collections.abc import Callable

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_EPERM_PATH = re.compile(r"\[Errno 1\] Operation not permitted: '([^']+)'")

# Cells get none of the harness's own keys. A prompt kernel gets only its target's key.
KERNEL_SECRETS = frozenset(
    {
        "ITERATE_BACKEND_API_KEY",
        "ITERATE_TARGET_API_KEY",
        "E2B_API_KEY",
        "KAGGLE_USERNAME",
        "KAGGLE_KEY",
        "GROQ_API_KEY",
        "TOGETHER_API_KEY",
        "DEEPSEEK_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "NOTION_API_KEY",
        "SLACK_WEBHOOK_URL",
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
    }
)

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

_MODULES_SNIPPET = (
    "(lambda: print(__import__('json').dumps(sorted("
    "{m.partition('.')[0] for m in list(__import__('sys').modules)}))))()\n"
)


def _strip_ansi(text: str) -> str:
    """IPython tracebacks come colour-coded; strip the escapes for clean feedback."""
    return _ANSI.sub("", text)


@dataclass(frozen=True)
class Blocked:
    """A path a confined cell was refused outside its folder. ``program`` is set when
    the refusal was to run it."""

    path: str
    program: bool


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

    def loaded_modules(self) -> list[str] | None:
        """Top-level names of every module the kernel process has imported, or None
        when they cannot be read."""
        ...

    def restart(self) -> None:
        """Restart the kernel process in place: same working directory, empty namespace."""
        ...

    def blocked(self, error: str | None) -> Blocked | None:
        """What a confined cell was refused outside its folder, or None."""
        ...

    def namespace_summary(self) -> str:
        """A compact listing of the user-defined variables currently live (names +
        shapes/types), so the agent builds on what exists instead of re-deriving or
        mis-naming it. Empty string if unavailable."""
        ...

    def read_output(self, name: str) -> bytes | None:
        """Read a file the session wrote (e.g. predictions.csv), or None if absent."""
        ...

    def keepalive(self) -> None:
        """Keep an idle kernel from being reaped (a paused session runs no cells,
        so a leased sandbox would expire mid-pause). Best-effort; local no-op."""
        ...

    def close(self) -> None:
        """Tear the kernel down and clean up."""
        ...


class LocalKernel:
    """A real IPython kernel on this machine, confined where the platform allows."""

    def __init__(
        self,
        *,
        confinement: confine.Confinement | None = None,
        target_key: str | None = None,
    ) -> None:
        self._confinement = confinement or confine.Confinement()
        self._target_key = target_key
        self._confined: bool | None = None
        self._programs: frozenset[str] = frozenset()
        self._km: Any = None
        self._kc: Any = None
        self._tmp: tempfile.TemporaryDirectory[str] | None = None
        self._scratch: tempfile.TemporaryDirectory[str] | None = None
        self._workdir: Path | None = None

    @property
    def confined(self) -> bool:
        if self._confined is None:
            self._confined = confine.sandbox_available()
        return self._confined

    @confined.setter
    def confined(self, value: bool) -> None:
        self._confined = value

    def start(self, inputs: dict[str, bytes]) -> None:
        from jupyter_client.manager import KernelManager

        self._tmp = tempfile.TemporaryDirectory(prefix="iterate-kernel-")
        self._workdir = Path(self._tmp.name)
        for name, content in inputs.items():
            (self._workdir / name).write_bytes(content)
        # torch_shm_manager binds a unix socket under TMPDIR, capped at 104 bytes.
        self._scratch = tempfile.TemporaryDirectory(prefix="itk-", dir="/tmp")
        scratch = Path(os.path.realpath(self._scratch.name))
        env = {
            k: v for k, v in os.environ.items() if k != "OLDPWD" and k.upper() not in KERNEL_SECRETS
        }
        env |= {
            "PWD": os.path.realpath(self._workdir),
            "TMPDIR": str(scratch),
            "JOBLIB_TEMP_FOLDER": str(scratch),
            "IPYTHONDIR": str(scratch / "ipython"),
        }
        if self._target_key:
            env["ITERATE_TARGET_API_KEY"] = self._target_key
        weights = self._confinement.weights
        if self.confined and weights is not None:
            env = {k: v for k, v in env.items() if k not in confine.WEIGHTS_OVERRIDES}
            # The sandbox lets a cell create inside the folder, never its parent.
            weights.mkdir(parents=True, exist_ok=True)
            env |= confine.weights_env(weights)
        # Pin the kernel to THIS interpreter. The registered "python3" kernelspec
        # can be a different environment, and `install()` pip-installs into
        # sys.executable.
        km = KernelManager(kernel_name="python3")
        km.connection_file = str(scratch / "kernel.json")
        argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
        if self.confined:
            self._programs = confine.installer_paths()
            sandbox = scratch / "kernel.sb"
            sandbox.write_text(confine.profile(self._workdir, scratch, self._confinement))
            argv = [confine.SANDBOX_EXEC, "-f", str(sandbox), *argv]
        if km.kernel_spec is not None:  # no python3 kernelspec at all: fall back to default
            km.kernel_spec.argv = argv
        km.start_kernel(cwd=str(self._workdir), env=env)
        kc = km.client()
        kc.start_channels()
        kc.wait_for_ready(timeout=60)
        self._km, self._kc = km, kc

    def run_cell(self, code: str, *, timeout: float) -> CellResult:
        if self._kc is None or self._km is None:
            raise RuntimeError("kernel not started")
        msg_id = self._kc.execute(code)
        out: list[str] = []
        err: list[str] = []
        error: str | None = None
        outputs: list[dict[str, Any]] = []
        # `timeout` bounds the CELL. The per-message wait is what is left of the
        # cell's deadline, or a chatty cell resets the clock on every print.
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._km.interrupt_kernel()
                return CellResult("".join(out), "".join(err), timed_out=True, outputs=outputs)
            try:
                msg = self._kc.get_iopub_msg(timeout=remaining)
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
                    {
                        "type": "stream",
                        "name": content.get("name", "stdout"),
                        "text": content.get("text", ""),
                    }
                )
            elif mtype == "execute_result":
                outputs.append(
                    {
                        "type": "execute_result",
                        "data": content.get("data", {}),
                        "metadata": content.get("metadata", {}),
                        "execution_count": content.get("execution_count"),
                    }
                )
            elif mtype == "display_data":
                outputs.append(
                    {
                        "type": "display_data",
                        "data": content.get("data", {}),
                        "metadata": content.get("metadata", {}),
                    }
                )
            elif mtype == "error":
                tb = "\n".join(content.get("traceback", []))
                error = _strip_ansi(tb) or f"{content.get('ename')}: {content.get('evalue')}"
                outputs.append(
                    {
                        "type": "error",
                        "ename": content.get("ename", ""),
                        "evalue": content.get("evalue", ""),
                        "traceback": content.get("traceback", []),
                    }
                )
            elif mtype == "status" and content.get("execution_state") == "idle":
                break
        return CellResult("".join(out), "".join(err), error=error, outputs=outputs)

    def install(self, packages: list[str]) -> str:
        from iterate.adapters.compute import deps

        if not packages:
            return ""
        with tempfile.TemporaryDirectory(prefix="iterate-pins-") as tmp:
            return deps.install(sys.executable, packages, deps.pin_everything(Path(tmp)))

    def loaded_modules(self) -> list[str] | None:
        if self._kc is None:
            return None
        result = self.run_cell(_MODULES_SNIPPET, timeout=15.0)
        if result.error or result.timed_out:
            return None
        try:
            return list(json.loads(result.stdout.strip().splitlines()[-1]))
        except (ValueError, IndexError):
            return None

    def restart(self) -> None:
        if self._km is None or self._kc is None:
            raise RuntimeError("kernel not started")
        # restart_kernel reuses the launch argv and env, so the sandbox stays on.
        self._km.restart_kernel(now=True)
        self._kc.wait_for_ready(timeout=60)

    def blocked(self, error: str | None) -> Blocked | None:
        if not self.confined or not error or self._workdir is None:
            return None
        inside = os.path.realpath(self._workdir) + os.sep
        for found in _EPERM_PATH.finditer(error):
            named = found.group(1)
            real = os.path.realpath(self._workdir / named)
            # A program run by bare name is refused under the name the cell gave.
            on_path = shutil.which(named) if os.sep not in named else None
            if (os.path.realpath(on_path) if on_path else real) in self._programs:
                return Blocked(named, program=True)
            if not real.startswith(inside):
                return Blocked(named, program=False)
        return None

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

    def keepalive(self) -> None:
        """No-op: a local kernel subprocess survives idle time indefinitely."""

    def close(self) -> None:
        if self._kc is not None:
            self._kc.stop_channels()
        if self._km is not None:
            self._km.shutdown_kernel(now=True)
        for folder in (self._tmp, self._scratch):
            if folder is not None:
                folder.cleanup()
        self._km = self._kc = self._tmp = self._scratch = self._workdir = None


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
        # A sliding lease renewed on every cell: the default 300s sandbox lifetime
        # is shorter than a session, and a crash then orphans at most one lease.
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
            # e2b SDK v2 ships the traceback as ONE string (v1 was a list). Joining
            # a string newlines every character, garbling the feedback the agent
            # debugs from, and a string traceback fails the notebook schema (both
            # caught on the first live cell-by-cell e2b run, 2026-07-12).
            tb = getattr(err, "traceback", None)
            if isinstance(tb, str):
                tb_lines = tb.splitlines()
            elif tb:
                tb_lines = [str(line) for line in tb]
            else:
                tb_lines = [str(getattr(err, "value", err))]
            error = _strip_ansi("\n".join(tb_lines)) or f"{getattr(err, 'name', 'Error')}"
            outputs.append(
                {
                    "type": "error",
                    "ename": getattr(err, "name", "Error"),
                    "evalue": str(getattr(err, "value", "")),
                    "traceback": tb_lines,
                }
            )
        return CellResult(stdout, stderr, error=error, outputs=outputs)

    def install(self, packages: list[str]) -> str:
        if self._sandbox is None or not packages:
            return ""
        self._renew_lease()  # a cold wheel install can be slow; don't let the lease lapse
        execution = self._sandbox.run_code(f"!pip install -q {' '.join(packages)}")
        return "".join(execution.logs.stderr)

    def loaded_modules(self) -> list[str] | None:
        return None

    def restart(self) -> None:
        raise NotImplementedError("an e2b session installs into its own sandbox and never restarts")

    def blocked(self, error: str | None) -> Blocked | None:
        return None

    def keepalive(self) -> None:
        """A paused session runs no cells, so the sliding lease (renewed only on
        activity) would reap the sandbox mid-pause; the pause loop ticks this."""
        self._renew_lease()

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
        # e2b SDK v2 returns a bytearray for format="bytes"; a bytearray is NOT
        # bytes, and str(bytearray(...)).encode() collapses a whole predictions
        # file into one literal line (caught live 2026-07-12).
        if isinstance(data, (bytes, bytearray)):
            return bytes(data)
        return str(data).encode()

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
            raise RuntimeError(
                "e2b_code_interpreter failed to import; reinstall iterate-ai"
            ) from exc
        # e2b SDK v2: Sandbox.create(), not the constructor (see runner.py note).
        return Sandbox.create(api_key=self._api_key)


__all__ = ["KERNEL_SECRETS", "Blocked", "CellResult", "E2BKernel", "LocalKernel", "StatefulKernel"]
