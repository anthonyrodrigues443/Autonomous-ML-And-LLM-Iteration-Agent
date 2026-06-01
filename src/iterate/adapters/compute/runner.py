"""Code runners — physically execute a Python script and return its outputs.

The low-level primitive under v0.2's sandbox: given a script + input files, run
it under a mandatory timeout and return stdout / stderr / exit code + the named
output files. Two venues, one `CodeRunner` protocol:

- `LocalCodeRunner` — runs the script in a subprocess on this machine (the
  `--compute local` path). No isolation: generated code runs with the user's
  permissions. Explicit opt-in only.
- `E2BCodeRunner` — runs it in an ephemeral e2b sandbox (the safe default for
  autonomously-generated code). `e2b_code_interpreter` is lazy-imported so this
  module loads without the `[sandbox]` extra; the sandbox factory is injectable
  for testing without e2b installed or a key.

Both place `inputs` in the script's working directory, run it, and read back the
files named in `outputs`. The code-gen *contract* (what a training script reads
and writes) is defined separately (Day 3); this is only the execution primitive.

Honest scope notes:
- A timeout is mandatory on every run.
- Network egress-deny for the e2b runner is NOT yet enforced (it needs a custom
  e2b sandbox template); flagged rather than assumed. The local runner offers no
  isolation at all.
- `E2BCodeRunner` is written to the documented e2b API and unit-tested with a
  fake sandbox, but not yet live-verified (no key in dev); the exact calls may
  need small fixes when first run against real e2b (Day 5).
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Callable

# The generated script is written under this name in the working directory.
_SCRIPT_NAME = "_iterate_run.py"


@dataclass(frozen=True)
class RunResult:
    """The outcome of running one script: streams, exit code, and output files."""

    stdout: str
    stderr: str
    exit_code: int
    outputs: dict[str, bytes] = field(default_factory=dict)
    timed_out: bool = False

    @property
    def succeeded(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


@runtime_checkable
class CodeRunner(Protocol):
    """Runs a Python script with input files and returns its outputs."""

    def run(
        self,
        script: str,
        *,
        inputs: dict[str, bytes],
        outputs: list[str],
        timeout: float,
    ) -> RunResult:
        """Run ``script`` with ``inputs`` available in its working directory.

        Reads back any file named in ``outputs`` that the script produced. Must
        not raise on a failing script — capture the failure in the `RunResult`
        (nonzero ``exit_code`` or ``timed_out``).
        """
        ...


class LocalCodeRunner:
    """Runs the script in a subprocess on this machine (no isolation)."""

    def run(
        self,
        script: str,
        *,
        inputs: dict[str, bytes],
        outputs: list[str],
        timeout: float,
    ) -> RunResult:
        with tempfile.TemporaryDirectory(prefix="iterate-run-") as tmp:
            workdir = Path(tmp)
            for name, content in inputs.items():
                (workdir / name).write_bytes(content)
            (workdir / _SCRIPT_NAME).write_text(script, encoding="utf-8")
            try:
                proc = subprocess.run(
                    [sys.executable, _SCRIPT_NAME],
                    cwd=workdir,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                return RunResult(
                    stdout=_as_text(exc.stdout),
                    stderr=_as_text(exc.stderr),
                    exit_code=-1,
                    timed_out=True,
                )
            collected = {
                name: (workdir / name).read_bytes()
                for name in outputs
                if (workdir / name).exists()
            }
            return RunResult(
                stdout=proc.stdout,
                stderr=proc.stderr,
                exit_code=proc.returncode,
                outputs=collected,
            )


class E2BCodeRunner:
    """Runs the script in an ephemeral e2b sandbox (isolated; the safe default)."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        work_dir: str = "/home/user",
        sandbox_factory: Callable[[float], Any] | None = None,
    ) -> None:
        self._api_key = api_key
        self._work_dir = work_dir
        self._sandbox_factory = sandbox_factory  # injected in tests; lazy e2b import otherwise

    def run(
        self,
        script: str,
        *,
        inputs: dict[str, bytes],
        outputs: list[str],
        timeout: float,
    ) -> RunResult:
        sandbox = self._make_sandbox(timeout)
        try:
            for name, content in inputs.items():
                sandbox.files.write(f"{self._work_dir}/{name}", content)
            execution = sandbox.run_code(script, timeout=timeout)
            stdout = "".join(execution.logs.stdout)
            stderr = "".join(execution.logs.stderr)
            error = getattr(execution, "error", None)
            exit_code = 1 if error else 0
            if error and not stderr:
                stderr = f"{getattr(error, 'name', 'Error')}: {getattr(error, 'value', error)}"
            collected: dict[str, bytes] = {}
            for name in outputs:
                data = _try_read(sandbox, f"{self._work_dir}/{name}")
                if data is not None:
                    collected[name] = data
            return RunResult(
                stdout=stdout, stderr=stderr, exit_code=exit_code, outputs=collected
            )
        finally:
            sandbox.kill()

    def _make_sandbox(self, timeout: float) -> Any:
        if self._sandbox_factory is not None:
            return self._sandbox_factory(timeout)
        try:
            from e2b_code_interpreter import Sandbox
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise RuntimeError(
                "the e2b runner needs the sandbox extra: "
                "pip install 'iterate-ai[sandbox]' and set E2B_API_KEY"
            ) from exc
        # Sandbox lifetime a bit beyond the run timeout so teardown is clean.
        return Sandbox(api_key=self._api_key, timeout=int(timeout) + 5)


def _as_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else value.decode(errors="replace")


def _try_read(sandbox: Any, path: str) -> bytes | None:
    try:
        data = sandbox.files.read(path, format="bytes")
    except Exception:  # a missing output file is expected, not an error
        return None
    return data if isinstance(data, bytes) else str(data).encode()


__all__ = ["CodeRunner", "E2BCodeRunner", "LocalCodeRunner", "RunResult"]
