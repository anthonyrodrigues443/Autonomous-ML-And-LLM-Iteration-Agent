"""Sandbox execution venue (v0.2) — runs the agent's generated training code.

Routes by candidate type:

- **code candidate** (``changes = {"code": ...}``) → the target assembles a
  `CodeJob`, the injected `CodeRunner` runs it (local subprocess or e2b sandbox),
  and the target scores the predictions. The venue is whichever runner was passed:
  `LocalCodeRunner` for ``--compute local``, `E2BCodeRunner` for ``--compute e2b``.
- **baseline / spec candidate** → run in-process (the spec path is our own trusted
  factory code), via the shared `run_in_process` helper.

Every failure mode — a runner that can't boot, a crashing or timed-out script, a
target that doesn't support code — is captured on `ExperimentResult.error`, never
raised, so a bad candidate never crashes the loop.

**Security boundary (permanent):** this runs the agent's OWN generated code only,
never the user's source. User-provided source is read as text by the Reconstructor.
"""

from __future__ import annotations

from time import perf_counter
from typing import TYPE_CHECKING

from iterate.adapters.compute.base import SupportsCodeGen
from iterate.adapters.compute.local import run_in_process
from iterate.core.codegen import is_code_candidate
from iterate.schemas.experiment import ExperimentResult

if TYPE_CHECKING:
    from iterate.adapters.compute.runner import CodeRunner
    from iterate.schemas.experiment import Candidate
    from iterate.targets.base import BenchmarkTarget


class SandboxExecutor:
    """Runs code candidates through a `CodeRunner`; everything else in-process."""

    def __init__(self, code_runner: CodeRunner, *, timeout: float = 300.0) -> None:
        self._runner = code_runner
        self._timeout = timeout

    def execute(
        self, target: BenchmarkTarget, candidate: Candidate | None = None
    ) -> ExperimentResult:
        if candidate is not None and is_code_candidate(candidate.changes):
            return self._execute_code(target, candidate)
        return run_in_process(target, candidate)

    def _execute_code(self, target: BenchmarkTarget, candidate: Candidate) -> ExperimentResult:
        if not isinstance(target, SupportsCodeGen):
            return ExperimentResult(
                experiment_id=candidate.id,
                error=f"target {target.name!r} does not support code candidates",
            )
        start = perf_counter()
        try:
            job = target.build_code_job(candidate)
            run = self._runner.run(
                job.script,
                inputs=job.inputs,
                outputs=job.outputs,
                packages=job.packages,
                timeout=self._timeout,
            )
            result = target.score_code_job(run, candidate.id)
        except Exception as exc:  # runner couldn't boot / upload / run, or scoring blew up
            return ExperimentResult(
                experiment_id=candidate.id,
                error=f"{type(exc).__name__}: {exc}",
                duration_seconds=perf_counter() - start,
            )
        return result.model_copy(update={"duration_seconds": perf_counter() - start})


__all__ = ["SandboxExecutor"]
