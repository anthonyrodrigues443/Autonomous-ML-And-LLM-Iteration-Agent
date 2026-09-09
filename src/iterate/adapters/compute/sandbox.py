"""Sandbox execution venue: runs the agent's generated code, never the user's.

A code candidate goes through the injected `CodeRunner`; a baseline or spec
candidate runs in-process. Every failure is captured on `ExperimentResult.error`,
never raised.
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
