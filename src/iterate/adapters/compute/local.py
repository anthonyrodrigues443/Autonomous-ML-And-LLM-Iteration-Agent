"""Local execution venue — run one experiment in-process, capture failures.

The first compute backend: it runs the target directly on this machine. A bad
candidate (broken params, a fit-time error, an off-list model) must not crash the
agentic loop, so the executor catches any exception and records it on
`ExperimentResult.error` instead of propagating — the loop reads that to avoid
re-proposing a known-broken change.

Intentionally minimal. Hard isolation (timeouts, memory/CPU caps, killing runaway
training, captured stdout/stderr) arrives with the e2b sandbox backend (v0.2), at
which point a `ComputeBackend` Protocol gets extracted and e2b/cloud become further
adapters on the same port. See the BUILD_LOG backlog.
"""

from __future__ import annotations

from time import perf_counter
from typing import TYPE_CHECKING

from iterate.schemas.experiment import ExperimentResult

if TYPE_CHECKING:
    from iterate.schemas.experiment import Candidate
    from iterate.targets.base import BenchmarkTarget


def run_in_process(target: BenchmarkTarget, candidate: Candidate | None = None) -> ExperimentResult:
    """Run one experiment on this machine, converting a crash into a failed result.

    The shared in-process path: `LocalExecutor` uses it for everything, and
    `SandboxExecutor` uses it for baselines + spec candidates (only code candidates
    go to the sandbox runner).
    """
    experiment_id = candidate.id if candidate is not None else "baseline"
    start = perf_counter()
    try:
        result = target.run(candidate) if candidate is not None else target.baseline()
    except Exception as exc:
        return ExperimentResult(
            experiment_id=experiment_id,
            error=f"{type(exc).__name__}: {exc}",
            duration_seconds=perf_counter() - start,
        )
    return result.model_copy(update={"duration_seconds": perf_counter() - start})


class LocalExecutor:
    """Runs one experiment in-process; converts a crash into a failed result."""

    def execute(
        self, target: BenchmarkTarget, candidate: Candidate | None = None
    ) -> ExperimentResult:
        return run_in_process(target, candidate)


__all__ = ["LocalExecutor", "run_in_process"]
