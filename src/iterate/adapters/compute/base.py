"""The compute backend contract.

A `ComputeBackend` runs one experiment (baseline or a candidate) and returns an
`ExperimentResult`, catching failures rather than letting them crash the loop. The
Orchestrator depends only on this protocol, so the execution venue is swappable:

- `LocalExecutor` (v0.1) — runs in-process on this machine.
- `SandboxExecutor` (v0.2) — runs the agent's generated training code in an e2b
  sandbox; a local variant can run it on the user's machine via `--compute local`.

Extracted at v0.2 Day 1, when the second backend (the sandbox) arrives — the same
"add the protocol when the second implementation lands" call we made for the data
source and the terminator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from iterate.adapters.compute.runner import RunResult
    from iterate.schemas.experiment import Candidate, ExperimentResult
    from iterate.targets.base import BenchmarkTarget


@runtime_checkable
class ComputeBackend(Protocol):
    """What every execution venue must provide."""

    def execute(
        self, target: BenchmarkTarget, candidate: Candidate | None = None
    ) -> ExperimentResult:
        """Run one experiment and return its result.

        ``candidate=None`` runs the baseline; otherwise the candidate. Implementations
        MUST capture failures and return a non-success `ExperimentResult` (error set)
        rather than raising, so a bad candidate never crashes the loop.
        """
        ...


@dataclass(frozen=True)
class CodeJob:
    """Everything a `CodeRunner` needs to run one generated training script.

    The target assembles it (it owns the data); the executor hands it to its
    runner (it owns the venue). ``packages`` are the pip distributions the code
    imports, for install-on-demand.
    """

    script: str
    inputs: dict[str, bytes]
    outputs: list[str]
    packages: list[str] = field(default_factory=list)


@runtime_checkable
class SupportsCodeGen(Protocol):
    """A target that can be evaluated on the code-gen path.

    Splits the code path cleanly: the target shapes the data + scores (it knows
    the sealed holdout); the executor owns where the script runs. Only targets
    implementing this can take code candidates.
    """

    def build_code_job(self, candidate: Candidate) -> CodeJob:
        """Assemble the runnable job (script + input files + needed packages)."""
        ...

    def score_code_job(self, run_result: RunResult, experiment_id: str) -> ExperimentResult:
        """Score a finished run against the held-back holdout labels.

        A failed/timed-out run is a captured failure (error set), never raised;
        the script's stdout is surfaced on ``logs`` so the next proposal sees it.
        """
        ...


__all__ = ["CodeJob", "ComputeBackend", "SupportsCodeGen"]
