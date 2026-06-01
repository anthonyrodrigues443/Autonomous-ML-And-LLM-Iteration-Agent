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

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
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


__all__ = ["ComputeBackend"]
