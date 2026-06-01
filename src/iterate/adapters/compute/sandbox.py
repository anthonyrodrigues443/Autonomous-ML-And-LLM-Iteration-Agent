"""Sandbox execution venue (v0.2) — runs the agent's generated training code.

Stub for now; the real implementation lands at v0.2 Day 2. It will boot an e2b
sandbox (the safe default for autonomously-generated code), upload the dataset,
run the generated script under a timeout with resource caps, capture the result,
and tear the sandbox down. A local variant runs the same code on the user's
machine via `--compute local` (explicit opt-in; e2b stays the default).

**Security boundary (permanent):** this runs the agent's OWN generated code only,
never the user's source. User-provided source is read as text by the Reconstructor
and never executed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from iterate.schemas.experiment import Candidate, ExperimentResult
    from iterate.targets.base import BenchmarkTarget


class SandboxExecutor:
    """Runs generated training code in an e2b sandbox. Implemented at v0.2 Day 2."""

    def execute(
        self, target: BenchmarkTarget, candidate: Candidate | None = None
    ) -> ExperimentResult:
        raise NotImplementedError(
            "SandboxExecutor lands in v0.2 (Day 2): boots an e2b sandbox, runs the "
            "agent's generated training code under a timeout, scores through our eval. "
            "Use LocalExecutor for v0.1 spec-candidates."
        )


__all__ = ["SandboxExecutor"]
