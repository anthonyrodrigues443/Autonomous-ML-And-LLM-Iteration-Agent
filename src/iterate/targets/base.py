"""The benchmark-target contract.

Every target the agent can iterate on — `ModelTarget` (tabular), `DLModelTarget`
(vision), `PromptTarget` (LLM prompts) — implements `BenchmarkTarget`, so the
orchestrator runs any of them the same way without knowing which kind it is.

A target only *measures*. `baseline()` evaluates the starting point and `run()`
evaluates a candidate change, both through the target's own eval so the two are
directly comparable. Deciding whether a candidate won, and when to stop, is the
loop's job — not the target's. Where execution happens (local / sandbox / cloud)
is also out of scope here; that belongs to the compute layer.

Implementations expose a ``name`` for logging and the audit trail.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from iterate.schemas.experiment import Candidate, ExperimentResult


@runtime_checkable
class BenchmarkTarget(Protocol):
    """What every target must provide."""

    name: str

    def baseline(self) -> ExperimentResult:
        """Measure the starting point through this target's own eval.

        Always re-measures — it never adopts an externally reported score — so the
        result is directly comparable to every `run()` result.
        """
        ...

    def run(self, candidate: Candidate) -> ExperimentResult:
        """Apply the candidate's changes, evaluate, and return the result.

        Uses the same eval as `baseline()`. Does not decide whether the candidate
        beat the baseline — the orchestrator compares the two.
        """
        ...
