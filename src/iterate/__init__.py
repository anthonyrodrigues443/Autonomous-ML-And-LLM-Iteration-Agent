"""iterate — Autonomous research-aware iteration agent for ML models and LLM prompts.

The framework runs an autonomous loop that improves a baseline (ML model or LLM
prompt) by researching literature, proposing experiments, executing them in a
sandbox, scoring results, and logging every decision with a reasoning trail —
with explicit termination criteria and human-approval safety gates.

Public API (re-exported here when implementations land):

    from iterate import BenchmarkTarget, ModelTarget, PromptTarget
    from iterate import LLMClient, Memory, Orchestrator

See README.md for the full architecture. See PRD.md (private) for scope.
"""

__version__ = "0.2.1"
__author__ = "Anthony Rodrigues"
__all__: list[str] = []  # populated as public API surfaces stabilize
