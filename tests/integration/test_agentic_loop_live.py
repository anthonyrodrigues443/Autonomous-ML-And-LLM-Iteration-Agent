"""Live integration: the full agentic loop end-to-end on real churn data.

Runs the Orchestrator with a real qwen3:14b Proposer (via OllamaClient), the real
ModelTarget on the prepared Telco data, and in-memory storage, for a couple of
iterations. This is the v0.1 headline path. Marked integration; skips when Ollama
or the data is unavailable.

Note: this is slow (can run several minutes) — each proposal is a real LLM call,
and if the agent proposes LightGBM its macOS-ARM wheel fit is ~128s (see the
BUILD_LOG known-issue). That's why it's opt-in and excluded from the fast suite.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from iterate.adapters.compute.local import LocalExecutor
from iterate.adapters.data.tabular import load_csv
from iterate.core.memory import InMemoryMemory
from iterate.core.orchestrator import Orchestrator
from iterate.core.proposer import Proposer, summarize_dataset
from iterate.core.terminator import MaxIterations
from iterate.llm.ollama_client import OllamaClient
from iterate.targets.model import ModelTarget

_CLEAN = Path(__file__).resolve().parents[2] / "examples" / "churn_tabular" / "data.clean.csv"


@pytest.mark.integration
def test_agentic_loop_runs_end_to_end_on_churn() -> None:
    if not _CLEAN.exists():
        pytest.skip("data.clean.csv not present; run examples/churn_tabular/prepare.py")

    dataset = load_csv(_CLEAN, target="Churn")
    target = ModelTarget(dataset, metric="f1")
    orchestrator = Orchestrator(
        target,
        Proposer(OllamaClient()),
        LocalExecutor(),
        MaxIterations(2),
        InMemoryMemory(),
        data_summary=summarize_dataset(dataset),
        baseline_model="sklearn.ensemble.HistGradientBoostingClassifier",
    )

    try:
        result = orchestrator.run()
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPError) as exc:
        pytest.skip(f"Ollama backend unavailable: {exc}")

    # The loop produced a re-measured baseline and terminated on the iteration cap.
    assert result.baseline.metrics is not None
    assert result.baseline.metrics.primary == "f1"
    assert result.stopped_because == "max_iterations"
    # Each of the 2 iterations either produced an Experiment or hit a proposer
    # error (which yields no history row by design). So history is 0..2; any
    # recorded experiment must carry a result.
    assert 0 <= len(result.history) <= 2
    assert all(exp.result is not None for exp in result.history)
