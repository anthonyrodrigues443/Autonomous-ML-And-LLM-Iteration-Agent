"""Live integration: the Proposer against a real Ollama model (qwen3:14b).

Uses the native OllamaClient with thinking OFF — the OpenAI /v1 path can't disable
thinking and takes ~128s/call. Confirms the model calls `propose_candidate` with
our schema and yields a usable Candidate. Marked integration; skips when Ollama is
unavailable.
"""

from __future__ import annotations

import httpx
import pytest

from iterate.core.proposer import Proposer
from iterate.llm.ollama_client import OllamaClient
from iterate.schemas.experiment import Candidate, ExperimentResult, Metrics


@pytest.mark.integration
def test_live_proposer_emits_a_candidate() -> None:
    baseline = ExperimentResult(
        experiment_id="baseline",
        metrics=Metrics(values={"f1": 0.75}, primary="f1", direction="maximize", n_samples=200),
    )
    proposer = Proposer(OllamaClient(), max_retries=2)
    try:
        candidate = proposer.propose(
            data_summary=(
                "Rows: 800 train / 200 test (sealed holdout). Target column: 'churn'. "
                "Features (12): 10 numeric, 2 categorical."
            ),
            baseline=baseline,
            current_model="sklearn.ensemble.HistGradientBoostingClassifier",
        )
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPError) as exc:
        pytest.skip(f"Ollama backend unavailable: {exc}")

    assert isinstance(candidate, Candidate)
    assert isinstance(candidate.changes.get("model"), str)
    assert candidate.source == "proposer"
    assert candidate.rationale
