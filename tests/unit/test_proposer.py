"""Tests for the Proposer — turns an LLM reply into the next Candidate.

Uses a deterministic fake LLM client (no network); a live qwen3:14b test lives in
the integration suite.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pytest

from iterate.adapters.data.tabular import load_csv
from iterate.core.proposer import Proposer, ProposerError, summarize_dataset
from iterate.llm.base import LLMClient
from iterate.schemas.experiment import Candidate, Experiment, ExperimentResult, Metrics
from iterate.schemas.llm import ChatResponse, Message, ToolCall

if TYPE_CHECKING:
    from pathlib import Path

    from iterate.schemas.llm import ToolSpec


class _FakeLLM:
    """Returns preset ChatResponses in order; records what it was called with."""

    def __init__(self, responses: list[ChatResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[list[Message]] = []

    @property
    def model(self) -> str:
        return "fake-model"

    def chat(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        self.calls.append(messages)
        return self._responses.pop(0)


def _baseline(score: float = 0.70) -> ExperimentResult:
    return ExperimentResult(
        experiment_id="baseline",
        metrics=Metrics(values={"f1": score}, primary="f1", direction="maximize", n_samples=100),
    )


def _tool_call(args: dict) -> ChatResponse:
    return ChatResponse(
        model="fake-model",
        tool_calls=[ToolCall(id="c1", name="propose_candidate", arguments=args)],
    )


def _text(content: str = "Here is what I think...") -> ChatResponse:
    return ChatResponse(model="fake-model", content=content)


_DEFAULT_MODEL = "sklearn.ensemble.HistGradientBoostingClassifier"


def test_fake_satisfies_the_llm_protocol() -> None:
    assert isinstance(_FakeLLM([]), LLMClient)


def test_builds_candidate_from_tool_call() -> None:
    fake = _FakeLLM(
        [
            _tool_call(
                {
                    "model": "xgboost.XGBClassifier",
                    "params": {"max_depth": 4},
                    "description": "XGBoost shallow trees",
                    "rationale": "curb overfitting",
                    "expected_metric_delta": 0.03,
                }
            )
        ]
    )
    candidate = Proposer(fake).propose(
        data_summary="120 rows", baseline=_baseline(), current_model=_DEFAULT_MODEL
    )
    assert candidate.changes == {"model": "xgboost.XGBClassifier", "params": {"max_depth": 4}}
    assert candidate.source == "proposer"
    assert candidate.description == "XGBoost shallow trees"
    assert candidate.rationale == "curb overfitting"
    assert candidate.expected_improvement == 0.03


def test_model_only_omits_params_key() -> None:
    fake = _FakeLLM(
        [
            _tool_call(
                {
                    "model": "sklearn.ensemble.RandomForestClassifier",
                    "description": "RF",
                    "rationale": "bagging",
                }
            )
        ]
    )
    candidate = Proposer(fake).propose(
        data_summary="d", baseline=_baseline(), current_model=_DEFAULT_MODEL
    )
    assert candidate.changes == {"model": "sklearn.ensemble.RandomForestClassifier"}


def test_missing_model_raises() -> None:
    fake = _FakeLLM(
        [_tool_call({"params": {"max_depth": 3}, "description": "d", "rationale": "r"})]
    )
    with pytest.raises(ProposerError, match="missing a 'model'"):
        Proposer(fake, max_retries=0).propose(
            data_summary="d", baseline=_baseline(), current_model=_DEFAULT_MODEL
        )


def test_no_tool_call_retries_then_raises() -> None:
    fake = _FakeLLM([_text(), _text()])
    with pytest.raises(ProposerError, match="no candidate"):
        Proposer(fake, max_retries=1).propose(
            data_summary="d", baseline=_baseline(), current_model=_DEFAULT_MODEL
        )
    assert len(fake.calls) == 2


def test_succeeds_on_retry_after_text_reply() -> None:
    fake = _FakeLLM(
        [
            _text(),
            _tool_call({"model": "xgboost.XGBClassifier", "description": "x", "rationale": "r"}),
        ]
    )
    candidate = Proposer(fake, max_retries=1).propose(
        data_summary="d", baseline=_baseline(), current_model=_DEFAULT_MODEL
    )
    assert candidate.changes["model"] == "xgboost.XGBClassifier"
    assert len(fake.calls) == 2


def test_baseline_without_metrics_raises() -> None:
    fake = _FakeLLM([])
    bad = ExperimentResult(experiment_id="x", error="boom")
    with pytest.raises(ProposerError, match="no metrics"):
        Proposer(fake).propose(data_summary="d", baseline=bad, current_model=_DEFAULT_MODEL)


def test_prompt_carries_data_baseline_current_model_and_history() -> None:
    history = [
        Experiment(
            candidate=Candidate(
                description="prev",
                changes={"model": "sklearn.linear_model.LogisticRegression"},
                rationale="linear baseline",
            ),
            target="churn-model",
            hypothesis="linear may suffice",
            result=ExperimentResult(
                experiment_id="e1",
                metrics=Metrics(values={"f1": 0.66}, primary="f1", direction="maximize"),
            ),
        )
    ]
    fake = _FakeLLM(
        [_tool_call({"model": "xgboost.XGBClassifier", "description": "x", "rationale": "r"})]
    )
    Proposer(fake).propose(
        data_summary="DATA_BRIEF_MARKER",
        baseline=_baseline(0.70),
        current_model=_DEFAULT_MODEL,
        history=history,
    )
    text = "\n".join(m.content or "" for m in fake.calls[0])
    assert "DATA_BRIEF_MARKER" in text
    assert "0.7000" in text
    assert "HistGradientBoostingClassifier" in text
    assert "LogisticRegression" in text


def test_summarize_dataset_describes_shape_and_target(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "age": range(40),
            "plan": (["a", "b"] * 20),
            "churn": [i % 2 for i in range(40)],
        }
    )
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "d.csv"
    frame.to_csv(path, index=False)
    summary = summarize_dataset(load_csv(path, target="churn"))
    assert "churn" in summary
    assert "train" in summary
    assert "numeric" in summary
    assert "categorical" in summary
