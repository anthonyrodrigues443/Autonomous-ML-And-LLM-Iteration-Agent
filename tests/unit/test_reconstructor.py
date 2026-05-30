"""Tests for the Reconstructor — turns a source document into a baseline Candidate.

Uses a deterministic fake LLM; the live integration test (real qwen3) is deferred.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from iterate.core.reconstructor import Reconstructor, ReconstructorError
from iterate.llm.base import LLMClient
from iterate.schemas.llm import ChatResponse, Message, ToolCall

if TYPE_CHECKING:
    from iterate.schemas.llm import ToolSpec


class _FakeLLM:
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


def _tool_call(args: dict) -> ChatResponse:
    return ChatResponse(
        model="fake-model",
        tool_calls=[ToolCall(id="c1", name="reconstruct_baseline", arguments=args)],
    )


def _text(content: str = "I think...") -> ChatResponse:
    return ChatResponse(model="fake-model", content=content)


def test_fake_satisfies_the_llm_protocol() -> None:
    assert isinstance(_FakeLLM([]), LLMClient)


def test_builds_candidate_from_tool_call() -> None:
    fake = _FakeLLM(
        [
            _tool_call(
                {
                    "model": "xgboost.XGBClassifier",
                    "params": {"max_depth": 6, "n_estimators": 500},
                    "description": "XGBoost mirroring the notebook's CatBoost",
                    "rationale": "Source used CatBoost; XGBoost is the closest allow-listed equivalent.",
                }
            )
        ]
    )
    candidate = Reconstructor(fake).reconstruct(
        data_summary="120 rows",
        source_text="# notebook content describing CatBoost...",
        metric="f1",
        direction="maximize",
    )
    assert candidate.changes == {
        "model": "xgboost.XGBClassifier",
        "params": {"max_depth": 6, "n_estimators": 500},
    }
    assert candidate.source == "human"
    assert "CatBoost" in candidate.rationale
    assert candidate.description.startswith("XGBoost")


def test_model_only_omits_params_key() -> None:
    fake = _FakeLLM(
        [
            _tool_call(
                {
                    "model": "sklearn.ensemble.RandomForestClassifier",
                    "description": "RandomForest",
                    "rationale": "source used a vanilla RF",
                }
            )
        ]
    )
    candidate = Reconstructor(fake).reconstruct(
        data_summary="d", source_text="s", metric="f1", direction="maximize"
    )
    assert candidate.changes == {"model": "sklearn.ensemble.RandomForestClassifier"}


def test_missing_model_raises() -> None:
    fake = _FakeLLM([_tool_call({"params": {"x": 1}, "description": "d", "rationale": "r"})])
    with pytest.raises(ReconstructorError, match="missing a 'model'"):
        Reconstructor(fake, max_retries=0).reconstruct(
            data_summary="d", source_text="s", metric="f1", direction="maximize"
        )


def test_no_tool_call_retries_then_raises() -> None:
    fake = _FakeLLM([_text(), _text()])
    with pytest.raises(ReconstructorError, match="no reconstruction"):
        Reconstructor(fake, max_retries=1).reconstruct(
            data_summary="d", source_text="s", metric="f1", direction="maximize"
        )
    assert len(fake.calls) == 2


def test_prompt_carries_source_text_and_data_summary() -> None:
    fake = _FakeLLM(
        [
            _tool_call(
                {
                    "model": "xgboost.XGBClassifier",
                    "description": "x",
                    "rationale": "r",
                }
            )
        ]
    )
    Reconstructor(fake).reconstruct(
        data_summary="DATA_BRIEF_MARKER",
        source_text="SOURCE_TEXT_MARKER",
        metric="f1",
        direction="maximize",
    )
    sent = "\n".join(m.content or "" for m in fake.calls[0])
    assert "DATA_BRIEF_MARKER" in sent
    assert "SOURCE_TEXT_MARKER" in sent
    assert "f1" in sent
