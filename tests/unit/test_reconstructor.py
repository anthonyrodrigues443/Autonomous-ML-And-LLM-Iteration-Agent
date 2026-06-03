"""Tests for the Reconstructor — turns a source document into a baseline Candidate.

It now WRITES a train_and_predict function (the code path) rather than emitting a
spec, so it can reproduce the source faithfully with no allow-list. Uses a
deterministic fake LLM; the live integration test (real qwen3) is deferred.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from iterate.core import codegen
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


_CATBOOST_FN = """
def train_and_predict(X_train, y_train, X_holdout):
    from catboost import CatBoostClassifier
    m = CatBoostClassifier(depth=6, iterations=500, verbose=0)
    m.fit(X_train, y_train)
    return m.predict(X_holdout)
"""


def _tool_call(args: dict) -> ChatResponse:
    return ChatResponse(
        model="fake-model",
        tool_calls=[ToolCall(id="c1", name="reconstruct_baseline", arguments=args)],
    )


def _text(content: str = "I think...") -> ChatResponse:
    return ChatResponse(model="fake-model", content=content)


def test_fake_satisfies_the_llm_protocol() -> None:
    assert isinstance(_FakeLLM([]), LLMClient)


def test_builds_code_candidate_faithful_to_the_source() -> None:
    fake = _FakeLLM(
        [
            _tool_call(
                {
                    "code": _CATBOOST_FN,
                    "description": "CatBoost mirroring the notebook (depth=6, 500 iters)",
                    "rationale": "Source used CatBoost; reproduced exactly, no approximation.",
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
    assert codegen.is_code_candidate(candidate.changes)
    assert "CatBoost" in candidate.changes["code"]  # the real library, not an equivalent
    assert candidate.source == "human"
    assert candidate.description.startswith("CatBoost")


def test_missing_code_retries_then_raises() -> None:
    fake = _FakeLLM([_tool_call({"description": "d", "rationale": "r"})] * 2)
    with pytest.raises(ReconstructorError, match="no reconstruction"):
        Reconstructor(fake, max_retries=1).reconstruct(
            data_summary="d", source_text="s", metric="f1", direction="maximize"
        )


def test_unparseable_code_retries_then_raises() -> None:
    bad = _tool_call({"code": "def train_and_predict(:\n", "description": "d", "rationale": "r"})
    fake = _FakeLLM([bad, bad])
    with pytest.raises(ReconstructorError):
        Reconstructor(fake, max_retries=1).reconstruct(
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
    fake = _FakeLLM([_tool_call({"code": _CATBOOST_FN, "description": "x", "rationale": "r"})])
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
