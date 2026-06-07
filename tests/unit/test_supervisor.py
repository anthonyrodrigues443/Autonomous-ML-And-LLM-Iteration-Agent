"""Tests for the Supervisor — briefs the coder from run history (fake LLM)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from iterate.core.supervisor import Supervisor, SupervisorError
from iterate.schemas.experiment import Candidate, Experiment, ExperimentResult, Metrics
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
        self.calls.append(list(messages))
        return self._responses.pop(0)


def _plan(stop: bool, title: str, brief: str) -> ChatResponse:
    return ChatResponse(
        model="fake-model",
        tool_calls=[ToolCall(id="p", name="plan_next", arguments={"stop": stop, "title": title, "brief": brief})],
    )


def _text() -> ChatResponse:
    return ChatResponse(model="fake-model", content="let me think")


def _baseline() -> ExperimentResult:
    return ExperimentResult(
        experiment_id="b",
        metrics=Metrics(values={"f1": 0.57}, primary="f1", direction="maximize", n_samples=100),
    )


def test_decide_returns_a_brief() -> None:
    fake = _FakeLLM([_plan(False, "target-encode", "so far: only one-hot tried. Try target encoding.")])
    d = Supervisor(fake, metric="f1").decide(data_summary="120 rows", baseline=_baseline(), history=[])
    assert not d.stop
    assert d.title == "target-encode"
    assert "target encoding" in d.brief


def test_decide_can_stop() -> None:
    fake = _FakeLLM([_plan(True, "", "")])
    d = Supervisor(fake, metric="f1").decide(data_summary="d", baseline=_baseline(), history=[])
    assert d.stop


def test_no_tool_call_retries_then_raises() -> None:
    fake = _FakeLLM([_text(), _text()])
    with pytest.raises(SupervisorError, match="no plan"):
        Supervisor(fake, metric="f1", max_retries=1).decide(
            data_summary="d", baseline=_baseline(), history=[]
        )
    assert len(fake.calls) == 2


def test_history_shows_components_to_the_supervisor() -> None:
    prior = Experiment(
        candidate=Candidate(
            description="logreg one-hot",
            changes={"code": "from sklearn.preprocessing import OneHotEncoder\nOneHotEncoder()"},
            rationale="r",
        ),
        target="t",
        hypothesis="h",
        status="completed",
        result=ExperimentResult(
            experiment_id="e",
            metrics=Metrics(values={"f1": 0.55}, primary="f1", direction="maximize"),
        ),
    )
    fake = _FakeLLM([_plan(False, "next", "try interactions")])
    Supervisor(fake, metric="f1").decide(data_summary="d", baseline=_baseline(), history=[prior])
    sent = "\n".join(m.content or "" for m in fake.calls[0])
    assert "used: OneHotEncoder" in sent
    assert "f1=0.5500" in sent


def test_history_shows_within_session_validation_trail() -> None:
    # a session that printed several validation scores as it iterated — the supervisor
    # must see the trail (incl. the attempts that lost), not just the final score.
    prior = Experiment(
        candidate=Candidate(
            description="iterated session",
            changes={
                "code": "from sklearn.linear_model import LogisticRegression\nLogisticRegression()",
                "cells": [
                    {"source": "agent", "stdout": "Validation F1 score: 0.5800", "error": None},
                    {"source": "agent", "stdout": "validation f1: 0.6100", "error": None},
                    {"source": "agent", "stdout": "Validation F1 score: 0.5900", "error": None},
                ],
            },
            rationale="r",
        ),
        target="t",
        hypothesis="h",
        status="completed",
        result=ExperimentResult(
            experiment_id="e",
            metrics=Metrics(values={"f1": 0.61}, primary="f1", direction="maximize"),
        ),
    )
    fake = _FakeLLM([_plan(False, "next", "refine")])
    Supervisor(fake, metric="f1").decide(data_summary="d", baseline=_baseline(), history=[prior])
    sent = "\n".join(m.content or "" for m in fake.calls[0])
    assert "val tries: 0.5800 -> 0.6100 -> 0.5900" in sent
