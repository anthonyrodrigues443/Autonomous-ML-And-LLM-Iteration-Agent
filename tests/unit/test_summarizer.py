"""Tests for the Summarizer — distills one finished experiment into a digest.

Driven by a deterministic fake LLM; no real backend. Proves the deterministic
skeleton (techniques + score + validation trail) is always filled, the LLM's
insight fields are merged in, and any LLM failure degrades to the skeleton rather
than raising (a digest must never cost the run).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from iterate.core.summarizer import Summarizer
from iterate.schemas.experiment import Candidate, Experiment, ExperimentResult, Metrics
from iterate.schemas.llm import ChatResponse, ToolCall

if TYPE_CHECKING:
    from iterate.schemas.llm import Message

_FIT_CODE = (
    "from sklearn.preprocessing import OneHotEncoder\n"
    "from sklearn.ensemble import HistGradientBoostingClassifier\n"
    "enc = OneHotEncoder(); model = HistGradientBoostingClassifier()\n"
)


class _FakeLLM:
    def __init__(self, responses: list[ChatResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[list[Message]] = []

    @property
    def model(self) -> str:
        return "fake"

    def chat(self, messages, *, tools=None, temperature=None, max_tokens=None) -> ChatResponse:  # type: ignore[no-untyped-def]
        self.calls.append(list(messages))
        return self._responses.pop(0)


class _RaisingLLM(_FakeLLM):
    def chat(self, *a: Any, **kw: Any) -> ChatResponse:
        raise RuntimeError("backend down")


def _digest_call(**fields: Any) -> ChatResponse:
    return ChatResponse(
        model="fake", tool_calls=[ToolCall(id="d", name="summarize_experiment", arguments=fields)]
    )


def _experiment(*, score: float | None = 0.61) -> Experiment:
    cells = [
        {"code": "# preamble", "stdout": "loaded", "error": None, "source": "preamble", "outputs": []},
        {"code": "print(X_train.nunique())", "stdout": "PaymentMethod 4", "error": None,
         "source": "agent", "outputs": []},
        {"code": _FIT_CODE, "stdout": "Validation f1: 0.55\nValidation f1: 0.61",
         "error": None, "source": "agent", "outputs": []},
    ]
    result = (
        ExperimentResult(
            experiment_id="e",
            metrics=Metrics(values={"f1": score}, primary="f1", direction="maximize", n_samples=100),
        )
        if score is not None
        else ExperimentResult(experiment_id="e", error="code-gen contract: no predictions")
    )
    return Experiment(
        candidate=Candidate(
            description="Target encoding attempt",
            changes={"code": _FIT_CODE, "cells": cells},
            rationale="r",
        ),
        target="t", hypothesis="h",
        status="completed" if score is not None else "failed",
        result=result,
    )


def test_digest_merges_deterministic_skeleton_with_llm_insight() -> None:
    fake = _FakeLLM([
        _digest_call(
            data_insights=["PaymentMethod cardinality 4"],
            what_helped=["target encoding: 0.55 -> 0.61"],
            what_hurt=[],
            takeaway="Push target encoding to the other high-cardinality columns.",
        )
    ])
    digest = Summarizer(fake, metric="f1").summarize(_experiment(score=0.61))
    # deterministic skeleton (no LLM needed for these)
    assert "OneHotEncoder" in digest.techniques
    assert any("HistGradientBoosting" in t for t in digest.techniques)
    assert digest.score == 0.61
    assert digest.val_trail == "0.55 -> 0.61"
    # LLM-filled insight
    assert digest.data_insights == ["PaymentMethod cardinality 4"]
    assert digest.what_helped == ["target encoding: 0.55 -> 0.61"]
    assert "target encoding" in digest.takeaway
    # the session was actually handed to the model
    sent = "\n".join(m.content or "" for m in fake.calls[0])
    assert "PaymentMethod" in sent
    assert "0.55" in sent


def test_failed_experiment_digest_has_no_score_but_still_summarizes() -> None:
    fake = _FakeLLM([_digest_call(what_hurt=["target encoding errored on NaN"], takeaway="Impute first.")])
    digest = Summarizer(fake, metric="f1").summarize(_experiment(score=None))
    assert digest.score is None
    assert digest.what_hurt == ["target encoding errored on NaN"]


def test_no_tool_call_falls_back_to_skeleton_not_raise() -> None:
    fake = _FakeLLM([ChatResponse(model="fake", content="here is a summary in prose"),
                     ChatResponse(model="fake", content="still prose")])
    digest = Summarizer(fake, metric="f1", max_retries=1).summarize(_experiment(score=0.61))
    assert digest.score == 0.61  # skeleton survives
    assert "OneHotEncoder" in digest.techniques
    assert digest.data_insights == []  # no insight, but no crash
    assert digest.takeaway == ""


def test_empty_takeaway_from_the_tool_is_synthesized_not_left_blank() -> None:
    # the model called the tool but returned an empty takeaway (observed once live);
    # the merge must synthesize a concrete one from the evidence, never leave it blank.
    fake = _FakeLLM([_digest_call(what_helped=["threshold tuning: 0.54 -> 0.62"], takeaway="")])
    digest = Summarizer(fake, metric="f1").summarize(_experiment(score=0.61))
    assert digest.takeaway != ""
    assert "threshold tuning" in digest.takeaway  # built from what_helped


def test_llm_exception_degrades_to_skeleton() -> None:
    digest = Summarizer(_RaisingLLM([]), metric="f1").summarize(_experiment(score=0.61))
    assert digest.score == 0.61  # a backend failure never costs the run a digest
    assert digest.takeaway == ""
