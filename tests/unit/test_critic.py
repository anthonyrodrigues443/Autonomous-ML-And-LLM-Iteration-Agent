"""Tests for the Critic specialist.

The contract has an asymmetry that most of these tests exist to pin: a proven leak
VETOES banking, a suspected mirage only annotates. A leak is visible in code and
checkable, so acting on it is safe. Whether a gain is "real" is probabilistic, and
the sealed holdout is already the ruler — a 12B must not get to overrule it.

No network, no real model: the LLM is a scripted fake.
"""

from __future__ import annotations

from typing import Any

from iterate.core.critic import FLAGGED, REJECTED, Critic, Verdict, stamp, was_rejected
from iterate.schemas.experiment import Candidate, Experiment, ExperimentResult, Metrics
from iterate.schemas.llm import ChatResponse, ToolCall

_CLEAN_CODE = """
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
scaler = StandardScaler().fit(X_train[num_cols])
Xt = scaler.transform(X_train[num_cols])
Xh = scaler.transform(X_holdout[num_cols])
model = HistGradientBoostingClassifier().fit(Xt, y_train)
predictions = model.predict(Xh)
pd.Series(predictions).to_csv('predictions.csv', index=False, header=False)
"""

_LEAKY_CODE = """
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
combined = pd.concat([X_train[num_cols], X_holdout[num_cols]])
scaler = StandardScaler().fit(combined)
Xt = scaler.transform(X_train[num_cols])
Xh = scaler.transform(X_holdout[num_cols])
model = HistGradientBoostingClassifier().fit(Xt, y_train)
predictions = model.predict(Xh)
pd.Series(predictions).to_csv('predictions.csv', index=False, header=False)
"""


class _FakeLLM:
    def __init__(self, replies: list[Any]) -> None:
        self._replies = list(replies)
        self.calls = 0
        self.seen: list[str] = []

    @property
    def model(self) -> str:
        return "fake"

    def chat(self, messages, *, tools=None, temperature=None, max_tokens=None) -> ChatResponse:  # type: ignore[no-untyped-def]
        self.calls += 1
        self.seen.append(str(messages[-1].content))
        reply = self._replies.pop(0) if self._replies else "prose"
        if isinstance(reply, str):
            return ChatResponse(model="fake", content=reply, tool_calls=[])
        return ChatResponse(
            model="fake",
            content="",
            tool_calls=[ToolCall(id="c1", name="review_experiment", arguments=reply)],
        )


class _RaisingLLM(_FakeLLM):
    def chat(self, *a: Any, **kw: Any) -> ChatResponse:
        raise RuntimeError("backend down")


def _experiment(
    code: str = _CLEAN_CODE, *, score: float | None = 0.68, cells: list[dict[str, Any]] | None = None
) -> Experiment:
    result = (
        ExperimentResult(
            experiment_id="e",
            metrics=Metrics(
                values={"f1": score}, primary="f1", direction="maximize", n_samples=100
            ),
        )
        if score is not None
        else ExperimentResult(experiment_id="e", error="no predictions")
    )
    return Experiment(
        candidate=Candidate(
            description="d",
            changes={"code": code, "cells": cells or []},
            rationale="r",
        ),
        target="t",
        hypothesis="h",
        status="completed" if score is not None else "failed",
        result=result,
    )


def _critic(llm: _FakeLLM) -> Critic:
    return Critic(llm, metric="f1", direction="maximize")


# ─── the asymmetry: leak vetoes, mirage does not ─────────────────────────────


def test_a_leak_rejects_the_experiment() -> None:
    v = _critic(_FakeLLM([{"leak": True, "mirage": False, "reason": "scaler fit on concat"}])).review(
        _experiment(_LEAKY_CODE)
    )
    assert v.leak
    assert v.rejected
    assert "scaler fit on concat" in v.render()


def test_a_mirage_flags_but_does_not_reject() -> None:
    """The sealed holdout is the ruler. A model may raise a hand about a gain; it
    may not overturn the measurement."""
    v = _critic(_FakeLLM([{"leak": False, "mirage": True, "reason": "val 0.55 vs holdout 0.68"}])).review(
        _experiment()
    )
    assert v.mirage
    assert not v.rejected
    assert not v.clean


def test_a_clean_review_is_neither() -> None:
    v = _critic(_FakeLLM([{"leak": False, "mirage": False, "reason": ""}])).review(_experiment())
    assert v.clean
    assert not v.rejected
    assert v.render() == ""


# ─── stamping uses the existing changes convention ───────────────────────────


def test_a_leak_stamp_marks_the_candidate_rejected() -> None:
    exp = _experiment(_LEAKY_CODE)
    stamp(exp, Verdict(leak=True, reason="fit on holdout"))
    assert was_rejected(exp)
    assert REJECTED in exp.candidate.changes


def test_a_mirage_stamp_annotates_without_rejecting() -> None:
    exp = _experiment()
    stamp(exp, Verdict(mirage=True, reason="val far below holdout"))
    assert not was_rejected(exp)
    assert FLAGGED in exp.candidate.changes


def test_a_clean_verdict_stamps_nothing() -> None:
    exp = _experiment()
    stamp(exp, Verdict())
    assert REJECTED not in exp.candidate.changes
    assert FLAGGED not in exp.candidate.changes


# ─── degradation: a review must never cost a real result ─────────────────────


def test_a_backend_failure_accepts_the_experiment() -> None:
    v = Critic(_RaisingLLM([]), metric="f1", direction="maximize").review(_experiment())
    assert v.clean
    assert not v.rejected


def test_a_model_that_never_calls_the_tool_accepts() -> None:
    llm = _FakeLLM(["I think this looks fine", "still prose"])
    assert _critic(llm).review(_experiment()).clean
    assert llm.calls == 2  # the retry nudge was spent


def test_the_retry_nudge_recovers_a_missing_tool_call() -> None:
    llm = _FakeLLM(["prose", {"leak": True, "mirage": False, "reason": "fit on holdout"}])
    assert _critic(llm).review(_experiment(_LEAKY_CODE)).leak


def test_string_booleans_are_coerced() -> None:
    """Weak models emit "false" as a string, and bool("false") is True — the
    supervisor hit this live on groq, and the same models drive the critic."""
    v = _critic(_FakeLLM([{"leak": "false", "mirage": "true", "reason": "r"}])).review(_experiment())
    assert not v.leak
    assert v.mirage


def test_a_failed_experiment_is_not_reviewed() -> None:
    """There is no score to disbelieve, so the call is never made."""
    llm = _FakeLLM([{"leak": True, "mirage": False, "reason": "x"}])
    assert _critic(llm).review(_experiment(score=None)).clean
    assert llm.calls == 0


def test_a_spec_candidate_without_code_is_not_reviewed() -> None:
    llm = _FakeLLM([{"leak": True, "mirage": False, "reason": "x"}])
    exp = Experiment(
        candidate=Candidate(description="d", changes={"model": "X"}, rationale="r"),
        target="t",
        hypothesis="h",
        status="completed",
        result=ExperimentResult(
            experiment_id="e",
            metrics=Metrics(values={"f1": 0.6}, primary="f1", direction="maximize", n_samples=10),
        ),
    )
    assert _critic(llm).review(exp).clean
    assert llm.calls == 0


# ─── what the critic is shown ────────────────────────────────────────────────


def test_the_review_sees_the_validation_trail_and_the_bar() -> None:
    """The val-vs-holdout gap is the mirage evidence, so it has to be in front of
    the model rather than left for it to infer."""
    cells = [{"code": "x", "stdout": "val f1 0.5500", "stderr": "", "error": None, "source": "agent"}]
    llm = _FakeLLM([{"leak": False, "mirage": False, "reason": ""}])
    _critic(llm).review(_experiment(cells=cells), previous_best=0.61)
    prompt = llm.seen[0]
    assert "0.5500" in prompt
    assert "0.6100" in prompt
    assert "0.6800" in prompt


def test_generated_code_containing_braces_does_not_break_the_prompt() -> None:
    """The submitted code is inserted into a template; a dict literal in it would
    blow up str.format, which is why the template is filled by replacement."""
    code = _CLEAN_CODE + "\nparams = {'max_depth': 5, 'lr': 0.1}\n"
    llm = _FakeLLM([{"leak": False, "mirage": False, "reason": ""}])
    assert _critic(llm).review(_experiment(code)).clean
    assert llm.calls == 1


# ─── precedence at the banking gate ──────────────────────────────────────────


def test_the_veto_only_ever_subtracts() -> None:
    """`_improves` is the deterministic ruler and stays the ruler: the critic can
    stop a winner banking, but nothing it says can make a loser bank."""
    from iterate.core.agent_loop import _improves

    baseline = ExperimentResult(
        experiment_id="b",
        metrics=Metrics(values={"f1": 0.60}, primary="f1", direction="maximize", n_samples=100),
    )
    better = _experiment(score=0.68)
    worse = _experiment(score=0.55)

    # the gate as the loop applies it
    def banks(exp: Experiment) -> bool:
        assert exp.result is not None
        return (
            exp.result.succeeded
            and _improves(exp.result, None, baseline, "maximize")
            and not was_rejected(exp)
        )

    assert banks(better)
    assert not banks(worse)

    stamp(better, Verdict(leak=True, reason="fit on holdout"))
    assert not banks(better), "a rejected winner must not bank"

    stamp(worse, Verdict())  # clean verdict on a losing experiment
    assert not banks(worse), "a clean verdict must not promote a loser"


def test_a_flagged_experiment_still_banks() -> None:
    """A mirage annotates. Only a proven leak takes the result away."""
    from iterate.core.agent_loop import _improves

    baseline = ExperimentResult(
        experiment_id="b",
        metrics=Metrics(values={"f1": 0.60}, primary="f1", direction="maximize", n_samples=100),
    )
    exp = _experiment(score=0.68)
    stamp(exp, Verdict(mirage=True, reason="val 0.55 vs holdout 0.68"))
    assert exp.result is not None
    assert _improves(exp.result, None, baseline, "maximize")
    assert not was_rejected(exp)


def test_a_rejected_experiment_stops_crediting_its_techniques() -> None:
    """Joins duplicate_submission and lever_unmeasured in the technique table's
    exclusion list, so a leaked score cannot teach the supervisor a false lesson —
    and it costs no extra prompt context to do it."""
    from iterate.core.supervisor import _technique_table
    from iterate.schemas.experiment import ExperimentDigest

    exp = _experiment(score=0.99)
    exp.digest = ExperimentDigest(techniques=["TargetEncoder"], score=0.99)
    assert "TargetEncoder" in _technique_table([exp], "f1")

    stamp(exp, Verdict(leak=True, reason="target encoded on full data"))
    assert "TargetEncoder" not in _technique_table([exp], "f1")


def test_a_rejected_score_is_annotated_in_the_supervisors_history() -> None:
    """Without this the scoreboard shows a leaked 0.81 as the run's high-water
    mark and the supervisor pushes that direction — the marker is correcting
    active misinformation, not adding optional colour."""
    from iterate.core.supervisor import _format_history

    exp = _experiment(_LEAKY_CODE, score=0.81)
    assert "REJECTED" not in _format_history([exp], "f1")[0]

    stamp(exp, Verdict(leak=True, reason="scaler fit on train+holdout"))
    line = _format_history([exp], "f1")[0]
    assert "REJECTED" in line
    assert "not a result" in line
    assert "0.8100" in line  # the number is still shown, just disowned


def test_a_flagged_score_is_annotated_more_gently() -> None:
    from iterate.core.supervisor import _format_history

    exp = _experiment(score=0.68)
    stamp(exp, Verdict(mirage=True, reason="val 0.55 vs holdout 0.68"))
    line = _format_history([exp], "f1")[0]
    assert "suspicious gain" in line
    assert "REJECTED" not in line
