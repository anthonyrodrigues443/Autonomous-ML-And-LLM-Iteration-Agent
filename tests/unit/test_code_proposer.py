"""Tests for the CodeProposer — the LLM that writes train_and_predict.

Uses a deterministic fake LLM. The bridge test feeds a known-good function through
the fake and runs the resulting candidate through the real Day-3 contract
(LocalCodeRunner + score_predictions), proving a CodeProposer candidate is directly
runnable. The live qwen3 integration test is deferred to Day 5.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pytest

from iterate.adapters.compute.runner import LocalCodeRunner
from iterate.adapters.data.tabular import load_csv
from iterate.core import codegen
from iterate.core.code_proposer import CodeProposer, CodeProposerError
from iterate.llm.base import LLMClient
from iterate.schemas.experiment import Candidate, Experiment, ExperimentResult, Metrics
from iterate.schemas.llm import ChatResponse, Message, ToolCall

if TYPE_CHECKING:
    from pathlib import Path

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
        tool_calls=[ToolCall(id="c1", name="propose_code", arguments=args)],
    )


def _text(content: str = "Here is my idea...") -> ChatResponse:
    return ChatResponse(model="fake-model", content=content)


def _baseline(score: float = 0.70) -> ExperimentResult:
    return ExperimentResult(
        experiment_id="baseline",
        metrics=Metrics(values={"f1": score}, primary="f1", direction="maximize", n_samples=100),
    )


_GOOD_FN = """
def train_and_predict(X_train, y_train, X_holdout):
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    Xtr = pd.get_dummies(X_train)
    Xho = pd.get_dummies(X_holdout).reindex(columns=Xtr.columns, fill_value=0)
    model = LogisticRegression(max_iter=1000).fit(Xtr, y_train)
    return model.predict(Xho)
"""


def _classification_csv(tmp_path: Path) -> Path:
    n = 120
    frame = pd.DataFrame(
        {
            "num": [i % 10 for i in range(n)],
            "cat": (["a", "b", "c"] * (n // 3 + 1))[:n],
            "churn": [1 if (i % 10) >= 6 else 0 for i in range(n)],
        }
    )
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "clf.csv"
    frame.to_csv(path, index=False)
    return path


def test_fake_satisfies_the_llm_protocol() -> None:
    assert isinstance(_FakeLLM([]), LLMClient)


def test_builds_code_candidate_from_tool_call() -> None:
    fake = _FakeLLM(
        [
            _tool_call(
                {
                    "code": _GOOD_FN,
                    "description": "logreg on one-hot encoded features",
                    "rationale": "linear baseline before trying boosting",
                    "expected_metric_delta": 0.04,
                }
            )
        ]
    )
    candidate = CodeProposer(fake).propose(data_summary="120 rows", baseline=_baseline())
    assert candidate.source == "proposer"
    assert codegen.is_code_candidate(candidate.changes)
    assert candidate.changes == {"code": _GOOD_FN.strip()}
    assert candidate.expected_improvement == 0.04
    assert candidate.description.startswith("logreg")


def test_candidate_output_runs_through_the_contract(tmp_path: Path) -> None:
    fake = _FakeLLM([_tool_call({"code": _GOOD_FN, "description": "logreg", "rationale": "r"})])
    candidate = CodeProposer(fake).propose(data_summary="d", baseline=_baseline())

    ds = load_csv(_classification_csv(tmp_path), target="churn")
    script = codegen.assemble_script(candidate.changes["code"])
    run = LocalCodeRunner().run(
        script,
        inputs=codegen.build_inputs(ds),
        outputs=[codegen.PREDICTIONS_CSV],
        timeout=60,
    )
    assert run.succeeded, run.stderr
    result = codegen.score_predictions(
        ds, run.outputs.get(codegen.PREDICTIONS_CSV), metric="f1", experiment_id="e1"
    )
    assert result.succeeded
    assert result.metrics is not None
    assert result.metrics.n_samples == ds.n_test


def test_non_parsing_code_retries_then_raises() -> None:
    bad = "def train_and_predict(X_train, y_train, X_holdout)\n    return []"  # missing colon
    fake = _FakeLLM([_tool_call({"code": bad, "description": "d", "rationale": "r"})] * 2)
    with pytest.raises(CodeProposerError, match="did not parse"):
        CodeProposer(fake, max_retries=1).propose(data_summary="d", baseline=_baseline())
    assert len(fake.calls) == 2


def test_missing_train_and_predict_retries_then_raises() -> None:
    wrong_name = "def fit_model(X_train, y_train, X_holdout):\n    return []"
    fake = _FakeLLM([_tool_call({"code": wrong_name, "description": "d", "rationale": "r"})] * 2)
    with pytest.raises(CodeProposerError, match="train_and_predict"):
        CodeProposer(fake, max_retries=1).propose(data_summary="d", baseline=_baseline())


def test_no_tool_call_retries_then_raises() -> None:
    fake = _FakeLLM([_text(), _text()])
    with pytest.raises(CodeProposerError, match="no usable code candidate"):
        CodeProposer(fake, max_retries=1).propose(data_summary="d", baseline=_baseline())
    assert len(fake.calls) == 2


def test_recovers_after_one_bad_attempt() -> None:
    fake = _FakeLLM(
        [
            _text(),  # first reply: no tool call
            _tool_call({"code": _GOOD_FN, "description": "logreg", "rationale": "r"}),
        ]
    )
    candidate = CodeProposer(fake, max_retries=2).propose(data_summary="d", baseline=_baseline())
    assert codegen.is_code_candidate(candidate.changes)
    assert len(fake.calls) == 2


def test_prompt_carries_data_summary_and_metric() -> None:
    fake = _FakeLLM([_tool_call({"code": _GOOD_FN, "description": "d", "rationale": "r"})])
    CodeProposer(fake).propose(data_summary="DATA_BRIEF_MARKER", baseline=_baseline())
    sent = "\n".join(m.content or "" for m in fake.calls[0])
    assert "DATA_BRIEF_MARKER" in sent
    assert "f1" in sent


def test_recent_run_output_and_errors_are_fed_back() -> None:
    # A succeeded prior attempt that printed diagnostics, and a failed one.
    printed = Experiment(
        candidate=Candidate(description="logreg with EDA", changes={"code": "x"}, rationale="r"),
        target="t",
        hypothesis="h",
        status="completed",
        result=ExperimentResult(
            experiment_id="p1",
            metrics=Metrics(values={"f1": 0.7}, primary="f1", direction="maximize"),
            logs="class balance: 0.4/0.6\nmissing: none",
        ),
    )
    crashed = Experiment(
        candidate=Candidate(description="xgboost attempt", changes={"code": "y"}, rationale="r"),
        target="t",
        hypothesis="h",
        status="failed",
        result=ExperimentResult(
            experiment_id="p2", error="code script failed:\nTraceback ...\nKeyError: 'age'"
        ),
    )
    fake = _FakeLLM([_tool_call({"code": _GOOD_FN, "description": "d", "rationale": "r"})])
    CodeProposer(fake).propose(data_summary="d", baseline=_baseline(), history=[printed, crashed])
    sent = "\n".join(m.content or "" for m in fake.calls[0])
    assert "class balance: 0.4/0.6" in sent  # stdout fed back so it can learn the data
    assert "KeyError: 'age'" in sent  # traceback fed back so it can self-correct


def test_history_shows_components_used_so_preprocessing_is_visible() -> None:
    prior = Experiment(
        candidate=Candidate(
            description="HistGB attempt",
            changes={
                "code": (
                    "def train_and_predict(a, b, c):\n"
                    "    from sklearn.preprocessing import OneHotEncoder\n"
                    "    from sklearn.ensemble import HistGradientBoostingClassifier\n"
                    "    enc = OneHotEncoder()\n"
                    "    model = HistGradientBoostingClassifier()\n"
                    "    return model.fit(enc.fit_transform(a), b).predict(c)\n"
                )
            },
            rationale="r",
        ),
        target="t",
        hypothesis="h",
        status="completed",
        result=ExperimentResult(
            experiment_id="p1",
            metrics=Metrics(values={"f1": 0.57}, primary="f1", direction="maximize"),
        ),
    )
    fake = _FakeLLM([_tool_call({"code": _GOOD_FN, "description": "d", "rationale": "r"})])
    CodeProposer(fake).propose(data_summary="d", baseline=_baseline(), history=[prior])
    sent = "\n".join(m.content or "" for m in fake.calls[0])
    assert "used: OneHotEncoder, HistGradientBoostingClassifier" in sent


def test_environment_note_is_injected() -> None:
    from iterate.core.code_proposer import ENV_NOTE_AMBIENT

    fake = _FakeLLM([_tool_call({"code": _GOOD_FN, "description": "d", "rationale": "r"})])
    CodeProposer(fake, environment_note=ENV_NOTE_AMBIENT).propose(
        data_summary="d", baseline=_baseline()
    )
    sent = "\n".join(m.content or "" for m in fake.calls[0])
    assert "available in the run environment" in sent  # the ambient phrasing, not "we install"


def test_history_is_summarized_without_raw_code() -> None:
    prior = Experiment(
        candidate=Candidate(
            description="ridge regression attempt",
            changes={"code": "def train_and_predict(a, b, c):\n    return SECRET_BLOB"},
            rationale="r",
        ),
        target="t",
        hypothesis="h",
        status="completed",
        result=ExperimentResult(
            experiment_id="p1",
            metrics=Metrics(values={"f1": 0.66}, primary="f1", direction="maximize"),
        ),
    )
    fake = _FakeLLM([_tool_call({"code": _GOOD_FN, "description": "d", "rationale": "r"})])
    CodeProposer(fake).propose(data_summary="d", baseline=_baseline(), history=[prior])
    sent = "\n".join(m.content or "" for m in fake.calls[0])
    assert "ridge regression attempt" in sent  # the description is summarized in
    assert "f1=0.6600" in sent
    assert "SECRET_BLOB" not in sent  # the raw function source is NOT echoed back
