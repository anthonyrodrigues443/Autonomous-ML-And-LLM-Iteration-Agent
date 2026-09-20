"""Tests for the coding agent — the cell-by-cell session.

End-to-end through a REAL LocalKernel + real ModelTarget scoring, driven by a
deterministic fake LLM that scripts cells. No real LLM, no e2b. Proves the loop
runs cells, persists state, scores the written predictions, captures the cells,
and recovers from a cell error mid-session.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

import pandas as pd
import pytest

from iterate.adapters.compute.deps import Plan, Route
from iterate.adapters.compute.kernel import Blocked, CellResult, LocalKernel
from iterate.adapters.data.tabular import load_csv
from iterate.core import codegen
from iterate.core.coder import CodingAgent
from iterate.prompts import PROMPTS
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
        self.calls.append(list(messages))
        return self._responses.pop(0)


def _run(code: str) -> ChatResponse:
    return ChatResponse(
        model="fake-model", tool_calls=[ToolCall(id="c", name="run_cell", arguments={"code": code})]
    )


def _finish() -> ChatResponse:
    return ChatResponse(
        model="fake-model", tool_calls=[ToolCall(id="f", name="finish", arguments={})]
    )


_FIT_AND_WRITE = """
import pandas as pd
from sklearn.linear_model import LogisticRegression
Xtr = pd.get_dummies(X_train)
Xho = pd.get_dummies(X_holdout).reindex(columns=Xtr.columns, fill_value=0)
preds = LogisticRegression(max_iter=1000).fit(Xtr, y_train).predict(Xho)
pd.Series(preds).to_csv('predictions.csv', index=False, header=False)
print('wrote', len(preds), 'predictions')
"""


def _dataset(tmp_path: Path):
    n = 120
    frame = pd.DataFrame(
        {
            "num": [i % 10 for i in range(n)],
            "cat": (["a", "b", "c"] * (n // 3 + 1))[:n],
            "churn": [1 if (i % 10) >= 6 else 0 for i in range(n)],
        }
    )
    path = tmp_path / "clf.csv"
    frame.to_csv(path, index=False)
    return load_csv(path, target="churn")


def test_session_runs_cells_scores_and_captures_transcript(tmp_path: Path) -> None:
    ds = _dataset(tmp_path)
    fake = _FakeLLM([_run("print(X_train.dtypes)"), _run(_FIT_AND_WRITE), _finish(), _finish()])
    agent = CodingAgent(fake, LocalKernel(), metric="f1", max_cells=8)
    out = agent.run(dataset=ds, brief="logreg on one-hot features", experiment_id="e1")

    assert out.result.succeeded, out.result.error
    assert out.result.metrics is not None
    assert out.result.metrics.primary == "f1"
    assert out.result.metrics.n_samples == ds.n_test
    # transcript: preamble + the two agent cells (EDA + fit), in order
    assert out.cells[0].source == "preamble"
    agent_cells = [c for c in out.cells if c.source == "agent"]
    assert len(agent_cells) == 2
    assert "wrote 24 predictions" in (out.result.logs or "")


def test_session_recovers_from_a_cell_error(tmp_path: Path) -> None:
    ds = _dataset(tmp_path)
    # cell 1 raises (undefined name); cell 2 fixes and writes predictions.
    fake = _FakeLLM([_run("preds = Undefined()"), _run(_FIT_AND_WRITE), _finish(), _finish()])
    out = CodingAgent(fake, LocalKernel(), metric="f1", max_cells=8).run(
        dataset=ds, brief="b", experiment_id="e2"
    )
    assert out.result.succeeded, out.result.error  # the kernel survived the error and finished
    errored = [c for c in out.cells if c.error]
    assert len(errored) == 1
    assert "NameError" in (errored[0].error or "")


class _FakeKernel:
    """A scripted kernel: returns preset CellResults; records installs; serves preds."""

    def __init__(
        self,
        results: list[CellResult],
        predictions: bytes | None = None,
        install_error: str = "",
    ) -> None:
        self._results = list(results)
        self._predictions = predictions
        self._install_error = install_error
        self.installed: list[str] = []

    def start(self, inputs: dict[str, bytes]) -> None:
        pass

    def run_cell(self, code: str, *, timeout: float) -> CellResult:
        return self._results.pop(0) if self._results else CellResult("ok", "")

    def install(self, packages: list[str]) -> str:
        self.installed += packages
        return self._install_error

    def namespace_summary(self) -> str:
        return ""

    def read_output(self, name: str) -> bytes | None:
        return self._predictions

    def blocked(self, error: str | None) -> Blocked | None:
        return None

    def close(self) -> None:
        pass


def test_auto_installs_a_missing_module_and_retries(tmp_path: Path) -> None:
    ds = _dataset(tmp_path)
    # run_cell calls: 1) preamble ok, 2) agent cell -> ModuleNotFoundError, 3) retry ok.
    kernel = _FakeKernel(
        [
            CellResult("loaded", ""),
            CellResult("", "", error="ModuleNotFoundError: No module named 'category_encoders'"),
            CellResult("worked after install", ""),
        ],
        predictions=b"0\n" * ds.n_test,  # so the verified finish accepts
    )
    fake = _FakeLLM([_run("import category_encoders"), _finish(), _finish()])
    out = CodingAgent(fake, kernel, metric="f1", max_cells=4).run(  # type: ignore[arg-type]
        dataset=ds, brief="b", experiment_id="e4"
    )
    assert kernel.installed == ["category_encoders"]  # mapped + installed on demand
    agent_cells = [c for c in out.cells if c.source == "agent"]
    assert agent_cells[0].error is None  # the retried (post-install) result was recorded


def test_failed_auto_install_is_surfaced_not_silently_retried(tmp_path: Path) -> None:
    ds = _dataset(tmp_path)
    # run_cell calls: 1) preamble ok, 2) agent cell -> ModuleNotFoundError. The pip
    # failure means NO transparent re-run; the agent must be told to pivot.
    kernel = _FakeKernel(
        [
            CellResult("loaded", ""),
            CellResult("", "", error="ModuleNotFoundError: No module named 'catboost'"),
        ],
        predictions=b"0\n" * ds.n_test,
        install_error="No module named pip",
    )
    fake = _FakeLLM([_run("import catboost"), _finish(), _finish()])
    out = CodingAgent(fake, kernel, metric="f1", max_cells=4).run(  # type: ignore[arg-type]
        dataset=ds, brief="b", experiment_id="e7"
    )
    agent_cells = [c for c in out.cells if c.source == "agent"]
    assert "ModuleNotFoundError" in (agent_cells[0].error or "")  # original result kept
    sent = "\n".join(m.content or "" for m in fake.calls[-1])
    assert "auto-install of 'catboost' FAILED" in sent  # ...and the failure is visible
    assert "switch to a library" in sent


def test_finish_shim_turns_finish_call_into_guidance(tmp_path: Path) -> None:
    ds = _dataset(tmp_path)
    # models append finish() to otherwise-perfect cells; the preamble shim must turn
    # that into printed guidance instead of a NameError that fails the whole cell.
    fake = _FakeLLM([_run(_FIT_AND_WRITE + "\nfinish()"), _finish(), _finish()])
    out = CodingAgent(fake, LocalKernel(), metric="f1").run(
        dataset=ds, brief="b", experiment_id="e8"
    )
    assert out.result.succeeded, out.result.error
    agent_cells = [c for c in out.cells if c.source == "agent"]
    assert agent_cells[0].error is None
    assert "finish is a tool call" in agent_cells[0].stdout


def test_zero_deadline_ends_before_any_llm_call(tmp_path: Path) -> None:
    ds = _dataset(tmp_path)
    fake = _FakeLLM([])  # any chat would raise IndexError — proves none happens
    out = CodingAgent(
        fake,
        _FakeKernel([CellResult("loaded", "")]),
        metric="f1",
        deadline_seconds=0.0,  # type: ignore[arg-type]
    ).run(dataset=ds, brief="b", experiment_id="e9")
    assert fake.calls == []
    assert not out.result.succeeded  # captured failure, not a crash


def test_deadline_does_not_charge_llm_latency(tmp_path: Path) -> None:
    import time as _time

    ds = _dataset(tmp_path)

    class _SlowLLM(_FakeLLM):
        def chat(self, messages, *, tools=None, temperature=None, max_tokens=None):  # type: ignore[override]
            _time.sleep(0.1)  # model latency far beyond the whole deadline
            return super().chat(
                messages, tools=tools, temperature=temperature, max_tokens=max_tokens
            )

    kernel = _FakeKernel(
        [CellResult("loaded", ""), CellResult("ok", ""), CellResult("ok", "")],
        predictions=b"0\n" * ds.n_test,
    )
    fake = _SlowLLM([_run("a=1"), _run("b=2"), _finish(), _finish()])
    out = CodingAgent(fake, kernel, metric="f1", deadline_seconds=0.05).run(  # type: ignore[arg-type]
        dataset=ds, brief="b", experiment_id="e10"
    )
    # under wall-clock accounting one 0.1s LLM call would exhaust the 0.05s budget;
    # charging kernel time only, all turns run (incl. the improve nudge) and finish.
    assert len(fake.calls) == 4
    assert out.result.succeeded, out.result.error


def test_context_budget_elides_oldest_observations(tmp_path: Path) -> None:
    from iterate.core.coder import _ELIDED

    ds = _dataset(tmp_path)
    kernel = _FakeKernel(
        [CellResult("loaded", "")] + [CellResult("X" * 3000, "") for _ in range(3)],
        predictions=b"0\n" * ds.n_test,
    )
    fake = _FakeLLM([_run("a=1"), _run("b=2"), _run("c=3"), _finish(), _finish()])
    out = CodingAgent(fake, kernel, metric="f1", context_budget_chars=6000).run(  # type: ignore[arg-type]
        dataset=ds, brief="b", experiment_id="e11"
    )
    assert out.result.succeeded, out.result.error
    last = fake.calls[-1]
    assert last[0].role == "system"
    assert last[0].content  # system never elided
    tool_msgs = [m for m in last if m.role == "tool"]
    assert tool_msgs[0].content == _ELIDED  # oldest gave way...
    # ...newest cell observation stays intact (the last tool msg is the improve nudge)
    assert "XXX" in (tool_msgs[-2].content or "")


def test_starting_code_seeds_the_prompt(tmp_path: Path) -> None:
    ds = _dataset(tmp_path)
    fake = _FakeLLM([_run(_FIT_AND_WRITE), _finish(), _finish()])
    CodingAgent(fake, LocalKernel(), metric="f1").run(
        dataset=ds,
        brief="b",
        experiment_id="e",
        starting_code="WINNING_PIPELINE = 1  # prior best",
        starting_score=0.55,
    )
    sent = "\n".join(m.content or "" for m in fake.calls[0])
    assert "BEST APPROACH SO FAR" in sent  # the prior best is offered as a starting point
    assert "WINNING_PIPELINE" in sent
    assert "0.5500" in sent


def test_predictions_with_an_index_column_are_rejected_at_finish(tmp_path: Path) -> None:
    # run 18 iter 3: to_csv without index=False shipped "0,0" lines — right line
    # count, unscorable values, the only FAILED iteration in 100+. The finish gate
    # must catch it in-session with the precise fix named.
    from iterate.core.coder import _validate_predictions

    n = 5
    indexed = "\n".join(f"{i},1" for i in range(n)).encode()
    reason = _validate_predictions(indexed, n)
    assert reason is not None
    assert "index=False" in reason
    # a clean single-column file still passes
    assert _validate_predictions(b"1\n0\n1\n0\n1\n", n) is None
    # string labels containing commas but NOT an index pattern also pass
    assert _validate_predictions(b"a,b\nc,d\nx\ny\nz\n", n) is None


def test_finish_is_rejected_until_valid_predictions_exist(tmp_path: Path) -> None:
    ds = _dataset(tmp_path)
    # 1st finish has no predictions yet -> rejected; then it writes them; 2nd finish -> ends.
    fake = _FakeLLM([_finish(), _run(_FIT_AND_WRITE), _finish(), _finish()])
    out = CodingAgent(fake, LocalKernel(), metric="f1").run(
        dataset=ds, brief="b", experiment_id="e5"
    )
    assert out.result.succeeded, out.result.error  # the early finish didn't end it empty-handed
    assert out.result.metrics is not None
    # finish(rejected) -> run(write) -> finish(improve nudge) -> finish(accepted)
    assert len(fake.calls) == 4
    agent_cells = [c for c in out.cells if c.source == "agent"]
    assert len(agent_cells) == 1  # only the write cell; finish runs no code


def test_early_finish_gets_one_improve_nudge_then_accepts(tmp_path: Path) -> None:
    ds = _dataset(tmp_path)
    # predictions are valid from the start, but ~no budget was spent: the first
    # finish is answered with the improve nudge; the second always ends the session.
    kernel = _FakeKernel([CellResult("loaded", "")], predictions=b"0\n" * ds.n_test)
    fake = _FakeLLM([_finish(), _finish()])
    out = CodingAgent(fake, kernel, metric="f1").run(  # type: ignore[arg-type]
        dataset=ds, brief="b", experiment_id="e12"
    )
    assert out.result.succeeded, out.result.error
    sent = "\n".join(m.content or "" for m in fake.calls[-1])
    assert "make at least one more data-justified improvement" in sent
    assert sent.count("data-justified improvement") == 1  # nudged exactly once


def test_no_improve_nudge_when_budget_mostly_spent(tmp_path: Path) -> None:
    import time as _time

    ds = _dataset(tmp_path)

    class _SlowKernel(_FakeKernel):
        def run_cell(self, code: str, *, timeout: float) -> CellResult:
            if "a=1" not in code:  # only the agent cell is slow, not the preamble
                return super().run_cell(code, timeout=timeout)
            _time.sleep(0.06)
            return super().run_cell(code, timeout=timeout)

    kernel = _SlowKernel(
        [CellResult("loaded", ""), CellResult("ok", "")], predictions=b"0\n" * ds.n_test
    )
    fake = _FakeLLM([_run("a=1"), _finish()])  # a lone finish — no nudge expected
    out = CodingAgent(fake, kernel, metric="f1", deadline_seconds=0.1).run(  # type: ignore[arg-type]
        dataset=ds, brief="b", experiment_id="e13"
    )
    assert out.result.succeeded, out.result.error
    assert len(fake.calls) == 2  # over half the budget spent -> finish accepted directly


class _CountingKernel(_FakeKernel):
    """Records how many cells actually reached the kernel (the breaker should keep
    duplicates from ever executing)."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.executed: list[str] = []

    def run_cell(self, code: str, *, timeout: float) -> CellResult:
        self.executed.append(code)
        return super().run_cell(code, timeout=timeout)


def test_identical_cell_is_not_executed_twice(tmp_path: Path) -> None:
    ds = _dataset(tmp_path)
    kernel = _CountingKernel([CellResult("loaded", "")], predictions=b"0\n" * ds.n_test)
    # the model submits the SAME cell twice, then finishes
    fake = _FakeLLM(
        [_run("print(X_train.shape)"), _run("print(X_train.shape)"), _finish(), _finish()]
    )
    out = CodingAgent(fake, kernel, metric="f1", max_cells=6).run(  # type: ignore[arg-type]
        dataset=ds, brief="b", experiment_id="e14"
    )
    assert out.result.succeeded, out.result.error
    # only ONE print cell reached the kernel (executed code carries a reset prefix)
    assert sum(1 for c in kernel.executed if "print(X_train.shape)" in c) == 1
    agent_cells = [c for c in out.cells if c.source == "agent"]
    assert len(agent_cells) == 1  # the duplicate was never recorded as a run cell
    sent = "\n".join(m.content or "" for m in fake.calls[-1])
    assert "already ran an identical cell" in sent


def test_breaker_catches_a_cycle_within_the_window(tmp_path: Path) -> None:
    ds = _dataset(tmp_path)
    kernel = _CountingKernel([CellResult("loaded", "")], predictions=b"0\n" * ds.n_test)
    # cycle A,B,A,B — the second A and second B must both be rejected (window > 1)
    fake = _FakeLLM([_run("a"), _run("b"), _run("a"), _run("b"), _finish(), _finish()])
    CodingAgent(fake, kernel, metric="f1", max_cells=10).run(  # type: ignore[arg-type]
        dataset=ds, brief="b", experiment_id="e15"
    )
    # executed cells carry a reset prefix, so match on the trailing agent code
    assert sum(1 for c in kernel.executed if c.endswith("\na")) == 1
    assert sum(1 for c in kernel.executed if c.endswith("\nb")) == 1


def test_breaker_treats_whitespace_noise_as_repeat_but_real_change_runs(tmp_path: Path) -> None:
    ds = _dataset(tmp_path)
    kernel = _CountingKernel(
        [CellResult("loaded", ""), CellResult("ok", ""), CellResult("ok", "")],
        predictions=b"0\n" * ds.n_test,
    )
    # only-whitespace/blank-line and full-line-comment edits normalize to the original
    # (a repeat); any real code change runs.
    noise = "  x = 1\n\n# a comment line"
    fake = _FakeLLM([_run("x = 1"), _run(noise), _run("x = 2"), _finish(), _finish()])
    CodingAgent(fake, kernel, metric="f1", max_cells=8).run(  # type: ignore[arg-type]
        dataset=ds, brief="b", experiment_id="e16"
    )
    # executed code carries a reset prefix; match on substring
    assert sum(1 for c in kernel.executed if c.endswith("\nx = 1")) == 1  # original ran once
    assert any("x = 2" in c for c in kernel.executed)  # real change ran
    assert not any("# a comment line" in c for c in kernel.executed)  # cosmetic repeat blocked


def test_same_error_breaker_escalates_when_one_error_recurs(tmp_path: Path) -> None:
    ds = _dataset(tmp_path)
    err = "ValueError: could not convert string to float: 'Male'"
    # three DISTINCT cells (so the repeated-cell breaker stays silent) all hitting the
    # SAME error, then a clean write + finish. The escalation must fire after the 3rd.
    kernel = _FakeKernel(
        [
            CellResult("loaded", ""),
            CellResult("", "", error=err),
            CellResult("", "", error=err),
            CellResult("", "", error=err),
        ],
        predictions=b"0\n" * ds.n_test,
    )
    fake = _FakeLLM([_run("a=1"), _run("b=2"), _run("c=3"), _finish(), _finish()])
    CodingAgent(fake, kernel, metric="f1", max_cells=8, install=False).run(  # type: ignore[arg-type]
        dataset=ds, brief="b", experiment_id="esame"
    )
    allmsgs = "\n".join(m.content or "" for call in fake.calls for m in call)
    assert "hit the SAME error repeatedly" in allmsgs  # escalated, naming the error
    assert "could not convert string to float" in allmsgs


def test_error_signature_collapses_to_last_traceback_line() -> None:
    from iterate.core.coder import _error_signature

    tb = "Traceback (most recent call last):\n  File x\nValueError: bad thing: 'Male'"
    assert _error_signature(tb) == "ValueError: bad thing: 'Male'"
    assert _error_signature("") == ""


def test_inputs_reset_to_pristine_before_each_cell(tmp_path: Path) -> None:
    ds = _dataset(tmp_path)  # X_train has 2 feature columns (num, cat)
    # cell 1 mutates X_train in place (adds a column); cell 2 must see it gone, because
    # the harness restores the canonical inputs at the top of every cell.
    fake = _FakeLLM(
        [
            _run("X_train['injected'] = 1\nprint('in-cell cols:', X_train.shape[1])"),
            _run("print('next-cell cols:', X_train.shape[1])"),
        ]
    )
    out = CodingAgent(fake, LocalKernel(), metric="f1", max_cells=2).run(
        dataset=ds, brief="b", experiment_id="ereset"
    )
    agent_cells = [c for c in out.cells if c.source == "agent"]
    assert "in-cell cols: 3" in agent_cells[0].stdout  # mutation visible within its cell
    assert "next-cell cols: 2" in agent_cells[1].stdout  # but reset for the next cell


def test_thinking_is_attached_to_the_cell_it_produced(tmp_path: Path) -> None:
    ds = _dataset(tmp_path)
    thought = ChatResponse(
        model="fake-model",
        thinking="Look at dtypes first, then build X_tr.",
        tool_calls=[ToolCall(id="c", name="run_cell", arguments={"code": "x = 1"})],
    )
    kernel = _FakeKernel(
        [CellResult("loaded", ""), CellResult("ok", "")], predictions=b"0\n" * ds.n_test
    )
    fake = _FakeLLM([thought, _finish(), _finish()])
    out = CodingAgent(fake, kernel, metric="f1", max_cells=4).run(  # type: ignore[arg-type]
        dataset=ds, brief="b", experiment_id="ethink"
    )
    agent_cells = [c for c in out.cells if c.source == "agent"]
    assert agent_cells[0].thinking == "Look at dtypes first, then build X_tr."
    preamble = [c for c in out.cells if c.source == "preamble"]
    assert preamble[0].thinking is None  # host cells never carry model reasoning


def test_session_without_predictions_banks_the_fallback_floor(tmp_path: Path) -> None:
    # The submission guarantee: the agent never writes predictions.csv and burns its
    # budget — the harness banks the canned baseline as a floor instead of a total loss.
    ds = _dataset(tmp_path)
    fake = _FakeLLM([_run("x = 1"), _run("y = 2")])
    out = CodingAgent(fake, LocalKernel(), metric="f1", max_cells=2).run(
        dataset=ds, brief="b", experiment_id="e3"
    )
    assert out.result.succeeded, out.result.error
    fallback = [c for c in out.cells if c.source == "fallback"]
    assert len(fallback) == 1
    assert "LogisticRegression" in fallback[0].code
    assert "fallback baseline banked" in fallback[0].stdout


def test_fallback_banks_the_carried_best_before_the_canned_baseline(tmp_path: Path) -> None:
    # With carried-forward best code available, a failed session degrades to roughly
    # the best score (re-run the carried code), not all the way down to the baseline.
    ds = _dataset(tmp_path)
    fake = _FakeLLM([_run("x = 1")])
    out = CodingAgent(fake, LocalKernel(), metric="f1", max_cells=1).run(
        dataset=ds,
        brief="b",
        experiment_id="e3b",
        starting_code=_FIT_AND_WRITE,
        starting_score=0.6,
    )
    assert out.result.succeeded, out.result.error
    fallback = [c for c in out.cells if c.source == "fallback"]
    assert len(fallback) == 1
    assert "LogisticRegression" in fallback[0].code  # the carried code, not the canned HGB


def test_fallback_falls_through_to_the_canned_baseline_when_carried_code_errors(
    tmp_path: Path,
) -> None:
    # the carried-forward best can itself be broken (e.g. it depended on session
    # state); the guarantee must then bank the canned baseline, not give up.
    ds = _dataset(tmp_path)
    fake = _FakeLLM([_run("x = 1")])
    out = CodingAgent(fake, LocalKernel(), metric="f1", max_cells=1).run(
        dataset=ds,
        brief="b",
        experiment_id="e3d",
        starting_code="raise RuntimeError('carried code broken')",
        starting_score=0.6,
    )
    assert out.result.succeeded, out.result.error
    fallback = [c for c in out.cells if c.source == "fallback"]
    assert len(fallback) == 2  # carried attempt (errored) + canned baseline (banked)
    assert fallback[0].error is not None
    assert "LogisticRegression" in fallback[1].code
    assert "fallback baseline banked" in fallback[1].stdout


def test_a_valid_submission_is_never_clobbered_by_the_floor(tmp_path: Path) -> None:
    # the session DID submit validly; the fallback must not run at all.
    ds = _dataset(tmp_path)
    fake = _FakeLLM([_run(_FIT_AND_WRITE), _finish()])
    out = CodingAgent(fake, LocalKernel(), metric="f1", max_cells=2).run(
        dataset=ds, brief="b", experiment_id="e3e"
    )
    assert out.result.succeeded, out.result.error
    assert not [c for c in out.cells if c.source == "fallback"]


def test_identical_submission_gets_one_corrective_nudge_then_accepts(tmp_path: Path) -> None:
    # live run: six byte-identical submissions in a row. A finish whose predictions
    # hash to ANY earlier experiment's digest gets ONE corrective message; the next
    # finish is accepted (a nudge, not a wall — a proven-worse lever may
    # legitimately end with the carried best re-submitted).
    import hashlib as _hashlib

    ds = _dataset(tmp_path)
    preds = b"0\n" * ds.n_test
    kernel = _FakeKernel([CellResult("loaded", ""), CellResult("ok", "")], predictions=preds)
    fake = _FakeLLM([_run("x = 1"), _finish(), _finish(), _finish()])
    out = CodingAgent(fake, kernel, metric="f1", max_cells=8).run(  # type: ignore[arg-type]
        dataset=ds,
        brief="b",
        experiment_id="g1",
        # the matching digest is a NON-best sibling's — the gate must still fire
        seen_digests={"unrelated-digest", _hashlib.sha256(preds).hexdigest()},
    )
    assert out.result.succeeded, out.result.error
    final_conversation = "\n".join(m.content or "" for m in fake.calls[-1])
    assert final_conversation.count("byte-identical to an earlier experiment") == 1  # fired once


def test_briefed_lever_missing_from_code_gets_one_corrective_nudge(tmp_path: Path) -> None:
    # live run: class_weight was briefed three times and never appeared in a single
    # cell. A finish without any lever marker in executed code gets ONE corrective.
    ds = _dataset(tmp_path)
    kernel = _FakeKernel(
        [CellResult("loaded", ""), CellResult("ok", "")], predictions=b"0\n" * ds.n_test
    )
    fake = _FakeLLM([_run("x = 1"), _finish(), _finish(), _finish()])
    out = CodingAgent(fake, kernel, metric="f1", max_cells=8).run(  # type: ignore[arg-type]
        dataset=ds,
        brief="b",
        experiment_id="g2",
        brief_markers=("class_weight", "scale_pos_weight", "smote", "threshold"),
    )
    assert out.result.succeeded, out.result.error
    final_conversation = "\n".join(m.content or "" for m in fake.calls[-1])
    assert final_conversation.count("does not appear in any cell") == 1


def test_lever_gate_ignores_markers_inherited_from_the_carried_code(tmp_path: Path) -> None:
    # run-5 false-pass: 'imbalance-or-threshold' was briefed (class_weight), the coder
    # only rebuilt the carried pipeline — whose inherited line contains 'threshold' —
    # and the gate stayed silent. Markers must count on NEW lines only.
    ds = _dataset(tmp_path)
    carried = "model = HGB().fit(Xa, ya)\npreds = (proba >= 0.4)  # threshold write"
    kernel = _FakeKernel(
        [CellResult("loaded", ""), CellResult("ok", "")], predictions=b"0\n" * ds.n_test
    )
    # the coder byte-copies the carried threshold line and adds nothing lever-shaped
    fake = _FakeLLM(
        [_run("preds = (proba >= 0.4)  # threshold write"), _finish(), _finish(), _finish()]
    )
    out = CodingAgent(fake, kernel, metric="f1", max_cells=8).run(  # type: ignore[arg-type]
        dataset=ds,
        brief="b",
        experiment_id="g4",
        starting_code=carried,
        brief_markers=("class_weight", "scale_pos_weight", "smote", "threshold"),
    )
    assert out.result.succeeded, out.result.error
    final_conversation = "\n".join(m.content or "" for m in fake.calls[-1])
    assert final_conversation.count("does not appear in any cell") == 1  # gate FIRED


def test_lever_gate_accepts_a_new_line_bearing_the_marker(tmp_path: Path) -> None:
    ds = _dataset(tmp_path)
    carried = "preds = (proba >= 0.4)  # threshold write"
    kernel = _FakeKernel(
        [CellResult("loaded", ""), CellResult("ok", "")], predictions=b"0\n" * ds.n_test
    )
    # a genuinely NEW threshold sweep line — the lever was pulled this session
    fake = _FakeLLM([_run("best_threshold = sweep(0.2, 0.6)"), _finish(), _finish()])
    out = CodingAgent(fake, kernel, metric="f1", max_cells=8).run(  # type: ignore[arg-type]
        dataset=ds,
        brief="b",
        experiment_id="g5",
        starting_code=carried,
        brief_markers=("class_weight", "scale_pos_weight", "smote", "threshold"),
    )
    assert out.result.succeeded, out.result.error
    final_conversation = "\n".join(m.content or "" for m in fake.calls[-1])
    assert "does not appear in any cell" not in final_conversation


def test_gates_stay_quiet_when_the_lever_landed_and_predictions_differ(tmp_path: Path) -> None:
    ds = _dataset(tmp_path)
    kernel = _FakeKernel(
        [CellResult("loaded", ""), CellResult("ok", "")], predictions=b"0\n" * ds.n_test
    )
    fake = _FakeLLM([_run("model = HGB(class_weight='balanced')"), _finish(), _finish()])
    out = CodingAgent(fake, kernel, metric="f1", max_cells=8).run(  # type: ignore[arg-type]
        dataset=ds,
        brief="b",
        experiment_id="g3",
        brief_markers=("class_weight",),
        seen_digests={"some-other-digest"},
    )
    assert out.result.succeeded, out.result.error
    sent = "\n".join(m.content or "" for call in fake.calls for m in call)
    assert "byte-identical" not in sent
    assert "does not appear in any cell" not in sent


def test_truncated_cell_is_rejected_unexecuted_and_the_retry_runs(tmp_path: Path) -> None:
    # live runs: cells arrived cut mid-token and died as 'unexpected EOF' — the
    # guard must reject them for free (no kernel time) and let the full cell run.
    ds = _dataset(tmp_path)
    kernel = _CountingKernel(
        [CellResult("loaded", ""), CellResult("ok", "")], predictions=b"0\n" * ds.n_test
    )
    truncated = "Xa_cat = pd.DataFrame(enc.fit_transform(Xa_raw[cat_cols]"  # never closed
    fake = _FakeLLM([_run(truncated), _run("x = 1"), _finish(), _finish()])
    out = CodingAgent(fake, kernel, metric="f1", max_cells=8).run(  # type: ignore[arg-type]
        dataset=ds, brief="b", experiment_id="t1"
    )
    assert out.result.succeeded, out.result.error
    assert len(kernel.executed) == 2  # preamble + the retry; the chopped cell never ran
    sent = "\n".join(m.content or "" for m in fake.calls[-1])
    assert "arrived INCOMPLETE" in sent
    # the truncated cell is not recorded as an executed (errored) cell
    assert all("fit_transform(Xa_raw" not in c.code for c in out.cells)


def test_ordinary_syntax_errors_still_execute_for_the_real_traceback(tmp_path: Path) -> None:
    ds = _dataset(tmp_path)
    fake = _FakeLLM([_run("def broken(:\n    pass"), _run(_FIT_AND_WRITE), _finish(), _finish()])
    out = CodingAgent(fake, LocalKernel(), metric="f1", max_cells=8).run(
        dataset=ds, brief="b", experiment_id="t2"
    )
    assert out.result.succeeded, out.result.error
    errored = [c for c in out.cells if c.error]
    assert len(errored) == 1  # the bad-syntax cell executed and its traceback was shown


def test_wall_clock_ceiling_ends_the_session_before_any_llm_call(tmp_path: Path) -> None:
    # The kernel-time deadline does not charge LLM latency, so a thrashing session
    # is otherwise unbounded in wall-clock. A spent ceiling ends it; the floor banks.
    ds = _dataset(tmp_path)
    fake = _FakeLLM([])  # any chat would raise IndexError — proves none happens
    out = CodingAgent(fake, LocalKernel(), metric="f1", wall_ceiling_seconds=0.0).run(
        dataset=ds, brief="b", experiment_id="e11"
    )
    assert fake.calls == []
    assert out.result.succeeded, out.result.error  # the floor was still banked
    assert [c.source for c in out.cells].count("fallback") == 1


def test_consecutive_distinct_errors_end_the_session_early(tmp_path: Path) -> None:
    # A different typo each cell evades the repeat and same-error breakers (live:
    # 'X_holdut', a truncated cell, a bad column name — 32 cells, 8s kernel time).
    # Six errored cells with no success in between must end the session.
    ds = _dataset(tmp_path)
    fake = _FakeLLM([_run(f"broken_name_{i}()") for i in range(6)])
    out = CodingAgent(fake, LocalKernel(), metric="f1").run(
        dataset=ds, brief="b", experiment_id="e12"
    )
    agent_cells = [c for c in out.cells if c.source == "agent"]
    assert len(agent_cells) == 6  # ended exactly at the breaker, no further turns
    assert all(c.error for c in agent_cells)
    assert out.result.succeeded, out.result.error  # floor banked by the guarantee
    assert [c.source for c in out.cells].count("fallback") == 1


def test_a_failed_fallback_stays_a_captured_failure(tmp_path: Path) -> None:
    # If even the fallback cannot produce predictions (kernel serves none), the
    # iteration is still a captured failure, never a crash.
    ds = _dataset(tmp_path)
    kernel = _FakeKernel([CellResult("loaded", "")], predictions=None)
    fake = _FakeLLM([_run("x = 1")])
    out = CodingAgent(fake, kernel, metric="f1", max_cells=1).run(  # type: ignore[arg-type]
        dataset=ds, brief="b", experiment_id="e3c"
    )
    assert not out.result.succeeded
    assert "no predictions" in (out.result.error or "")
    # the fallback WAS attempted (canned baseline; no carried code) and recorded
    assert [c.source for c in out.cells].count("fallback") == 1


# ─── Interactive controller: user notes, pause, stop (v0.3) ──────────────────


def test_user_note_is_injected_at_the_cell_boundary(tmp_path: Path) -> None:
    from iterate.core.interactive import RunController

    fake = _FakeLLM([_run(_FIT_AND_WRITE), _finish(), _finish()])
    ctrl = RunController()
    ctrl.add_session_note("try a smaller learning rate")
    agent = CodingAgent(fake, LocalKernel(), metric="f1", max_cells=8, controller=ctrl)
    coding = agent.run(dataset=_dataset(tmp_path), brief="b", experiment_id="note-test")
    assert coding.result.succeeded
    flat = "\n".join(m.content or "" for call in fake.calls for m in call)
    assert "USER NOTE" in flat
    assert "smaller learning rate" in flat


def test_pause_suspends_the_wall_ceiling(tmp_path: Path) -> None:
    import threading

    from iterate.core.interactive import RunController

    write_floor = (
        "import pandas as pd\n"
        "pd.Series([0]*len(X_holdout)).to_csv('predictions.csv', index=False, header=False)\n"
        "print('ok')\n"
    )
    fake = _FakeLLM([_run(write_floor), _finish(), _finish()])
    ctrl = RunController()
    ctrl.submit_line("pause")
    threading.Timer(1.2, lambda: ctrl.submit_line("resume")).start()
    agent = CodingAgent(
        fake,
        LocalKernel(),
        metric="f1",
        max_cells=8,
        controller=ctrl,
        wall_ceiling_seconds=0.8,  # smaller than the pause: fires unless suspended
    )
    coding = agent.run(dataset=_dataset(tmp_path), brief="b", experiment_id="pause-test")
    # Without the pause credit, iteration 2's ceiling check sees the 1.2s pause and
    # ends the session before the finish turn; with it, the scripted finish runs.
    assert len(fake.calls) >= 2, "the ceiling fired during the pause — no credit applied"
    assert coding.result.succeeded


def test_stop_ends_the_session_before_any_llm_turn_and_banks_a_floor(tmp_path: Path) -> None:
    from iterate.core.interactive import RunController

    fake = _FakeLLM([])  # the model must never be consulted
    ctrl = RunController()
    ctrl.submit_line("stop")
    agent = CodingAgent(fake, LocalKernel(), metric="f1", max_cells=8, controller=ctrl)
    coding = agent.run(dataset=_dataset(tmp_path), brief="b", experiment_id="stop-test")
    assert fake.calls == []
    assert any(c.source == "fallback" for c in coding.cells)
    assert coding.result.succeeded  # the canned baseline floor was banked


def test_cell_events_carry_the_code_for_the_ui(tmp_path: Path) -> None:
    from iterate.core.interactive import RunController

    fake = _FakeLLM([_run(_FIT_AND_WRITE), _finish(), _finish()])
    ctrl = RunController()
    events: list[tuple[str, dict]] = []
    ctrl.on_event = lambda kind, payload: events.append((kind, payload))
    agent = CodingAgent(fake, LocalKernel(), metric="f1", max_cells=8, controller=ctrl)
    agent.run(dataset=_dataset(tmp_path), brief="b", experiment_id="ui-test")
    cell_events = [p for k, p in events if k == "cell"]
    assert len(cell_events) == 1
    assert "LogisticRegression" in str(cell_events[0]["code"])
    assert cell_events[0]["ok"] is True
    assert cell_events[0]["index"] == 1


def test_live_cells_are_shared_during_the_session_and_cleared_after(tmp_path: Path) -> None:
    from iterate.core.interactive import RunController

    fake = _FakeLLM([_run(_FIT_AND_WRITE), _finish(), _finish()])
    ctrl = RunController()
    seen_live: list[int] = []

    def _spy_interpreter(batch: list[str], live_session: bool) -> None:
        seen_live.append(len(ctrl.live_cells or []))

    ctrl.interpreter = _spy_interpreter
    ctrl.submit_line("what is happening?")
    agent = CodingAgent(fake, LocalKernel(), metric="f1", max_cells=8, controller=ctrl)
    agent.run(dataset=_dataset(tmp_path), brief="b", experiment_id="live-test")
    assert seen_live, "the interpreter never saw a checkpoint"
    assert seen_live[0] >= 1  # at least the preamble cell was visible live
    assert ctrl.live_cells is None  # cleared when the session ended


# ─── Timeout resilience (v0.3.1: a live run lost an iteration to a fit-timeout spiral) ─


class _TimeoutKernel:
    """Every cell times out — simulates a fit that can never finish in the cap."""

    def start(self, inputs: dict) -> None:
        pass

    def run_cell(self, code: str, *, timeout: float) -> CellResult:
        return CellResult("", "", error=None, timed_out=True)

    def install(self, packages: list) -> str:
        return ""

    def namespace_summary(self) -> str:
        return ""

    def read_output(self, name: str) -> bytes | None:
        return None

    def keepalive(self) -> None:
        pass

    def blocked(self, error: str | None) -> Blocked | None:
        return None

    def close(self) -> None:
        pass


def test_timeouts_get_a_nudge_count_toward_the_breaker_and_enrich_the_failure(
    tmp_path: Path,
) -> None:
    from iterate.core.coder import _MAX_CONSECUTIVE_ERRORS

    # Each turn's cell differs slightly (as a real model retries) so the
    # repeated-cell breaker does not swallow them before the timeout counter.
    fake = _FakeLLM(
        [
            _run(f"model_{i} = HistGradientBoostingClassifier(random_state=42).fit(Xa, ya)")
            for i in range(_MAX_CONSECUTIVE_ERRORS + 2)
        ]
    )
    agent = CodingAgent(fake, _TimeoutKernel(), metric="f1", max_cells=20)  # type: ignore[arg-type]
    coding = agent.run(dataset=_dataset(tmp_path), brief="b", experiment_id="to-test")
    # the breaker sees timeouts: the session ended after the cap, not max_cells
    assert len(fake.calls) == _MAX_CONSECUTIVE_ERRORS
    # the second turn carried the timeout nudge, naming the limit and the pivot
    second_turn = "\n".join(m.content or "" for m in fake.calls[1])
    assert "per-cell limit" in second_turn
    assert "faster model family" in second_turn
    # the failure record carries the WHY, not just the contract violation
    assert coding.result.error is not None
    assert "timed out" in coding.result.error
    assert "fit(" in coding.result.error


# ─── the probability contract at the finish gate (v0.4) ──────────────────────


def test_proba_requirement_is_only_added_for_probability_metrics() -> None:
    """Every line of the coder system prompt competes for a weak model's attention,
    so an f1 run must not carry an instruction about a file it should never write."""
    from iterate.core.coder import _proba_requirement

    assert _proba_requirement("f1") == ""
    assert _proba_requirement("rmse") == ""
    for metric in ("roc_auc", "average_precision", "log_loss", "brier"):
        text = _proba_requirement(metric)
        assert "probabilities.csv" in text
        assert "predict_proba" in text


def test_validate_probabilities_mirrors_the_parser() -> None:
    from iterate.core.coder import _validate_probabilities

    assert _validate_probabilities(b"0.1\n0.9\n", 2) is None
    assert _validate_probabilities(None, 2) is not None
    assert "expected 2" in (_validate_probabilities(b"0.1\n", 2) or "")


def test_a_misspelled_name_gets_named_and_corrected() -> None:
    """Measured across two live runs: one variable was misspelled four different
    ways (BASEL_PROMPT, BASELINES_PROMPT, BASELIN_PROMPT), each costing a cell —
    while the live namespace was ALREADY appended to every observation. A listing
    the model has to scan is not the same as being told which name was wrong.
    """
    from iterate.core.coder import name_error_hint

    namespace = "X_train DataFrame (1200, 1)\nBASELINE_PROMPT Prompt\nBASE Prompt\n"

    hint = name_error_hint("NameError: name 'BASELIN_PROMPT' is not defined", namespace)

    assert "BASELIN_PROMPT" in hint
    assert "BASELINE_PROMPT" in hint


def test_the_hint_stays_quiet_when_it_has_nothing_useful() -> None:
    """A wrong suggestion is worse than none: it sends the next cell somewhere new."""
    from iterate.core.coder import name_error_hint

    assert name_error_hint("NameError: name 'zzzz' is not defined", "X_train DataFrame\n") == ""
    assert name_error_hint("ValueError: could not convert", "X_train DataFrame\n") == ""
    assert name_error_hint(None, "X_train DataFrame\n") == ""


def test_the_hint_is_not_prompt_specific() -> None:
    """A mistyped variable is an every-path mistake."""
    from iterate.core.coder import name_error_hint

    hint = name_error_hint(
        "NameError: name 'X_trian' is not defined", "X_train DataFrame (100, 5)\ny_train Series\n"
    )

    assert "X_train" in hint


# ─── the free inspect step (carry-in 5) ───────────────────────────────────────
# Cut from v0.4 for a blast radius it turned out not to have: modelling it as an
# experiment is what demanded a fourth AttemptOutcome and the patience carve-outs.
# It submits nothing, so it produces no experiment and touches no terminator.

_PRINT_FACTS = """
print("class balance:", y_train.value_counts(normalize=True).round(2).to_dict())
print("unique values in cat:", X_train['cat'].nunique())
print("fitting model")
"""


def test_an_inspection_returns_the_facts_its_cells_printed(tmp_path: Path) -> None:
    ds = _dataset(tmp_path)
    fake = _FakeLLM([_run(_PRINT_FACTS), _finish()])

    findings = CodingAgent(fake, LocalKernel(), metric="f1", max_cells=8).inspect(dataset=ds)

    assert "class balance" in findings
    assert "unique values in cat" in findings
    # Progress chatter is not a data fact; the dossier's extractor is what draws
    # that line, and the inspect step deliberately reuses it rather than redrawing.
    assert "fitting model" not in findings


def test_an_inspection_finishes_without_writing_predictions(tmp_path: Path) -> None:
    """Every gate on a finish asks a question about a SUBMISSION, and an inspection
    has none — so a finish is accepted on the model's word."""
    ds = _dataset(tmp_path)
    fake = _FakeLLM([_run("print('rows:', len(X_train))"), _finish()])

    agent = CodingAgent(fake, LocalKernel(), metric="f1", max_cells=8)
    findings = agent.inspect(dataset=ds)

    assert "rows:" in findings
    assert len(fake.calls) == 2  # not re-driven for a missing predictions.csv


def test_an_inspection_that_prints_nothing_useful_returns_empty(tmp_path: Path) -> None:
    ds = _dataset(tmp_path)
    fake = _FakeLLM([_run("print('working on it')"), _finish()])

    assert CodingAgent(fake, LocalKernel(), metric="f1", max_cells=8).inspect(dataset=ds) == ""


def test_a_broken_inspection_costs_the_run_nothing(tmp_path: Path) -> None:
    """A free step must never be able to fail a run: the caller gets "" and moves on."""

    class _DeadKernel(_FakeKernel):
        def run_cell(self, code: str, *, timeout: float) -> CellResult:
            raise RuntimeError("kernel died")

    ds = _dataset(tmp_path)
    agent = CodingAgent(_FakeLLM([]), _DeadKernel([]), metric="f1")  # type: ignore[arg-type]

    assert agent.inspect(dataset=ds) == ""


# ─── a confined cell refused something (Sprint 4 Day 5) ───────────────────────


class _BlockedKernel(_FakeKernel):
    def blocked(self, error: str | None) -> Blocked | None:
        if not error or "[Errno 1]" not in error:
            return None
        return Blocked(error.rsplit("'", 2)[-2], program=error.endswith("'uv'"))


def _observation_after(error: str, tmp_path: Path, family: str = "tabular") -> str:
    ds = _dataset(tmp_path)
    kernel = _BlockedKernel(
        [CellResult("loaded", ""), CellResult("", "", error=error)],
        predictions=b"0\n" * ds.n_test,
    )
    fake = _FakeLLM([_run("pd.read_csv('/Users/someone/data.csv')"), _finish(), _finish()])
    CodingAgent(fake, kernel, metric="f1", max_cells=2, family=family).run(  # type: ignore[arg-type]
        dataset=ds, brief="b", experiment_id="e"
    )
    return next(m.content or "" for m in fake.calls[1] if m.role == "tool")


def test_a_cell_refused_a_path_outside_its_folder_is_told_which(tmp_path: Path) -> None:
    denied = "PermissionError: [Errno 1] Operation not permitted: '/Users/someone/data.csv'"
    observation = _observation_after(denied, tmp_path)
    assert observation.startswith(
        "BLOCKED: /Users/someone/data.csv is outside this session's folder"
    )
    assert "Your data is already in X_train, y_train and X_holdout" in observation


def test_the_data_line_follows_the_family(tmp_path: Path) -> None:
    from iterate.prompts import PROMPTS

    lines = PROMPTS["coder"]["outside_folder_inputs"]
    assert set(lines) >= {"tabular", "prompt"}
    denied = "PermissionError: [Errno 1] Operation not permitted: '/Users/someone/data.csv'"
    prompt = _observation_after(denied, tmp_path, family="prompt")
    assert lines["prompt"] in prompt
    assert lines["tabular"] not in prompt


def test_a_refused_program_gets_its_own_note(tmp_path: Path) -> None:
    observation = _observation_after(
        "PermissionError: [Errno 1] Operation not permitted: 'uv'", tmp_path
    )
    assert observation.startswith("BLOCKED: uv is a program a cell may not run")
    assert "outside this session's folder" not in observation


def test_a_permission_error_naming_no_path_gets_no_folder_note(tmp_path: Path) -> None:
    assert "BLOCKED" not in _observation_after("RuntimeError: Operation not permitted", tmp_path)


# ─── cells never install; the harness routes a missing import (Sprint 4 Day 5) ───


class _RouteKernel(_FakeKernel):
    def __init__(self, results: list[CellResult], **kw: Any) -> None:
        super().__init__(results, **kw)
        self.executed: list[str] = []
        self.restarts = 0

    def run_cell(self, code: str, *, timeout: float) -> CellResult:
        self.executed.append(code)
        return super().run_cell(code, timeout=timeout)

    def loaded_modules(self) -> list[str]:
        return ["pandas", "narwhals"]

    def restart(self) -> None:
        self.restarts += 1


class _FakeInstaller:
    def __init__(self, plan: Plan, install_error: str = "") -> None:
        self._plan = plan
        self._error = install_error
        self.installed: list[Plan] = []
        self.saved: list[tuple[Plan, str]] = []
        self.asked: list[tuple[str, list[str] | None]] = []

    def plan(
        self, package: str, *, kernel_modules: list[str] | None, module: str | None = None
    ) -> Plan:
        self.asked.append((package, kernel_modules))
        return self._plan

    def install(self, plan: Plan) -> str:
        self.installed.append(plan)
        return self._error

    def save_for_next_run(self, plan: Plan, *, module: str) -> None:
        self.saved.append((plan, module))


def _missing(name: str) -> CellResult:
    return CellResult("", "", error=f"ModuleNotFoundError: No module named '{name}'")


def _tool_replies(fake: _FakeLLM) -> list[str]:
    return [m.content or "" for m in fake.calls[-1] if m.role == "tool"]


def _route_session(
    tmp_path: Path,
    plan: Plan,
    llm: list[ChatResponse],
    results: list[CellResult],
    install_error: str = "",
) -> tuple[Any, _RouteKernel, _FakeInstaller, _FakeLLM]:
    ds = _dataset(tmp_path)
    kernel = _RouteKernel(results, predictions=b"0\n" * ds.n_test)
    installer = _FakeInstaller(plan, install_error)
    fake = _FakeLLM([*llm, _finish(), _finish()])
    agent = CodingAgent(fake, kernel, metric="f1", max_cells=len(llm) + 2, installer=installer)  # type: ignore[arg-type]
    out = agent.run(dataset=ds, brief="b", experiment_id="route")
    return out, kernel, installer, fake


@pytest.mark.parametrize(
    "cell",
    [
        "!pip install catboost",
        "%pip install -q catboost",
        "import subprocess, sys\nsubprocess.check_call(\n"
        "    [sys.executable, '-m', 'pip', 'install', 'catboost']\n)",
        "from pip._internal.cli.main import main\nmain(['install', 'catboost'])",
    ],
)
@pytest.mark.parametrize("install", [True, False])
def test_a_cell_that_installs_is_refused_unrun_and_free(
    tmp_path: Path, cell: str, install: bool
) -> None:
    ds = _dataset(tmp_path)
    kernel = _RouteKernel([CellResult("loaded", "")], predictions=b"0\n" * ds.n_test)
    fake = _FakeLLM([_run(cell), _finish(), _finish(), _finish()])
    CodingAgent(fake, kernel, metric="f1", max_cells=4, install=install).run(  # type: ignore[arg-type]
        dataset=ds, brief="b", experiment_id="inst"
    )
    assert not any("catboost" in code for code in kernel.executed)
    assert kernel.installed == []
    key = "installer_refused" if install else "installer_refused_no_install"
    assert _tool_replies(fake)[0] == PROMPTS["coder"][key]


def test_a_cell_that_only_asks_what_is_installed_is_pointed_at_importlib_metadata(
    tmp_path: Path,
) -> None:
    _, kernel, _, fake = _route_session(
        tmp_path, Plan("x", Route.INSTALL), [_run("!pip list")], [CellResult("loaded", "")]
    )
    assert not any("pip list" in code for code in kernel.executed)
    assert _tool_replies(fake)[0] == PROMPTS["coder"]["installer_query_refused"]
    assert "importlib.metadata" in PROMPTS["coder"]["installer_query_refused"]


def test_a_resent_installer_cell_is_refused_as_an_installer_not_as_a_repeat(
    tmp_path: Path,
) -> None:
    cell = "!pip install catboost"
    _, _, _, fake = _route_session(
        tmp_path, Plan("x", Route.INSTALL), [_run(cell), _run(cell)], [CellResult("loaded", "")]
    )
    assert _tool_replies(fake)[:2] == [PROMPTS["coder"]["installer_refused"]] * 2


def test_six_installer_cells_in_a_row_end_the_session(tmp_path: Path) -> None:
    ds = _dataset(tmp_path)
    kernel = _RouteKernel([CellResult("loaded", "")], predictions=b"0\n" * ds.n_test)
    cells = [f"!pip install pkg{i}" for i in range(6)]
    fake = _FakeLLM([_run(c) for c in cells] + [_run("print('never')")])
    CodingAgent(fake, kernel, metric="f1", max_cells=20).run(  # type: ignore[arg-type]
        dataset=ds, brief="b", experiment_id="inst6"
    )
    assert len(fake.calls) == 6
    assert not any("never" in code for code in kernel.executed)


def test_the_installer_rule_holds_on_an_e2b_kernel(tmp_path: Path) -> None:
    from iterate.adapters.compute.kernel import E2BKernel

    ds = _dataset(tmp_path)
    ran: list[str] = []

    class _Sandbox:
        files = type(
            "_Files",
            (),
            {
                "write": lambda self, path, content: None,
                "read": lambda self, path, format="bytes": b"0\n" * ds.n_test,
            },
        )()

        def run_code(self, code: str, timeout: float | None = None) -> Any:
            ran.append(code)
            logs = type("_Logs", (), {"stdout": ["ok\n"], "stderr": []})()
            return type("_Execution", (), {"logs": logs, "error": None})()

        def kill(self) -> None:
            pass

    fake = _FakeLLM([_run("!pip install catboost"), _finish(), _finish()])
    CodingAgent(fake, E2BKernel(sandbox_factory=_Sandbox), metric="f1", max_cells=3).run(
        dataset=ds, brief="b", experiment_id="e2b"
    )
    assert not any("catboost" in code for code in ran)
    assert PROMPTS["coder"]["installer_refused"] in "\n".join(_tool_replies(fake))


def test_a_missing_import_with_installs_off_says_so(tmp_path: Path) -> None:
    ds = _dataset(tmp_path)
    kernel = _RouteKernel(
        [CellResult("loaded", ""), _missing("catboost")], predictions=b"0\n" * ds.n_test
    )
    fake = _FakeLLM([_run("import catboost"), _finish(), _finish()])
    CodingAgent(fake, kernel, metric="f1", max_cells=4, install=False).run(  # type: ignore[arg-type]
        dataset=ds, brief="b", experiment_id="off"
    )
    assert _tool_replies(fake)[0].startswith(
        "('catboost' is not installed and installs are off for this run."
    )
    assert kernel.installed == []


def test_install_route_reruns_the_cell_with_fresh_import_caches(tmp_path: Path) -> None:
    plan = Plan("tabulate", Route.INSTALL, "0.10.0")
    _, kernel, installer, fake = _route_session(
        tmp_path,
        plan,
        [_run("import tabulate")],
        [CellResult("loaded", ""), _missing("tabulate"), CellResult("ok", "")],
    )
    assert installer.asked == [("tabulate", ["pandas", "narwhals"])]
    assert installer.installed == [plan]
    assert kernel.restarts == 0
    assert kernel.executed[-1].startswith("__import__('importlib').invalidate_caches()")
    assert _tool_replies(fake)[0].startswith(
        "('tabulate' 0.10.0 was installed and your cell re-ran.)"
    )


def test_restart_route_restarts_reruns_the_preamble_and_says_variables_are_gone(
    tmp_path: Path,
) -> None:
    plan = Plan("plotly", Route.RESTART, "7.1.0", {"narwhals": ("1.14.0", "2.26.0")})
    results = [
        CellResult("loaded", ""),
        _missing("plotly"),
        CellResult("loaded again", ""),
        CellResult("plotly ok", ""),
    ]
    out, kernel, _, fake = _route_session(tmp_path, plan, [_run("import plotly")], results)
    assert kernel.restarts == 1
    assert [c.source for c in out.cells[:3]] == ["preamble", "preamble", "agent"]
    assert out.cells[2].stdout == "plotly ok"
    reply = _tool_replies(fake)[0]
    for phrase in (
        "KERNEL RESTARTED",
        "narwhals 1.14.0 -> 2.26.0",
        "Every variable you made is GONE",
        "loaded again",
    ):
        assert phrase in reply


def test_a_cell_resent_after_a_restart_runs_again(tmp_path: Path) -> None:
    plan = Plan("plotly", Route.RESTART, "7.1.0", {"narwhals": ("1.14.0", "2.26.0")})
    build = "feats = X_train.assign(double=X_train['num'] * 2)"
    results = [
        CellResult("loaded", ""),
        CellResult("built", ""),
        _missing("plotly"),
        CellResult("loaded again", ""),
        CellResult("plotly ok", ""),
        CellResult("rebuilt", ""),
    ]
    _, kernel, _, fake = _route_session(
        tmp_path, plan, [_run(build), _run("import plotly"), _run(build)], results
    )
    assert sum(build in code for code in kernel.executed) == 2
    assert not any("already ran an identical cell" in reply for reply in _tool_replies(fake))


def test_a_restart_starts_the_error_breakers_afresh(tmp_path: Path) -> None:
    plan = Plan("plotly", Route.RESTART, "7.1.0", {"narwhals": ("1.14.0", "2.26.0")})
    boom = CellResult("", "", error="ValueError: boom")
    llm = [
        _run("a = boom()"),
        _run("b = boom()"),
        _run("c = x()"),
        _run("d = y()"),
        _run("import plotly"),
        _run("e = boom()"),
        _run("print('still here')"),
    ]
    results = [
        CellResult("loaded", ""),
        boom,
        boom,
        CellResult("", "", error="NameError: name 'x' is not defined"),
        CellResult("", "", error="NameError: name 'y' is not defined"),
        _missing("plotly"),
        CellResult("loaded again", ""),
        CellResult("", "", error="NameError: name 'z' is not defined"),
        boom,
        CellResult("still here", ""),
    ]
    _, kernel, _, fake = _route_session(tmp_path, plan, llm, results)
    assert any("still here" in code for code in kernel.executed)
    assert not any("hit the SAME error repeatedly" in reply for reply in _tool_replies(fake))


def test_next_run_route_saves_and_does_not_rerun(tmp_path: Path) -> None:
    plan = Plan("sktime", Route.NEXT_RUN, "1.1.0", {"pandas": ("3.0.3", "2.3.3")})
    _, kernel, installer, fake = _route_session(
        tmp_path, plan, [_run("import sktime")], [CellResult("loaded", ""), _missing("sktime")]
    )
    assert installer.saved == [(plan, "sktime")]
    assert installer.installed == []
    assert sum("import sktime" in code for code in kernel.executed) == 1
    assert "will be installed when the next run starts" in _tool_replies(fake)[0]


@pytest.mark.parametrize(
    ("reason", "phrase"),
    [
        ("not_found", "does not exist on the package index"),
        ("frozen", "never installed or changed during a run"),
        ("frozen_dep", "needs a different torch or torchvision"),
        ("needs_torch", "installed only at the start of an image run"),
        ("iterate", "beside the packages iterate runs on"),
        ("installed", "Nothing to install"),
        ("network", "the package index is unreachable"),
        ("no_installer", "the environment, not the package"),
        ("something_new", "cannot be installed in this environment"),
    ],
)
def test_refuse_route_names_its_reason_and_installs_nothing(
    tmp_path: Path, reason: str, phrase: str
) -> None:
    plan = Plan("x", Route.REFUSE, reason=reason, detail="d")
    _, kernel, installer, fake = _route_session(
        tmp_path, plan, [_run("import x")], [CellResult("loaded", ""), _missing("x")]
    )
    assert (installer.installed, installer.saved) == ([], [])
    assert kernel.restarts == 0
    assert phrase in _tool_replies(fake)[0]


def test_a_failed_route_install_keeps_the_original_error_and_never_restarts(
    tmp_path: Path,
) -> None:
    plan = Plan("plotly", Route.RESTART, "7.1.0", {"narwhals": ("1.14.0", "2.26.0")})
    out, kernel, _, fake = _route_session(
        tmp_path,
        plan,
        [_run("import plotly")],
        [CellResult("loaded", ""), _missing("plotly")],
        install_error="boom",
    )
    assert kernel.restarts == 0
    assert "No module named 'plotly'" in (out.cells[1].error or "")
    assert "auto-install of 'plotly' FAILED: boom" in _tool_replies(fake)[0]


# ─── an import a cell caught itself (sprint 4 Day 6) ───────────────────────────


class _WatchKernel(_RouteKernel):
    """A kernel with a working directory: what the watch recorded, and the predictions."""

    def __init__(self, results: list[CellResult], *, files: dict[str, bytes], **kw: Any) -> None:
        super().__init__(results, **kw)
        self.files = dict(files)
        self.reads: list[str] = []
        self.started: dict[str, bytes] = {}

    def start(self, inputs: dict[str, bytes]) -> None:
        self.started = dict(inputs)

    def read_output(self, name: str) -> bytes | None:
        self.reads.append(name)
        return self.files.get(name)


def _caught_session(
    tmp_path: Path,
    plan: Plan,
    results: list[CellResult],
    *,
    recorded: bytes = b"catboost\n",
    install: bool = True,
    installer: bool = True,
    code: str = "try:\n    import catboost\nexcept ImportError:\n    pass",
) -> tuple[Any, _WatchKernel, _FakeInstaller, _FakeLLM]:
    ds = _dataset(tmp_path)
    kernel = _WatchKernel(
        results,
        files={
            codegen.MISSING_IMPORTS: recorded,
            codegen.PREDICTIONS_CSV: b"0\n" * ds.n_test,
        },
    )
    fake_installer = _FakeInstaller(plan)
    fake = _FakeLLM([_run(code), _finish(), _finish()])
    agent = CodingAgent(
        fake,
        kernel,  # type: ignore[arg-type]
        metric="f1",
        max_cells=4,
        install=install,
        installer=fake_installer if installer else None,
    )
    out = agent.run(dataset=ds, brief="b", experiment_id="caught")
    return out, kernel, fake_installer, fake


def test_a_caught_import_is_installed_and_the_cell_is_not_rerun(tmp_path: Path) -> None:
    plan = Plan("catboost", Route.INSTALL, "1.2.10")
    _, kernel, installer, fake = _caught_session(
        tmp_path, plan, [CellResult("loaded", ""), CellResult("used the fallback", "")]
    )
    assert installer.installed == [plan]
    assert sum("import catboost" in code for code in kernel.executed) == 1
    note = _tool_replies(fake)[0]
    assert note.startswith(PROMPTS["coder"]["install_caught"].format(module="catboost"))
    assert PROMPTS["coder"]["install_caught_ran_without"].strip() in note
    assert "Import it plainly" in note


def test_a_caught_import_is_planned_once_a_session(tmp_path: Path) -> None:
    ds = _dataset(tmp_path)
    kernel = _WatchKernel(
        [CellResult("loaded", ""), CellResult("first", ""), CellResult("second", "")],
        files={
            codegen.MISSING_IMPORTS: b"catboost\n",
            codegen.PREDICTIONS_CSV: b"0\n" * ds.n_test,
        },
    )
    installer = _FakeInstaller(Plan("catboost", Route.INSTALL, "1.2.10"))
    fake = _FakeLLM([_run("import x"), _run("import y"), _finish(), _finish()])
    CodingAgent(fake, kernel, metric="f1", max_cells=5, installer=installer).run(  # type: ignore[arg-type]
        dataset=ds, brief="b", experiment_id="once"
    )
    assert len(installer.asked) == 1


@pytest.mark.parametrize(
    ("result", "finished"),
    [
        (CellResult("banked the fallback", ""), True),
        (CellResult("", "", error="ValueError: something else broke"), False),
        (CellResult("", "", timed_out=True), False),
    ],
)
def test_a_caught_import_never_reruns_the_cell_however_it_ended(
    tmp_path: Path, result: CellResult, finished: bool
) -> None:
    """An image cell is a fit of minutes: re-running one to reach an import it already
    worked around would spend the session's budget twice."""
    plan = Plan("timm", Route.INSTALL, "1.0.29")
    _, kernel, installer, fake = _caught_session(
        tmp_path, plan, [CellResult("loaded", ""), result], recorded=b"timm\n"
    )
    assert installer.installed == [plan]
    assert kernel.restarts == 0
    assert sum("import catboost" in code for code in kernel.executed) == 1
    note = _tool_replies(fake)[0]
    assert (PROMPTS["coder"]["install_caught_ran_without"].strip() in note) is finished


def test_a_caught_import_that_would_restart_the_kernel_does_not(tmp_path: Path) -> None:
    plan = Plan("plotly", Route.RESTART, "7.1.0", {"narwhals": ("1.14.0", "2.26.0")})
    _, kernel, installer, fake = _caught_session(
        tmp_path, plan, [CellResult("loaded", ""), CellResult("ok", "")], recorded=b"plotly\n"
    )
    assert (kernel.restarts, installer.installed) == (0, [])
    assert "in a cell of its own" in _tool_replies(fake)[0]


def test_a_caught_import_saved_for_the_next_run_is_saved(tmp_path: Path) -> None:
    plan = Plan("sktime", Route.NEXT_RUN, "1.1.0", {"pandas": ("3.0.3", "2.3.3")})
    _, _, installer, fake = _caught_session(
        tmp_path, plan, [CellResult("loaded", ""), CellResult("ok", "")], recorded=b"sktime\n"
    )
    assert installer.saved == [(plan, "sktime")]
    assert "will be installed when the next run starts" in _tool_replies(fake)[0]


def test_a_caught_import_that_is_refused_says_why(tmp_path: Path) -> None:
    plan = Plan("notathing", Route.REFUSE, reason="not_found", detail="d")
    _, _, installer, fake = _caught_session(
        tmp_path, plan, [CellResult("loaded", ""), CellResult("ok", "")], recorded=b"notathing\n"
    )
    assert installer.installed == []
    assert "does not exist on the package index" in _tool_replies(fake)[0]


def test_with_installs_off_a_caught_import_is_only_reported(tmp_path: Path) -> None:
    """The pair a local run without `--install` builds: no installer at all. The agent is
    still told the module was missing, as the raised-import path already tells it."""
    _, _, installer, fake = _caught_session(
        tmp_path,
        Plan("catboost", Route.INSTALL, "1.2.10"),
        [CellResult("loaded", ""), CellResult("ok", "")],
        install=False,
        installer=False,
    )
    assert installer.installed == []
    assert PROMPTS["coder"]["install_off_note"].format(module="catboost") in _tool_replies(fake)[0]


def test_on_e2b_a_caught_import_is_left_to_the_sandbox(tmp_path: Path) -> None:
    """Installs on with no installer of the harness's own is e2b, which installs through
    its own kernel: telling that session installs are off would be false."""
    _, _, installer, fake = _caught_session(
        tmp_path,
        Plan("catboost", Route.INSTALL, "1.2.10"),
        [CellResult("loaded", ""), CellResult("ok", "")],
        installer=False,
    )
    assert installer.asked == []
    reply = _tool_replies(fake)[0]
    assert PROMPTS["coder"]["install_caught"].format(module="catboost") not in reply


def test_an_environment_probe_a_cell_caught_is_never_planned(tmp_path: Path) -> None:
    """google.colab and the kaggle names are missing here by definition, and what PyPI
    holds under them is not what the cell was asking for. Where no installed package owns
    the `google` folder, the parent import fails first and the bare top name is the one
    the watch is asked for."""
    _, _, installer, fake = _caught_session(
        tmp_path,
        Plan("google", Route.INSTALL, "3.0.0"),
        [CellResult("loaded", ""), CellResult("ok", "")],
        recorded=b"google\ngoogle.colab\nkaggle_secrets\n",
    )
    assert installer.asked == []
    assert "was missing" not in _tool_replies(fake)[0]


def test_a_record_file_that_is_not_a_module_name_plans_nothing(tmp_path: Path) -> None:
    _, _, installer, _ = _caught_session(
        tmp_path,
        Plan("x", Route.INSTALL, "1.0"),
        [CellResult("loaded", ""), CellResult("ok", "")],
        recorded=b"0\n0\n0\n",
    )
    assert installer.asked == []


def test_a_sandbox_kernel_never_reads_the_record_file(tmp_path: Path) -> None:
    """e2b installs through the kernel itself and keeps main's behaviour."""
    _, kernel, _, _ = _caught_session(
        tmp_path,
        Plan("catboost", Route.INSTALL, "1.2.10"),
        [CellResult("loaded", ""), CellResult("ok", "")],
        installer=False,
    )
    assert codegen.MISSING_IMPORTS not in kernel.reads
    assert kernel.installed == []


def test_a_raised_missing_import_still_installs_and_reruns_the_cell(tmp_path: Path) -> None:
    """The plain path is unchanged: a traceback names the module, so the cell is worth
    re-running, and the watch must not handle it a second time."""
    plan = Plan("category_encoders", Route.INSTALL, "2.9.0")
    _, kernel, installer, _ = _caught_session(
        tmp_path,
        plan,
        [
            CellResult("loaded", ""),
            _missing("category_encoders"),
            CellResult("worked after install", ""),
        ],
        recorded=b"category_encoders\n",
        code="import category_encoders",
    )
    assert len(installer.asked) == 1
    assert sum("import category_encoders" in code for code in kernel.executed) == 2


# ─── the family's own cell prefix, floor and profile ──────────────────────────


def test_the_cell_prefix_is_the_familys_own(tmp_path: Path) -> None:
    ds = _dataset(tmp_path)
    kernel = _RouteKernel([CellResult("loaded", "")], predictions=b"0\n" * ds.n_test)
    fake = _FakeLLM([_run("print(1)"), _finish(), _finish()])
    CodingAgent(
        fake,
        kernel,  # type: ignore[arg-type]
        metric="f1",
        max_cells=3,
        cell_prefix="PREFIX\n",
    ).run(dataset=ds, brief="b", experiment_id="prefix")
    assert kernel.executed[-1] == "PREFIX\nprint(1)"


def test_the_floor_runs_the_familys_prefix_and_may_skip_the_carried_code(
    tmp_path: Path,
) -> None:
    """The image floor writes predictions from the labels alone; re-running the carried
    best there would be a fit of minutes, at the moment the session has no budget left."""
    ds = _dataset(tmp_path)
    kernel = _RouteKernel([CellResult("loaded", "")], predictions=None)
    fake = _FakeLLM([_run("print('nothing submitted')"), _finish(), _finish(), _finish()])
    CodingAgent(
        fake,
        kernel,  # type: ignore[arg-type]
        metric="f1",
        max_cells=3,
        cell_prefix="PREFIX\n",
        floor_carries_code=False,
        floor_cell="FLOOR\n",
    ).run(dataset=ds, brief="b", experiment_id="floor", starting_code="CARRIED\n")
    assert any(c.startswith("PREFIX\nFLOOR") for c in kernel.executed)
    assert not any("CARRIED" in code for code in kernel.executed)


def test_the_carried_best_is_still_the_first_floor_on_a_table(tmp_path: Path) -> None:
    ds = _dataset(tmp_path)
    kernel = _RouteKernel([CellResult("loaded", "")], predictions=None)
    fake = _FakeLLM([_run("print('nothing submitted')"), _finish(), _finish(), _finish()])
    CodingAgent(fake, kernel, metric="f1", max_cells=3).run(  # type: ignore[arg-type]
        dataset=ds, brief="b", experiment_id="floor", starting_code="CARRIED\n"
    )
    assert any("CARRIED" in code for code in kernel.executed)


def test_the_family_may_hand_the_coder_its_own_profile(tmp_path: Path) -> None:
    """A table profile of one column of file names says nothing about images."""
    ds = _dataset(tmp_path)
    kernel = _RouteKernel([CellResult("loaded", "")], predictions=b"0\n" * ds.n_test)
    fake = _FakeLLM([_finish(), _finish()])
    CodingAgent(
        fake,
        kernel,  # type: ignore[arg-type]
        metric="f1",
        max_cells=2,
        data_summary="Images: 40 train / 10 holdout at 64px.",
    ).run(dataset=ds, brief="b", experiment_id="profile")
    assert any("Images: 40 train" in (m.content or "") for m in fake.calls[0])


def test_the_data_line_has_wording_for_every_family() -> None:
    """A BLOCKED cell reads this by family; a missing key would be a KeyError in the
    middle of a live session."""
    lines = PROMPTS["coder"]["outside_folder_inputs"]
    assert set(lines) == {"tabular", "prompt", "vision"}
    assert "train_px" in lines["vision"]


# ─── what the host carries in, and what it keeps ──────────────────────────────


def test_starting_files_reach_the_kernels_working_directory(tmp_path: Path) -> None:
    ds = _dataset(tmp_path)
    kernel = _WatchKernel(
        [CellResult("loaded", "")], files={codegen.PREDICTIONS_CSV: b"0\n" * ds.n_test}
    )
    fake = _FakeLLM([_finish(), _finish()])
    CodingAgent(fake, kernel, metric="f1", max_cells=2).run(  # type: ignore[arg-type]
        dataset=ds,
        brief="b",
        experiment_id="carry",
        starting_files={codegen.INCUMBENT_JSON: b'{"backbone": "resnet18"}'},
    )
    assert kernel.started[codegen.INCUMBENT_JSON] == b'{"backbone": "resnet18"}'
    assert codegen.TRAIN_CSV in kernel.started


def test_the_recipe_is_kept_only_while_it_describes_the_predictions(tmp_path: Path) -> None:
    ds = _dataset(tmp_path)
    predictions = b"0\n" * ds.n_test
    digest = hashlib.sha256(predictions).hexdigest()
    for recorded, kept in (
        (json.dumps({"backbone": "resnet18", "predictions_sha256": digest}).encode(), True),
        (json.dumps({"backbone": "resnet18", "predictions_sha256": "beef"}).encode(), False),
        (b"not json", False),
    ):
        kernel = _WatchKernel(
            [CellResult("loaded", "")],
            files={codegen.PREDICTIONS_CSV: predictions, codegen.RECIPE_JSON: recorded},
        )
        fake = _FakeLLM([_finish(), _finish()])
        out = CodingAgent(fake, kernel, metric="f1", max_cells=2).run(  # type: ignore[arg-type]
            dataset=ds, brief="b", experiment_id="recipe"
        )
        assert (codegen.RECIPE_JSON in out.result.artifacts) is kept


def _network_session(
    tmp_path: Path, *, recipe: dict[str, Any], network: bytes | None, keep: Path | None
) -> _WatchKernel:
    ds = _dataset(tmp_path)
    predictions = b"0\n" * ds.n_test
    recorded = {"predictions_sha256": hashlib.sha256(predictions).hexdigest(), **recipe}
    files = {codegen.PREDICTIONS_CSV: predictions, codegen.RECIPE_JSON: json.dumps(recorded)}
    if network is not None:
        files[codegen.NETWORK_PT] = network
    kernel = _WatchKernel(
        [CellResult("loaded", "")], files={k: _bytes(v) for k, v in files.items()}
    )
    fake = _FakeLLM([_finish(), _finish()])
    CodingAgent(fake, kernel, metric="f1", max_cells=2, keep_model=keep).run(  # type: ignore[arg-type]
        dataset=ds, brief="b", experiment_id="keep"
    )
    return kernel


def _bytes(value: str | bytes) -> bytes:
    return value if isinstance(value, bytes) else value.encode()


def test_the_network_recipe_json_names_is_copied_out_before_the_kernel_closes(
    tmp_path: Path,
) -> None:
    slot = tmp_path / "slot" / "best_model.pt"
    slot.parent.mkdir()
    digest = hashlib.sha256(b"weights").hexdigest()
    _network_session(tmp_path, recipe={"model_sha256": digest}, network=b"weights", keep=slot)
    assert slot.read_bytes() == b"weights"


@pytest.mark.parametrize(
    ("recipe", "network"),
    [
        ({"model_sha256": "beef"}, b"weights"),
        ({"model_sha256": hashlib.sha256(b"weights").hexdigest()}, None),
        ({}, b"weights"),
        ({"model_sha256": hashlib.sha256(b"weights").hexdigest(), "predictions_sha256": "x"}, b"w"),
    ],
    ids=["another file's digest", "no file", "a recipe that names no network", "a floor submit"],
)
def test_a_network_the_recipe_does_not_vouch_for_is_never_copied(
    tmp_path: Path, recipe: dict[str, Any], network: bytes | None
) -> None:
    slot = tmp_path / "slot" / "best_model.pt"
    slot.parent.mkdir()
    _network_session(tmp_path, recipe=recipe, network=network, keep=slot)
    assert not slot.exists()


def test_a_slot_whose_folder_is_gone_is_not_recreated(tmp_path: Path) -> None:
    """A hard quit deletes the slot folder while the loop thread is still running."""
    slot = tmp_path / "gone" / "best_model.pt"
    digest = hashlib.sha256(b"weights").hexdigest()
    _network_session(tmp_path, recipe={"model_sha256": digest}, network=b"weights", keep=slot)
    assert not slot.parent.exists()


def test_the_slot_is_cleared_when_a_session_starts(tmp_path: Path) -> None:
    """A session that crashed must not leave its network for the next one to win with."""
    slot = tmp_path / "best_model.pt"
    slot.write_bytes(b"the last session's")
    _network_session(tmp_path, recipe={}, network=None, keep=slot)
    assert not slot.exists()


def test_a_copy_that_fails_costs_the_network_and_not_the_score(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pathlib

    slot = tmp_path / "slot" / "best_model.pt"
    slot.parent.mkdir()
    ds = _dataset(tmp_path)
    predictions = b"0\n" * ds.n_test
    recorded = {
        "predictions_sha256": hashlib.sha256(predictions).hexdigest(),
        "model_sha256": hashlib.sha256(b"weights").hexdigest(),
    }
    kernel = _WatchKernel(
        [CellResult("loaded", "")],
        files={
            codegen.PREDICTIONS_CSV: predictions,
            codegen.RECIPE_JSON: json.dumps(recorded).encode(),
            codegen.NETWORK_PT: b"weights",
        },
    )

    def full(self: pathlib.Path, data: bytes) -> int:
        self.touch()
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(pathlib.Path, "write_bytes", full)
    fake = _FakeLLM([_finish(), _finish()])
    out = CodingAgent(fake, kernel, metric="f1", max_cells=2, keep_model=slot).run(  # type: ignore[arg-type]
        dataset=ds, brief="b", experiment_id="keep"
    )
    assert out.result.succeeded
    assert codegen.RECIPE_JSON in out.result.artifacts
    assert not slot.exists()


def test_a_run_with_no_slot_never_asks_the_kernel_for_a_network(tmp_path: Path) -> None:
    """Every table and prompt run: `keep_model` is None and the session is today's."""
    digest = hashlib.sha256(b"weights").hexdigest()
    kernel = _network_session(
        tmp_path, recipe={"model_sha256": digest}, network=b"weights", keep=None
    )
    assert codegen.NETWORK_PT not in kernel.reads


# ─── a cell that would not stop ───────────────────────────────────────────────


def test_a_cell_the_kernel_had_to_be_restarted_for_re_runs_the_preamble(tmp_path: Path) -> None:
    ds = _dataset(tmp_path)
    kernel = _RouteKernel(
        [
            CellResult("loaded", ""),
            CellResult("", "", timed_out=True, restarted=True),
            CellResult("session ready again", ""),
        ],
        predictions=b"0\n" * ds.n_test,
    )
    fake = _FakeLLM([_run("while True: pass"), _finish(), _finish()])
    out = CodingAgent(fake, kernel, metric="f1", max_cells=4).run(  # type: ignore[arg-type]
        dataset=ds, brief="b", experiment_id="stuck"
    )
    assert [c.source for c in out.cells] == ["preamble", "preamble", "agent"]
    note = _tool_replies(fake)[0]
    assert "kernel was RESTARTED" in note
    assert "session ready again" in note


def test_a_cell_that_would_not_stop_still_counts_towards_ending_the_session(
    tmp_path: Path,
) -> None:
    """Resetting the breaker on every restart would let one runaway cell repeat forever."""
    ds = _dataset(tmp_path)
    stuck = CellResult("", "", timed_out=True, restarted=True)
    kernel = _RouteKernel(
        [CellResult("loaded", ""), *[x for _ in range(6) for x in (stuck, CellResult("ok", ""))]],
        predictions=b"0\n" * ds.n_test,
    )
    fake = _FakeLLM([_run(f"while True: pass  # {i}") for i in range(7)])
    CodingAgent(fake, kernel, metric="f1", max_cells=10).run(  # type: ignore[arg-type]
        dataset=ds, brief="b", experiment_id="stuck6"
    )
    assert len(fake.calls) == 6
