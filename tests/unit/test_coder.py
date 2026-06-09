"""Tests for the coding agent — the cell-by-cell session.

End-to-end through a REAL LocalKernel + real ModelTarget scoring, driven by a
deterministic fake LLM that scripts cells. No real LLM, no e2b. Proves the loop
runs cells, persists state, scores the written predictions, captures the cells,
and recovers from a cell error mid-session.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from iterate.adapters.compute.kernel import CellResult, LocalKernel
from iterate.adapters.data.tabular import load_csv
from iterate.core.coder import CodingAgent
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
    return ChatResponse(model="fake-model", tool_calls=[ToolCall(id="f", name="finish", arguments={})])


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
        fake, _FakeKernel([CellResult("loaded", "")]), metric="f1", deadline_seconds=0.0  # type: ignore[arg-type]
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
        dataset=ds, brief="b", experiment_id="e",
        starting_code="WINNING_PIPELINE = 1  # prior best", starting_score=0.55,
    )
    sent = "\n".join(m.content or "" for m in fake.calls[0])
    assert "BEST APPROACH SO FAR" in sent  # the prior best is offered as a starting point
    assert "WINNING_PIPELINE" in sent
    assert "0.5500" in sent


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
    fake = _FakeLLM([_run("print(X_train.shape)"), _run("print(X_train.shape)"), _finish(), _finish()])
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
    fake = _FakeLLM(
        [_run("a"), _run("b"), _run("a"), _run("b"), _finish(), _finish()]
    )
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


def test_session_without_predictions_is_a_captured_failure(tmp_path: Path) -> None:
    ds = _dataset(tmp_path)
    # the agent never writes predictions.csv and burns its budget.
    fake = _FakeLLM([_run("x = 1"), _run("y = 2")])
    out = CodingAgent(fake, LocalKernel(), metric="f1", max_cells=2).run(
        dataset=ds, brief="b", experiment_id="e3"
    )
    assert not out.result.succeeded
    assert "no predictions" in (out.result.error or "")
