"""The coding agent — drives one experiment as a cell-by-cell kernel session.

Unlike the one-shot proposer (write a whole `train_and_predict` blind), the coding
agent works like a human in a notebook: it writes a cell, sees the cell's REAL
output, then writes the next — in a live `StatefulKernel` whose namespace persists.
It can inspect the data, engineer features, check intermediate shapes, and fix its
own errors from the actual traceback, all within one experiment.

It runs a multi-turn tool loop: the LLM calls ``run_cell(code)`` (we execute it and
feed the output back) or ``finish`` (it has written `predictions.csv`). A trusted
host-authored preamble loads `X_train`/`y_train`/`X_holdout` first; the holdout
labels never enter the kernel, and we score the written predictions host-side
through `core.scoring`, so the sealed-holdout guarantee is unchanged.

For now the `brief` (what to try) is supplied by the caller; the Supervisor agent
will produce it from run history next.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from iterate.core import codegen
from iterate.core.scoring import direction
from iterate.prompts import PROMPTS
from iterate.schemas.llm import Message, ToolSpec

if TYPE_CHECKING:
    from iterate.adapters.compute.kernel import CellResult, StatefulKernel
    from iterate.adapters.data.tabular import TabularDataset
    from iterate.llm.base import LLMClient
    from iterate.schemas.experiment import ExperimentResult


class CodingAgentError(RuntimeError):
    """The coding session failed irrecoverably (e.g. the LLM backend errored)."""


_PROMPTS = PROMPTS["coder"]
# Feed the agent the FULL cell output (EDA tables, value_counts, etc.) — modern
# context windows are large and that output is what it reasons from. Only a very
# high ceiling remains, to bound a pathological dump (e.g. printing the whole frame).
_OBSERVATION_TAIL = 20000
_MISSING_MODULE = re.compile(r"No module named ['\"]([\w.]+)['\"]")
_REPEAT_WINDOW = 6  # how many recent executed cells the repeated-cell breaker remembers
_SAME_ERROR_LIMIT = 3  # recurrences of one error signature before escalating the nudge


def _error_signature(error: str) -> str:
    """A stable one-line fingerprint of an error for the same-error breaker: the last
    non-empty traceback line (the `ExceptionType: message`), so two cells that fail
    the same way collapse to one signature even when their code differs."""
    lines = [ln.strip() for ln in (error or "").splitlines() if ln.strip()]
    return lines[-1][:160] if lines else ""


def _normalize_code(code: str) -> str:
    """Code reduced to what actually runs, for duplicate detection: strip each line,
    drop blank and comment-only lines. So a re-submitted cell with only whitespace or
    comment edits still reads as a repeat, but any real code change does not."""
    lines = []
    for raw in code.splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            lines.append(line)
    return "\n".join(lines)


def _missing_module(error: str | None) -> str | None:
    match = _MISSING_MODULE.search(error or "")
    return match.group(1) if match else None


def _validate_predictions(preds: bytes | None, n_test: int) -> str | None:
    """Why a `finish` should be rejected, or None if predictions are valid. Same
    front-half checks as scoring, run at finish time so we don't end empty-handed."""
    if not preds:
        return "predictions.csv was not found in the working directory"
    raw = preds.decode(errors="replace").strip()
    if not raw:
        return "predictions.csv is empty"
    count = len(raw.splitlines())
    if count != n_test:
        return f"expected {n_test} predictions (one per holdout row), found {count}"
    return None


@dataclass(frozen=True)
class Cell:
    """One cell of the session — for the notebook deliverable + memory."""

    code: str
    stdout: str
    stderr: str
    error: str | None
    source: str  # "preamble" (trusted host cell) | "agent"
    outputs: list[dict[str, Any]] = field(default_factory=list)  # nbformat-ready cell outputs


@dataclass(frozen=True)
class CodingResult:
    """A finished session: the scored result + the full cell transcript."""

    result: ExperimentResult
    cells: list[Cell]


def _build_tool(key: str) -> ToolSpec:
    tool = _PROMPTS["tools"][key]
    spec: dict[str, Any] = {"type": "object", "properties": {}, "required": []}
    if key == "run_cell":
        spec["properties"] = {"code": {"type": "string", "description": tool["code"]}}
        spec["required"] = ["code"]
    return ToolSpec(name=tool["name"], description=tool["description"], parameters=spec)


RUN_CELL = _build_tool("run_cell")
FINISH = _build_tool("finish")


class CodingAgent:
    """Runs one experiment as a cell-by-cell session on a stateful kernel."""

    def __init__(
        self,
        client: LLMClient,
        kernel: StatefulKernel,
        *,
        metric: str,
        deadline_seconds: float = 300.0,
        cell_timeout: float = 120.0,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        install: bool = True,
        max_cells: int = 50,  # safety backstop against a non-finishing loop; time is the real bound
        context_budget_chars: int = 400_000,  # prompt cap; oldest observations elide first
    ) -> None:
        self._client = client
        self._kernel = kernel
        self._metric = metric
        self._deadline_seconds = deadline_seconds
        self._cell_timeout = cell_timeout
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._install = install
        self._max_cells = max_cells
        self._context_budget_chars = context_budget_chars

    def run(
        self,
        *,
        dataset: TabularDataset,
        brief: str,
        experiment_id: str,
        starting_code: str | None = None,
        starting_score: float | None = None,
    ) -> CodingResult:
        from iterate.core.proposer import summarize_dataset

        cells: list[Cell] = []
        self._kernel.start(codegen.build_inputs(dataset))
        try:
            pre = codegen.session_preamble()
            pre_result = self._kernel.run_cell(pre, timeout=self._cell_timeout)
            cells.append(_cell(pre, pre_result, "preamble"))

            messages = _build_messages(
                data_summary=summarize_dataset(dataset),
                metric=self._metric,
                direction=direction(self._metric),
                brief=brief,
                preamble_output=_observation(pre_result),
                starting_code=starting_code,
                starting_score=starting_score,
            )
            self._drive(messages, cells, n_test=dataset.n_test)

            preds = self._kernel.read_output(codegen.PREDICTIONS_CSV)
            result = codegen.score_predictions(
                dataset, preds, metric=self._metric, experiment_id=experiment_id
            )
            logs = _tail("\n".join(c.stdout for c in cells if c.stdout))
            return CodingResult(result=result.model_copy(update={"logs": logs or None}), cells=cells)
        finally:
            self._kernel.close()

    def _drive(self, messages: list[Message], cells: list[Cell], *, n_test: int) -> None:
        """The tool loop: run_cell → feed output back → repeat, until a VERIFIED finish
        or the deadline. The deadline charges KERNEL-EXECUTION seconds only — LLM
        latency is free, so a slow local model gets the same working budget as a fast
        cloud one (`max_cells` backstops a runaway loop). `finish` is rejected unless
        valid predictions are actually written; a first VALID finish with most of the
        budget unspent is met once with an improve nudge (the next is accepted)."""
        work = 0.0  # kernel-execution seconds spent — the budget the deadline bounds
        warned = False
        improve_nudged = False
        recent_cells: list[str] = []  # normalized code of the last few executed cells
        error_sig_counts: dict[str, int] = {}  # how often each error signature has recurred
        for _ in range(self._max_cells):
            if work >= self._deadline_seconds:
                break
            if work >= 0.8 * self._deadline_seconds and not warned:
                messages.append(
                    Message(
                        role="user",
                        content=_PROMPTS["budget_nudge"].format(
                            predictions_csv=codegen.PREDICTIONS_CSV
                        ),
                    )
                )
                warned = True
            _fit_context(messages, self._context_budget_chars)
            response = self._client.chat(
                messages, tools=[RUN_CELL, FINISH], temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
            call = next(
                (c for c in response.tool_calls if c.name in (RUN_CELL.name, FINISH.name)), None
            )
            if call is None:
                messages.append(Message(role="user", content=_PROMPTS["retry_nudge"]))
                continue
            messages.append(Message(role="assistant", tool_calls=[call]))
            if call.name == FINISH.name:
                # Verified finish: only end if valid predictions were actually written.
                reason = _validate_predictions(
                    self._kernel.read_output(codegen.PREDICTIONS_CSV), n_test
                )
                if reason is None:
                    # Improve nudge (once): a first valid submission with most of the
                    # working budget unspent should be iterated on, not banked. The
                    # next finish is always accepted — a nudge, not a wall.
                    if not improve_nudged and work < 0.5 * self._deadline_seconds:
                        improve_nudged = True
                        messages.append(
                            _tool_msg(
                                call,
                                _PROMPTS["improve_nudge"].format(
                                    predictions_csv=codegen.PREDICTIONS_CSV
                                ),
                            )
                        )
                        continue
                    messages.append(_tool_msg(call, "predictions verified; ending session"))
                    return
                messages.append(
                    _tool_msg(
                        call,
                        _PROMPTS["finish_rejected"].format(
                            reason=reason, predictions_csv=codegen.PREDICTIONS_CSV
                        ),
                    )
                )
                continue
            code = call.arguments.get("code")
            if not isinstance(code, str) or not code.strip():
                messages.append(_tool_msg(call, "error: run_cell needs non-empty 'code'"))
                continue
            # Repeated-cell breaker: a weak model can lock into re-running an identical
            # cell (or cycling a few) whose output never changes — burning turns for no
            # new evidence. Refuse the duplicate (free, no kernel time) and tell it to
            # act differently. Window, not just last cell, because the loop can cycle.
            normalized = _normalize_code(code)
            if normalized in recent_cells:
                messages.append(_tool_msg(call, _PROMPTS["repeat_rejected"]))
                continue
            recent_cells.append(normalized)
            del recent_cells[:-_REPEAT_WINDOW]
            cell_result, note, spent = self._run_with_autoinstall(code)
            work += spent
            cells.append(_cell(code, cell_result, "agent"))
            obs = _observation(cell_result)
            if note:
                obs = note + "\n\n" + obs
            # Same-error breaker: the repeated-cell breaker only catches IDENTICAL code;
            # a weak model also thrashes by re-hitting the SAME error with cosmetically
            # different cells (e.g. swapping the encoder while the real cause — a string
            # column reaching a numeric step — is unchanged). When an error signature
            # recurs, escalate with a nudge that names it and forbids cosmetic retries.
            if cell_result.error:
                sig = _error_signature(cell_result.error)
                error_sig_counts[sig] = error_sig_counts.get(sig, 0) + 1
                if error_sig_counts[sig] >= _SAME_ERROR_LIMIT:
                    obs = _PROMPTS["same_error_nudge"].format(error=sig) + "\n\n" + obs
            if not cell_result.timed_out:
                namespace = self._kernel.namespace_summary()
                if namespace:
                    obs += "\n\nVariables defined now (build on these; don't re-import or re-load):\n"
                    obs += namespace
            messages.append(_tool_msg(call, obs))

    def _run_with_autoinstall(self, code: str) -> tuple[CellResult, str | None, float]:
        """Run a cell; if it fails on a missing module and install is on, install it
        and re-run once. Returns (result, note-for-the-agent, kernel-exec seconds).
        The note makes the install VISIBLE — especially a failed one, so the agent
        pivots to another library instead of retrying an import the environment can
        never satisfy. Install subprocess time is harness overhead, not charged.

        Each run is prefixed with RESET_INPUTS so the canonical X_train/y_train/
        X_holdout are pristine at the top of every cell, no matter what a prior cell
        did to them in place."""
        runnable = codegen.RESET_INPUTS + code
        t0 = time.monotonic()
        result = self._kernel.run_cell(runnable, timeout=self._cell_timeout)
        spent = time.monotonic() - t0
        if not self._install or not result.error:
            return result, None, spent
        missing = _missing_module(result.error)
        if missing is None:
            return result, None, spent
        package = codegen.package_for_import(missing)
        error_log = self._kernel.install([package])
        if error_log:
            note = (
                f"(auto-install of {package!r} FAILED: {error_log.strip()[-300:]} — do not "
                "retry this import; switch to a library that is already available)"
            )
            return result, note, spent
        t1 = time.monotonic()
        retried = self._kernel.run_cell(runnable, timeout=self._cell_timeout)
        spent += time.monotonic() - t1
        return retried, f"({package!r} was auto-installed and the cell re-ran)", spent


def _build_messages(
    *,
    data_summary: str,
    metric: str,
    direction: str,
    brief: str,
    preamble_output: str,
    starting_code: str | None = None,
    starting_score: float | None = None,
) -> list[Message]:
    system = _PROMPTS["system"].format(
        metric=metric,
        direction=direction,
        predictions_csv=codegen.PREDICTIONS_CSV,
    )
    if starting_code and starting_code.strip():
        score = f" (scored {metric}={starting_score:.4f})" if starting_score is not None else ""
        # Built by concatenation, not str.format — the code may contain braces.
        starting_point = (
            f"\nBEST APPROACH SO FAR{score} — start from this and apply the brief's "
            "change; do NOT rebuild from scratch:\n```python\n"
            + starting_code.strip()
            + "\n```\n"
        )
    else:
        starting_point = ""
    user = _PROMPTS["user_template"].format(
        data_summary=data_summary,
        brief=brief or "(no brief — choose a strong first approach yourself)",
        metric=metric,
        predictions_csv=codegen.PREDICTIONS_CSV,
        preamble_output=preamble_output,
        starting_point=starting_point,
    )
    return [Message(role="system", content=system), Message(role="user", content=user)]


_ELIDED = "(this earlier cell's output was elided to fit the context window — rely on more recent cells and the variables list)"


def _fit_context(messages: list[Message], budget_chars: int) -> None:
    """Keep the prompt under `budget_chars` by eliding the OLDEST cell observations
    first (in place). The system prompt, the task message, and the two newest
    observations are never touched — the model always keeps its rules, the brief,
    and its recent state; only stale middle output gives way. Without this, a long
    session overflows the backend's context and the FRONT (system prompt + tool
    schema) is what gets silently truncated."""

    def size() -> int:
        total = 0
        for m in messages:
            total += len(m.content or "")
            for tc in m.tool_calls or []:
                total += sum(len(str(v)) for v in tc.arguments.values())
        return total

    if size() <= budget_chars:
        return
    observations = [i for i, m in enumerate(messages) if m.role == "tool"]
    for i in observations[:-2]:  # oldest first; never the two newest
        content = messages[i].content or ""
        if content == _ELIDED or len(content) <= len(_ELIDED):
            continue
        messages[i] = messages[i].model_copy(update={"content": _ELIDED})
        if size() <= budget_chars:
            return


def _tool_msg(call: Any, content: str) -> Message:
    # name (Ollama) + tool_call_id (OpenAI) so the result threads back on either backend.
    return Message(role="tool", name=call.name, tool_call_id=call.id, content=content)


def _observation(result: CellResult) -> str:
    parts = []
    if result.stdout.strip():
        parts.append(result.stdout.strip())
    if result.error:
        parts.append(f"ERROR:\n{result.error.strip()}")
    elif result.stderr.strip():
        parts.append(result.stderr.strip())
    if result.timed_out:
        parts.append("(cell timed out)")
    return _tail("\n".join(parts)) or "(no output)"


def _cell(code: str, result: CellResult, source: str) -> Cell:
    return Cell(
        code=code, stdout=result.stdout, stderr=result.stderr, error=result.error,
        source=source, outputs=list(result.outputs),
    )


def _tail(text: str, limit: int = _OBSERVATION_TAIL) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return "...(truncated)\n" + text[-limit:]


__all__ = ["FINISH", "RUN_CELL", "Cell", "CodingAgent", "CodingAgentError", "CodingResult"]
