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

import ast
import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from iterate.core import codegen
from iterate.core.scoring import direction, task_for_metric
from iterate.prompts import PROMPTS
from iterate.schemas.llm import Message, ToolSpec

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

    from iterate.adapters.compute.kernel import CellResult, StatefulKernel
    from iterate.adapters.data.tabular import TabularDataset
    from iterate.llm.base import LLMClient
    from iterate.schemas.experiment import ExperimentResult


class CodingAgentError(RuntimeError):
    """The coding session failed irrecoverably (e.g. the LLM backend errored)."""


log = logging.getLogger(__name__)

_PROMPTS = PROMPTS["coder"]
# Feed the agent the FULL cell output (EDA tables, value_counts, etc.) — modern
# context windows are large and that output is what it reasons from. Only a very
# high ceiling remains, to bound a pathological dump (e.g. printing the whole frame).
_OBSERVATION_TAIL = 20000
_MISSING_MODULE = re.compile(r"No module named ['\"]([\w.]+)['\"]")
_REPEAT_WINDOW = 6  # how many recent executed cells the repeated-cell breaker remembers
_SAME_ERROR_LIMIT = 3  # recurrences of one error signature before escalating the nudge
# Consecutive errored cells (no success in between) before the session is ended
# early. The repeat/same-error breakers catch identical code and identical error
# signatures; a weak model can also thrash with a DIFFERENT typo each cell (live:
# 'X_holdut', a truncated cell, a bad column name — 32 cells, 8s of kernel time).
# Ending the session hands over to the submission guarantee, which banks a floor.
_MAX_CONSECUTIVE_ERRORS = 6


def _error_signature(error: str) -> str:
    """A stable one-line fingerprint of an error for the same-error breaker: the last
    non-empty traceback line (the `ExceptionType: message`), so two cells that fail
    the same way collapse to one signature even when their code differs."""
    lines = [ln.strip() for ln in (error or "").splitlines() if ln.strip()]
    return lines[-1][:160] if lines else ""


# SyntaxError messages that mean the CELL WAS CUT OFF mid-generation, not that the
# model wrote wrong syntax — executing a chopped cell wastes a turn on a misleading
# traceback (live: cells ending "Xa_cat_" / "class_weight='balanced')." died as
# 'unexpected EOF while parsing').
_TRUNCATION_SIGNS = (
    "unexpected EOF",
    "was never closed",
    "unterminated string literal",
    "unterminated triple-quoted string",
)


def _looks_truncated(code: str) -> str | None:
    """The truncation reason when a cell reads as cut off mid-token, else None.
    Deliberately narrow: ordinary syntax errors still execute so the model sees the
    real traceback, and IPython magics (which ast cannot parse) pass through."""
    try:
        ast.parse(code)
    except SyntaxError as exc:
        message = str(exc)
        if any(sign in message for sign in _TRUNCATION_SIGNS):
            return message.splitlines()[0][:120]
    return None


def _carried_lines(starting_code: str | None) -> frozenset[str]:
    """The carried best's code, as a set of stripped non-comment lines — the
    reference for what counts as NEW code in this session."""
    if not starting_code:
        return frozenset()
    return frozenset(
        line.strip()
        for line in starting_code.splitlines()
        if line.strip() and not line.strip().startswith("#")
    )


def lever_executed(
    cells: list[Cell], markers: tuple[str, ...], starting_code: str | None
) -> bool:
    """True when the briefed lever's markers appear on a NEW line of a SUCCESSFULLY
    executed agent cell — the post-session verdict on whether the commissioned lever
    was actually measured (an errored attempt does not count as measured)."""
    successful = [c for c in cells if not c.error]
    return _markers_present(successful, markers, _carried_lines(starting_code))


def _markers_present(
    cells: list[Cell], markers: tuple[str, ...], carried: frozenset[str] = frozenset()
) -> bool:
    """True when any lever-class marker appears on a NEW agent-cell code line.

    Diff-scoped on purpose: sessions rebuild the carried best, so its code (with
    the PREVIOUS levers' markers, e.g. an inherited threshold write) is present in
    every session — scanning the whole corpus let the gate false-pass while the
    briefed lever was never fit (live: class_weight briefed, never coded, gate
    silent because the carried pipeline contained the string 'threshold'). Only
    lines not byte-copied from the carried code count as evidence of the lever."""
    new_lines = [
        line.strip().lower()
        for c in cells
        if c.source == "agent"
        for line in c.code.splitlines()
        if line.strip() and not line.strip().startswith("#") and line.strip() not in carried
    ]
    # Same neutralization as the supervisor's marker matching: the model-swap
    # marker "gradientboosting" must not match inside the default estimator's
    # "histgradientboosting" (rebuilt in every session).
    corpus = "\n".join(new_lines).replace("histgradientboosting", "hgb")
    return any(m in corpus for m in markers)


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
    lines = raw.splitlines()
    count = len(lines)
    if count != n_test:
        return f"expected {n_test} predictions (one per holdout row), found {count}"
    # An index column rode into the file (to_csv without index=False): every line
    # reads "i,value" with i counting 0..n-1. Passed the count check in a live run,
    # then scoring choked on "could not convert string to float: '0,0'" — the one
    # FAILED iteration in 100+. Catch it here so the coder fixes it in-session.
    first_fields = [line.split(",", 1)[0] for line in lines[:50]]
    if all("," in line for line in lines) and first_fields == [
        str(i) for i in range(len(first_fields))
    ]:
        return (
            "each line must be ONE value, but the file has an index column "
            "(lines look like '0,1'); rewrite with .to_csv(..., index=False, header=False)"
        )
    return None


@dataclass(frozen=True)
class Cell:
    """One cell of the session — for the notebook deliverable + memory."""

    code: str
    stdout: str
    stderr: str
    error: str | None
    source: str  # "preamble" (trusted host cell) | "agent" | "fallback" (host floor submit)
    outputs: list[dict[str, Any]] = field(default_factory=list)  # nbformat-ready cell outputs
    thinking: str | None = None  # the model's reasoning that produced this cell (think mode)


@dataclass(frozen=True)
class CodingResult:
    """A finished session: the scored result + the full cell transcript."""

    result: ExperimentResult
    cells: list[Cell]
    # sha256 of the submitted predictions bytes — the loop hands the best's digest
    # to the NEXT session so a byte-identical re-submission can be caught in-session.
    predictions_sha256: str | None = None


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
        wall_ceiling_seconds: float = 1800.0,  # hard wall-clock bound on one session
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
        # The kernel-time deadline deliberately does NOT charge LLM latency (a slow
        # local model gets the same working budget as a fast cloud one) — but that
        # leaves a thrashing session unbounded in wall-clock: tiny errored cells
        # spend almost no kernel time, so max_cells is the only stop and each cell
        # costs minutes of model latency. This ceiling bounds the real-world time
        # one experiment can consume; generous enough that healthy sessions (even
        # think-mode ones) never feel it.
        self._wall_ceiling_seconds = wall_ceiling_seconds

    def run(
        self,
        *,
        dataset: TabularDataset,
        brief: str,
        experiment_id: str,
        starting_code: str | None = None,
        starting_score: float | None = None,
        brief_markers: Sequence[str] | None = None,
        seen_digests: Collection[str] | None = None,
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
            self._drive(
                messages, cells, n_test=dataset.n_test, experiment_id=experiment_id,
                brief_markers=tuple(brief_markers or ()),
                seen_digests=frozenset(seen_digests or ()),
                carried=_carried_lines(starting_code),
            )

            preds = self._kernel.read_output(codegen.PREDICTIONS_CSV)
            if _validate_predictions(preds, dataset.n_test) is not None:
                self._bank_floor(
                    cells, starting_code=starting_code, n_test=dataset.n_test,
                    experiment_id=experiment_id,
                )
                preds = self._kernel.read_output(codegen.PREDICTIONS_CSV)
            result = codegen.score_predictions(
                dataset, preds, metric=self._metric, experiment_id=experiment_id
            )
            logs = _tail("\n".join(c.stdout for c in cells if c.stdout))
            return CodingResult(
                result=result.model_copy(update={"logs": logs or None}),
                cells=cells,
                predictions_sha256=hashlib.sha256(preds).hexdigest() if preds else None,
            )
        finally:
            self._kernel.close()

    def _drive(
        self,
        messages: list[Message],
        cells: list[Cell],
        *,
        n_test: int,
        experiment_id: str,
        brief_markers: tuple[str, ...] = (),
        seen_digests: frozenset[str] = frozenset(),
        carried: frozenset[str] = frozenset(),
    ) -> None:
        """The tool loop: run_cell → feed output back → repeat, until a VERIFIED finish
        or the deadline. The deadline charges KERNEL-EXECUTION seconds only — LLM
        latency is free, so a slow local model gets the same working budget as a fast
        cloud one (`max_cells` backstops a runaway loop). `finish` is rejected unless
        valid predictions are actually written; a first VALID finish with most of the
        budget unspent is met once with an improve nudge (the next is accepted).

        Two no-op gates guard a valid finish, each firing at most once (nudges, not
        walls — the next finish is accepted). Live runs produced six byte-identical
        submissions in a row: briefed levers never reached a single code cell, and
        sessions ended by deliberately re-writing the carried best's predictions.
        The lever gate fires when none of the brief's lever-class markers appear in
        any executed cell; the identical gate fires when the submitted bytes hash to
        ANY earlier experiment's submission (not just the best — a later run wasted
        4 of 10 iterations on sibling duplicates the best-only check could not see)."""
        work = 0.0  # kernel-execution seconds spent — the budget the deadline bounds
        warned = False
        improve_nudged = False
        lever_nudged = False
        identical_nudged = False
        recent_cells: list[str] = []  # normalized code of the last few executed cells
        error_sig_counts: dict[str, int] = {}  # how often each error signature has recurred
        consecutive_errors = 0  # errored cells since the last successful one
        truncation_rejections = 0  # consecutive truncated cells rejected unexecuted
        session_start = time.monotonic()
        for _ in range(self._max_cells):
            if work >= self._deadline_seconds:
                break
            if time.monotonic() - session_start >= self._wall_ceiling_seconds:
                log.info(
                    "coder[%s]: wall-clock ceiling (%.0fs) reached after %d cells; ending session",
                    experiment_id, self._wall_ceiling_seconds,
                    sum(1 for c in cells if c.source == "agent"),
                )
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
                preds = self._kernel.read_output(codegen.PREDICTIONS_CSV)
                reason = _validate_predictions(preds, n_test)
                if reason is None:
                    # Lever gate (once): the briefed change never reached a single
                    # executed cell — a submission without it is not this experiment.
                    if (
                        brief_markers
                        and not lever_nudged
                        and not _markers_present(cells, brief_markers, carried)
                    ):
                        lever_nudged = True
                        log.info(
                            "coder[%s]: lever gate fired — briefed change absent from all cells",
                            experiment_id,
                        )
                        messages.append(
                            _tool_msg(
                                call,
                                _PROMPTS["lever_missing_nudge"].format(
                                    lever=", ".join(brief_markers[:3]),
                                    predictions_csv=codegen.PREDICTIONS_CSV,
                                ),
                            )
                        )
                        continue
                    # Identical gate (once): the submission is byte-identical to an
                    # earlier experiment's — zero information gained this iteration.
                    if (
                        seen_digests
                        and not identical_nudged
                        and preds is not None
                        and hashlib.sha256(preds).hexdigest() in seen_digests
                    ):
                        identical_nudged = True
                        log.info(
                            "coder[%s]: identical gate fired — submission matches an earlier experiment",
                            experiment_id,
                        )
                        messages.append(
                            _tool_msg(
                                call,
                                _PROMPTS["identical_submission_nudge"].format(
                                    predictions_csv=codegen.PREDICTIONS_CSV
                                ),
                            )
                        )
                        continue
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
            # Truncated-cell guard: a mid-token cutoff is a transport artifact, not
            # the model's intent — reject it unexecuted (free) so the retry sends
            # the complete cell. Capped so a persistently-truncating model still
            # falls through to real execution and the error breakers.
            truncation = _looks_truncated(code)
            if truncation and truncation_rejections < 3:
                truncation_rejections += 1
                messages.append(
                    _tool_msg(call, _PROMPTS["truncated_cell_nudge"].format(error=truncation))
                )
                continue
            truncation_rejections = 0
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
            cells.append(_cell(code, cell_result, "agent", thinking=response.thinking))
            # Per-cell progress: a long local-model session is otherwise silent for
            # minutes. One line per executed cell shows it IS moving — which cell, how
            # it landed, and how much of the kernel-time budget is gone.
            n_agent_cells = sum(1 for c in cells if c.source == "agent")
            if cell_result.timed_out:
                status = "timed out"
            elif cell_result.error:
                status = f"error: {_error_signature(cell_result.error)}"
            else:
                status = "ok"
            log.info(
                "coder[%s]: cell %d %s (%.1fs, %.0f/%.0fs budget)",
                experiment_id, n_agent_cells, status, spent, work, self._deadline_seconds,
            )
            obs = _observation(cell_result)
            if note:
                obs = note + "\n\n" + obs
            # Same-error breaker: the repeated-cell breaker only catches IDENTICAL code;
            # a weak model also thrashes by re-hitting the SAME error with cosmetically
            # different cells (e.g. swapping the encoder while the real cause — a string
            # column reaching a numeric step — is unchanged). When an error signature
            # recurs, escalate with a nudge that names it and forbids cosmetic retries.
            if cell_result.error:
                consecutive_errors += 1
                if consecutive_errors >= _MAX_CONSECUTIVE_ERRORS:
                    # A DIFFERENT error each time evades the repeat and same-error
                    # breakers; this many in a row means the session is lost. End it
                    # — the submission guarantee banks a floor.
                    log.info(
                        "coder[%s]: %d consecutive errored cells; ending session early",
                        experiment_id, consecutive_errors,
                    )
                    return
                sig = _error_signature(cell_result.error)
                error_sig_counts[sig] = error_sig_counts.get(sig, 0) + 1
                if error_sig_counts[sig] >= _SAME_ERROR_LIMIT:
                    obs = _PROMPTS["same_error_nudge"].format(error=sig) + "\n\n" + obs
            else:
                consecutive_errors = 0
            if not cell_result.timed_out:
                namespace = self._kernel.namespace_summary()
                if namespace:
                    obs += "\n\nVariables defined now (build on these; don't re-import or re-load):\n"
                    obs += namespace
            messages.append(_tool_msg(call, obs))

    def _bank_floor(
        self, cells: list[Cell], *, starting_code: str | None, n_test: int, experiment_id: str
    ) -> None:
        """The submission guarantee: the session ended with NO valid predictions (a
        heavy lever ate the whole budget, or the approach was abandoned), so bank a
        floor instead of recording a total loss. Host-authored and run AFTER the
        agent's kernel budget: first the carried-forward best code (reproduces
        roughly the best score), else a canned deterministic baseline. The cell is
        labeled "fallback" so the notebook shows the harness banked it, not the
        coder — and the coder's prompt never mentions it, so the pressure to submit
        stays on the model."""
        candidates: list[tuple[str, str]] = []
        if starting_code and starting_code.strip():
            candidates.append(("carried-forward best", starting_code.strip()))
        candidates.append(
            ("canned baseline", codegen.fallback_baseline(task_for_metric(self._metric)))
        )
        # A dead session's namespace is untrusted: a stale or mutated variable could
        # steer the carried code into a silently different pipeline. Wipe it and
        # reload the pristine inputs first (plumbing, not recorded as cells).
        self._kernel.run_cell("%reset -f", timeout=self._cell_timeout)
        self._kernel.run_cell(codegen.session_preamble(), timeout=self._cell_timeout)
        for label, code in candidates:
            cell_result = self._kernel.run_cell(
                codegen.RESET_INPUTS + code, timeout=self._cell_timeout
            )
            cells.append(_cell(code, cell_result, "fallback"))
            if _validate_predictions(self._kernel.read_output(codegen.PREDICTIONS_CSV), n_test) is None:
                log.info(
                    "coder[%s]: session ended without a valid submission; banked the %s as a floor",
                    experiment_id, label,
                )
                return
        log.warning(
            "coder[%s]: fallback submission failed too; recording the iteration as a failure",
            experiment_id,
        )

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


def _cell(code: str, result: CellResult, source: str, *, thinking: str | None = None) -> Cell:
    return Cell(
        code=code, stdout=result.stdout, stderr=result.stderr, error=result.error,
        source=source, outputs=list(result.outputs), thinking=thinking,
    )


def _tail(text: str, limit: int = _OBSERVATION_TAIL) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return "...(truncated)\n" + text[-limit:]


__all__ = [
    "FINISH",
    "RUN_CELL",
    "Cell",
    "CodingAgent",
    "CodingAgentError",
    "CodingResult",
    "lever_executed",
]
