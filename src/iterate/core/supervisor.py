"""The Supervisor — the strategist that briefs the coding agent across experiments.

It reads the full run history (what was tried, the components used, the scores),
COMPRESSES it into a short summary, and hands the coding agent a brief for the next
experiment: 1-2 lines of "what's been learned so far" plus a specific strategy to
try. It can also decide to stop.

For now the Supervisor is a single LLM that does this via its own tool
(`plan_next`); the summarization work is a natural candidate to graduate into a
dedicated Summarizer agent later — the tool boundary is the future agent boundary.
The coding agent never sees raw history; it only sees the Supervisor's brief, which
is what keeps each experiment's context tight and lifts a weaker coder.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from iterate.core import codegen
from iterate.core.scoring import direction
from iterate.prompts import PROMPTS
from iterate.schemas.llm import Message, ToolSpec

if TYPE_CHECKING:
    from iterate.llm.base import LLMClient
    from iterate.schemas.experiment import Experiment, ExperimentResult


class SupervisorError(RuntimeError):
    """The supervisor failed to return a usable plan after retries."""


log = logging.getLogger(__name__)

_PROMPTS = PROMPTS["supervisor"]
_HISTORY_LIMIT = 12  # most recent experiments shown


@dataclass(frozen=True)
class SupervisorDecision:
    """What to do next: stop, or run an experiment described by ``brief``."""

    stop: bool
    title: str  # short label for the leaderboard
    brief: str  # the instruction handed to the coding agent (summary + strategy)


def _build_tool() -> ToolSpec:
    tool = _PROMPTS["tool"]
    fields = tool["fields"]
    return ToolSpec(
        name=tool["name"],
        description=tool["description"],
        parameters={
            "type": "object",
            "properties": {
                "stop": {"type": "boolean", "description": fields["stop"]},
                "title": {"type": "string", "description": fields["title"]},
                "brief": {"type": "string", "description": fields["brief"]},
            },
            "required": ["stop", "brief"],
        },
    )


PLAN_NEXT = _build_tool()


class Supervisor:
    """Briefs the coding agent for the next experiment from run history."""

    def __init__(
        self,
        client: LLMClient,
        *,
        metric: str,
        temperature: float = 0.4,
        max_tokens: int = 1024,
        max_retries: int = 1,
    ) -> None:
        self._client = client
        self._metric = metric
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_retries = max_retries

    def decide(
        self,
        *,
        data_summary: str,
        baseline: ExperimentResult,
        history: list[Experiment],
        carried_best: Experiment | None = None,
    ) -> SupervisorDecision:
        """Plan the next experiment. ``carried_best`` is the loop's CURRENT best —
        the experiment whose code the coder will actually receive as its starting
        point — so the grounded "so far:" slot describes exactly that pipeline and
        never a cross-run best the coder does not hold."""
        if baseline.metrics is None:
            raise SupervisorError("baseline has no metrics")
        messages = _build_messages(
            data_summary=data_summary,
            metric=self._metric,
            direction=direction(self._metric),
            score=baseline.metrics.primary_value,
            history=history,
        )
        detail = ""
        rebrief_nudged = False
        dup_class_nudged = False
        banked_nudged = False
        lost_nudged = False
        lint_nudged = False
        for attempt in range(self._max_retries + 1):
            last_attempt = attempt == self._max_retries
            try:
                response = self._client.chat(
                    messages, tools=[PLAN_NEXT], temperature=self._temperature,
                    max_tokens=self._max_tokens,
                )
            except Exception as exc:  # the backend is an I/O boundary; never fatal here
                # A backend reject (e.g. groq's tool_use_failed when a weak model emits
                # `stop` as the string "false") must NOT crash the whole run: nudge a
                # well-formed retry, and if it never lands, fall through to a graceful
                # SupervisorError that the agent loop records as a proposer failure.
                detail = f"backend error: {type(exc).__name__}: {exc}"
                messages.append(Message(role="user", content=_PROMPTS["tool_error_nudge"]))
                continue
            call = next((c for c in response.tool_calls if c.name == PLAN_NEXT.name), None)
            if call is not None:
                decision = _to_decision(call.arguments)
                if decision.stop or not history:
                    return decision  # experiment 1 has no so-far slot to ground
                # Guards, in order. Each violation gets ONE corrective retry; if the
                # retry violates again (or the budget is spent), the harness composes
                # a deterministic fallback brief from an untried lever class instead
                # of accepting the violation — a live run's guard fired, the model
                # re-emitted the same lever on the retry, and the known re-commission
                # was accepted (detection without conversion).
                violation: tuple[str, str, bool] | None = None  # (log reason, nudge, seen)
                if _is_baseline_rebrief(decision.title, decision.brief):
                    violation = (
                        f"baseline re-brief ({decision.title!r})",
                        _PROMPTS["baseline_rebrief_nudge"],
                        rebrief_nudged,
                    )
                    rebrief_nudged = True
                elif _rebriefs_a_just_duplicated_class(decision.brief, history):
                    violation = (
                        f"re-brief of the just-duplicated class ({decision.title!r})",
                        _PROMPTS["duplicate_class_rebrief_nudge"],
                        dup_class_nudged,
                    )
                    dup_class_nudged = True
                elif banked_reason := _recommissions_banked_work(decision.brief, carried_best):
                    violation = (
                        f"banked-work re-brief — {banked_reason}",
                        _PROMPTS["banked_rebrief_nudge"].format(reason=banked_reason),
                        banked_nudged,
                    )
                    banked_nudged = True
                elif lost_reason := _recommissions_a_measured_lost_technique(
                    decision.brief, history, carried_best
                ):
                    violation = (
                        f"measured-lost re-brief — {lost_reason}",
                        _PROMPTS["banked_rebrief_nudge"].format(reason=lost_reason),
                        lost_nudged,
                    )
                    lost_nudged = True
                elif lint_reason := _move_lint(
                    decision.brief,
                    baseline_score=baseline.metrics.primary_value,
                    carried=carried_best,
                ):
                    violation = (
                        f"ill-formed move — {lint_reason}",
                        _PROMPTS["move_lint_nudge"].format(reason=lint_reason),
                        lint_nudged,
                    )
                    lint_nudged = True
                if violation is not None:
                    reason, nudge, seen = violation
                    fallback = _fallback_move(history) if seen or last_attempt else None
                    if fallback is not None:
                        title, move = fallback
                        log.info(
                            "supervisor: %s persisted; falling back to an untried lever (%s)",
                            reason, title,
                        )
                        decision = SupervisorDecision(stop=False, title=title, brief=move)
                    elif not (seen or last_attempt):
                        log.info("supervisor: rejected a %s", reason)
                        messages.append(Message(role="user", content=nudge))
                        continue
                    # else: no untried class remains — accept the model's brief.
                return SupervisorDecision(
                    stop=False,
                    title=decision.title,
                    brief=_grounded_brief(
                        decision.brief,
                        metric=self._metric,
                        baseline_score=baseline.metrics.primary_value,
                        carried=carried_best,
                        dead_ends=_dead_ends(history),
                    ),
                )
            detail = "model replied without calling plan_next"
            messages.append(Message(role="user", content=_PROMPTS["retry_nudge"]))
        raise SupervisorError(f"no plan after {self._max_retries + 1} attempt(s): {detail}")


_NEXT_SLOT = re.compile(r"\bnext\s*:", re.IGNORECASE)
# A leading so-far-style claim (through its first sentence end), for stripping when
# the model wrote no "next:" marker — the hallucination-prone part must not survive.
_SO_FAR_CLAIM = re.compile(r"(?is)^\s*so\s*far\s*:.*?(?:\.\s+|\n+|$)")
# The APPLIED decision threshold: a proba comparison against a literal or a
# variable, resolved from the final submit path only. The old any-line literal scan
# extracted a sweep value (0.4) instead of the banked threshold (0.2802) in a live
# run, and the corrupted fact poisoned every subsequent brief.
_THRESHOLD_LITERAL = re.compile(r">=?\s*(0?\.\d+)")
_THRESHOLD_VARIABLE = re.compile(r">=?\s*([A-Za-z_]\w*)")


def _grounded_brief(
    llm_brief: str,
    *,
    metric: str,
    baseline_score: float,
    carried: Experiment | None,
    dead_ends: str = "",
) -> str:
    """The brief with its "so far:" slot composed by CODE from the loop's real state.

    Live runs showed the supervisor hallucinating the so-far facts — scores that
    never happened on the holdout, techniques absent from the carried code — which
    steered the coder against a false model of the run. So the facts come from the
    record, and the LLM contributes only the "next:" move. Anything it wrote before
    "next:" is discarded; if it wrote no marker at all, a leading so-far-style claim
    is stripped before the rest is wrapped as the move. ``dead_ends`` (one compact
    code-composed sentence) trails the brief so the CODER inherits which ideas
    already failed — without it, sessions re-probed the same pet feature in 8 of 10
    notebooks."""
    match = _NEXT_SLOT.search(llm_brief)
    if match:
        move = llm_brief[match.start() :].strip()
    else:
        cleaned = _SO_FAR_CLAIM.sub("", llm_brief, count=1).strip()
        move = f"next: {cleaned or llm_brief.strip()}"
    tail = f" {dead_ends}" if dead_ends else ""
    return f"{_so_far(metric, baseline_score, carried)} {move}{tail}"


def _so_far(metric: str, baseline_score: float, carried: Experiment | None) -> str:
    """One compact factual sentence about the pipeline the coder actually inherits.

    Grounded on the loop's carried best — the same experiment whose code becomes the
    coder's BEST APPROACH block — never on a cross-run or worse-than-baseline score,
    so the brief and the starting code can never contradict each other. Kept lean on
    purpose: an earlier, denser status line regressed the weak local model."""
    if carried is None or carried.result is None or carried.result.metrics is None:
        return (
            f"so far: baseline {metric}={baseline_score:.4f} is the bar to beat; "
            "no carried best yet."
        )
    code = carried.candidate.changes.get("code")
    parts: list[str] = []
    if isinstance(code, str):
        config = _carried_config(code)
        used = [c for c in _submit_components(code) if not config or c not in config][:3]
        if used:
            parts.append("used: " + ", ".join(used))
        if config:
            # The EXACT banked config — without it the supervisor re-briefed levers
            # already embodied in the best (class_weight already set, a grid already
            # searched, a threshold already banked): 4 of 5 residual duplicates.
            parts.append("final model: " + config)
        threshold = _carried_threshold(code)
        if threshold:
            parts.append(f"decision threshold {threshold} already applied")
        grid = _searched_grid(code)
        if grid:
            parts.append(f"already grid-searched: {grid}")
    note = f" ({'; '.join(parts)})" if parts else ""
    return (
        f"so far: best {metric}={carried.result.metrics.primary_value:.4f} via "
        f"'{_word_cut(carried.candidate.description.strip(), 60)}'{note}."
    )


_FINAL_ESTIMATOR = re.compile(r"([A-Z]\w*(?:Classifier|Regressor))\s*\(([^()]*)\)")
_GRID_LITERAL = re.compile(r"param_grid\s*=\s*\{([^}]*)\}", re.DOTALL)


_PREDICT_VAR = re.compile(r"(\w+)\s*\.\s*predict(?:_proba)?\s*\(")


def _carried_config(code: str) -> str | None:
    """The banked configuration the next brief must not re-search: the estimator
    that actually produced the final submission, with its explicit arguments.

    Resolved from the submit path, not the last constructor in the text — a live
    run's last-match extraction picked a probe's plain constructor while the banked
    model had class_weight='balanced', and the corrupted fact let the brief
    re-commission the already-applied lever (the coder even noted 'already using
    class_weight=balanced' and executed anyway). Method: find the model variable in
    the LAST predict call, take that variable's last constructor assignment; fall
    back to the last constructor when the variable cannot be resolved."""
    matches = list(_FINAL_ESTIMATOR.finditer(code))
    if not matches:
        return None
    chosen = matches[-1]
    predict_vars = _PREDICT_VAR.findall(code)
    if predict_vars:
        var = predict_vars[-1]
        assign = re.compile(rf"^\s*{re.escape(var)}\s*=")
        for match in reversed(matches):
            line_start = code.rfind("\n", 0, match.start()) + 1
            if assign.match(code[line_start : match.start()]):
                chosen = match
                break
    name, args = chosen.group(1), re.sub(r"\s+", " ", chosen.group(2).strip())
    if len(args) > 90:
        args = _word_cut(args, 90)
    return f"{name}({args})"


def _searched_grid(code: str) -> str | None:
    """The hyperparameter names of a grid already searched in the carried code, so
    the supervisor does not brief the same search again (observed: the exact
    lr x depth grid re-run twice)."""
    match = _GRID_LITERAL.search(code)
    if not match:
        return None
    keys = re.findall(r"['\"](\w+)['\"]\s*:", match.group(1))
    return "/".join(keys[:4]) if keys else None


def _word_cut(text: str, limit: int) -> str:
    """Truncate at a word boundary — mid-word chops shipped visibly broken lines
    into the deliverables' Hypothesis cells. An unbalanced open parenthesis is
    dropped whole (a clipped mid-parenthetical shipped in three briefs)."""
    if len(text) <= limit:
        return text
    cut = text[:limit]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    if cut.count("(") > cut.count(")") and "(" in cut:
        cut = cut[: cut.rfind("(")].rstrip()
    return cut


_DEAD_ENDS_LIMIT = 3
_DEAD_END_ITEM_CHARS = 90
# Generic ML-report words that carry no identity — kept out of the grouping key so
# "PowerTransformer application: no change" and "PowerTransformer on numerics: hurt"
# read as the SAME dead idea.
_DEAD_END_STOPWORDS = frozenset(
    ["score", "feature", "features", "improvement", "change", "model", "added", "adding", "caused", "likely", "without", "measurable", "decreased", "lowered", "applied", "application", "threshold"]
)
_DEAD_END_TOKEN = re.compile(r"[a-z_]{5,}")


def _idea_tokens(text: str) -> frozenset[str]:
    # split camelCase and snake_case into atomic words, so "tenure_monthly
    # interaction" groups with "Interaction feature (tenure * MonthlyCharges)"
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text).replace("_", " ").lower()
    return frozenset(
        t for t in _DEAD_END_TOKEN.findall(spaced) if t not in _DEAD_END_STOPWORDS
    )


# Phrasings that claim an API or parameter DOES NOT EXIST — when the run's own
# executed code used that parameter successfully, the claim is a hallucination and
# must not enter the dead-ends channel (live: "HGB lacks class_weight" banked while
# the prior session printed a successful class_weight fit).
_FALSE_API_CLAIM = (
    "lacks", "does not have", "doesn't have", "no such", "not supported",
    "does not accept", "doesn't accept", "does not support", "doesn't support",
)


def _holdout_score(exp: Experiment) -> float | None:
    if exp.result is None or not exp.result.succeeded or exp.result.metrics is None:
        return None
    return exp.result.metrics.primary_value


def _best_holdout(history: list[Experiment]) -> tuple[float, str] | None:
    """(best holdout score, direction) across the run, or None if nothing scored."""
    scores = [s for s in (_holdout_score(e) for e in history) if s is not None]
    if not scores:
        return None
    metrics = next(
        e.result.metrics for e in history
        if e.result is not None and e.result.metrics is not None
    )
    assert metrics is not None
    if metrics.direction == "minimize":
        return min(scores), "minimize"
    return max(scores), "maximize"


def _executed_ok_corpus(history: list[Experiment]) -> str:
    """All successfully executed agent-cell code across the run, lowercased."""
    parts: list[str] = []
    for exp in history:
        cells = exp.candidate.changes.get("cells")
        if not isinstance(cells, list):
            continue
        parts.extend(
            str(cell.get("code") or "")
            for cell in cells
            if isinstance(cell, dict) and cell.get("source") == "agent" and not cell.get("error")
        )
    return "\n".join(parts).lower()


def _dead_ends(history: list[Experiment]) -> str:
    """One compact sentence naming ideas already tried and rejected, composed from
    the digests' what_hurt across the run — the failure knowledge that otherwise
    never reaches the coder (only the supervisor reads digests). Deliberately lean
    (one line, {limit} items): an earlier, denser additive block regressed the weak
    model.

    Ranked by RECURRENCE, not recency: a live run recorded the coder's pet dead
    idea early, but so many distinct one-off failures followed that a most-recent
    window evicted it — and the re-probing resumed the moment it left the list.
    The ideas that keep failing are exactly the ones that must stick, so variant
    phrasings are grouped by shared distinctive tokens and groups ranked by how
    often they recur (latest phrasing shown)."""
    texts: dict[frozenset[str], list[str]] = {}  # group -> phrasings, in order
    counts: dict[frozenset[str], int] = {}  # group -> total occurrences
    last_seen: dict[frozenset[str], int] = {}  # group -> occurrence index, for ties
    seen = 0
    executed_ok = _executed_ok_corpus(history)
    best = _best_holdout(history)
    for exp in history:
        if exp.digest is None:
            continue
        entries = list(exp.digest.what_hurt)
        # Settled results: a "helped" claim from an experiment that LOST on the
        # holdout is a val-vs-holdout mirage, and it invited a re-commission in a
        # live run (a GB swap sold as 0.6593 that had stamped 0.6180). It enters
        # the same do-NOT-retry channel, labeled with the real stamp.
        score = _holdout_score(exp)
        if best is not None and score is not None:
            best_score, sense = best
            lost = score > best_score if sense == "minimize" else score < best_score
            if lost:
                # cut the CLAIM, never the settled reason — a capped cut applied to
                # the whole entry shipped "settled: its;" mid-sentence in a brief
                entries.extend(
                    _word_cut(helped.strip(), _DEAD_END_ITEM_CHARS)
                    + f" — settled: its holdout was {score:.4f}, not the best"
                    for helped in exp.digest.what_helped
                )
        for hurt in entries:
            # settled entries are pre-cut with their suffix protected; cap the rest
            text = (
                hurt.strip() if " — settled:" in hurt
                else _word_cut(hurt.strip(), _DEAD_END_ITEM_CHARS)
            )
            if not text:
                continue
            low = text.lower()
            if any(claim in low for claim in _FALSE_API_CLAIM) and any(
                token in executed_ok for token in _SNAKE_PARAM.findall(low)
            ):
                continue  # hallucinated "X does not exist" while the run used X
            seen += 1
            tokens = _idea_tokens(text)
            key = next((k for k in texts if len(k & tokens) >= 2), None)
            if key is None:
                key = tokens
                texts[key] = []
            counts[key] = counts.get(key, 0) + 1
            last_seen[key] = seen
            if text.lower() not in (t.lower() for t in texts[key]):
                texts[key].append(text)
    if not texts:
        return ""
    ranked = sorted(texts, key=lambda k: (-counts[k], -last_seen[k]))
    picks = [texts[k][-1] for k in ranked[:_DEAD_ENDS_LIMIT]]  # latest phrasing per idea
    return "Known dead ends this run (do NOT retry): " + "; ".join(picks) + "."


_IDENT = re.compile(r"[A-Za-z_]\w*")
_ASSIGN_TARGET = re.compile(r"\s*([A-Za-z_]\w*)\s*(?:\[[^\]]*\])?\s*=[^=]")
_FIT_MUTATION = re.compile(r"\s*([A-Za-z_]\w*)\s*\.\s*fit\b")


def _submit_path_code(code: str) -> str:
    """A backward slice of the carried code from its LAST predictions write: only
    lines that assign (or fit) something the submission transitively depends on.
    The whole-code component list credited null probes to the banked pipeline in a
    live run (PowerTransformer, measured a no-op and excluded from the submission,
    still showed in the 'used:' facts and steered the next brief)."""
    lines = code.splitlines()
    write_idx = next(
        (i for i in range(len(lines) - 1, -1, -1) if "predictions.csv" in lines[i]), None
    )
    if write_idx is None:
        return code
    needed = set(_IDENT.findall(lines[write_idx]))
    sliced = [lines[write_idx]]
    for i in range(write_idx - 1, -1, -1):
        line = lines[i]
        if not line.strip() or line.strip().startswith("#"):
            continue
        target = _ASSIGN_TARGET.match(line)
        fitted = _FIT_MUTATION.match(line)
        if (target and target.group(1) in needed) or (fitted and fitted.group(1) in needed):
            needed |= set(_IDENT.findall(line))
            sliced.append(line)
    return "\n".join(reversed(sliced))


def _submit_components(code: str) -> list[str]:
    """The components the final submission actually uses; whole-code fallback when
    the slice is unparseable (multi-line statements defeat line slicing)."""
    components = codegen.components_used(_submit_path_code(code))
    return components if components else codegen.components_used(code)


_PROBA_ASSIGN = re.compile(r"^\s*(\w+)\s*=.*predict_proba")


def _carried_threshold(code: str) -> str | None:
    """The decision threshold the winning code APPLIES, or None when it cannot be
    resolved with certainty — a wrong fact is worse than no fact (a live run's
    briefs carried '0.4 already applied' extracted from sweep code while the banked
    threshold was 0.2802). Resolution: take the LAST comparison against
    predict_proba output — same-line, or via a variable the coder assigned from
    predict_proba first (the split idiom `probs_h = model.predict_proba(..)` then
    `(probs_h > 0.4)` hid the banked threshold from the same-line-only check and
    disarmed the re-tune guard in a live run). A literal is the answer; a variable
    is followed to its last literal assignment; anything else resolves to None."""
    lines = code.splitlines()
    proba_vars = {
        m.group(1).lower() for line in lines if (m := _PROBA_ASSIGN.match(line))
    }
    for line in reversed(lines):
        low = line.lower()
        if ">" not in low:
            continue
        if "proba" not in low and not any(
            re.search(rf"\b{re.escape(v)}\b", low) for v in proba_vars
        ):
            continue
        literal = _THRESHOLD_LITERAL.search(low)
        if literal:
            return literal.group(1)
        variable = _THRESHOLD_VARIABLE.search(low)
        if variable:
            assign = re.compile(
                rf"^\s*{re.escape(variable.group(1))}\s*=\s*(0?\.\d+)\s*(?:#.*)?$"
            )
            for earlier in reversed(lines):
                match = assign.match(earlier.lower())
                if match:
                    return match.group(1)
        return None  # the LAST proba comparison decides; do not scan further back
    return None


def _coerce_bool(value: Any) -> bool:
    """A weak model often emits `stop` as the STRING "false"/"true" rather than a JSON
    boolean. `bool("false")` is True — the dangerous default that would silently STOP a
    run — so coerce textual booleans by meaning, not truthiness."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    return bool(value)


def _to_decision(args: dict[str, Any]) -> SupervisorDecision:
    stop = _coerce_bool(args.get("stop"))
    brief = str(args.get("brief") or "").strip()
    title = str(args.get("title") or "").strip() or (brief[:60] or "experiment")
    if not stop and not brief:
        raise SupervisorError("plan_next returned neither stop nor a brief")
    return SupervisorDecision(stop=stop, title=title, brief=brief)


def _build_messages(
    *, data_summary: str, metric: str, direction: str, score: float, history: list[Experiment]
) -> list[Message]:
    system = _PROMPTS["system"].format(metric=metric, direction=direction)
    if history:
        recent = history[-_HISTORY_LIMIT:]
        lines = _format_history(recent, metric)
        # the ledger scans the FULL history: a lever tried before the display
        # window is still tried.
        extras = "".join(
            block + "\n\n"
            for block in (_technique_table(recent, metric), _lever_ledger(history))
            if block
        )
        history_section = _PROMPTS["history_header"] + "\n" + "\n".join(lines) + "\n\n" + extras
    else:
        history_section = "No experiments yet — brief the first one.\n\n"
    user = _PROMPTS["user_template"].format(
        data_summary=data_summary,
        metric=metric,
        score=f"{score:.4f}",
        direction=direction,
        history_section=history_section,
    )
    return [Message(role="system", content=system), Message(role="user", content=user)]


_FLOAT = re.compile(r"\d+\.\d+")


def _validation_trail(exp: Experiment) -> str:
    """The within-session validation scores an experiment printed, in order — the
    knowledge that otherwise dies with the session (every IMPROVE attempt that lost,
    and how far a FAILED session actually got before it died). Scans stdout lines
    that look like a validation/score print and takes the last float on each (the
    metric value usually trails labels like 'Validation F1 score:')."""
    cells = exp.candidate.changes.get("cells")
    if not isinstance(cells, list):
        return ""
    seen: list[str] = []
    for cell in cells:
        stdout = cell.get("stdout") if isinstance(cell, dict) else None
        for line in (stdout or "").splitlines():
            low = line.lower()
            if "val" not in low and "score" not in low:
                continue
            floats = _FLOAT.findall(line)
            if floats and (not seen or seen[-1] != floats[-1]):
                seen.append(floats[-1])
    if len(seen) < 2:  # one value adds nothing beyond the final score
        return ""
    return f" (val tries: {' -> '.join(seen[-6:])})"


# A move that RE-TUNES the threshold (a tuning word within reach of "threshold"),
# as opposed to one that merely KEEPS it as context ("grid-search depth, keeping
# the 0.28 threshold") — proximity-bounded to avoid the over-blocking failure mode.
_THRESHOLD_TUNING_MOVE = re.compile(
    r"(?:tun\w*|optimi\w*|refin\w*|sweep\w*|adjust\w*)\W{1,20}threshold"
    r"|threshold\W{1,20}(?:tun\w*|optimi\w*|refin\w*|sweep\w*|adjust\w*)"
)
_COLUMN_ASSIGN = re.compile(r"\[['\"](\w+)['\"]\]\s*=")
_SNAKE_PARAM = re.compile(r"\b[a-z][a-z0-9]*_[a-z0-9_]+\b")  # learning_rate, l2_regularization


def _recommissions_banked_work(brief: str, carried: Experiment | None) -> str | None:
    """The reason a brief re-commissions work the carried best already contains, or
    None when the move is genuinely new. Three deterministic checks, each from a
    live duplicate: class_weight briefed when the final model already sets it; the
    already-searched grid re-briefed with no new dimension; an engineered feature
    the carried pipeline already builds briefed as an addition."""
    if carried is None:
        return None
    code = carried.candidate.changes.get("code")
    if not isinstance(code, str):
        return None
    move = _move_text(brief)
    config = (_carried_config(code) or "").lower()
    if ("class_weight" in move or "balanced" in move) and "class_weight" in config:
        return "class_weight is already set in the carried best's final model"
    threshold = _carried_threshold(code)
    if (
        threshold
        and _THRESHOLD_TUNING_MOVE.search(move)
        and not any(m in move for m in ("class_weight", "scale_pos_weight", "smote", "resampl"))
    ):
        # A tuned threshold is already applied in the banked best — re-tuning it
        # deterministically re-produces the incumbent (two live iterations burned
        # on exactly this before the check existed).
        return (
            f"a tuned decision threshold ({threshold}) is already applied in the "
            "carried best; re-tuning it re-produces the incumbent — name a different "
            "technique"
        )
    grid = _searched_grid(code)
    if grid:
        # No lever-class-name requirement: live briefs labeled these moves with the
        # ladder rung ("refine-best:") instead of a class, sailing past a
        # class-gated check while literally naming the already-searched params.
        searched = set(grid.lower().split("/"))
        named = set(_SNAKE_PARAM.findall(move))
        re_searches_same_params = bool(named) and named <= searched
        vague_re_search = not named and ("hyperparamet" in move or "grid" in move)
        if re_searches_same_params or vague_re_search:
            return (
                f"a {grid} grid was already searched on the carried best; name a NEW "
                "search dimension or pull a different lever"
            )
    for column in _COLUMN_ASSIGN.findall(code):
        if len(column) >= 6 and column.lower() in move:
            return f"the engineered feature '{column}' already exists in the carried best"
    return None


def _move_text(brief: str) -> str:
    """The move part of a raw brief, lowercased — mirroring _grounded_brief: text
    after "next:", or the brief with any leading so-far claim stripped (the claim
    is discarded before the coder ever sees it, so guards must not judge it)."""
    match = _NEXT_SLOT.search(brief)
    if match:
        return brief[match.end() :].lower()
    return _SO_FAR_CLAIM.sub("", brief, count=1).strip().lower()


def _technique_mentions(text: str) -> str:
    """The move part of a brief where a marker mention means the TECHNIQUE itself:
    class tags and the dead-ends tail are removed. Without this the guard
    over-fired on every brief — 'imbalance-or-threshold' (the class name the prompt
    REQUIRES) and the so-far's 'decision threshold ... already applied' both
    contain the 'threshold' marker."""
    move = _neutralize_hgb(_move_text(text))
    move = move.split("known dead ends", 1)[0]
    for lever in _LEVER_MARKERS:
        move = move.replace(lever, " ")
    return move


def _recommissions_a_measured_lost_technique(
    brief: str, history: list[Experiment], carried: Experiment | None
) -> str | None:
    """The reason a move re-commissions a SPECIFIC technique already measured this
    run that did not beat the best, or None. Matched via the technique marker in
    both the move and a prior experiment's HYPOTHESIS (its brief) — hypothesis, not
    code, because every session's code contains the carried pipeline's markers.
    Only strictly-losing priors count: re-refining the winning technique stays
    legitimate."""
    if carried is None or carried.result is None or carried.result.metrics is None:
        return None
    best = carried.result.metrics.primary_value
    move = _technique_mentions(brief)
    for markers in _LEVER_MARKERS.values():
        for marker in markers:
            if marker not in move:
                continue
            for exp in reversed(history):
                if exp is carried or exp.result is None or exp.result.metrics is None:
                    continue
                if not exp.result.succeeded:
                    continue
                # Match against the prior brief AND the prior session's SUBMITTED
                # final estimator: a lever pulled as unbriefed in-session drift
                # (live: a GB swap recorded only in Findings) is settled by its
                # holdout stamp just the same, and was invisible to a
                # hypothesis-only check.
                code = exp.candidate.changes.get("code")
                submitted = (
                    _neutralize_hgb((_carried_config(code) or "").lower())
                    if isinstance(code, str)
                    else ""
                )
                if marker in _technique_mentions(exp.hypothesis or "") or marker in submitted:
                    score = exp.result.metrics.primary_value
                    if score < best:
                        return (
                            f"'{marker}' was already measured this run (holdout "
                            f"{score:.4f}, did not beat {best:.4f}); pick a different "
                            "technique"
                        )
    return None


_MOVE_FLOAT = re.compile(r"0?\.\d{2,4}")
_SCORE_CLAIM_CONTEXT = (
    "best", "previous", "achieved", "scored", "top-performing", "top performing", "winning",
)


def _move_lint(brief: str, *, baseline_score: float, carried: Experiment | None) -> str | None:
    """The reason a move is ill-formed, or None. Two deterministic checks from a
    live brief that steered a wasted iteration: (a) two lever classes fused into
    one move; (b) a score claimed as best/previous that is neither the banked best
    nor the baseline (a hallucinated or validation-split number)."""
    move = _move_text(brief)
    classes = [lever for lever in _LEVER_MARKERS if lever in move]
    if len(classes) >= 2:
        return f"the move names two lever classes ({' and '.join(classes[:2])}); brief exactly ONE"
    known = {round(baseline_score, 4)}
    if carried is not None and carried.result is not None and carried.result.metrics is not None:
        known.add(round(carried.result.metrics.primary_value, 4))
    for match in _MOVE_FLOAT.finditer(move):
        context = move[max(0, match.start() - 30) : match.start()]
        if not any(word in context for word in _SCORE_CLAIM_CONTEXT):
            continue
        value = round(float(match.group()), 4)
        if all(abs(value - k) > 0.0005 for k in known):
            return (
                f"the move claims a best/previous score ({match.group()}) that is neither "
                "the banked best nor the baseline"
            )
    return None


# One canonical, safe move per lever class — used ONLY when the model twice fails
# to produce a novel brief and the harness must compose one itself.
_CANONICAL_MOVES: dict[str, str] = {
    "categorical-encoding": "frequency-encode the highest-cardinality categorical column instead of one-hot",
    "imbalance-or-threshold": "train with class_weight='balanced' on the current best pipeline",
    "feature-selection": "drop the lowest-importance features via SelectFromModel and re-measure",
    "interactions-or-ratios": "add one ratio feature between the two strongest numeric columns",
    "hyperparameter-search": "grid-search learning_rate and max_depth on the current best model",
    "model-swap": "swap the estimator for XGBClassifier with defaults, same preprocessing",
    "ensembling": "soft-vote the current best model with a LogisticRegression on the same features",
    "numeric-transform": "quantile-transform the most skewed numeric column",
}


def _fallback_move(history: list[Experiment]) -> tuple[str, str] | None:
    """A harness-composed (title, move) from the first lever class never tried this
    run, or None when every class is tried. The last resort after a guard fires
    twice: guaranteed novel by construction."""
    tried: set[str] = set()
    for exp in history:
        code = exp.candidate.changes.get("code")
        if not isinstance(code, str):
            continue
        low = _neutralize_hgb(code.lower())
        for lever, markers in _LEVER_MARKERS.items():
            if lever not in tried and any(m in low for m in markers):
                tried.add(lever)
    for lever, move in _CANONICAL_MOVES.items():
        if lever not in tried:
            return f"untried lever: {lever}", f"next: {lever}: {move}."
    return None


def _rebriefs_a_just_duplicated_class(brief: str, history: list[Experiment]) -> bool:
    """True when the brief names the SAME lever class as the immediately-previous
    experiment and that experiment's submission was a stamped duplicate — pushing
    the family that just no-opped again is how tail duplicates happen. Legitimate
    refinements are untouched: the guard needs the duplicate stamp, not merely a
    repeated class."""
    if not history:
        return False
    prev = history[-1]
    if not prev.candidate.changes.get("duplicate_submission"):
        return False
    prev_class = lever_class_for_brief(prev.hypothesis or "")
    new_class = lever_class_for_brief(brief)
    return prev_class is not None and prev_class == new_class


def _is_baseline_rebrief(title: str, brief: str) -> bool:
    """True when a brief reads as a plain baseline rebuild: baseline-flavored wording
    with NO lever class named. A lever brief that merely references the baseline
    score names its class, so it does not trip this."""
    text = f"{title} {brief}".lower()
    return "baseline" in text and not lever_markers_for_brief(brief)


def _was_floor_banked(exp: Experiment) -> bool:
    """True when this experiment's submission came from the harness fallback cell."""
    cells = exp.candidate.changes.get("cells")
    return isinstance(cells, list) and any(
        isinstance(c, dict) and c.get("source") == "fallback" for c in cells
    )


def _format_history(history: list[Experiment], metric: str) -> list[str]:
    lines = []
    for exp in history:
        desc = exp.candidate.description.strip()[:90]
        code = exp.candidate.changes.get("code")
        used = ""
        if isinstance(code, str):
            comps = codegen.components_used(code)
            if comps:
                used = f" [used: {', '.join(comps)}]"
        result = exp.result
        if result is not None and result.succeeded and result.metrics is not None:
            outcome = f"{metric}={result.metrics.primary_value:.4f}"
            if _was_floor_banked(exp):
                # The score came from the harness's fallback submit, not the lever —
                # the supervisor must not credit the lever for it.
                outcome += " [floor: harness fallback submission, the lever itself did not score]"
            elif exp.candidate.changes.get("duplicate_submission"):
                # Byte-identical to an earlier submission: a re-run, not a result.
                # Without this marker the scoreboard re-credits the lever and the
                # supervisor keeps orbiting it (observed: 4 wasted iterations/run).
                outcome += " [duplicate submission of an earlier experiment — no new information]"
            elif exp.candidate.changes.get("lever_unmeasured"):
                # The commissioned lever never executed successfully — the score
                # belongs to the carried pipeline, not the briefed idea.
                outcome += " [commissioned lever never executed — the score is the carried pipeline's]"
        elif result is not None and result.error:
            outcome = f"FAILED ({result.error.splitlines()[0][:60]})"
        else:
            outcome = "not run"
        lines.append(f"- {desc}{used} -> {outcome}{_validation_trail(exp)}")
        # The Summarizer's digest is the cross-notebook knowledge: what the data
        # showed and what helped or hurt, so the supervisor can compound winners.
        if exp.digest is not None:
            for label, items in (
                ("data", exp.digest.data_insights),
                ("helped", exp.digest.what_helped),
                ("hurt", exp.digest.what_hurt),
            ):
                if items:
                    lines.append(f"    {label}: " + "; ".join(items[:4]))
            if exp.digest.takeaway:
                lines.append(f"    next-idea: {exp.digest.takeaway}")
    return lines


# Technique LEVER classes, each detected by deterministic code markers. The ledger
# built from these makes coverage explicit ("what is done and what is not") — run
# c7ddda92 left the imbalance lever untouched for 9 of 10 experiments while the
# supervisor orbited model swaps, because nothing surfaced the untried classes.
# Markers are matched case-insensitively against each experiment's full code.
_LEVER_MARKERS: dict[str, tuple[str, ...]] = {
    "categorical-encoding": ("onehotencoder", "targetencoder", "ordinalencoder", "get_dummies"),
    "numeric-transform": ("powertransformer", "quantiletransformer", "log1p", "np.log"),
    "imbalance-or-threshold": ("class_weight", "scale_pos_weight", "smote", "threshold"),
    "interactions-or-ratios": ("polynomialfeatures", "interaction", "ratio", "_per_"),
    "feature-selection": ("selectkbest", "selectfrommodel", "rfe(", "feature_importances"),
    "ensembling": ("votingclassifier", "stackingclassifier", "votingregressor", "stackingregressor"),
    "hyperparameter-search": ("gridsearchcv", "randomizedsearchcv", "halvinggridsearchcv"),
    # Estimator identity: without this class every guard was blind to model swaps —
    # a run orbited GB/RF swaps re-stamping settled scores three times. The default
    # HGB family is deliberately absent (it is the baseline, not a swap); matching
    # text is neutralized via _neutralize_hgb so "gradientboosting" cannot match
    # inside "histgradientboosting".
    "model-swap": (
        "randomforest", "extratrees", "gradientboosting", "xgb", "lgbm", "lightgbm",
        "catboost", "logisticregression", "kneighbors",
    ),
}


def _neutralize_hgb(text: str) -> str:
    """Rewrite the default estimator's name so model-swap markers cannot match
    inside it ("gradientboosting" is a substring of "histgradientboosting")."""
    return text.replace("histgradientboosting", "hgb")


def lever_class_for_brief(brief: str) -> str | None:
    """The lever-class name a brief mentions, or None when it names no known class."""
    low = brief.lower()
    for lever in _LEVER_MARKERS:
        if lever in low:
            return lever
    return None


def lever_markers_for_brief(brief: str) -> tuple[str, ...]:
    """The deterministic code markers for the lever class a brief names, so the
    harness can verify the briefed change actually appears in the coder's executed
    code (live runs briefed class_weight three times; the string never reached a
    single cell). Empty when the brief names no known class — the gate stays off."""
    lever = lever_class_for_brief(brief)
    return _LEVER_MARKERS[lever] if lever else ()


def _lever_ledger(history: list[Experiment]) -> str:
    """One explicit done/not-done line across the technique lever classes, scanned
    from every experiment's code (full history, not just the display window). The
    supervisor should pull from the NOT-yet-tried side instead of re-orbiting the
    tried side."""
    if not history:
        return ""
    tried: set[str] = set()
    for exp in history:
        code = exp.candidate.changes.get("code")
        if not isinstance(code, str):
            continue
        low = _neutralize_hgb(code.lower())
        for lever, markers in _LEVER_MARKERS.items():
            if lever not in tried and any(m in low for m in markers):
                tried.add(lever)
    tried_s = ", ".join(lv for lv in _LEVER_MARKERS if lv in tried) or "none"
    untried_s = ", ".join(lv for lv in _LEVER_MARKERS if lv not in tried) or "none"
    return f"Levers tried: {tried_s} | Levers NOT yet tried: {untried_s}"


def _technique_table(history: list[Experiment], metric: str) -> str:
    """Best score reached whenever each technique appeared, aggregated across all
    digests, so the pattern 'this technique tends to score well' is explicit rather
    than left for the model to infer from scattered lines."""
    best: dict[str, float] = {}
    seen: dict[str, int] = {}
    for exp in history:
        if exp.digest is None or exp.digest.score is None:
            continue
        changes = exp.candidate.changes
        if changes.get("duplicate_submission") or changes.get("lever_unmeasured") or _was_floor_banked(exp):
            # A stamped experiment's score belongs to the incumbent (or the harness
            # floor), not its techniques — crediting them re-invited the settled
            # lever in a live run.
            continue
        score = exp.digest.score
        for tech in exp.digest.techniques:
            seen[tech] = seen.get(tech, 0) + 1
            if tech not in best or score > best[tech]:
                best[tech] = score
    if not best:
        return ""
    ranked = sorted(best.items(), key=lambda kv: -kv[1])
    cells = [f"{t} {best[t]:.4f} (x{seen[t]})" for t, _ in ranked]
    return f"Technique scoreboard (best {metric} when each appeared): " + " | ".join(cells)


__all__ = [
    "PLAN_NEXT",
    "Supervisor",
    "SupervisorDecision",
    "SupervisorError",
    "lever_markers_for_brief",
]
