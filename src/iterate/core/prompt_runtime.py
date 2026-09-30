"""`ask()`: one prompt across every record, from inside a session cell.

The harness owns the call so the model, temperature and endpoint cannot change
between experiments. Calls run concurrently and every answer is cached. The
allowed answers are a tool schema, not an instruction. A row the model answered
unusably is retried, then recorded as unparseable and scored as wrong. A row the
provider never answered is not the model's answer: it is asked once more after the
pass, and if it is still unanswered it comes back as NO_REPLY and is left out of the
score, or, past one row in five, the pass raises.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from iterate.schemas.llm import Message, ToolSpec

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

    from iterate.core.prompting import Prompt
    from iterate.llm.base import LLMClient

log = logging.getLogger(__name__)

# What a row's answer is when the model could not be made to produce a usable one.
# A distinct sentinel rather than an empty string or a guessed label: it must score
# as wrong, and it must be countable afterwards.
UNPARSEABLE = "__unparseable__"
# What a row's answer is when the provider never answered it, asked twice. Not the
# model's answer: left out of the score, and out of the score it is compared with.
NO_REPLY = "__no_reply__"

_ANSWER_TOOL = "answer"
_DEFAULT_WORKERS = 8
_DEFAULT_RETRIES = 2

_ANSWER_MAX_TOKENS = 64
_FREE_TEXT_MAX_TOKENS = 512
# Long enough for a rate limit's minute or a short outage to pass.
_SECOND_ASK_WAIT = 30.0

# What a provider says when it turns down the model's own tool call: groq's 400, and
# Ollama's 500 for a call it could not parse. The model answered.
_TURNED_DOWN = ("tool_use_failed", "error parsing tool call")


class NoReplyError(RuntimeError):
    """The provider answered none of the tries for more than one record in five, asked
    twice: an outage, a rate limit that outlasted the retries, a key or a model it
    refuses. Too few records are left to score the pass on."""


@dataclass
class AskStats:
    """What one pass over the data cost."""

    calls: int = 0
    cached: int = 0
    unparseable: int = 0
    no_reply: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        parts = [f"{self.calls} calls", f"{self.cached} cached"]
        if self.unparseable:
            parts.append(f"{self.unparseable} unparseable")
        if self.no_reply:
            parts.append(f"{self.no_reply} not answered by the provider")
        return ", ".join(parts)


class AnswerCache:
    """Answers keyed by (who answered, model, prompt, rendered record).

    File-backed so the cache survives across experiments in a run: the baseline
    prompt gets re-run, and a session re-tries a prompt it has already scored. Both
    become free. Falls back to memory if the path cannot be opened, because a broken
    cache must never stop a run.
    """

    def __init__(self, path: Path | str | None) -> None:
        self._lock = threading.Lock()
        self._memory: dict[str, str] = {}
        self._conn: sqlite3.Connection | None = None
        if path is None:
            return
        try:
            conn = sqlite3.connect(str(path), check_same_thread=False)
            conn.execute("CREATE TABLE IF NOT EXISTS answers (key TEXT PRIMARY KEY, value TEXT)")
            conn.commit()
            self._conn = conn
        except sqlite3.Error as exc:
            log.info("answer cache unavailable at %s (%s); using memory only", path, exc)

    def get(self, key: str) -> str | None:
        with self._lock:
            if key in self._memory:
                return self._memory[key]
            if self._conn is None:
                return None
            row = self._conn.execute("SELECT value FROM answers WHERE key = ?", (key,)).fetchone()
            if row is None:
                return None
            self._memory[key] = str(row[0])
            return self._memory[key]

    def put(self, key: str, value: str) -> None:
        with self._lock:
            self._memory[key] = value
            if self._conn is None:
                return
            try:
                self._conn.execute(
                    "INSERT OR REPLACE INTO answers (key, value) VALUES (?, ?)", (key, value)
                )
                self._conn.commit()
            except sqlite3.Error:
                pass


def _key(scope: str, model: str, prompt: Prompt, rendered: str) -> str:
    # With no scope the key is the one every answer before v0.7 was filed under.
    fields = {"model": model, "system": prompt.system, "user": rendered}
    if scope:
        fields["scope"] = scope
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()


# A number the model wrote, possibly wrapped in prose: "4", "4.5", "I'd say 3".
# Anchored to a token boundary so the 5 in "5 stars" is read and the 3 in "GPT-3"
# is not — a stray identifier scoring as a rating is a silent wrong answer.
_NUMBER = re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?(?![\w.])")


def numeric_answer_tool(low: float | None, high: float | None) -> ToolSpec:
    """The tool for a NUMERIC target: a number, bounded where bounds are known.

    Same idea as the enum. A rating outside the observed range is not something the
    model can express, so the commonest scoring failure — answering 7 on a 1-to-5
    scale — is structurally impossible rather than something to catch afterwards.
    """
    value: dict[str, Any] = {"type": "number", "description": "The value for this record."}
    if low is not None:
        value["minimum"] = low
    if high is not None:
        value["maximum"] = high
    return ToolSpec(
        name=_ANSWER_TOOL,
        description="Give your answer for this record as a number. Call this exactly once.",
        parameters={"type": "object", "properties": {"value": value}, "required": ["value"]},
    )


def coerce_number(text: str | None, low: float | None, high: float | None) -> str:
    """A number out of a reply, or the sentinel.

    A reply containing more than one number is UNPARSEABLE rather than
    first-wins: "somewhere between 3 and 4" names two and picking either invents a
    precision the model did not express. Same rule as a reply naming two labels.
    """
    if text is None:
        return UNPARSEABLE
    found = _NUMBER.findall(text.strip())
    if len(found) != 1:
        return UNPARSEABLE
    value = float(found[0])
    if low is not None and value < low:
        return UNPARSEABLE
    if high is not None and value > high:
        return UNPARSEABLE
    return repr(value)


def answer_tool(labels: Sequence[str] | None) -> ToolSpec:
    """The one tool the model under test may call.

    With a known label set the argument is an enum, so an out-of-vocabulary answer is
    not something the model can express. Without one (a free-text target) it is a
    plain string and parsing does the work instead.
    """
    value: dict[str, Any] = {"type": "string", "description": "The answer for this record."}
    if labels:
        value["enum"] = [str(label) for label in labels]
    return ToolSpec(
        name=_ANSWER_TOOL,
        description="Give your answer for this record. Call this exactly once.",
        parameters={
            "type": "object",
            "properties": {"value": value},
            "required": ["value"],
        },
    )


def coerce(text: str | None, labels: Sequence[str] | None) -> str:
    """Map a free-text reply onto the label set, or report it unparseable.

    Only used when the model answered with prose instead of calling the tool. Exact
    match first, then a UNIQUE substring match — "I think this is toxic" resolves,
    while a reply naming two labels does not, because guessing which one it meant is
    how a wrong answer becomes a right one by accident.
    """
    if text is None:
        return UNPARSEABLE
    cleaned = text.strip()
    if not cleaned:
        return UNPARSEABLE
    if not labels:
        return cleaned

    normalised = cleaned.casefold()
    for label in labels:
        if normalised == str(label).strip().casefold():
            return str(label)

    # Scanned by POSITION, longest alternative first, with word boundaries. Label
    # comparison gets both of these wrong: "the intent is timer" must not read as
    # ambiguous with "time", and "toxic or not toxic" must.
    ordered = sorted((str(label).strip() for label in labels), key=len, reverse=True)
    pattern = "|".join(re.escape(label.casefold()) for label in ordered)
    found = re.findall(rf"(?<!\w)(?:{pattern})(?!\w)", normalised)
    distinct = set(found)
    if len(distinct) != 1:
        return UNPARSEABLE
    matched = distinct.pop()
    return next(str(label) for label in labels if str(label).strip().casefold() == matched)


def _one(
    client: LLMClient,
    prompt: Prompt,
    rendered: str,
    labels: Sequence[str] | None,
    retries: int,
    numeric_range: tuple[float | None, float | None] | None = None,
) -> tuple[str, int, int, str | None, bool]:
    """One record. Returns (answer, prompt_tokens, completion_tokens, error, replied):
    ``replied`` is False when no try got a reply from the provider."""
    messages = [
        Message(role="system", content=prompt.system),
        Message(role="user", content=rendered),
    ]
    tools = [
        numeric_answer_tool(*numeric_range) if numeric_range else answer_tool(labels)
    ]
    last_error: str | None = None
    replied = False
    prompt_tokens = 0
    completion_tokens = 0

    cap = _ANSWER_MAX_TOKENS if labels or numeric_range else _FREE_TEXT_MAX_TOKENS

    for _ in range(retries + 1):
        try:
            reply = client.chat(messages, tools=tools, temperature=0.0, max_tokens=cap)
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            replied = replied or _turned_down(exc)
            continue
        replied = True
        # Counted even on a retry: a run that burned three calls to get one answer
        # spent three calls, and cost that only counts successes is not cost.
        prompt_tokens += reply.usage.prompt_tokens
        completion_tokens += reply.usage.completion_tokens
        raw = (
            reply.tool_calls[0].arguments.get("value")
            if reply.tool_calls
            else reply.content
        )
        text = None if raw is None else str(raw)
        answer = (
            coerce_number(text, *numeric_range) if numeric_range else coerce(text, labels)
        )
        if answer != UNPARSEABLE:
            return answer, prompt_tokens, completion_tokens, None, True
        last_error = "model did not produce a usable answer"

    return UNPARSEABLE, prompt_tokens, completion_tokens, last_error, replied


def _turned_down(exc: BaseException) -> bool:
    """The provider replied, and what it refused was the model's own tool call."""
    response = getattr(exc, "response", None)
    try:
        body = str(getattr(response, "text", "") or "")
    except Exception:
        body = ""
    said = f"{exc} {body}"
    return any(words in said for words in _TURNED_DOWN)


def _humanise(seconds: float) -> str:
    """A duration a person can act on, not a float."""
    if seconds < 90:
        return f"{int(seconds)}s"
    minutes = seconds / 60
    if minutes < 90:
        return f"{minutes:.0f} min"
    return f"{minutes / 60:.1f} hours"


def ask(
    prompt: Prompt,
    rows: Sequence[Mapping[str, Any]],
    *,
    client_factory: Callable[[], LLMClient],
    columns: Sequence[str],
    labels: Sequence[str] | None = None,
    numeric_range: tuple[float | None, float | None] | None = None,
    cache: AnswerCache | None = None,
    cache_scope: str = "",
    model: str | None = None,
    max_workers: int = _DEFAULT_WORKERS,
    retries: int = _DEFAULT_RETRIES,
    stats: AskStats | None = None,
) -> list[str]:
    """Run `prompt` over every record and return one answer per record, in order.

    A fresh client per worker thread: the backends are HTTP clients that were never
    promised to be thread-safe, and one shared connection quietly serialising the
    whole pass would silently undo the concurrency this exists for.
    """
    from iterate.core.prompting import render

    counters = stats if stats is not None else AskStats()
    answers: list[str | None] = [None] * len(rows)
    missed: dict[int, str] = {}
    local = threading.local()

    def client() -> LLMClient:
        existing = getattr(local, "client", None)
        if existing is None:
            existing = client_factory()
            local.client = existing
        return existing

    def handle(index: int, *, again: bool = False) -> None:
        rendered = render(prompt.user_template, rows[index], columns)
        # With the model's name in hand no client is built for an answer already
        # cached, so a pass served whole from the cache needs no key.
        key = _key(cache_scope, model if model is not None else client().model, prompt, rendered)

        if cache is not None and (hit := cache.get(key)) is not None:
            answers[index] = hit
            counters.cached += 1
            if hit == UNPARSEABLE:
                counters.unparseable += 1
            return

        answer, prompt_tokens, completion_tokens, error, replied = _one(
            client(), prompt, rendered, labels, retries, numeric_range
        )
        answers[index] = answer
        if not again:
            counters.calls += 1
        counters.prompt_tokens += prompt_tokens
        counters.completion_tokens += completion_tokens
        if not replied:
            missed[index] = error or "no reply"
            return
        missed.pop(index, None)
        if answer == UNPARSEABLE:
            counters.unparseable += 1
            if error:
                counters.errors.append(error)
        elif cache is not None:
            # Only successes are cached. Caching a failure would make a transient
            # network blip permanent for the rest of the run.
            cache.put(key, answer)

    if rows:
        # A pass over a few hundred records is minutes of model calls with nothing
        # to show for it. The first live run printed NOTHING for nine minutes while
        # the baseline scored, which reads as a hang; a heartbeat every 10% costs
        # nothing and tells the user it is moving.
        step = max(1, len(rows) // 10)
        # Enough completed calls to project from without waiting long to say
        # anything. The first few are the slowest (a cold model loads), so this is a
        # pessimistic estimate, which is the right direction to be wrong in.
        probe = min(5, len(rows))
        started = time.monotonic()
        done = 0
        lock = threading.Lock()

        def tracked(index: int) -> None:
            handle(index)
            nonlocal done
            with lock:
                done += 1
                if done == probe and len(rows) > probe:
                    # An unbounded wait is the actual pain; a known 16 minutes is a
                    # decision the user can make. Measured per pass rather than
                    # guessed for the whole run, because how many passes the agent
                    # will take is not knowable up front.
                    per_call = (time.monotonic() - started) / done
                    log.info(
                        "prompt pass: %d records at ~%.1fs each -> about %s remaining",
                        len(rows),
                        per_call,
                        _humanise(per_call * (len(rows) - done)),
                    )
                if done % step == 0 or done == len(rows):
                    log.info("prompt pass: %d/%d records", done, len(rows))

        with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(rows)))) as pool:
            list(pool.map(tracked, range(len(rows))))

    if missed:
        log.warning(
            "prompt pass: the provider did not answer %d records; asking them once more in %ds",
            len(missed),
            int(_SECOND_ASK_WAIT),
        )
        time.sleep(_SECOND_ASK_WAIT)
        with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(missed)))) as pool:
            list(pool.map(lambda index: handle(index, again=True), sorted(missed)))
    if missed:
        counters.no_reply += len(missed)
        counters.errors.extend(missed.values())
        if len(missed) * 5 > len(rows):
            first = missed[min(missed)]
            raise NoReplyError(
                f"the provider did not answer {len(missed)} of {len(rows)} records, each "
                f"asked twice ({first}); more than one in five is too many to score"
            )
        for index in missed:
            answers[index] = NO_REPLY
    return [a if a is not None else UNPARSEABLE for a in answers]


def make_ask(
    *,
    columns: Sequence[str],
    labels: Sequence[str] | None,
    numeric_range: tuple[float | None, float | None] | None = None,
    backend: str,
    model: str,
    base_url: str | None = None,
    api_key: str | None = None,
    cache_path: Path | str | None = None,
    max_workers: int = _DEFAULT_WORKERS,
    scoped: bool = False,
) -> Callable[..., list[str]]:
    """Build the `ask` a session cell calls. Bound to ONE model, on purpose.

    The client is the model under test's own: its endpoint is the one given or the
    provider's public address, and its key is the provider's. Neither ever comes from
    the harness's settings.

    The host passes ``api_key``. A cell passes none: its kernel holds the one key it was
    handed, as `ITERATE_TARGET_API_KEY`. A delivered notebook run by hand finds the
    provider's own variable (`GROQ_API_KEY` for groq). The key is never written into
    meta.json, so it does not land on disk beside data the generated code reads.

    ``scoped`` files each answer under the provider and the host that gave it. A
    notebook delivered before v0.7 passes nothing, and finds the answers it paid for
    where it left them.
    """
    from iterate.llm import factory

    cache = AnswerCache(cache_path)
    scope = factory.cache_scope(backend, base_url) if scoped else ""

    def client_factory() -> LLMClient:
        key = (
            api_key
            or (os.environ.get(factory.TARGET_KEY_ENV) or "").strip()
            or factory.own_key_for(factory.provider_name(backend, base_url))
        )
        return factory.build_target_client(backend, model=model, base_url=base_url, api_key=key)

    def bound(
        prompt: Prompt,
        rows: Sequence[Mapping[str, Any]],
        *,
        stats: AskStats | None = None,
    ) -> list[str]:
        return ask(
            prompt,
            rows,
            client_factory=client_factory,
            columns=columns,
            labels=labels,
            numeric_range=numeric_range,
            cache=cache,
            cache_scope=scope,
            model=model,
            max_workers=max_workers,
            stats=stats,
        )

    return bound


__all__ = [
    "NO_REPLY",
    "UNPARSEABLE",
    "AnswerCache",
    "AskStats",
    "NoReplyError",
    "answer_tool",
    "ask",
    "coerce",
    "make_ask",
]
