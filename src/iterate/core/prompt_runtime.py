"""`ask()` — running one prompt across every record, from inside a session cell.

This is the prompt path's equivalent of calling scikit-learn. The agent writes the
prompt, decides the placeholders and writes the evaluation cells; it does not
hand-roll the transport, exactly as the tabular coder does not implement gradient
boosting. Four reasons the harness owns the call rather than the generated code:

*Comparability.* A model writing its own HTTP call can quietly change the model,
the temperature or the endpoint between experiments. Then two experiments differ by
more than the prompt and the comparison is worthless — the same rule that keeps the
metric fixed for a whole run.

*Speed.* One call per record, run sequentially, is what makes a prompt loop too slow
to finish. Calls go out concurrently and every answer is cached, so re-running a
prompt the session has already tried costs nothing.

*Format.* The allowed answers are a tool schema with an enum, not an instruction in
the text. Asking a model to reply with one of three labels is a request; giving it a
tool whose only argument is one of three labels makes anything else impossible. Same
shape as the Researcher picking a paper by number rather than writing a DOI.

*Failure.* A flaky call, a rate limit or a garbled reply is retried and then recorded
as an unparseable answer for that row. It never raises, because one bad row must not
cost the whole experiment — but it does count as wrong, since a prompt that provokes
unusable output IS a worse prompt and hiding that would reward vagueness.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import threading
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

_ANSWER_TOOL = "answer"
_DEFAULT_WORKERS = 8
_DEFAULT_RETRIES = 2


@dataclass
class AskStats:
    """What one pass over the data cost."""

    calls: int = 0
    cached: int = 0
    unparseable: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        parts = [f"{self.calls} calls", f"{self.cached} cached"]
        if self.unparseable:
            parts.append(f"{self.unparseable} unparseable")
        if self.errors:
            parts.append(f"{len(self.errors)} errored")
        return ", ".join(parts)


class AnswerCache:
    """Answers keyed by (model, prompt, rendered record).

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


def _key(model: str, prompt: Prompt, rendered: str) -> str:
    blob = json.dumps(
        {"model": model, "system": prompt.system, "user": rendered}, sort_keys=True
    ).encode()
    return hashlib.sha256(blob).hexdigest()


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

    hits = [str(label) for label in labels if str(label).strip().casefold() in normalised]
    return hits[0] if len(hits) == 1 else UNPARSEABLE


def _one(
    client: LLMClient,
    prompt: Prompt,
    rendered: str,
    labels: Sequence[str] | None,
    retries: int,
) -> tuple[str, int, int, str | None]:
    """One record. Returns (answer, prompt_tokens, completion_tokens, error)."""
    messages = [
        Message(role="system", content=prompt.system),
        Message(role="user", content=rendered),
    ]
    tools = [answer_tool(labels)]
    last_error: str | None = None
    prompt_tokens = 0
    completion_tokens = 0

    for _ in range(retries + 1):
        try:
            reply = client.chat(messages, tools=tools, temperature=0.0)
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            continue
        # Counted even on a retry: a run that burned three calls to get one answer
        # spent three calls, and cost that only counts successes is not cost.
        prompt_tokens += reply.usage.prompt_tokens
        completion_tokens += reply.usage.completion_tokens
        if reply.tool_calls:
            raw = reply.tool_calls[0].arguments.get("value")
            answer = coerce(str(raw) if raw is not None else None, labels)
        else:
            answer = coerce(reply.content, labels)
        if answer != UNPARSEABLE:
            return answer, prompt_tokens, completion_tokens, None
        last_error = "model did not produce a usable answer"

    return UNPARSEABLE, prompt_tokens, completion_tokens, last_error


def ask(
    prompt: Prompt,
    rows: Sequence[Mapping[str, Any]],
    *,
    client_factory: Callable[[], LLMClient],
    columns: Sequence[str],
    labels: Sequence[str] | None = None,
    cache: AnswerCache | None = None,
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
    local = threading.local()

    def client() -> LLMClient:
        existing = getattr(local, "client", None)
        if existing is None:
            existing = client_factory()
            local.client = existing
        return existing

    def handle(index: int) -> None:
        rendered = render(prompt.user_template, rows[index], columns)
        model = client().model
        key = _key(model, prompt, rendered)

        if cache is not None and (hit := cache.get(key)) is not None:
            answers[index] = hit
            counters.cached += 1
            if hit == UNPARSEABLE:
                counters.unparseable += 1
            return

        answer, prompt_tokens, completion_tokens, error = _one(
            client(), prompt, rendered, labels, retries
        )
        answers[index] = answer
        counters.calls += 1
        counters.prompt_tokens += prompt_tokens
        counters.completion_tokens += completion_tokens
        if answer == UNPARSEABLE:
            counters.unparseable += 1
            if error:
                counters.errors.append(error)
        elif cache is not None:
            # Only successes are cached. Caching a failure would make a transient
            # network blip permanent for the rest of the run.
            cache.put(key, answer)

    if rows:
        with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(rows)))) as pool:
            list(pool.map(handle, range(len(rows))))

    return [a if a is not None else UNPARSEABLE for a in answers]


def make_ask(
    *,
    columns: Sequence[str],
    labels: Sequence[str] | None,
    backend: str,
    model: str,
    base_url: str | None = None,
    cache_path: Path | str | None = None,
    max_workers: int = _DEFAULT_WORKERS,
) -> Callable[..., list[str]]:
    """Build the `ask` a session cell calls. Bound to ONE model, on purpose.

    No URL is constructed here. Calls go through the same `build_client` factory the
    agent itself runs on, so a backend alias resolves its own endpoint and whatever
    the user configured with `iterate setup` applies unchanged.

    The key resolves through the same table too: `api_key_for` reads `GROQ_API_KEY`,
    `OPENAI_API_KEY` and the rest exactly as the driving model does, so
    `--target-backend groq` picks up the key already in the environment. It is read
    at call time and never written into meta.json, so it does not land on disk
    beside data the generated code reads. `ITERATE_TARGET_API_KEY` overrides it for
    the case where the model under test needs a different key from the same provider.
    """
    from iterate.llm.factory import api_key_for, build_client

    cache = AnswerCache(cache_path)

    def client_factory() -> LLMClient:
        key = os.environ.get("ITERATE_TARGET_API_KEY") or api_key_for(backend)
        return build_client(backend, model=model, base_url=base_url, api_key=key)

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
            cache=cache,
            max_workers=max_workers,
            stats=stats,
        )

    return bound


__all__ = [
    "UNPARSEABLE",
    "AnswerCache",
    "AskStats",
    "answer_tool",
    "ask",
    "coerce",
    "make_ask",
]
