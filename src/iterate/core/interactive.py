"""Interactive run control — the seam between a human at the terminal and the loop.

One ``RunController`` is shared by three parties: a CLI-owned listener thread that
feeds it the lines the user types, the supervised loop that drains them at safe
boundaries, and the coding agent that parks at a cell boundary while paused.

The controller itself never touches the console (replies go through an injected
callback), never calls an LLM (interpretation is an injected callback the loop
binds, because that is where the Supervisor lives), and never touches Memory
(sqlite is single-thread). The listener thread only parses the three control
words and enqueues text; everything heavy happens on the MAIN thread inside
``checkpoint``, so the rest of the codebase stays single-threaded in practice.

Timing model: a message can only take effect at a boundary — after a cell
finishes, or just before the next supervisor call — because the main thread
blocks inside kernel/LLM calls between boundaries. The instant ack is therefore
TIMING-only ("delivering at the next safe point"); what the message *means* is
decided later by the interpreter, and the interpreter replies again with the
real routing.
"""

from __future__ import annotations

import contextlib
import logging
import queue
import threading
import time
from collections.abc import Callable, Sequence
from typing import Protocol

log = logging.getLogger(__name__)

# An interpreter takes the drained batch of user lines plus whether a coding
# session is live right now, and distributes each line (session note, brief
# note, standing rule, or an answered question via ``reply``). Bound by the
# supervised loop at run start; absent before that.
Interpreter = Callable[[list[str], bool], None]

_RULES_LIMIT = 3  # standing rules shown to the supervisor — lean context, dead-ends style
_NOTE_CHARS = 200  # cap on one steer note; what is stored is exactly what the model sees
_RULE_CHARS = 90  # cap on one standing rule (dead-ends style) — stored = stamped = shown


class _Keepalive(Protocol):
    def keepalive(self) -> None: ...


def _clip(text: str, limit: int = _NOTE_CHARS) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0]


class RunController:
    """Shared state between the stdin listener and the supervised loop.

    Thread contract: ``submit_line`` (and nothing else) may be called from the
    listener thread; every other method runs on the main thread at a boundary.
    """

    def __init__(
        self,
        *,
        reply: Callable[[str], None] | None = None,
        keepalive_interval: float = 240.0,
    ) -> None:
        self._reply_fn = reply
        self._keepalive_interval = keepalive_interval
        self._lines: queue.Queue[str] = queue.Queue()
        self._running = threading.Event()  # set = running; cleared = pause requested
        self._running.set()
        self._lock = threading.Lock()
        self._abort = False
        self._session_notes: list[str] = []  # for the LIVE coding session
        self._brief_notes: list[str] = []  # for the next supervisor planning turn
        self._rules: list[str] = []  # standing, every future planning turn
        self.paused_seconds_total = 0.0  # the run-level deadline subtracts this
        self.status = "starting"  # one short phrase the ack quotes
        self.interpreter: Interpreter | None = None
        # Structured UI events (brief issued / cell executed / score recorded) —
        # presentation only. The TUI subscribes; plain mode leaves it None.
        self.on_event: Callable[[str, dict[str, object]], None] | None = None
        # The RUNNING session's cell transcript (the coder shares its live list
        # for the session's duration) — Q&A about "the work on screen" needs it,
        # because an in-flight experiment is not in Memory yet.
        self.live_cells: Sequence[object] | None = None
        # Hard-quit hook: bound by the CLI/TUI. When set, a typed stop means
        # "terminate NOW, keep what is on disk" (notebooks + memory persist
        # incrementally, so nothing already earned is lost). When absent (tests,
        # or no front end bound yet), stop degrades to the graceful wind-down.
        self.on_stop_now: Callable[[], None] | None = None
        # Live run snapshot, bound by the loop: returns a RunResult built from
        # whatever has FINISHED so far, callable at any instant — the hard quit
        # uses it to print the summary table even while a cell is mid-flight.
        self.snapshot: Callable[[], object] | None = None

    # ── listener side (the only cross-thread entry point) ────────────────────

    def submit_line(self, line: str) -> None:
        """Accept one typed line. Control words act immediately; anything else
        queues for the next boundary with a timing-only ack."""
        text = line.strip()
        if not text:
            return
        word = text.lower().rstrip(" .!?")  # "stop." means stop
        if word.startswith("/"):
            # Slash form: explicitly a command, never chat. An unknown one is
            # rejected here rather than delivered to the agent as guidance.
            command = word[1:].strip()
            if command in ("pause", "resume", "stop"):
                word = command
            else:
                self.reply(f"unknown command: {text} — commands are /pause, /resume, /stop")
                return
        if word == "pause":
            if self.abort_requested:
                self.reply("already stopping — the run is winding down")
                return
            self._running.clear()
            self.reply("pausing — the loop parks once the current step finishes")
        elif word == "resume":
            if self.abort_requested:
                self.reply("stop was already requested — the run is winding down")
                return
            self._running.set()
            self.reply("resuming")
        elif word == "stop":
            with self._lock:
                self._abort = True
            self._running.set()  # a paused loop must wake to see the abort
            if self.on_stop_now is not None:
                self.reply("stopped — quitting now; everything already saved is kept")
                with contextlib.suppress(Exception):
                    self.on_stop_now()
            else:
                self.reply("stopping — winding down at the next safe point; finished work is kept")
        else:
            self._lines.put(text)
            self.reply(f"queued ({self.status}) — delivering at the next safe point")

    def request_graceful_stop(self) -> None:
        """Wind the run down at the next safe point (single Ctrl-C's polite
        form): the in-flight session still banks its floor and Memory gets
        finalized. A typed stop is the HARD form (see ``on_stop_now``)."""
        with self._lock:
            self._abort = True
        self._running.set()
        self.reply(
            "stopping gracefully — winding down at the next safe point "
            "(Ctrl-C again, or /stop, quits immediately)"
        )

    def reply(self, text: str) -> None:
        """Show ``text`` to the user; a broken console must never kill the run."""
        if self._reply_fn is not None:
            with contextlib.suppress(Exception):
                self._reply_fn(text)

    def bind_reply(self, fn: Callable[[str], None] | None) -> None:
        """Attach (or detach) the user-facing output channel — the plain path's
        console printer or the TUI's log pane."""
        self._reply_fn = fn

    def emit(self, kind: str, **payload: object) -> None:
        """Publish one structured UI event. Free when nobody subscribes; a
        broken renderer must never kill the run."""
        if self.on_event is not None:
            with contextlib.suppress(Exception):
                self.on_event(kind, payload)

    # ── loop/coder side (main thread only) ────────────────────────────────────

    @property
    def abort_requested(self) -> bool:
        with self._lock:
            return self._abort

    @property
    def rules(self) -> tuple[str, ...]:
        return tuple(self._rules)

    def add_session_note(self, text: str) -> None:
        self._session_notes.append(_clip(text))

    def add_brief_note(self, text: str) -> None:
        self._brief_notes.append(_clip(text))

    def add_rule(self, text: str) -> None:
        clipped = _clip(text, _RULE_CHARS)
        self._rules.append(clipped)
        if len(clipped) < len(text.strip()):
            self.reply(f"rule stored shortened to fit the lean cap: {clipped!r}")
        if len(self._rules) > _RULES_LIMIT:  # lean context: oldest rule gives way
            dropped = self._rules.pop(0)
            self.reply(f"rule list is full ({_RULES_LIMIT}); dropped the oldest: {dropped!r}")

    def take_session_notes(self) -> list[str]:
        notes, self._session_notes = self._session_notes, []
        return notes

    def take_brief_notes(self) -> list[str]:
        notes, self._brief_notes = self._brief_notes, []
        return notes

    def checkpoint(self, kernel: _Keepalive | None = None) -> float:
        """The boundary: interpret queued messages, honor a pause, return the
        seconds spent paused so the CALLER credits its own clocks (the coder
        shifts its wall-ceiling anchor; the loop's deadline subtracts the shared
        total). ``kernel`` (when a session is live) gets a keepalive at every
        blocking edge — before/after interpretation (whole LLM calls can hide in
        a drain) and periodically while paused — so an e2b lease outlives all of
        it. Pausing blocks IN PLACE, never by returning, so the kernel and every
        local survive untouched; an abort is never pausable."""
        self._tick(kernel)
        if self._drain(live_session=kernel is not None):
            self._tick(kernel)  # interpretation may have burned minutes of lease
        if self.abort_requested or self._running.is_set():
            return 0.0
        self.reply(
            "paused (kernel kept alive) — type resume to continue, stop to end the run"
            if kernel is not None
            else "paused between experiments — type resume to continue, stop to end the run"
        )
        prior_status, self.status = self.status, "paused"
        started = time.monotonic()
        last_keepalive = started
        while not self._running.wait(timeout=min(1.0, self._keepalive_interval)):
            if self.abort_requested:  # stop typed after pause must never park forever
                break
            now = time.monotonic()
            if kernel is not None and now - last_keepalive >= self._keepalive_interval:
                self._tick(kernel)
                last_keepalive = now
            if self._drain(live_session=kernel is not None):  # chatting while paused works
                self._tick(kernel)
                last_keepalive = time.monotonic()
        paused = time.monotonic() - started
        self.paused_seconds_total += paused
        self.status = prior_status
        self._drain(live_session=kernel is not None)
        self._tick(kernel)
        if not self.abort_requested:
            self.reply(f"resumed after {paused:.0f}s (clocks were suspended)")
        return paused

    @staticmethod
    def _tick(kernel: _Keepalive | None) -> None:
        if kernel is not None:
            with contextlib.suppress(Exception):
                kernel.keepalive()

    def _drain(self, *, live_session: bool) -> bool:
        """Route everything queued; returns whether any message was processed."""
        batch: list[str] = []
        while True:
            try:
                batch.append(self._lines.get_nowait())
            except queue.Empty:
                break
        if not batch:
            return False
        if self.interpreter is None:
            self._default_route(batch, live_session)
            return True
        try:
            self.interpreter(batch, live_session)
        except Exception:  # interpretation degrades, it never crashes the run
            log.warning("interactive: interpreter failed; treating messages as steers", exc_info=True)
            self._default_route(batch, live_session)
        return True

    def _default_route(self, batch: list[str], live_session: bool) -> None:
        """The least-destructive route when nothing can interpret: a steer for
        the work in front of us, visible to the live session (if any) and to the
        next planning turn."""
        for text in batch:
            if live_session:
                self.add_session_note(text)
            self.add_brief_note(text)
        where = "the running session and the next brief" if live_session else "the next brief"
        self.reply(f"delivered {len(batch)} message(s) as guidance for {where}")


__all__ = ["Interpreter", "RunController"]
