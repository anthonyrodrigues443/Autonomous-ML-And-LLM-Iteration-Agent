"""The interactive terminal UI for supervised runs (v0.3).

A scrollable run log on top, a pinned input box at the bottom, the run itself on
a worker thread. Everything the plain path can do (pause / resume / stop as
words, questions, steers, rules) works identically here — the Input widget
simply replaces the stdin listener thread as the source of ``submit_line``
calls, and log records route into the log pane instead of stdout (the alternate
screen would swallow a stdout handler).

Kept deliberately thin: the loop, the controller, and the interpreter know
nothing about Textual; this module only moves text between widgets and the
controller. ``--plain`` (or a non-tty) never imports it.
"""

from __future__ import annotations

import contextlib
import logging
import threading
from typing import TYPE_CHECKING, Any, ClassVar

from pygments.token import (
    Comment,
    Error,
    Generic,
    Keyword,
    Literal,
    Name,
    Number,
    Operator,
    Punctuation,
    String,
    Token,
)
from pygments.token import Text as TextToken
from rich.panel import Panel
from rich.rule import Rule
from rich.style import Style
from rich.syntax import ANSISyntaxTheme, Syntax
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Input, OptionList, RichLog, Static
from textual.widgets.option_list import Option

if TYPE_CHECKING:
    from collections.abc import Callable

    from rich.console import RenderableType
    from textual import events

    from iterate.core.interactive import RunController

# Two-tone code, on purpose: full syntax rainbows compete with the transcript's
# role colors (borders, headings, you/agent lines). Code is white; comments keep
# the muted gray they already had. Every top-level token family is mapped
# explicitly — the theme lookup inherits within a family (Comment.Single from
# Comment) but not from the root Token.
_WHITE = Style(color="white")
_CODE_THEME = ANSISyntaxTheme(
    {
        Token: _WHITE,
        TextToken: _WHITE,
        Keyword: _WHITE,
        Name: _WHITE,
        String: _WHITE,
        Number: _WHITE,
        Literal: _WHITE,
        Operator: _WHITE,
        Punctuation: _WHITE,
        Generic: _WHITE,
        Error: _WHITE,
        Comment: Style(color="bright_black", italic=True),
    }
)


# The command palette's contents: every run-control command with a one-line
# description. Typed "/" opens the list; plain bare words keep working too.
_COMMANDS: tuple[tuple[str, str], ...] = (
    ("pause", "Park the run at the next safe point — kernel kept alive, clocks suspended"),
    ("resume", "Continue a paused run"),
    ("stop", "Quit NOW — prints the summary of everything finished; saved work is kept"),
)


class _WidgetLogHandler(logging.Handler):
    """Routes log records into the app's log pane; a broken UI never kills the run."""

    def __init__(self, app: IterateTUI) -> None:
        super().__init__(level=logging.INFO)
        self._app = app
        self.setFormatter(logging.Formatter("%(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        with contextlib.suppress(Exception):
            message = self.format(record)
            # The styled event renderer already covers per-cell and per-iteration
            # lines with richer blocks; keep the rest, dimmed, as ambient detail.
            if message.startswith("coder[") and ": cell " in message:
                return
            if message.startswith("agent loop: iteration") and "->" in message:
                return
            self._app.post_line(Text(message, style="dim"))


class IterateTUI(App[Any]):
    """Scrollable log + pinned input; the supervised run executes on a worker."""

    CSS = """
    #status { dock: top; height: 1; padding: 0 1; background: $panel; color: $text-muted; }
    #log { height: 1fr; padding: 0 1; }
    #palette { height: auto; max-height: 8; padding: 0 1; background: $panel; }
    #chat { dock: bottom; }
    """

    BINDINGS: ClassVar[list[Binding | tuple[str, str] | tuple[str, str, str]]] = [
        # Ctrl-C keeps its plain-path meaning (graceful stop, keep the work) —
        # never a hard app quit that would abandon a live kernel mid-run.
        Binding("ctrl+c", "request_stop", "stop the run", priority=True),
        Binding("ctrl+q", "request_stop", "stop the run", show=False, priority=True),
    ]

    def __init__(
        self, run_fn: Callable[[], Any], controller: RunController, *, title: str
    ) -> None:
        super().__init__()
        self._run_fn = run_fn
        self._controller = controller
        self._title_line = title
        self.result: Any = None
        self.error: BaseException | None = None
        self.quit_requested = False
        self._ui_thread: int | None = None
        self._log_widget: RichLog | None = None
        self._stop_asked = False
        self._palette: OptionList | None = None
        self._palette_names: list[str] = []
        self._suppress_palette_for = ""

    def compose(self) -> ComposeResult:
        yield Static(self._title_line, id="status")
        # markup/highlight off: log lines contain bracketed text like
        # "coder[iter-01]" that rich markup would misparse.
        yield RichLog(id="log", wrap=True, markup=False, highlight=False, auto_scroll=True)
        yield OptionList(id="palette")
        yield ChatInput(
            placeholder="type anything — a question or an instruction; / for commands",
            id="chat",
        )

    def on_mount(self) -> None:
        self._ui_thread = threading.get_ident()
        self._log_widget = self.query_one(RichLog)
        self._palette = self.query_one(OptionList)
        self._palette.display = False
        self._palette.can_focus = False  # driven by keys from the input, not focus
        self.query_one(Input).focus()
        # A plain daemon thread, deliberately NOT a Textual worker: a hard quit
        # must never wait for a thread blocked inside a cell or an LLM call —
        # daemon threads are simply abandoned at process exit (kernels self-reap:
        # ipykernel's parent poller locally, the lease on e2b).
        threading.Thread(target=self._execute, daemon=True, name="iterate-run").start()

    # ── text in ───────────────────────────────────────────────────────────

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.value = ""
        if not text:
            return
        # "bold blue" reads as violet-magenta on common dark terminal palettes —
        # the approved look; literal ANSI magenta renders pink on those palettes.
        self.post_line(Text(f"you ▸ {text}", style="bold blue"))
        # The controller normalizes /commands and rejects unknown ones itself,
        # so plain mode and the TUI behave identically.
        self._controller.submit_line(text)

    # ── the command palette ("/" opens; arrows move; enter completes) ──────

    def on_input_changed(self, event: Input.Changed) -> None:
        value = event.value
        if value.startswith("/"):
            if value == self._suppress_palette_for:
                # Just completed via the palette: the command sits in the box,
                # waiting for a deliberate second Enter (or edits).
                self._hide_palette()
                return
            self._suppress_palette_for = ""
            self._show_palette(value[1:].lower())
        else:
            self._suppress_palette_for = ""
            self._hide_palette()

    def palette_handle_key(self, event: events.Key) -> bool:
        """Navigation keys the ChatInput hands over while the palette is open.
        Returns True when the key was consumed."""
        palette = self._palette
        if palette is None or not palette.display:
            return False
        if event.key == "down":
            palette.action_cursor_down()
        elif event.key == "up":
            palette.action_cursor_up()
        elif event.key in ("enter", "tab"):
            index = palette.highlighted if palette.highlighted is not None else 0
            self._complete_from_palette(index)
        elif event.key == "escape":
            self._hide_palette()
        else:
            return False
        event.stop()
        event.prevent_default()
        return True

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self._complete_from_palette(event.option_index)  # mouse click on an option
        self.query_one(Input).focus()

    def _show_palette(self, prefix: str) -> None:
        palette = self._palette
        if palette is None:
            return
        matches = [(name, desc) for name, desc in _COMMANDS if name.startswith(prefix)]
        if not matches:
            self._hide_palette()
            return
        self._palette_names = [name for name, _ in matches]
        palette.clear_options()
        palette.add_options(
            Option(Text.assemble((f"/{name}".ljust(12), "bold"), (desc, "dim")))
            for name, desc in matches
        )
        palette.highlighted = 0
        palette.display = True

    def _hide_palette(self) -> None:
        if self._palette is not None:
            self._palette.display = False
        self._palette_names = []

    def _complete_from_palette(self, index: int) -> None:
        """Put the selected command INTO the input box — never send it. The user
        sends with a second Enter, or edits/deletes it like any text."""
        if not 0 <= index < len(self._palette_names):
            return
        value = "/" + self._palette_names[index]
        chat = self.query_one(Input)
        self._suppress_palette_for = value
        chat.value = value
        chat.cursor_position = len(value)
        self._hide_palette()

    def action_request_stop(self) -> None:
        # First Ctrl-C: graceful wind-down (the in-flight attempt banks its
        # floor, Memory finalizes). Second Ctrl-C: quit immediately.
        if not self._stop_asked:
            self._stop_asked = True
            self._controller.request_graceful_stop()
        else:
            self.quit_immediately()

    def quit_immediately(self) -> None:
        """Hard exit NOW: normal app shutdown restores the terminal, then the
        caller prints the summary from the controller's snapshot. The run thread
        may still be blocked inside a cell — it is a daemon, nobody waits."""
        self.quit_requested = True
        if threading.get_ident() == self._ui_thread:
            self.exit(None)
        else:
            self.call_from_thread(self.exit, None)

    # ── text out (safe from any thread) ───────────────────────────────────

    def post_line(self, text: RenderableType) -> None:
        widget = self._log_widget
        if widget is None:  # a record before mount: drop rather than crash
            return
        # expand=True: panels and rules span the full pane width, edge to edge,
        # instead of shrinking to their content — uniform blocks, easier to scan.
        if threading.get_ident() == self._ui_thread:
            widget.write(text, expand=True)
        else:
            self.call_from_thread(widget.write, text, expand=True)

    def post_reply(self, text: str) -> None:
        self.post_line(Text(f"agent ▸ {text}", style="cyan"))

    def render_event(self, kind: str, payload: dict[str, object]) -> None:
        """Structured run events, styled like a session transcript: an experiment
        header per brief, each executed cell as a bordered syntax panel with its
        status and seconds, a score row per iteration. Presentation only — any
        rendering surprise is swallowed, never surfaced to the run."""
        with contextlib.suppress(Exception):
            if kind == "brief":
                self.post_line(Rule(style="magenta"))
                self.post_line(
                    Text(
                        f"◆ experiment {payload.get('iteration')} — {payload.get('title')}",
                        style="bold magenta",
                    )
                )
                brief = str(payload.get("brief") or "")
                if brief:
                    self.post_line(Text(brief, style="dim"))
            elif kind == "cell":
                code = str(payload.get("code") or "")
                lines = code.splitlines()
                shown = "\n".join(lines[:30]) + ("\n# ... truncated" if len(lines) > 30 else "")
                ok = bool(payload.get("ok"))
                mark = "✓" if ok else "!"
                header = (
                    f"{mark} cell {payload.get('index')} · {payload.get('status')} · "
                    f"{float(str(payload.get('seconds') or 0)):.1f}s · budget "
                    f"{float(str(payload.get('budget_spent') or 0)):.0f}/"
                    f"{float(str(payload.get('budget_total') or 0)):.0f}s"
                )
                self.post_line(
                    Panel(
                        # word_wrap: long lines fold inside the panel instead of
                        # cropping at the border — a transcript should never hide
                        # code off-screen.
                        Syntax(shown, "python", theme=_CODE_THEME, word_wrap=True),
                        title=header, title_align="left",
                        border_style="green" if ok else "red",
                    )
                )
            elif kind == "score":
                error = payload.get("error")
                if error:
                    self.post_line(
                        Text(f"! iteration {payload.get('iteration')} failed: {error}",
                             style="bold red")
                    )
                else:
                    best = "  ← best so far" if payload.get("is_best") else ""
                    score = payload.get("score")
                    shown_score = f"{float(str(score)):.4f}" if score is not None else "?"
                    self.post_line(
                        Text(f"✓ iteration {payload.get('iteration')} scored {shown_score}{best}",
                             style="bold green")
                    )

    # ── the run ───────────────────────────────────────────────────────────

    def _execute(self) -> None:
        try:
            self.result = self._run_fn()
        except BaseException as exc:  # re-raised by run_in_tui once the screen closes
            self.error = exc
        finally:
            # After a hard quit the app is already gone; the abandoned daemon
            # thread must die silently, not with a traceback on stderr.
            with contextlib.suppress(Exception):
                self.call_from_thread(self.exit, self.result)


class ChatInput(Input):
    """The chat box: yields navigation keys to the command palette while it is
    open (arrows move the highlight, Enter/Tab complete into the box, Escape
    closes), and behaves as a plain Input otherwise."""

    async def _on_key(self, event: events.Key) -> None:
        app = self.app
        if isinstance(app, IterateTUI) and app.palette_handle_key(event):
            return
        await super()._on_key(event)


def run_in_tui(
    run_fn: Callable[[], Any],
    controller: RunController,
    *,
    title: str,
    configure_logging: Callable[..., None],
) -> Any:
    """Run the supervised loop inside the TUI and return the loop's result.

    Log records route into the log pane for the app's lifetime and the handler
    is removed afterwards, so the post-run summary (rendered by the CLI on the
    normal screen) prints exactly as in plain mode. An exception from the run
    surfaces after the screen closes, never into the alternate screen."""
    app = IterateTUI(run_fn, controller, title=title)
    controller.bind_reply(app.post_reply)
    controller.on_event = app.render_event
    controller.on_stop_now = app.quit_immediately
    handler = _WidgetLogHandler(app)
    configure_logging(handler=handler)
    try:
        app.run()
    finally:
        logging.getLogger().removeHandler(handler)
        controller.bind_reply(None)
        controller.on_event = None
        controller.on_stop_now = None
    if app.quit_requested:
        # Hard quit: hand back the loop's live snapshot so the CLI still prints
        # the summary table of everything that finished before the stop.
        snapshot = controller.snapshot
        return snapshot() if callable(snapshot) else None
    if app.error is not None:
        raise app.error
    return app.result


__all__ = ["IterateTUI", "run_in_tui"]
