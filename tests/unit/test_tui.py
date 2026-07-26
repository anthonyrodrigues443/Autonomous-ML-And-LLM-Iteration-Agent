"""Tests for the TUI face — typed input reaches the controller, logs reach the pane.

Textual's Pilot drives a headless app; the fake "run" is a thread worker parked
on an event, exactly like the real loop parked inside a cell.
"""

from __future__ import annotations

import logging
import threading

import pytest
from textual.widgets import Input

from iterate.core.interactive import RunController
from iterate.ui.tui import IterateTUI, _WidgetLogHandler


def test_widget_log_handler_formats_and_forwards() -> None:
    lines: list[object] = []

    class _AppStub:
        def post_line(self, text: object) -> None:
            lines.append(text)

    handler = _WidgetLogHandler(_AppStub())  # type: ignore[arg-type]
    record = logging.LogRecord("t", logging.INFO, __file__, 1, "hello %s", ("world",), None)
    handler.emit(record)
    assert len(lines) == 1
    assert getattr(lines[0], "plain", str(lines[0])) == "hello world"


def test_widget_log_handler_never_raises() -> None:
    class _BrokenApp:
        def post_line(self, text: object) -> None:
            raise RuntimeError("pane gone")

    handler = _WidgetLogHandler(_BrokenApp())  # type: ignore[arg-type]
    record = logging.LogRecord("t", logging.INFO, __file__, 1, "x", (), None)
    handler.emit(record)  # must not raise


@pytest.mark.asyncio
async def test_typed_message_reaches_the_controller() -> None:
    ctrl = RunController()
    release = threading.Event()

    def run_fn() -> str:
        release.wait(timeout=10)
        return "done"

    app = IterateTUI(run_fn, ctrl, title="t")
    ctrl.bind_reply(app.post_reply)
    async with app.run_test() as pilot:
        await pilot.pause()
        app.query_one(Input).value = "prefer catboost"
        await pilot.press("enter")
        await pilot.pause()
        release.set()
        await pilot.pause(0.2)
    ctrl.checkpoint(None)  # drain at a boundary, as the loop would
    assert ctrl.take_brief_notes() == ["prefer catboost"]
    assert app.result == "done"
    assert app.error is None


@pytest.mark.asyncio
async def test_ctrl_c_is_a_graceful_stop_not_a_quit() -> None:
    ctrl = RunController()
    release = threading.Event()

    def run_fn() -> str:
        release.wait(timeout=10)
        return "wound-down"

    app = IterateTUI(run_fn, ctrl, title="t")
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("ctrl+c")
        await pilot.pause()
        assert ctrl.abort_requested  # the run winds down; the app did not die
        release.set()
        await pilot.pause(0.2)
    assert app.result == "wound-down"


@pytest.mark.asyncio
async def test_worker_thread_can_own_a_sqlite_memory(tmp_path) -> None:
    """Regression: the run (and its sqlite Memory) execute on the TUI's worker
    thread — the connection must be created there, never on the main thread."""
    from iterate.core.memory import SqliteMemory
    from iterate.schemas.experiment import ExperimentResult, Metrics

    ctrl = RunController()

    def run_fn() -> str:
        memory = SqliteMemory(tmp_path / "m.db")  # born on the worker, used on the worker
        run_id = memory.start_run(
            "t",
            ExperimentResult(
                experiment_id="b",
                metrics=Metrics(values={"f1": 0.5}, primary="f1", direction="maximize", n_samples=10),
            ),
        )
        memory.finish_run(run_id, "done")
        return run_id

    app = IterateTUI(run_fn, ctrl, title="t")
    async with app.run_test() as pilot:
        await pilot.pause(0.3)
    assert app.error is None, f"worker-thread Memory failed: {app.error}"
    assert isinstance(app.result, str)
    assert app.result


@pytest.mark.asyncio
async def test_slash_opens_the_palette_and_enter_completes_without_sending() -> None:
    from textual.widgets import OptionList

    ctrl = RunController()
    release = threading.Event()

    def run_fn() -> str:
        release.wait(timeout=10)
        return "ok"

    app = IterateTUI(run_fn, ctrl, title="t")
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("/")
        palette = app.query_one(OptionList)
        assert palette.display
        assert palette.option_count == 3
        # arrows move the highlight; enter completes INTO the box, sends nothing
        await pilot.press("down", "down", "enter")
        chat = app.query_one(Input)
        assert chat.value == "/stop"
        assert not palette.display
        assert not ctrl.abort_requested  # nothing was sent yet
        # the second, deliberate enter sends it
        await pilot.press("enter")
        await pilot.pause()
        assert ctrl.abort_requested
        release.set()
        await pilot.pause(0.2)
    assert app.result == "ok"


@pytest.mark.asyncio
async def test_palette_filters_and_escape_closes() -> None:
    from textual.widgets import OptionList

    ctrl = RunController()
    release = threading.Event()

    def run_fn() -> str:
        release.wait(timeout=10)
        return "ok"

    app = IterateTUI(run_fn, ctrl, title="t")
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("/", "p")
        palette = app.query_one(OptionList)
        assert palette.display
        assert palette.option_count == 1  # only /pause matches "/p"
        await pilot.press("escape")
        assert not palette.display
        release.set()
        await pilot.pause(0.2)


def test_unknown_slash_command_is_rejected_by_the_controller() -> None:
    replies: list[str] = []
    ctrl = RunController(reply=replies.append)
    ctrl.submit_line("/dance")
    assert any("unknown command" in r for r in replies)
    ctrl.checkpoint(None)
    assert ctrl.take_brief_notes() == []  # never delivered as chat


def test_slash_forms_of_control_words_work_everywhere() -> None:
    ctrl = RunController()
    ctrl.submit_line("/stop")
    assert ctrl.abort_requested


@pytest.mark.asyncio
async def test_double_ctrl_c_quits_immediately() -> None:
    ctrl = RunController()
    release = threading.Event()

    def run_fn() -> str:
        release.wait(timeout=10)
        return "x"

    app = IterateTUI(run_fn, ctrl, title="t")
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("ctrl+c")
        await pilot.pause()
        assert ctrl.abort_requested  # first press: graceful request only
        assert not app.quit_requested
        await pilot.press("ctrl+c")
        await pilot.pause()
    assert app.quit_requested
    release.set()


@pytest.mark.asyncio
async def test_typed_stop_quits_immediately_when_the_hook_is_bound() -> None:
    ctrl = RunController()
    release = threading.Event()

    def run_fn() -> str:
        release.wait(timeout=10)
        return "x"

    app = IterateTUI(run_fn, ctrl, title="t")
    ctrl.on_stop_now = app.quit_immediately
    async with app.run_test() as pilot:
        await pilot.pause()
        app.query_one(Input).value = "stop"
        await pilot.press("enter")
        await pilot.pause()
    assert app.quit_requested
    release.set()
