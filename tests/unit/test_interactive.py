"""Tests for the RunController — the pause/chat seam between a human and the loop."""

from __future__ import annotations

import threading

from iterate.core.interactive import RunController


class _FakeKernel:
    def __init__(self) -> None:
        self.keepalives = 0

    def keepalive(self) -> None:
        self.keepalives += 1


def test_a_plain_message_queues_with_a_timing_only_ack() -> None:
    replies: list[str] = []
    ctrl = RunController(reply=replies.append)
    ctrl.status = "session iter-01: at a cell boundary"
    ctrl.submit_line("try a smaller learning rate\n")
    assert any("queued" in r and "iter-01" in r for r in replies)
    # nothing is routed until a boundary
    assert ctrl.take_brief_notes() == []


def test_default_routing_without_an_interpreter_is_a_steer() -> None:
    ctrl = RunController()
    ctrl.submit_line("use class weights")
    ctrl.checkpoint(None)  # between experiments: brief only
    assert ctrl.take_brief_notes() == ["use class weights"]
    assert ctrl.take_session_notes() == []

    ctrl.submit_line("use class weights")
    ctrl.checkpoint(_FakeKernel())  # a session is live: both routes
    assert ctrl.take_session_notes() == ["use class weights"]
    assert ctrl.take_brief_notes() == ["use class weights"]


def test_stop_sets_abort_and_wakes_a_pause() -> None:
    ctrl = RunController()
    ctrl.submit_line("pause")
    ctrl.submit_line("stop")
    assert ctrl.abort_requested
    # the pause must not block a stopping run: stop re-set the running event
    assert ctrl.checkpoint(None) == 0.0


def test_pause_blocks_until_resume_credits_time_and_keeps_the_kernel_alive() -> None:
    ctrl = RunController(keepalive_interval=0.05)
    kernel = _FakeKernel()
    ctrl.submit_line("pause")
    threading.Timer(0.4, lambda: ctrl.submit_line("resume")).start()
    paused = ctrl.checkpoint(kernel)
    assert paused >= 0.3
    assert ctrl.paused_seconds_total == paused
    assert kernel.keepalives >= 1


def test_messages_typed_while_paused_are_routed_on_resume() -> None:
    ctrl = RunController()
    ctrl.submit_line("pause")

    def _chat_then_resume() -> None:
        ctrl.submit_line("skip lightgbm here")
        ctrl.submit_line("resume")

    threading.Timer(0.2, _chat_then_resume).start()
    ctrl.checkpoint(None)
    assert ctrl.take_brief_notes() == ["skip lightgbm here"]


def test_rules_are_capped_and_the_oldest_gives_way() -> None:
    ctrl = RunController()
    for i in range(4):
        ctrl.add_rule(f"rule {i}")
    assert ctrl.rules == ("rule 1", "rule 2", "rule 3")


def test_a_failing_interpreter_degrades_to_the_default_route() -> None:
    ctrl = RunController()

    def _boom(batch: list[str], live_session: bool) -> None:
        raise RuntimeError("classifier down")

    ctrl.interpreter = _boom
    ctrl.submit_line("do the thing")
    ctrl.checkpoint(None)
    assert ctrl.take_brief_notes() == ["do the thing"]


def test_stored_notes_are_clipped() -> None:
    ctrl = RunController()
    ctrl.add_brief_note("word " * 100)
    (note,) = ctrl.take_brief_notes()
    assert len(note) <= 200


def test_stop_then_pause_never_parks_the_run() -> None:
    ctrl = RunController()
    ctrl.submit_line("stop")
    ctrl.submit_line("pause")  # must not re-enter a pause the stop cannot escape
    assert ctrl.abort_requested
    assert ctrl.checkpoint(None) == 0.0  # returns promptly, no deadlock


def test_control_words_tolerate_trailing_punctuation() -> None:
    ctrl = RunController()
    ctrl.submit_line("Stop.")
    assert ctrl.abort_requested


def test_resume_after_stop_says_so_instead_of_lying() -> None:
    replies: list[str] = []
    ctrl = RunController(reply=replies.append)
    ctrl.submit_line("stop")
    ctrl.submit_line("resume")
    assert any("winding down" in r for r in replies)
    assert ctrl.abort_requested  # resume did not cancel the stop


def test_checkpoint_ticks_keepalive_around_interpretation() -> None:
    kernel = _FakeKernel()
    ctrl = RunController()

    def _slow_interpreter(batch: list[str], live_session: bool) -> None:
        pass  # stands in for LLM calls; the ticks bracket it

    ctrl.interpreter = _slow_interpreter
    ctrl.submit_line("some question about the run")
    ctrl.checkpoint(kernel)
    assert kernel.keepalives >= 2  # entry tick + post-drain tick


def test_emit_is_free_without_a_subscriber_and_safe_with_a_broken_one() -> None:
    ctrl = RunController()
    ctrl.emit("cell", index=1)  # no subscriber: no-op

    seen: list[tuple[str, dict]] = []
    ctrl.on_event = lambda kind, payload: seen.append((kind, payload))
    ctrl.emit("cell", index=2, ok=True)
    assert seen == [("cell", {"index": 2, "ok": True})]

    def _boom(kind: str, payload: dict) -> None:
        raise RuntimeError("renderer died")

    ctrl.on_event = _boom
    ctrl.emit("score", iteration=1)  # must not raise


def test_typed_stop_calls_the_hard_quit_hook() -> None:
    fired: list[bool] = []
    ctrl = RunController()
    ctrl.on_stop_now = lambda: fired.append(True)
    ctrl.submit_line("/stop")
    assert fired == [True]
    assert ctrl.abort_requested  # belt: the loop still winds down if the quit fails


def test_graceful_stop_never_hard_quits() -> None:
    fired: list[bool] = []
    ctrl = RunController()
    ctrl.on_stop_now = lambda: fired.append(True)
    ctrl.request_graceful_stop()
    assert fired == []
    assert ctrl.abort_requested
