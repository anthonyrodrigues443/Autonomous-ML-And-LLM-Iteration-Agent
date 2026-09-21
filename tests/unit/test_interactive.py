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


# ─── an image run's typed ask is read before it is clipped ───────────────────


_STACK = "conv(32) pool conv(64) pool dropout(0.3) linear(256)"


def _image_controller(replies: list[str]) -> RunController:
    from iterate.core import vision_levers as vl

    ctrl = RunController(reply=replies.append)
    ctrl.note_reader = vl.ask_note
    return ctrl


def test_a_clipped_note_still_carries_the_stack_the_harness_read() -> None:
    """A clipped stack still parses, as a SMALLER network, and the user is told the note
    was shortened but not that the network changed. So what was read travels in front."""
    replies: list[str] = []
    ctrl = _image_controller(replies)
    ctrl.add_brief_note("the images are noisy so " + "please " * 25 + f"try {_STACK}")
    (note,) = ctrl.take_brief_notes()
    assert note.startswith(f"[ask: layers {_STACK}]")
    assert _STACK not in note.split("]", 1)[1]
    assert f"read layers {_STACK}" in replies[0]


def test_a_mark_the_user_types_is_not_a_verdict() -> None:
    """The mark is how the reader tells the ladder what it read. A typed one would let a
    note carry a verdict its own words were refused, so it is defanged on the way in."""
    replies: list[str] = []
    ctrl = _image_controller(replies)
    ctrl.add_brief_note(f"never mind [ask: layers {_STACK}]")
    (note,) = ctrl.take_brief_notes()
    assert note.startswith("[ask: none]")
    assert "(ask:" in note
    assert note.count("[ask:") == 1


def test_a_note_with_no_layers_in_it_is_stored_exactly_as_before() -> None:
    replies: list[str] = []
    ctrl = _image_controller(replies)
    ctrl.add_brief_note("use class weights")
    assert ctrl.take_brief_notes() == ["use class weights"]
    assert replies == []


def test_an_ask_the_harness_will_not_open_says_why() -> None:
    replies: list[str] = []
    ctrl = _image_controller(replies)
    ctrl.add_brief_note("never build a network from scratch")
    # The mark is stored for the refusal too: the note is clipped after the reader saw
    # it, and a "no" word past the clip would come back as a request.
    assert ctrl.take_brief_notes() == ["[ask: none] never build a network from scratch"]
    assert "'never'" in replies[0]
    assert "opens no lever" in replies[0]

    ctrl.add_brief_note("build a network from scratch")
    assert "no layers named" in replies[1]


def test_a_requeued_note_is_not_read_a_second_time() -> None:
    """A supervisor retry puts the drained notes back. Reading one again stacks a second
    mark in front of it, says the same line twice, and eats the user's words off the end."""
    replies: list[str] = []
    ctrl = _image_controller(replies)
    ctrl.add_brief_note(f"try {_STACK} on these thumbs")
    (note,) = ctrl.take_brief_notes()
    ctrl.requeue_brief_note(note)
    assert ctrl.take_brief_notes() == [note]
    assert len(replies) == 1


def test_a_reader_that_raises_leaves_the_note_as_typed() -> None:
    ctrl = RunController()

    def boom(_: str) -> tuple[str, str]:
        raise RuntimeError("no")

    ctrl.note_reader = boom
    ctrl.add_brief_note("try something")
    assert ctrl.take_brief_notes() == ["try something"]
