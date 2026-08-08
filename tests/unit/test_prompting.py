"""The prompt itself: rendering a record into it, and the baseline it starts from."""

from __future__ import annotations

import pytest

from iterate.core.prompting import (
    Prompt,
    baseline_prompt,
    render,
    render_all,
    render_value,
)

pytestmark = pytest.mark.unit


def test_a_single_input_column_is_rendered_bare() -> None:
    """Labelling a lone `text` field with "text:" adds noise to every single call."""
    assert render_all({"text": "you are the worst"}, ["text"]) == "you are the worst"


def test_several_input_columns_are_labelled() -> None:
    rendered = render_all({"title": "Bad", "body": "Awful"}, ["title", "body"])

    assert rendered == "title: Bad\nbody: Awful"


def test_a_json_column_becomes_readable_lines() -> None:
    """Users arrive with one column holding a JSON record as often as with several
    columns, and a model reads labelled fields better than brace soup."""
    rendered = render_value('{"age": 31, "city": "Mumbai"}')

    assert rendered == "age: 31\ncity: Mumbai"


def test_a_column_that_only_looks_like_json_is_left_alone() -> None:
    assert render_value("{not json at all}") == "{not json at all}"


def test_placeholders_are_filled_from_the_record() -> None:
    template = "Review: {body}\nRating so far: {stars}"

    assert render(template, {"body": "great", "stars": 4}, ["body", "stars"]) == (
        "Review: great\nRating so far: 4"
    )


def test_input_expands_to_every_column() -> None:
    template = "Record:\n{input}\nAnswer:"

    assert "title: Bad" in render(template, {"title": "Bad", "body": "x"}, ["title", "body"])


def test_braces_in_the_users_own_data_are_never_re_substituted() -> None:
    """A record containing braces is normal. A second substitution pass would read
    the user's data as prompt structure, or crash on it."""
    rendered = render(
        "{text}", {"text": "use {input} carefully", "input": "SHOULD NOT APPEAR"}, ["text"]
    )

    assert rendered == "use {input} carefully"


def test_an_unknown_placeholder_is_left_as_written() -> None:
    """A prompt saying {format} as English is likelier than the agent inventing a
    column, and blanking it would delete an instruction."""
    assert render("Reply in {format}", {"text": "x"}, ["text"]) == "Reply in {format}"


def test_a_very_long_field_is_trimmed_visibly() -> None:
    rendered = render_value("x" * 9000)

    assert "trimmed" in rendered
    assert len(rendered) < 9000


def test_the_baseline_names_the_task_and_the_allowed_answers() -> None:
    prompt = baseline_prompt("Say whether the comment is toxic.", ["toxic", "not toxic"])

    assert "Say whether the comment is toxic." in prompt.system
    assert "toxic, not toxic" in prompt.system
    assert prompt.user_template == "{input}"


def test_the_baseline_is_minimal_and_not_a_strawman() -> None:
    """A deliberately terrible baseline makes every later change look like a
    triumph. It must be the simplest prompt a competent person writes, and nothing
    more: no examples, no reasoning instructions, no role play."""
    prompt = baseline_prompt("Classify the intent.", ["book", "cancel"])

    lowered = prompt.system.casefold()
    for tell in ("for example", "step by step", "you are an expert", "think carefully"):
        assert tell not in lowered
    assert len(prompt.system.splitlines()) <= 3


def test_a_baseline_without_a_label_set_still_works() -> None:
    prompt = baseline_prompt("Summarise the complaint in one word.", None)

    assert "Answer with exactly one of" not in prompt.system


def test_a_prompt_round_trips_through_a_dict() -> None:
    prompt = Prompt(system="be terse", user_template="{input}")

    assert Prompt.from_dict(prompt.as_dict()) == prompt
