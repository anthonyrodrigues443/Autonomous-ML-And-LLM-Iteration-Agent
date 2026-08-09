"""CLI wiring for a prompt run: reading the user's prompt, and refusing bad metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import typer

from iterate.adapters.data.tabular import load_csv
from iterate.cli import _build_prompt_target, _read_starting_prompt

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit


@pytest.fixture
def eval_csv(tmp_path: Path) -> Path:
    lines = ["text,label"] + [
        f"comment {i},{'toxic' if i % 3 == 0 else 'clean'}" for i in range(30)
    ]
    path = tmp_path / "eval.csv"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _target(eval_csv: Path, metric: str = "f1", **kwargs: object) -> object:
    defaults: dict[str, object] = {
        "metric": metric,
        "average": None,
        "task": "say whether the comment is toxic",
        "prompt_file": None,
        "backend": "ollama",
        "model": "gemma4:12b",
        "base_url": None,
        "cache_path": eval_csv.parent / "answers.db",
    }
    return _build_prompt_target(load_csv(eval_csv, target="label"), **{**defaults, **kwargs})


def test_a_plain_text_prompt_becomes_the_system_message(tmp_path: Path) -> None:
    """What someone pasting the prompt they already run actually means."""
    path = tmp_path / "prod.txt"
    path.write_text("You are a strict content moderator.", encoding="utf-8")

    prompt = _read_starting_prompt(path)

    assert prompt.system == "You are a strict content moderator."
    assert prompt.user_template == "{input}"


def test_a_yaml_prompt_keeps_both_halves(tmp_path: Path) -> None:
    path = tmp_path / "prod.yaml"
    path.write_text(
        "system: You are a moderator.\nuser_template: 'Comment: {text}'\n", encoding="utf-8"
    )

    prompt = _read_starting_prompt(path)

    assert prompt.system == "You are a moderator."
    assert prompt.user_template == "Comment: {text}"


def test_yaml_without_a_system_key_is_read_as_text(tmp_path: Path) -> None:
    """Rather than silently producing an empty prompt that scores like a broken run."""
    path = tmp_path / "prod.yaml"
    path.write_text("just some instructions, no keys\n", encoding="utf-8")

    assert "just some instructions" in _read_starting_prompt(path).system


def test_an_empty_prompt_file_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "prod.txt"
    path.write_text("   \n", encoding="utf-8")

    with pytest.raises(typer.BadParameter, match="empty"):
        _read_starting_prompt(path)


def test_malformed_yaml_is_rejected_with_the_filename(tmp_path: Path) -> None:
    path = tmp_path / "prod.yaml"
    path.write_text("system: [unclosed\n", encoding="utf-8")

    with pytest.raises(typer.BadParameter, match="not valid yaml"):
        _read_starting_prompt(path)


def test_a_probability_metric_is_refused_before_the_run_starts(eval_csv: Path) -> None:
    """A text model returns a label, not a calibrated probability, so this run could
    never score. Same spirit as validating a metric against the target column."""
    with pytest.raises(typer.BadParameter, match="probabilities"):
        _target(eval_csv, metric="roc_auc")


def test_a_label_metric_builds_the_target(eval_csv: Path) -> None:
    target = _target(eval_csv, metric="f1")

    assert target.model_under_test == "gemma4:12b"
    assert target.labels == ["clean", "toxic"]
    assert "say whether the comment is toxic" in target.baseline_prompt.system


def test_the_users_prompt_becomes_the_baseline(eval_csv: Path, tmp_path: Path) -> None:
    path = tmp_path / "prod.txt"
    path.write_text("my production prompt", encoding="utf-8")

    target = _target(eval_csv, prompt_file=path)

    assert target.baseline_prompt.system == "my production prompt"
