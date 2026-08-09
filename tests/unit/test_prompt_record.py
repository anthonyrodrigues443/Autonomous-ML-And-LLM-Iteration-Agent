"""`prompts.yaml`: the harness owns the scoreboard, and the Critic outranks a score."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import yaml

from iterate.core import codegen
from iterate.core.critic import REJECTED
from iterate.core.prompting import Prompt
from iterate.deliver import prompt_record
from iterate.schemas.experiment import Candidate, Experiment, ExperimentResult, Metrics

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit

BASELINE = Prompt(system="Say whether the comment is toxic.", user_template="{input}")


def _experiment(
    *,
    description: str,
    score: float | None,
    system: str,
    rejected: str = "",
    submitted: bool = True,
) -> Experiment:
    changes: dict[str, object] = {"code": "..."}
    if rejected:
        changes[REJECTED] = rejected
    artifacts = {}
    if submitted:
        artifacts[codegen.PROMPT_JSON] = (
            '{"system": ' + f'"{system}"' + ', "user_template": "{input}"}'
        )
    result = ExperimentResult(
        experiment_id="e",
        metrics=(
            Metrics(values={"accuracy": score}, primary="accuracy", direction="maximize")
            if score is not None
            else None
        ),
        error=None if score is not None else "kernel died",
        artifacts=artifacts,
    )
    return Experiment(
        candidate=Candidate(description=description, changes=changes, rationale="r"),
        target="prompt",
        hypothesis="h",
        status="completed" if score is not None else "failed",
        result=result,
    )


def _build(history: list[Experiment], baseline_score: float | None = 0.61) -> dict:
    text = prompt_record.build(
        task="Say whether the comment is toxic.",
        metric="accuracy",
        direction="maximize",
        model_under_test="gemma4:12b",
        baseline_prompt=BASELINE,
        baseline_score=baseline_score,
        history=history,
    )
    return yaml.safe_load(text)


def test_the_baseline_is_version_zero() -> None:
    document = _build([])

    assert document["versions"][0]["version"] == "v0"
    assert document["versions"][0]["kind"] == "baseline"
    assert document["versions"][0]["score"] == pytest.approx(0.61)


def test_the_best_scoring_version_is_marked() -> None:
    document = _build(
        [
            _experiment(description="added examples", score=0.70, system="A"),
            _experiment(description="tightened wording", score=0.66, system="B"),
        ]
    )

    marked = [v["version"] for v in document["versions"] if v["best"]]

    assert marked == ["v1"]


def test_a_rejected_score_can_never_be_marked_best() -> None:
    """That number is not a result. Marking it best would hand the user a prompt the
    Critic proved was cheating."""
    document = _build(
        [
            _experiment(description="honest gain", score=0.70, system="A"),
            _experiment(
                description="suspicious jump",
                score=0.95,
                system="B",
                rejected="few-shot examples restated the answer key",
            ),
        ]
    )

    best = [v["version"] for v in document["versions"] if v["best"]]

    assert best == ["v1"]
    assert document["versions"][2]["rejected"].startswith("few-shot")


def test_a_rejected_version_still_appears() -> None:
    """A prompt that looked good and was not is worth being able to see."""
    document = _build([_experiment(description="x", score=0.95, system="B", rejected="leak")])

    assert len(document["versions"]) == 2


def test_a_minimising_metric_picks_the_lowest() -> None:
    text = prompt_record.build(
        task="t",
        metric="rmse",
        direction="minimize",
        model_under_test="m",
        baseline_prompt=BASELINE,
        baseline_score=10.0,
        history=[
            _experiment(description="a", score=8.0, system="A"),
            _experiment(description="b", score=12.0, system="B"),
        ],
    )
    document = yaml.safe_load(text)

    assert [v["version"] for v in document["versions"] if v["best"]] == ["v1"]


def test_a_failed_experiment_with_no_prompt_is_skipped() -> None:
    document = _build([_experiment(description="died", score=None, system="", submitted=False)])

    assert len(document["versions"]) == 1


def test_when_nothing_scored_nothing_is_best() -> None:
    document = _build([], baseline_score=None)

    assert not any(v["best"] for v in document["versions"])


def test_prompts_render_as_readable_blocks(tmp_path: Path) -> None:
    """A multi-line prompt dumped in flow style is unusable as a deliverable."""
    multiline = Prompt(system="line one\nline two\nline three", user_template="{input}")
    text = prompt_record.build(
        task="t",
        metric="accuracy",
        direction="maximize",
        model_under_test="m",
        baseline_prompt=multiline,
        baseline_score=0.5,
        history=[],
    )

    assert "system: |" in text
    assert "\\n" not in text


def test_the_header_explains_the_placeholders() -> None:
    text = prompt_record.build(
        task="t",
        metric="accuracy",
        direction="maximize",
        model_under_test="m",
        baseline_prompt=BASELINE,
        baseline_score=0.5,
        history=[],
    )

    assert "{input}" in text
    assert "production" in text


def test_write_puts_the_file_in_the_run_directory(tmp_path: Path) -> None:
    path = prompt_record.write(
        tmp_path / "run-1",
        task="t",
        metric="accuracy",
        direction="maximize",
        model_under_test="m",
        baseline_prompt=BASELINE,
        baseline_score=0.5,
        history=[],
    )

    assert path.name == "prompts.yaml"
    assert yaml.safe_load(path.read_text())["model_under_test"] == "m"
