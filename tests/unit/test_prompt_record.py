"""`prompts.yaml`: the harness owns the scoreboard, and the Critic outranks a score."""

from __future__ import annotations

import json
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
    over_budget: float | None = None,
) -> Experiment:
    changes: dict[str, object] = {"code": "..."}
    if rejected:
        changes[REJECTED] = rejected
    if over_budget is not None:
        changes["over_budget"] = over_budget
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


def test_the_full_holdout_number_leads_the_file() -> None:
    """Per-version scores come from the cheap slice the search ranked on. That is the
    right basis for choosing between prompts and the wrong basis for quoting one, so
    the winner's full-holdout score is stated separately and first."""
    text = prompt_record.build(
        task="t",
        metric="f1",
        direction="maximize",
        model_under_test="gemma4:12b",
        baseline_prompt=BASELINE,
        baseline_score=0.86,
        history=[_experiment(description="one change", score=0.91, system="A")],
        final_score={"score": 0.8814123, "n": 300},
    )
    document = yaml.safe_load(text)

    assert document["best_score_on_full_holdout"] == pytest.approx(0.8814)
    assert document["full_holdout_records"] == 300
    assert "number to quote" in document["note"]
    # the loop score is still recorded, just not presented as the headline
    assert document["versions"][1]["score"] == pytest.approx(0.91)


def test_no_final_score_means_no_misleading_headline() -> None:
    """A run that used the full holdout throughout, or whose re-score failed, must
    not gain a field implying a measurement that never happened."""
    document = _build([_experiment(description="x", score=0.9, system="A")])

    assert "best_score_on_full_holdout" not in document
    assert "note" not in document


def test_the_serving_profile_sits_above_the_versions() -> None:
    serving = {
        "requests_per_hour": 1000,
        "chosen": None,
        "by_cloud": [],
        "usd_per_1k_requests": None,
        "prices_as_of": "2026-09-25",
        "unpriced_because": "no public price for m on groq",
        "basis": [],
    }
    text = prompt_record.build(
        task="t",
        metric="accuracy",
        direction="maximize",
        model_under_test="m",
        baseline_prompt=BASELINE,
        baseline_score=0.5,
        history=[],
        serving=serving,
    )
    document = yaml.safe_load(text)

    assert document["serving"] == serving
    assert list(document).index("serving") < list(document).index("versions")


def test_a_version_over_the_serving_budget_is_never_marked_best() -> None:
    """The loop never banks it, so the file must not tell the user to ship it: a best
    the user cannot afford to serve is the one thing the wall exists to prevent."""
    document = _build(
        [
            _experiment(description="within the budget", score=0.70, system="A"),
            _experiment(description="bigger few-shot", score=0.95, system="B", over_budget=147.17),
        ]
    )

    assert [v["version"] for v in document["versions"] if v["best"]] == ["v1"]
    assert document["versions"][2]["over_budget"] == (
        "costs about $147 a month to serve, over the serving budget"
    )
    assert "over_budget" not in document["versions"][1]


def test_when_every_version_is_over_the_budget_the_baseline_is_best() -> None:
    document = _build([_experiment(description="x", score=0.95, system="B", over_budget=40.0)])

    assert [v["version"] for v in document["versions"] if v["best"]] == ["v0"]
    assert len(document["versions"]) == 2


def test_the_wall_stamp_on_the_experiment_is_what_the_record_reads() -> None:
    """The loop stamps each finished experiment and hands the same objects to the
    record, so the flag needs no plumbing: priced here the way the loop prices them."""
    from functools import partial

    from iterate.core import serving

    def measured(system: str, score: float, tokens_in: float) -> Experiment:
        experiment = _experiment(description=system, score=score, system=system)
        assert experiment.result is not None
        experiment.result.artifacts[codegen.PROMPT_JSON] = json.dumps(
            {
                "system": system,
                "user_template": "{input}",
                "records_measured": 100,
                "tokens_in_per_record": tokens_in,
                "tokens_out_per_record": 8,
            }
        )
        return experiment

    wall = serving.Wall(
        requests_per_hour=1000,
        prices=serving.load_prices(),
        budget=50.0,
        facts_of=partial(
            serving.experiment_facts, family="prompt", provider="openai", model="gpt-4o-mini"
        ),
    )
    short, long = measured("short", 0.70, 100), measured("long", 0.95, 2000)
    for experiment in (short, long):
        wall.stamp(experiment)

    document = _build([short, long])

    assert "over_budget" not in short.candidate.changes
    assert [v["version"] for v in document["versions"] if v["best"]] == ["v1"]
    assert document["versions"][2]["over_budget"].startswith("costs about $")
