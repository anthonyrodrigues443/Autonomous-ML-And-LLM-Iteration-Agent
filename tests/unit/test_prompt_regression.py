"""Regression on the prompt path: numbers in, a score out."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
import pytest

from iterate.adapters.data.tabular import load_csv
from iterate.core.prompt_runtime import UNPARSEABLE, coerce_number, numeric_answer_tool
from iterate.core.scoring import REGISTRY, direction, metric_module, score, task_for_metric
from iterate.schemas.llm import ChatResponse, ToolCall
from iterate.targets.prompt import PromptTarget, median_answer, numeric_range

if TYPE_CHECKING:
    from pathlib import Path

    from iterate.schemas.llm import Message

pytestmark = pytest.mark.unit


@pytest.fixture
def ratings(tmp_path: Path) -> Any:
    path = tmp_path / "ratings.csv"
    pd.DataFrame(
        {
            "a": [f"sentence {i}" for i in range(60)],
            "b": [f"other {i}" for i in range(60)],
            "score": [round(i % 6 + 0.5, 1) for i in range(60)],
        }
    ).to_csv(path, index=False)
    return load_csv(path, target="score", stratify=False)


# ─── the correlations ───────────────────────────────────────────────────────


@pytest.mark.parametrize("name", ["pearson", "spearman", "kendall"])
def test_the_correlations_are_selectable_and_importable(name: str) -> None:
    """None of the three is an sklearn scorer. The constraint that matters is not
    "lives in sklearn" but "the agent can import and compute it"."""
    import importlib

    assert name in REGISTRY
    assert task_for_metric(name) == "regression"
    assert direction(name) == "maximize"
    assert metric_module(name) == "scipy.stats"
    assert hasattr(importlib.import_module(metric_module(name)), REGISTRY[name].sklearn_func)


def test_the_correlations_stay_out_of_the_always_computed_panel() -> None:
    """Adding them to PANEL would change the recorded metric set of every regression
    run that has ever happened and make old history incomparable to new."""
    from iterate.core.scoring import PANEL

    for name in ("pearson", "spearman", "kendall"):
        assert name not in PANEL


def test_a_well_ranked_but_badly_calibrated_prompt_is_seen_as_useful() -> None:
    """The reason to have correlations at all: a prompt that rates everyone three
    points high but in exactly the right order is useful, because the offset can be
    calibrated away. rmse calls it a failure."""
    truth = list(range(1, 11))
    generous = [x + 3 for x in truth]

    values = score("regression", truth, generous, include=("pearson", "kendall", "rmse"))

    assert values["pearson"] == pytest.approx(1.0)
    assert values["kendall"] == pytest.approx(1.0)
    assert values["rmse"] == pytest.approx(3.0)


def test_a_constant_answer_scores_zero_not_nan() -> None:
    """Answering "5" to everything is the likeliest failure of a rating prompt, and
    scipy calls the correlation undefined there. An unguarded nan either crashes the
    run or compares falsely against every real score."""
    values = score("regression", list(range(1, 11)), [5] * 10, include=("pearson", "spearman"))

    assert values["pearson"] == 0.0
    assert values["spearman"] == 0.0


# ─── the numeric answer ─────────────────────────────────────────────────────


def test_the_answer_tool_bounds_the_number() -> None:
    """Same trick as the enum: a rating outside the observed range is not something
    the model can express."""
    spec = numeric_answer_tool(0, 5).parameters["properties"]["value"]

    assert spec["type"] == "number"
    assert spec["minimum"] == 0
    assert spec["maximum"] == 5


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("4", "4.0"),
        ("4.5", "4.5"),
        ("I'd say 3", "3.0"),
        ("5 stars", "5.0"),
        ("GPT-3 rates it 4", UNPARSEABLE),  # two numbers, one of them an identifier
        ("between 3 and 4", UNPARSEABLE),  # naming two invents a precision
        ("9", UNPARSEABLE),  # out of range
        ("no idea", UNPARSEABLE),
    ],
)
def test_a_number_is_read_out_of_a_reply(reply: str, expected: str) -> None:
    assert coerce_number(reply, 0, 5) == expected


# ─── the target ─────────────────────────────────────────────────────────────


def test_bounds_and_floor_come_from_training_answers_only(ratings: Any) -> None:
    """A range widened by the holdout would be a leak: it would tell the model which
    values the answer key uses."""
    low, high = numeric_range(ratings)

    assert low == pytest.approx(float(ratings.train_target.min()))
    assert high == pytest.approx(float(ratings.train_target.max()))
    assert median_answer(ratings) == pytest.approx(float(ratings.train_target.median()))


def test_the_metric_decides_the_task_not_the_column(ratings: Any, monkeypatch: Any) -> None:
    """The same rating column is ten ordered classes under f1 and a score under a
    correlation. Both are legitimate; the user's metric says which."""

    class Scripted:
        model = "scripted"

        def chat(self, messages: list[Message], **kwargs: Any) -> ChatResponse:
            return ChatResponse(
                model="scripted",
                tool_calls=[ToolCall(id="1", name="answer", arguments={"value": 3.0})],
            )

    monkeypatch.setattr("iterate.llm.factory.build_client", lambda *a, **k: Scripted())

    numeric = PromptTarget(
        ratings, metric="rmse", task="rate it", target_backend="ollama", target_model="s"
    )
    assert numeric.labels is None  # a number, so no enum
    assert numeric.baseline().succeeded


def test_an_unusable_answer_becomes_the_median_never_a_free_pass() -> None:
    """Measured across four options on a 300-row rating task with 10% refusals:
    DROPPING the unparseable rows scored BETTER than answering them honestly
    (rmse 0.691 vs 0.701, pearson 0.894 vs 0.890), so refusing the hard rows would
    have been a winning strategy. The median penalises, and unlike a worst-case
    substitution it cannot let a few refusals swamp the score.
    """
    truth = [1.0, 2.0, 3.0, 4.0, 5.0]
    honest = [1.2, 2.1, 2.8, 4.3, 4.9]
    median = 3.0
    refused_two = [1.2, 2.1, median, median, 4.9]  # rows 3 and 4 unusable

    honest_rmse = score("regression", truth, honest, include=("rmse",))["rmse"]
    refused_rmse = score("regression", truth, refused_two, include=("rmse",))["rmse"]

    assert refused_rmse > honest_rmse, "refusing must never score better than answering"


def test_the_session_asks_and_scores_the_way_the_host_does(
    ratings: Any, tmp_path: Path, monkeypatch: Any
) -> None:
    """The gap my other tests missed: they exercised PromptTarget.baseline(), not
    the SESSION, and the session is how real runs happen.

    Its `ask` fell back to the label tool and its `evaluate` hardcoded
    'classification', so on a regression run the agent iterated against a different
    ruler than the one deciding its score.
    """
    import json

    from iterate.core import codegen
    from iterate.schemas.experiment import Candidate

    class Scripted:
        model = "scripted"

        def chat(self, messages: list[Message], *, tools: Any = None, **kw: Any) -> ChatResponse:
            # record what tool shape the session offered
            Scripted.seen = tools[0].parameters["properties"]["value"]
            return ChatResponse(
                model="scripted",
                tool_calls=[ToolCall(id="1", name="answer", arguments={"value": 2.5})],
            )

    monkeypatch.setattr("iterate.llm.factory.build_client", lambda *a, **k: Scripted())

    target = PromptTarget(
        ratings, metric="pearson", task="rate it", target_backend="ollama", target_model="s"
    )
    job = target.build_code_job(
        Candidate(description="x", changes={"code": "pass"}, rationale="r")
    )
    for name, blob in job.inputs.items():
        (tmp_path / name).write_bytes(blob)
    monkeypatch.chdir(tmp_path)

    meta = json.loads(job.inputs[codegen.META_JSON])
    assert meta["task_kind"] == "regression"
    assert meta["numeric_range"] is not None
    assert meta["median"] is not None

    namespace: dict[str, Any] = {}
    exec(codegen.prompt_session_preamble(), namespace)

    sample = namespace["X_train"].head(4)
    answers = namespace["ask"](namespace["BASE"], sample)

    # the session offered a NUMBER tool, not an enum
    assert Scripted.seen["type"] == "number"
    assert "minimum" in Scripted.seen

    # and scored with the run's regression metric rather than a classification one
    value = namespace["evaluate"](answers, namespace["y_train"].head(4))
    assert isinstance(value, float)
    assert -1.0 <= value <= 1.0  # a correlation, not an f1
