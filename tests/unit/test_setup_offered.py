"""What the image metric step may offer, what it refuses before a baseline trains, and
the table proposals it must leave alone."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from iterate.adapters.data.tabular import TabularDataset, dataset_from_frames
from iterate.core import setup
from iterate.core.researcher import Setup
from iterate.core.scoring import PROBA_METRICS, score

pytestmark = pytest.mark.unit


def _data(train: list[Any], holdout: list[Any], task: str | None = None) -> TabularDataset:
    def frame(y: list[Any]) -> pd.DataFrame:
        return pd.DataFrame({"x": range(len(y)), "y": y})

    return dataset_from_frames(frame(train), frame(holdout), "y", task=task)


@pytest.mark.parametrize("k", [2, 3, 10, 102])
def test_every_offered_class_metric_scores_on_that_many_classes(k: int) -> None:
    labels = np.repeat(np.arange(k), 3).tolist()
    data = _data(labels, labels, task="classification")
    proba = np.random.default_rng(1).dirichlet(np.ones(k), len(labels))
    offered = setup.offered_for_images(data)
    for name in offered:
        values = score("classification", labels, proba.argmax(1), y_proba=proba, include=(name,))
        assert np.isfinite(values[name]), name
    # On two classes a top-3 score is 1.000 for any model at all.
    assert ("top_k_accuracy" in offered) is (k > 2)


def test_a_holdout_missing_a_class_offers_no_probability_metric() -> None:
    data = _data(["a", "b", "c", "d"] * 4, ["a", "b", "c"] * 3)
    offered = setup.offered_for_images(data)
    assert not set(offered) & PROBA_METRICS
    assert "accuracy" in offered
    assert "no image of 1 training class(es)" in str(
        setup.refuse_labels(data, metric="roc_auc", average=None)
    )


def test_a_two_class_holdout_under_more_training_classes_is_refused_whatever_the_metric() -> None:
    data = _data(["a", "b", "c"] * 4, ["a", "b"] * 3)
    for metric in (None, "accuracy", "f1_macro", "roc_auc"):
        assert "add holdout images of the missing classes" in str(
            setup.refuse_labels(data, metric=metric, average=None)
        )
    # Why: the panel reads two classes off the holdout, so the first prediction of the
    # missing class fails the scoring, after a baseline has trained for its full length.
    with pytest.raises(ValueError, match="multiclass"):
        score("classification", ["a", "b"] * 3, ["a", "b", "c"] * 2, include=("accuracy",))


def test_an_average_the_class_count_cannot_take_is_refused() -> None:
    three = _data(["a", "b", "c"] * 4, ["a", "b", "c"] * 2)
    assert "needs a two-class target" in str(
        setup.refuse_labels(three, metric=None, average="binary")
    )
    assert setup.refuse_labels(three, metric=None, average="macro") is None
    two = _data(["a", "b"] * 4, ["a", "b"] * 2)
    assert setup.refuse_labels(two, metric=None, average="binary") is None


def test_numbers_offer_one_name_per_ruler_and_nothing_a_negative_prediction_breaks() -> None:
    names = setup.offered_for_images(_data([1.5, 2.5, 3.5] * 4, [2.0, 3.0]))
    assert {"rmse", "mae", "mse", "r2"} <= set(names)
    assert not {
        "root_mean_squared_error",
        "mean_squared_log_error",
        "mean_poisson_deviance",
    } & set(names)


def test_an_explicit_number_metric_is_refused_only_where_scikit_learn_refuses_it() -> None:
    zero = _data([0.0, 10.0, 20.0] * 4, [5.0, 15.0], task="regression")
    assert "cannot score labels as low as 0" in str(
        setup.refuse_labels(zero, metric="mean_gamma_deviance", average=None)
    )
    # Counts with zeros are what poisson is for, and msle only refuses below -1.
    assert setup.refuse_labels(zero, metric="mean_poisson_deviance", average=None) is None
    assert setup.refuse_labels(zero, metric="mean_squared_log_error", average=None) is None
    below = _data([-2.0, 10.0, 20.0] * 4, [5.0, 15.0], task="regression")
    assert "cannot score labels as low as -2" in str(
        setup.refuse_labels(below, metric="root_mean_squared_log_error", average=None)
    )
    assert setup.refuse_labels(below, metric="rmse", average=None) is None


def test_only_an_image_run_chooses_its_metric_with_no_papers() -> None:
    """A failed search must not decide how an image run is measured; a table or prompt
    run keeps main's deterministic default rather than taking an ungrounded pick."""
    from iterate.core.researcher import Researcher
    from iterate.schemas.llm import ChatResponse, ToolCall

    class Capture:
        def __init__(self) -> None:
            self.called: list[str] = []

        def chat(self, messages: list[Any], **kw: Any) -> ChatResponse:
            tool = kw["tools"][0]
            self.called.append(tool.name)
            arguments = {"queries": []} if tool.name == "plan_queries" else {"metric": "mae"}
            call = ToolCall(id="1", name=tool.name, arguments=arguments)
            return ChatResponse(model="m", tool_calls=[call])

    def pass_with(family: str, allow: bool) -> tuple[list[str], Any]:
        client = Capture()
        found = Researcher(
            client,
            metric="",
            direction="maximize",  # type: ignore[arg-type]
            family=family,
            sources=[],
        ).research(
            profile="p", choose_setup=True, allowed_metrics=["mae"], allow_without_papers=allow
        )
        return client.called, found.setup

    assert pass_with("tabular", False) == (["plan_queries"], None)
    assert pass_with("prompt", False) == (["plan_queries"], None)
    called, chosen = pass_with("vision", True)
    assert called == ["plan_queries", "choose_setup"]
    assert chosen is not None
    assert chosen.metric == "mae"


def test_a_proposal_outside_the_offered_list_falls_back_and_a_table_keeps_its_own() -> None:
    data = _data(["a", "b", "c"] * 4, ["a", "b", "c"])
    offered = setup.offered_for_images(data)
    resolved = setup.resolve(data, proposed=Setup(metric="jaccard"), offered=offered)
    assert (resolved.metric, resolved.chosen_by_agent) == ("f1_macro", False)
    # Tables pass no list, so a name they always accepted is still accepted.
    numbers = _data([1.5, 2.5, 3.5] * 4, [2.0, 3.0])
    for name in ("mean_absolute_error", "root_mean_squared_error", "mean_squared_log_error"):
        table = setup.resolve(numbers, proposed=Setup(metric=name))
        assert (table.metric, table.chosen_by_agent) == (name, True)
