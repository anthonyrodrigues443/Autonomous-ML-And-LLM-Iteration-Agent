"""The image family's four roles and its lever gate, on sessions shaped as the helpers print.

The cells here carry the exact lines `vision_session` writes: the runner's epoch lines,
`FIT {...}` with the recipe it started from under "from", `MODEL {...}` from an own-code
`evaluate(model=...)`, and `SUBMITTED {...}`. test_vision_contract.py holds the other end
of that contract, where a real session prints them.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from iterate.core import coder as coder_mod
from iterate.core import critic as critic_mod
from iterate.core import researcher as researcher_mod
from iterate.core import supervisor as sup
from iterate.core import vision_levers as vl
from iterate.schemas.experiment import Candidate, Experiment, ExperimentResult, Metrics
from iterate.schemas.llm import ChatResponse, ToolCall

RECIPE: dict[str, Any] = {
    "backbone": "resnet18",
    "image_size": 64,
    "unfreeze": "all",
    "epochs": 3,
    "batch_size": 64,
    "lr": 0.001,
    "optimizer": "adamw",
    "schedule": "onecycle",
    "augment": "flip",
    "label_smoothing": 0.0,
    "head_init": "random",
    "seed": 42,
}
BASELINE: dict[str, Any] = {
    **RECIPE,
    "backbone": "simple_cnn",
    "epochs": 20,
    "schedule": "cosine",
    "augment": "none",
}
BENCH: dict[str, Any] = {**RECIPE, "augment": "flip"}
FINDINGS = (
    "- timm_vit_base_patch16_224 <doi:10.3390/rs13030516>\n"
    "- fine-tune timm efficientnet_b0 on all layers <doi:10.3390/rs71114680>"
)
# Recorded from gemma4:12b on 2026-09-17 (roles-and-levers/live/calls): the brief it wrote
# from a facts-only ready line, and the one it wrote from "class: change (because fact)".
LIVE_BRIEF_1 = (
    "next: own-model: a literature finding names efficientnet_b0 and vit_base_patch16_224."
)
LIVE_BRIEF_2 = (
    "next: epochs: keep the recipe and train 6 epochs (because training was still rising at "
    "the last epoch (0.912 -> 0.981))."
)
LIVE_BRIEF_OWN = (
    "next: own-model: write torch code for timm efficientnet_b0 with pretrained weights, all "
    "layers, 3 epochs at 64 px, and score it with evaluate() (because a literature finding "
    "names efficientnet_b0)."
)


def fit_cell(
    recipe: dict[str, Any],
    trains: list[float],
    val: float,
    *,
    secs: int = 12,
    planned: int | None = None,
    start: dict[str, Any] | None = None,
    code: str = "f = fit()",
) -> tuple[dict[str, Any], dict[str, Any]]:
    planned = planned if planned is not None else int(recipe["epochs"])
    lines = [
        f"epoch {i + 1}/{planned} loss=0.5000 train_acc={t:.4f} {secs}s"
        for i, t in enumerate(trains)
    ]
    if len(trains) < planned:
        lines.append(f"stopped in epoch {len(trains) + 1}/{planned}: the fit budget ran out")
    payload = {
        **recipe,
        "epochs_planned": planned,
        "epochs_run": len(trains),
        "seconds": secs * len(trains),
        "val": val,
        "val_accuracy": val,
        "from": dict(start if start is not None else recipe),
    }
    lines.append("FIT " + json.dumps(payload))
    return {"code": code, "stdout": "\n".join(lines), "error": None, "source": "agent"}, payload


def submit_cell(payload: dict[str, Any], code: str = "submit(f)") -> dict[str, Any]:
    return {
        "code": code,
        "stdout": "SUBMITTED " + json.dumps(payload),
        "error": None,
        "source": "agent",
    }


def model_cell(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "code": "evaluate(val, model='x')",
        "stdout": "MODEL " + json.dumps(payload),
        "error": None,
        "source": "agent",
    }


def experiment(
    cells: list[dict[str, Any]],
    brief: str,
    score: float | None,
    *,
    carried: dict[str, Any],
    error: str | None = None,
    metric: str = "accuracy",
    direction: str = "maximize",
) -> Experiment:
    changes: dict[str, Any] = {
        "code": "\n".join(c["code"] for c in cells),
        "cells": cells,
        "levers_moved": vl.moved_levers(cells, carried),
    }
    if (recipe := vl.submitted(cells)) is not None:
        changes["recipe"] = recipe
    result = (
        ExperimentResult(experiment_id="e", error=error)
        if error
        else ExperimentResult(
            experiment_id="e",
            metrics=Metrics(
                values={metric: score}, primary=metric, direction=direction, n_samples=100
            ),
        )
    )
    return Experiment(
        candidate=Candidate(description="try", changes=changes, rationale=brief),
        target="vision-model",
        hypothesis=brief,
        status="failed" if error else "completed",
        iteration=1,
        result=result,
        started_at=datetime.now(UTC),
        finished_at=datetime.now(UTC),
    )


def first_try(
    *,
    secs: int = 12,
    trains: tuple[float, ...] = (0.912, 0.958, 0.981),
    val: float = 0.9771,
    score: float = 0.9785,
) -> Experiment:
    cell, payload = fit_cell(RECIPE, list(trains), val, secs=secs, start=BENCH)
    return experiment(
        [cell, submit_cell(payload)], "next: backbone: fine-tune resnet18", score, carried=BASELINE
    )


def own_try(model: str = "efficientnet_b0", score: float = 0.9830) -> Experiment:
    payload = {
        "model": model,
        "image_size": 64,
        "epochs": 3,
        "seconds": 90,
        "val": 0.983,
        "val_accuracy": 0.983,
    }
    cells = [
        model_cell(payload),
        submit_cell(payload, code=f"submit_probabilities(h, model='{model}')"),
    ]
    return experiment(
        cells, f"next: own-model: write torch code for {model}", score, carried=RECIPE
    )


def ready_for(history: list[Experiment], carried: Experiment | None, **kw: Any) -> list[vl.Ready]:
    kw.setdefault("task", "classification")
    kw.setdefault("direction", "maximize")
    return vl.ready(history, carried, **kw)


# ─── reading the brief ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("brief", "expected"),
    [
        ("next: epochs: keep the backbone and train 6 epochs", ["epochs"]),
        ("next: FINE-TUNE THE BENCH: unfreeze all, 3 epochs", []),
        ("next: backbone swap: convnext_tiny", ["backbone"]),
        ("next: image size and epochs: 128 px", ["image-size", "epochs"]),
        ("Next: Learning Rate: 3e-4", ["fine-tune-depth"]),
    ],
)
def test_the_class_is_read_from_the_move_tag_only(brief: str, expected: list[str]) -> None:
    assert vl.classes_named(brief) == expected


def test_the_change_clause_drops_the_reason_so_a_quoted_value_is_not_a_proposal() -> None:
    brief = (
        "next: backbone: keep the recipe and swap the backbone to convnext_tiny (because "
        "resnet18 took 12s an epoch, so convnext_tiny fits the budget at about 126s)."
    )
    assert vl.change_clause(brief) == "keep the recipe and swap the backbone to convnext_tiny"
    assert vl.proposed_value("backbone", vl.change_clause(brief)) == "convnext_tiny"
    assert vl.missing_value(brief) is None
    # The same text read whole names two backbones and states no one value.
    assert vl.proposed_value("backbone", brief) is None


def test_the_clause_survives_the_other_shape_a_move_arrives_in() -> None:
    fallback = "next: epochs: resnet18 took 12s an epoch, so keep the recipe and train 6 epochs."
    assert vl.proposed_value("epochs", vl.change_clause(fallback)) == 6
    assert vl.proposed_value("epochs", vl.change_clause(LIVE_BRIEF_2)) == 6
    assert (
        vl.proposed_value("image-size", vl.change_clause("next: image-size: train at 128 px"))
        == 128
    )
    named = "next: epochs: retry that try as fit(backbone='resnet18', image_size=128, epochs=2)"
    assert vl.proposed_value("epochs", vl.change_clause(named)) == 2
    assert vl.proposed_value("image-size", vl.change_clause(named)) == 128


@pytest.mark.parametrize(
    "word",
    ["evaluate", "evaluation", "clipping", "train/val", "flip/rotation", "swing", "training"],
)
def test_an_english_word_names_no_model(word: str) -> None:
    assert vl.models_named(f"write torch code for {word} with pretrained weights") == []


@pytest.mark.parametrize(
    "name",
    [
        "maxvit_tiny_tf_224",
        "coatnet_0_rw_224",
        "convnextv2_tiny",
        "resnet34",
        "edgenext_small",
        "tf_efficientnetv2_s",
        "efficientnet_b0",
        "vit_base_patch16_224",
    ],
)
def test_every_pretrained_architecture_timm_ships_is_a_model(name: str) -> None:
    assert vl.models_named(f"fine-tune timm {name} on all layers") == [name]


def test_a_network_fit_already_trains_is_a_backbone_move_not_an_own_model_one() -> None:
    assert vl.models_named("swap the backbone to convnext_tiny") == []
    assert vl.models_named("write torch code for facebook/convnextv2-tiny-1k-224") == [
        "facebook/convnextv2-tiny-1k-224"
    ]


def test_an_own_model_move_naming_one_model_passes_and_a_doi_names_nothing() -> None:
    assert vl.missing_value(LIVE_BRIEF_OWN) is None
    assert vl.missing_value(LIVE_BRIEF_1) is not None  # two models, and no change stated
    assert vl.models_named("a finding <doi:10.3390/rs13030516> names efficientnet_b0") == [
        "efficientnet_b0"
    ]


def test_every_entry_the_ready_line_can_emit_passes_the_guard() -> None:
    """A ready entry reaches the model two ways: copied word for word, and as the
    harness's own fallback. The supervisor refusing its own line is the failure the
    skeptic measured, so every shape ready() can build is checked here."""
    best, own = first_try(), own_try()
    probe_cell, probe_payload = fit_cell(
        {**RECIPE, "unfreeze": "none", "epochs": 0}, [], 0.90, planned=0, start=BENCH
    )
    probe = experiment(
        [probe_cell, submit_cell(probe_payload)],
        "next: backbone: probe resnet18",
        0.90,
        carried=BASELINE,
    )
    overfit_cell, overfit_payload = fit_cell(RECIPE, [0.90, 0.95, 0.999], 0.955, start=BENCH)
    overfit = experiment(
        [overfit_cell, submit_cell(overfit_payload)],
        "next: backbone: fine-tune resnet18",
        0.955,
        carried=BASELINE,
    )
    cut_cell, cut_payload = fit_cell(
        {**RECIPE, "backbone": "convnext_tiny", "image_size": 128},
        [0.9, 0.95],
        0.97,
        planned=2,
        secs=150,
    )
    cut = experiment(
        [cut_cell, submit_cell(cut_payload)],
        "next: backbone: swap the backbone to convnext_tiny",
        0.975,
        carried=RECIPE,
    )
    oom = experiment(
        [
            {
                "code": "f = fit(image_size=384)",
                "stdout": "",
                "source": "agent",
                "error": "out of memory on mps at batch_size=128, image_size=384: halve one of them",
            }
        ],
        "next: image-size: keep the best and train at 384 px",
        None,
        carried=RECIPE,
        error="no predictions were written",
    )
    failed_own = experiment(
        [
            {
                "code": "model = timm.create_model(NAME)",
                "stdout": "",
                "source": "agent",
                "error": "RuntimeError: shape mismatch",
            }
        ],
        "next: own-model: write torch code for efficientnet_b0",
        None,
        carried=RECIPE,
        error="no predictions were written",
    )
    rising = first_try(trains=(0.80, 0.85, 0.95))
    cases: list[tuple[str, list[Experiment], Experiment | None, dict[str, Any]]] = [
        ("experiment 1", [], None, {"findings": FINDINGS}),
        ("experiment 1, no findings", [], None, {}),
        ("a fine-tune carried", [best], best, {"findings": FINDINGS, "median_width": 64}),
        ("a slow fine-tune", [first_try(secs=60)], first_try(secs=60), {"findings": FINDINGS}),
        ("a rising trail", [rising], rising, {}),
        ("an overfitting fine-tune", [overfit], overfit, {"findings": FINDINGS}),
        ("a linear probe", [probe], probe, {}),
        ("an own-code best", [own], own, {"findings": FINDINGS}),
        ("a budget cut", [best, cut], best, {"findings": FINDINGS}),
        ("out of memory", [best, oom], best, {"findings": FINDINGS}),
        ("a failed own try", [best, failed_own], best, {"findings": FINDINGS}),
        ("a number run", [best], best, {"task": "regression", "direction": "minimize"}),
    ]
    seen = 0
    for label, history, carried, kw in cases:
        items = ready_for(history, carried, **kw)
        assert items, f"{label} opens no lever"
        shapes = [f"next: {r.lever}: {r.move} (because {r.reason})." for r in items]
        fallback = vl.fallback_move(items)
        assert fallback is not None
        shapes.append(fallback[1])
        for text in shapes:
            seen += 1
            assert vl.classes_named(text)[:1] == [vl.lever_class(text)], f"{label}: {text}"
            assert vl.missing_value(text) is None, f"{label}: {text}"
    assert seen >= 30


def test_the_recorded_live_briefs_the_guard_should_pass_do() -> None:
    assert vl.missing_value(LIVE_BRIEF_2) is None
    assert vl.lever_class(LIVE_BRIEF_2) == "epochs"
    assert vl.missing_value(LIVE_BRIEF_OWN) is None
    assert vl.lever_class(LIVE_BRIEF_OWN) == "own-model"
    # The one the model wrote from a facts-only line is refused on purpose: it names two
    # models and no change at all. The nudge asks for one, and the retry passes.
    assert "exactly ONE model" in (vl.missing_value(LIVE_BRIEF_1) or "")


# ─── reading what a session printed ──────────────────────────────────────────


def test_the_first_pretrained_fit_moves_the_backbone_and_nothing_its_kind_change_implies() -> None:
    assert first_try().candidate.changes["levers_moved"] == ["backbone"]


def test_an_epochs_brief_whose_session_swapped_the_backbone_is_unmeasured() -> None:
    cells = [fit_cell({**RECIPE, "backbone": "resnet50"}, [0.9, 0.95, 0.97], 0.97, start=RECIPE)[0]]
    assert not vl.moved("epochs", cells, RECIPE)
    assert vl.moved("backbone", cells, RECIPE)


def test_a_second_fit_is_measured_against_what_it_started_from() -> None:
    first, first_payload = fit_cell(
        {**RECIPE, "epochs": 6}, [0.9, 0.92, 0.94, 0.95, 0.96, 0.97], 0.97, start=RECIPE
    )
    second, _ = fit_cell(
        {**RECIPE, "epochs": 6, "augment": "flip_crop"}, [0.9] * 6, 0.98, start=first_payload
    )
    assert vl.moved("epochs", [first, second], RECIPE)
    assert vl.moved("augmentation", [first, second], RECIPE)


def test_a_tagged_line_printed_by_a_cell_that_calls_no_helper_counts_for_nothing() -> None:
    forged, _ = fit_cell(
        {**RECIPE, "epochs": 6}, [0.9], 0.97, start=RECIPE, code="print('FIT ...')"
    )
    assert not vl.moved("epochs", [forged], RECIPE)


def test_an_errored_cell_is_not_evidence_but_its_out_of_memory_is() -> None:
    oom = {
        "code": "f = fit(image_size=384)",
        "stdout": "",
        "source": "agent",
        "error": "DeviceOutOfMemoryError: out of memory on mps at batch_size=64, image_size=384",
    }
    assert not vl.moved("image-size", [oom], RECIPE)
    assert "OUT OF MEMORY" in vl.evidence([oom], "accuracy")


def test_a_cell_that_raised_after_its_submit_keeps_the_submission_and_the_credit() -> None:
    """The host scores predictions.csv, which the helper wrote before the cell died, so
    dropping the payload would carry a recipe the run did not score."""
    won, payload = fit_cell(RECIPE, [0.9, 0.95, 0.97], 0.97, start=BASELINE)
    late = {
        "code": won["code"] + "\nsubmit(f)\nfit(image_size=224)",
        "stdout": won["stdout"] + "\nSUBMITTED " + json.dumps(payload),
        "source": "agent",
        "error": "DeviceOutOfMemoryError: out of memory on mps at batch_size=64, image_size=224",
    }
    clean = {**late, "error": None}
    assert vl.submitted([late]) == payload
    assert vl.moved_levers([late], BASELINE) == vl.moved_levers([clean], BASELINE)
    assert "backbone" in vl.moved_levers([late], BASELINE)
    # The fit that finished comes before the fit the memory refused.
    assert [t.kind for t in vl.tries([late])] == ["fit", "oom"]


def test_an_own_model_best_carries_its_code_and_a_fit_recipe_the_kernel_can_start_from() -> None:
    own = model_cell({"model": "efficientnet_b0", "image_size": 64, "epochs": 3, "val": 0.94})
    fit_line, _ = fit_cell(RECIPE, [0.9, 0.95, 0.97], 0.97, start=BASELINE)
    exp = experiment(
        [fit_line, own, submit_cell({"model": "efficientnet_b0", "image_size": 64, "epochs": 3})],
        "next: own-model: timm efficientnet_b0",
        0.978,
        carried=BASELINE,
    )
    assert vl.recipe_of(exp)["model"] == "efficientnet_b0"
    assert "model" not in vl.fit_recipe_of(exp)
    assert vl.fit_recipe_of(exp)["backbone"] == "resnet18"
    assert vl.fit_recipe_of(None) == {}


def test_own_code_measures_own_model_through_a_named_model_line() -> None:
    cells = [model_cell({"model": "timm/efficientnet_b0", "image_size": 64, "val": 0.97})]
    assert vl.moved("own-model", cells, RECIPE)
    assert not vl.moved("own-model", [model_cell({"val": 0.97})], RECIPE)


def test_the_fit_line_names_the_budget_cut_the_trail_and_the_gap() -> None:
    cut = [
        fit_cell(
            {**RECIPE, "image_size": 224, "epochs": 3},
            [0.90, 0.999],
            0.955,
            planned=3,
            secs=230,
            start=RECIPE,
        )[0]
    ]
    line = vl.evidence(cut, "accuracy")
    assert "ran 2 of 3 epochs at 230s each" in line
    assert "STOPPED BY THE FIT BUDGET" in line
    assert "(still rising)" in line
    assert "train above val by 0.044" in line


def test_a_plan_the_budget_shrank_before_the_first_epoch_is_still_a_cut() -> None:
    """dl.plan_epochs lowers the epochs before epoch 1, so a fit that ran every epoch it
    planned can still have run fewer than the brief asked for."""
    shrunk = [
        fit_cell(
            {**RECIPE, "epochs": 12}, [0.90, 0.95, 0.97], 0.97, planned=3, secs=150, start=RECIPE
        )[0]
    ]
    line = vl.evidence(shrunk, "accuracy")
    assert "planned 3 of the 12 asked" in line
    assert "STOPPED BY THE FIT BUDGET" in line


def test_a_still_rising_trail_needs_a_gain_worth_doubling_for() -> None:
    small = first_try(trains=(0.90, 0.95, 0.981))  # +0.031, EuroSAT's measured shape
    big = first_try(trains=(0.80, 0.85, 0.95))  # +0.10, Flowers102's
    assert "epochs" not in [r.lever for r in ready_for([small], small)]
    assert "epochs" in [r.lever for r in ready_for([big], big)]


# ─── what this run's numbers open ────────────────────────────────────────────


def test_a_budget_cut_opens_only_the_epochs_that_ran_on_the_recipe_that_was_cut() -> None:
    best = first_try()
    cut_cell, cut_payload = fit_cell(
        {**RECIPE, "backbone": "convnext_tiny", "image_size": 128},
        [0.9, 0.95],
        0.97,
        planned=3,
        secs=150,
    )
    cut = experiment(
        [cut_cell, submit_cell(cut_payload)],
        "next: backbone: swap the backbone to convnext_tiny",
        0.975,
        carried=RECIPE,
    )
    (repair,) = ready_for([best, cut], best)
    assert repair.lever == "epochs"
    # The coder starts from the carried best (resnet18, 3 epochs), so the repair has to
    # name the recipe that was cut or it repairs a recipe that never failed.
    assert "convnext_tiny" in repair.move
    assert "image_size=128" in repair.move
    assert vl.proposed_value("epochs", vl.change_clause(f"next: epochs: {repair.move}")) == 2


def test_an_out_of_memory_fit_opens_its_own_retry_with_a_smaller_batch() -> None:
    best = first_try()
    oom = experiment(
        [
            {
                "code": "f = fit(backbone='convnext_tiny')",
                "stdout": "",
                "source": "agent",
                "error": "out of memory on mps at batch_size=128, image_size=384: halve one of them",
            }
        ],
        "next: backbone: keep the recipe and swap the backbone to convnext_tiny",
        None,
        carried=RECIPE,
        error="no predictions were written",
    )
    (repair,) = ready_for([best, oom], best)
    assert repair.lever == "backbone"
    assert "convnext_tiny" in repair.move
    assert "batch_size=64" in repair.move


def test_an_out_of_memory_the_session_recovered_from_opens_the_ladder_not_a_repair() -> None:
    best = first_try()
    oom_cell = {
        "code": "f = fit(image_size=384)",
        "stdout": "",
        "source": "agent",
        "error": "out of memory on mps at batch_size=64, image_size=384",
    }
    won_cell, won_payload = fit_cell(
        {**RECIPE, "image_size": 128}, [0.9, 0.95, 0.97], 0.98, start=RECIPE
    )
    recovered = experiment(
        [oom_cell, won_cell, submit_cell(won_payload)],
        "next: image-size: keep the best and train at 128 px",
        0.987,
        carried=RECIPE,
    )
    items = ready_for([best, recovered], recovered)
    assert items
    assert not any("out of memory" in r.reason for r in items)


def test_a_failed_own_code_try_is_counted_as_tried_so_it_cannot_reopen_for_ever() -> None:
    best = first_try()
    failed = experiment(
        [
            {
                "code": "model = timm.create_model(NAME)",
                "stdout": "",
                "source": "agent",
                "error": "RuntimeError: shape mismatch",
            }
        ],
        "next: own-model: write torch code for efficientnet_b0 with pretrained weights",
        None,
        carried=RECIPE,
        error="no predictions were written",
    )
    (repair,) = ready_for([best, failed], best, findings=FINDINGS)
    assert repair.lever == "backbone"  # the failure's own repair, once
    # Two experiments spent on own-model without beating the best close the class.
    assert "own-model" not in [
        r.lever for r in ready_for([best, failed, failed], best, findings=FINDINGS)
    ]
    # And once the run moves on, the model that failed is not offered a third time: the
    # other name in the finding is.
    later = first_try(score=0.9790)
    own = [
        r
        for r in ready_for([failed, best, later], later, findings=FINDINGS)
        if r.lever == "own-model"
    ]
    assert own
    assert "efficientnet_b0" not in own[0].move


def test_two_experiments_spending_one_class_without_beating_the_best_close_it() -> None:
    best = first_try()
    losers = []
    for backbone in ("resnet50", "convnext_tiny"):
        cell, payload = fit_cell(
            {**RECIPE, "backbone": backbone}, [0.9, 0.95, 0.97], 0.95, secs=45, start=RECIPE
        )
        losers.append(
            experiment(
                [cell, submit_cell(payload)], f"next: backbone: {backbone}", 0.95, carried=RECIPE
            )
        )
    assert "backbone" not in [r.lever for r in ready_for([best, *losers], best)]


def test_capacity_opens_only_when_the_seconds_per_epoch_leave_room() -> None:
    fast = first_try()
    assert "backbone" in [r.lever for r in ready_for([fast], fast)]
    slow = first_try(secs=60)
    levers = [r.lever for r in ready_for([slow], slow)]
    assert "backbone" not in levers
    assert "image-size" not in levers


def test_label_smoothing_never_opens_for_a_number_and_a_citation_is_not_a_model() -> None:
    cell, payload = fit_cell(RECIPE, [0.95, 0.99, 0.999], 0.955, start=BENCH)
    best = experiment(
        [cell, submit_cell(payload)],
        "next: backbone: resnet18",
        9.1,
        carried=BASELINE,
        metric="rmse",
        direction="minimize",
    )
    items = ready_for([best], best, task="regression", direction="minimize", findings=FINDINGS)
    levers = {r.lever: r for r in items}
    assert "regularisation" not in levers
    assert "efficientnet_b0" in levers["own-model"].reason
    assert "3390" not in vl.ready_line(items)


def test_an_own_code_best_still_opens_levers_from_what_its_model_line_carried() -> None:
    own = own_try()
    levers = {r.lever for r in ready_for([own], own, findings=FINDINGS)}
    assert {"image-size", "epochs", "own-model"} <= levers


def test_measured_lost_and_banked_read_the_clause_and_the_direction() -> None:
    cell, payload = fit_cell({**RECIPE, "epochs": 3}, [0.9, 0.95, 0.97], 0.97, start=RECIPE)
    best = experiment(
        [cell, submit_cell(payload)], "b", 8.9, carried=RECIPE, metric="rmse", direction="minimize"
    )
    lost_cell, lost_payload = fit_cell({**RECIPE, "epochs": 6}, [0.9] * 6, 0.97, start=RECIPE)
    lost = experiment(
        [lost_cell, submit_cell(lost_payload)],
        "l",
        9.8,
        carried=RECIPE,
        metric="rmse",
        direction="minimize",
    )
    brief = "next: epochs: keep the recipe and train 6 epochs"
    assert vl.measured_lost(brief, [lost, best], best, "minimize")
    assert vl.measured_lost(brief, [lost, best], best, "maximize") is None
    assert vl.banked("next: epochs: keep the recipe and train 3 epochs", best)


# ─── the supervisor ──────────────────────────────────────────────────────────


class Scripted:
    def __init__(self, *briefs: str) -> None:
        self.replies = [
            ChatResponse(
                model="m",
                tool_calls=[
                    ToolCall(
                        id=str(i),
                        name="plan_next",
                        arguments={"stop": False, "brief": b, "title": "t"},
                    )
                ],
            )
            for i, b in enumerate(briefs)
        ]
        self.seen: list[Any] = []

    def chat(self, messages: list[Any], **kw: Any) -> ChatResponse:
        self.seen.append((list(messages), kw["tools"]))
        return self.replies.pop(0)


def vision_supervisor(client: Scripted, **kw: Any) -> sup.Supervisor:
    return sup.Supervisor(
        client,
        metric="accuracy",
        family="vision",
        multiclass=True,  # type: ignore[arg-type]
        image_size=64,
        **kw,
    )


def baseline_result() -> ExperimentResult:
    return ExperimentResult(
        experiment_id="b",
        metrics=Metrics(
            values={"accuracy": 0.9496}, primary="accuracy", direction="maximize", n_samples=100
        ),
    )


def test_the_vision_supervisor_has_its_own_prompt_tool_and_no_inspect_field() -> None:
    client = Scripted(
        "next: backbone: fine-tune resnet18 through fit(), all layers, 3 epochs (because none scored)"
    )
    vision_supervisor(client).decide(
        data_summary="Images: 10", baseline=baseline_result(), history=[]
    )
    messages, tools = client.seen[0]
    assert "IMAGE experimentation loop" in messages[0].content
    assert "Levers ready now: backbone:" in messages[1].content
    assert "want_inspect" not in tools[0].parameters["properties"]
    assert "HistGradientBoosting" not in tools[0].parameters["properties"]["brief"]["description"]


def test_a_brief_mentioning_the_baseline_is_not_replaced_by_a_tabular_lever() -> None:
    best = first_try()
    client = Scripted(
        "next: backbone: the baseline CNN scored 0.9496, so keep the recipe and swap the backbone "
        "to convnext_tiny"
    )
    decision = vision_supervisor(client).decide(
        data_summary="Images: 10",
        baseline=baseline_result(),
        history=[best],
        carried_best=best,
    )
    assert "categorical" not in decision.brief
    assert "convnext_tiny" in decision.brief
    assert len(client.seen) == 1


def test_a_lever_that_is_not_ready_is_nudged_once_then_replaced_by_the_first_ready_move() -> None:
    best = first_try()
    client = Scripted(
        "next: augmentation: set augment to flip_crop", "next: augmentation: flip_crop again"
    )
    decision = vision_supervisor(client).decide(
        data_summary="Images: 10",
        baseline=baseline_result(),
        history=[best],
        carried_best=best,
    )
    assert "nothing in this run's numbers opens augmentation" in client.seen[1][0][-1].content
    assert decision.brief.split("next: ", 1)[1].startswith("backbone:")


def test_an_own_model_brief_naming_two_models_is_nudged_on_experiment_one_too() -> None:
    client = Scripted(
        LIVE_BRIEF_1, "next: own-model: write torch code for timm efficientnet_b0, all layers"
    )
    decision = vision_supervisor(client).decide(
        data_summary="Images: 10",
        baseline=baseline_result(),
        history=[],
        known_findings=FINDINGS,
    )
    assert "names exactly ONE model" in client.seen[1][0][-1].content
    assert "efficientnet_b0" in decision.brief
    assert "vit" not in decision.brief.split("next:", 1)[1]


def test_a_number_run_is_never_briefed_label_smoothing() -> None:
    cell, payload = fit_cell(RECIPE, [0.95, 0.99, 0.999], 0.955, start=BENCH)
    best = experiment(
        [cell, submit_cell(payload)],
        "next: backbone: fine-tune resnet18",
        9.1,
        carried=BASELINE,
        metric="rmse",
        direction="minimize",
    )
    client = Scripted(
        "next: regularisation: set label_smoothing to 0.1",
        "next: image-size: keep the best and train at 128 px",
    )
    supervisor = sup.Supervisor(
        client,
        metric="rmse",
        family="vision",
        task="regression",
        image_size=64,  # type: ignore[arg-type]
    )
    baseline = ExperimentResult(
        experiment_id="b",
        metrics=Metrics(
            values={"rmse": 13.12}, primary="rmse", direction="minimize", n_samples=100
        ),
    )
    decision = supervisor.decide(
        data_summary="Images: 10",
        baseline=baseline,
        history=[best],
        carried_best=best,
        this_run=[best],
    )
    assert "label smoothing spreads probability across classes" in client.seen[1][0][-1].content
    assert "128 px" in decision.brief


def test_with_nothing_ready_the_guards_stay_on_and_the_supervisor_asks_for_papers() -> None:
    """An own-code best used to switch the gate off: ready() was empty, the fallback was
    None, and every retry was accepted whatever it briefed."""
    payload = {
        "model": "efficientnet_b0",
        "image_size": 224,
        "epochs": 12,
        "seconds": 3000,
        "val": 0.983,
        "val_accuracy": 0.983,
    }
    own = experiment(
        [model_cell(payload), submit_cell(payload, code="submit_probabilities(h, model='e')")],
        "next: own-model: write torch code for efficientnet_b0",
        0.9830,
        carried=RECIPE,
    )
    assert ready_for([own], own) == []
    client = Scripted(
        "next: backbone and epochs: try something stronger",
        "next: bigger model: try something stronger",
    )
    decision = vision_supervisor(client).decide(
        data_summary="Images: 10",
        baseline=baseline_result(),
        history=[own],
        carried_best=own,
        this_run=[own],
    )
    assert decision.want_research
    assert len(client.seen) == 2  # nudged once, then asked for research rather than accepted


def test_the_image_supervisor_reads_this_run_and_not_the_one_before_it() -> None:
    """Every image run is named vision-model, so memory hands decide() the earlier run's
    rows; a lever opens on numbers this run produced."""
    before = experiment(
        [
            {
                "code": "f = fit(backbone='convnext_tiny')",
                "stdout": "",
                "source": "agent",
                "error": "out of memory on mps at batch_size=64, image_size=224",
            }
        ],
        "next: image-size: keep the best and train at 224 px",
        None,
        carried=RECIPE,
        error="no predictions were written",
    )
    client = Scripted(
        "next: backbone: fine-tune resnet18 through fit(), all layers, 3 epochs (because none scored)"
    )
    decision = vision_supervisor(client).decide(
        data_summary="Images: 10",
        baseline=baseline_result(),
        history=[before],
        this_run=[],
    )
    assert "backbone" in decision.brief
    assert len(client.seen) == 1  # not rejected in favour of the other run's repair


def test_the_history_carries_the_fit_line_the_ledger_and_the_ready_line() -> None:
    best = first_try()
    client = Scripted("next: backbone: keep the recipe and swap the backbone to convnext_tiny")
    vision_supervisor(client).decide(
        data_summary="Images: 10",
        baseline=baseline_result(),
        history=[best],
        carried_best=best,
        this_run=[best],
    )
    user = client.seen[0][0][1].content
    assert "    fit: resnet18 64px all layers, 3 epochs" in user
    assert "val accuracy 0.9771" in user
    assert "Levers tried: backbone" in user
    assert "Levers ready now:" in user
    assert "[used:" not in user  # AdamW and Linear name no image lever
    assert "FIT {" not in user  # the payload is the harness's, not the model's to read


def test_the_so_far_slot_states_the_carried_recipe() -> None:
    best = first_try()
    brief = sup._grounded_brief(
        "next: epochs: keep the recipe and train 6 epochs",
        metric="accuracy",
        baseline_score=0.9496,
        carried=best,
        family="vision",
    )
    assert brief.startswith(
        "so far: best accuracy=0.9785 via 'try' (recipe: resnet18 64px all layers, 3 epochs"
    )


# ─── the coder, the critic and the researcher ────────────────────────────────


def test_the_image_coder_gets_the_recipe_not_code_and_the_image_nudges() -> None:
    messages = coder_mod._build_messages(
        data_summary="Images: 10",
        metric="accuracy",
        direction="maximize",
        brief="next: epochs: train 6 epochs",
        preamble_output="loaded",
        family="vision",
        starting_code="f = fit()",
        starting_score=0.9785,
        starting_recipe=RECIPE,
    )
    system, user = messages[0].content, messages[1].content
    assert "YOUR OWN CODE" in system
    assert "model= name is REQUIRED" in system
    assert "Never wrap an import in try/except" in system
    assert "pretrained_cfg" in system  # a 224 px model refuses a 64 px image
    assert "BEST RECIPE SO FAR (holdout accuracy 0.9785): resnet18 64px all layers" in user
    assert "f = fit()" not in user
    agent = coder_mod.CodingAgent(object(), object(), metric="accuracy", family="vision")  # type: ignore[arg-type]
    assert "fit() stops" in agent._nudge("timeout_nudge")
    table = coder_mod.CodingAgent(object(), object(), metric="f1")  # type: ignore[arg-type]
    assert "LogisticRegression" in table._nudge("timeout_nudge")


def test_an_own_code_best_travels_as_its_code_because_no_recipe_rebuilds_it() -> None:
    recipe = {"model": "efficientnet_b0", "image_size": 64, "epochs": 3}
    messages = coder_mod._build_messages(
        data_summary="Images: 10",
        metric="accuracy",
        direction="maximize",
        brief="next: epochs: train 6 epochs",
        preamble_output="loaded",
        family="vision",
        starting_code="model = timm.create_model(NAME)",
        starting_score=0.983,
        starting_recipe=recipe,
    )
    assert "model = timm.create_model(NAME)" in messages[1].content
    assert "own code: efficientnet_b0, 64px, 3 epochs" in messages[1].content


def test_the_image_critic_keeps_an_errored_cell_and_drops_what_followed_the_submission() -> None:
    cells = [
        {
            "code": "MEAN = np.concatenate([train_px, holdout_px]).mean()",
            "source": "agent",
            "error": "NameError: build_model",
        },
        {"code": "submit_probabilities(hold, model='resnet50')", "source": "agent", "error": None},
        {"code": "print('after')", "source": "agent", "error": None},
    ]
    code = critic_mod.vision_submit_code(cells)
    # The errored cell bound MEAN before it died, and the submitting cell used it.
    assert "holdout_px" in code
    assert "after" not in code
    assert "batch-norm statistics from holdout pixels" in critic_mod._PROMPTS["vision_system"]


def test_the_image_critic_reviews_through_its_own_prompt_and_slice() -> None:
    class Capture:
        def __init__(self) -> None:
            self.seen: list[Any] = []

        def chat(self, messages: list[Any], **kw: Any) -> ChatResponse:
            self.seen.append(messages)
            return ChatResponse(
                model="m",
                tool_calls=[
                    ToolCall(
                        id="1",
                        name="review_experiment",
                        arguments={"leak": True, "mirage": False, "reason": "holdout stats"},
                    )
                ],
            )

    cells = [
        {
            "code": "allpx = np.concatenate([train_px, holdout_px])",
            "stdout": "",
            "source": "agent",
            "error": None,
        },
        {
            "code": "submit_probabilities(hold, model='resnet50')",
            "stdout": "",
            "source": "agent",
            "error": None,
        },
        {"code": "print('after the submission')", "stdout": "", "source": "agent", "error": None},
    ]
    exp = experiment(cells, "next: own-model: resnet50", 0.98, carried=RECIPE)
    client = Capture()
    verdict = critic_mod.Critic(
        client,
        metric="accuracy",
        family="vision",  # type: ignore[arg-type]
        direction="maximize",
    ).review(exp, previous_best=0.97)
    system, user = client.seen[0][0].content, client.seen[0][1].content
    assert "IMAGE experiment" in system
    assert "holdout_px" in user
    assert "after the submission" not in user
    assert verdict.leak


def test_the_image_researcher_uses_image_queries_and_an_image_suggest_tool() -> None:
    class Capture:
        def __init__(self) -> None:
            self.seen: list[Any] = []

        def chat(self, messages: list[Any], **kw: Any) -> ChatResponse:
            self.seen.append((messages, kw["tools"]))
            return ChatResponse(
                model="m",
                tool_calls=[
                    ToolCall(
                        id="1",
                        name=kw["tools"][0].name,
                        arguments={"queries": [], "suggestions": []},
                    )
                ],
            )

    client = Capture()
    researcher = researcher_mod.Researcher(
        client,
        metric="accuracy",
        direction="maximize",  # type: ignore[arg-type]
        family="vision",
        sources=[],
    )
    researcher._plan_queries("Images: 10", "nothing yet")
    researcher._suggest("Images: 10", "nothing yet", [])
    assert "IMAGE problem" in client.seen[0][0][0].content
    assert "THESE images" in client.seen[1][0][0].content
    assert "target-encode" not in json.dumps(client.seen[1][1][0].parameters)


# ─── the loop ────────────────────────────────────────────────────────────────


def test_the_loop_carries_the_recipe_stamps_what_moved_and_marks_what_did_not() -> None:
    from iterate.core import codegen
    from iterate.core.agent_loop import run_supervised
    from iterate.core.coder import Cell, CodingResult
    from iterate.core.memory import InMemoryMemory
    from iterate.core.supervisor import SupervisorDecision
    from iterate.core.terminator import MaxIterations

    def scored(score: float) -> ExperimentResult:
        return ExperimentResult(
            experiment_id="x",
            metrics=Metrics(
                values={"accuracy": score}, primary="accuracy", direction="maximize", n_samples=100
            ),
        )

    class Target:
        name = "vision-model"

        def baseline(self) -> ExperimentResult:
            return scored(0.9496).model_copy(
                update={
                    "artifacts": {
                        "recipe.json": json.dumps(
                            {**BASELINE, "epochs_planned": 20, "epochs_run": 20}
                        )
                    }
                }
            )

        def meta_json(self) -> bytes:
            return json.dumps({"image_size": 64}).encode()

    class Supervisor:
        def __init__(self) -> None:
            self.briefs = [
                "next: backbone: fine-tune resnet18 through fit()",
                "next: epochs: keep the recipe and train 6 epochs",
            ]
            self.seen: list[dict[str, Any]] = []

        def decide(self, **kw: Any) -> SupervisorDecision:
            self.seen.append(kw)
            return SupervisorDecision(False, "t", self.briefs.pop(0))

    class Coder:
        def __init__(self, cells: list[dict[str, Any]], score: float) -> None:
            self.cells, self.score = cells, score
            self.kwargs: dict[str, Any] = {}
            self.gate_says: bool | None = None

        def run(self, **kw: Any) -> CodingResult:
            self.kwargs = kw
            cells = [Cell(c["code"], c["stdout"], "", None, "agent") for c in self.cells]
            gate = kw.get("lever_gate")
            self.gate_says = gate(cells) if gate else None
            return CodingResult(
                result=scored(self.score), cells=cells, predictions_sha256=str(self.score)
            )

    won_cell, won_payload = fit_cell(RECIPE, [0.912, 0.958, 0.981], 0.9771, start=BENCH)
    first = Coder([won_cell, submit_cell(won_payload)], 0.9785)
    swapped = {**RECIPE, "backbone": "resnet50"}
    swap_cell, swap_payload = fit_cell(swapped, [0.9, 0.95, 0.97], 0.97, start=RECIPE)
    second = Coder([swap_cell, submit_cell(swap_payload)], 0.97)
    coders = iter([first, second])
    supervisor = Supervisor()
    out = run_supervised(
        target=Target(),
        dataset=object(),
        supervisor=supervisor,  # type: ignore[arg-type]
        make_coder=lambda: next(coders),
        terminator=MaxIterations(2),
        memory=InMemoryMemory(),
        data_summary="d",
        family="vision",
    )
    one, two = out.history
    assert one.candidate.changes["recipe"]["backbone"] == "resnet18"
    assert one.candidate.changes["levers_moved"] == ["backbone"]
    assert "lever_unmeasured" not in one.candidate.changes
    assert first.gate_says is True
    # The recipe the kernel reads out of incumbent.json is the one the gate measures against.
    assert (
        json.loads(first.kwargs["starting_files"][codegen.INCUMBENT_JSON])["backbone"]
        == "simple_cnn"
    )
    assert (
        json.loads(second.kwargs["starting_files"][codegen.INCUMBENT_JSON])["backbone"]
        == "resnet18"
    )
    assert second.kwargs["starting_recipe"]["backbone"] == "resnet18"
    assert second.kwargs["starting_code"] is None  # a fit() best is a recipe, not code to rebuild
    assert second.kwargs["brief_markers"] == ()
    assert second.kwargs["lever_name"] == "epochs"
    assert two.candidate.changes["levers_moved"] == ["backbone"]
    assert two.candidate.changes["lever_unmeasured"] is True
    assert supervisor.seen[1]["this_run"] == out.history[:1]


def test_an_own_model_win_writes_a_fit_recipe_into_the_file_the_kernel_opens() -> None:
    """The prompt and the gate keep the whole payload, because that is what won; the file
    `fit()` departs from cannot, since an own-model payload names no backbone and counts
    loops the agent wrote."""
    from iterate.core import codegen
    from iterate.core.agent_loop import run_supervised
    from iterate.core.coder import Cell, CodingResult
    from iterate.core.memory import InMemoryMemory
    from iterate.core.supervisor import SupervisorDecision
    from iterate.core.terminator import MaxIterations

    def scored(score: float) -> ExperimentResult:
        return ExperimentResult(
            experiment_id="x",
            metrics=Metrics(
                values={"accuracy": score}, primary="accuracy", direction="maximize", n_samples=100
            ),
        )

    class Target:
        name = "vision-model"

        def baseline(self) -> ExperimentResult:
            return scored(0.9496).model_copy(
                update={"artifacts": {"recipe.json": json.dumps(BASELINE)}}
            )

        def meta_json(self) -> bytes:
            return json.dumps({"image_size": 64}).encode()

    class Supervisor:
        def decide(self, **kw: Any) -> SupervisorDecision:
            return SupervisorDecision(False, "t", "next: epochs: train 6 epochs")

    class Coder:
        def __init__(self, cells: list[dict[str, Any]], score: float) -> None:
            self.cells, self.score, self.kwargs = cells, score, {}

        def run(self, **kw: Any) -> CodingResult:
            self.kwargs = kw
            cells = [Cell(c["code"], c["stdout"], "", None, "agent") for c in self.cells]
            return CodingResult(
                result=scored(self.score), cells=cells, predictions_sha256=str(self.score)
            )

    fit_line, _ = fit_cell(RECIPE, [0.912, 0.958, 0.981], 0.9771, start=BASELINE)
    own = {"model": "efficientnet_b0", "image_size": 64, "epochs": 3, "val": 0.9417}
    first = Coder(
        [fit_line, model_cell(own), submit_cell(own, "submit_probabilities(p, model='x')")], 0.9785
    )
    second = Coder([fit_line], 0.9)
    coders = iter([first, second])
    run_supervised(
        target=Target(),
        dataset=object(),
        supervisor=Supervisor(),  # type: ignore[arg-type]
        make_coder=lambda: next(coders),
        terminator=MaxIterations(2),
        memory=InMemoryMemory(),
        data_summary="d",
        family="vision",
    )
    carried = json.loads(second.kwargs["starting_files"][codegen.INCUMBENT_JSON])
    assert "model" not in carried
    assert carried["backbone"] == "resnet18"
    assert second.kwargs["starting_recipe"]["model"] == "efficientnet_b0"


def test_two_iterations_through_the_real_supervisor_read_each_other() -> None:
    """The seam the live run's pass criteria name: iteration 1's fit line and ready line
    reach iteration 2's prompt, and the brief it writes from them survives the guard."""
    from iterate.core.agent_loop import run_supervised
    from iterate.core.coder import Cell, CodingResult
    from iterate.core.memory import InMemoryMemory
    from iterate.core.terminator import MaxIterations

    def scored(score: float) -> ExperimentResult:
        return ExperimentResult(
            experiment_id="x",
            metrics=Metrics(
                values={"accuracy": score}, primary="accuracy", direction="maximize", n_samples=100
            ),
        )

    class Target:
        name = "vision-model"

        def baseline(self) -> ExperimentResult:
            return scored(0.9496).model_copy(
                update={"artifacts": {"recipe.json": json.dumps(BASELINE)}}
            )

        def meta_json(self) -> bytes:
            return json.dumps({"image_size": 64}).encode()

    won_cell, won_payload = fit_cell(RECIPE, [0.80, 0.85, 0.95], 0.9771, start=BENCH)
    swapped = {**RECIPE, "backbone": "convnext_tiny"}
    swap_cell, swap_payload = fit_cell(swapped, [0.9, 0.95, 0.97], 0.98, secs=40, start=RECIPE)
    sessions = iter([[won_cell, submit_cell(won_payload)], [swap_cell, submit_cell(swap_payload)]])
    scores = iter([0.9785, 0.9820])

    class Coder:
        def run(self, **kw: Any) -> CodingResult:
            cells = [Cell(c["code"], c["stdout"], "", None, "agent") for c in next(sessions)]
            score = next(scores)
            return CodingResult(result=scored(score), cells=cells, predictions_sha256=str(score))

    client = Scripted(
        "next: backbone: fine-tune resnet18 through fit(), all layers, 3 epochs",
        "next: backbone: keep the recipe and swap the backbone to convnext_tiny",
    )
    run_supervised(
        target=Target(),
        dataset=object(),  # type: ignore[arg-type]
        supervisor=vision_supervisor(client),
        make_coder=Coder,  # type: ignore[arg-type]
        terminator=MaxIterations(2),
        memory=InMemoryMemory(),
        data_summary="Images: 21600 train",
        family="vision",
    )
    second = client.seen[1][0][1].content
    assert "    fit: resnet18 64px all layers, 3 epochs, augment flip, lr 0.001; " in second
    assert "ran 3 of 3 epochs at 12s each" in second
    assert "train_acc 0.800 -> 0.950 (still rising)" in second
    assert "Levers tried: backbone" in second
    assert "convnext_tiny" in second  # the ready line read iteration 1's seconds per epoch
    assert len(client.seen) == 2  # neither brief was nudged
