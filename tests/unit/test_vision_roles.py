"""The image family's four roles and its lever gate, on sessions shaped as the helpers print.

The cells here carry the exact lines `vision_session` writes: the runner's epoch lines,
`FIT {...}` with the recipe it started from under "from", `MODEL {...}` from an own-code
`evaluate(model=...)`, and `SUBMITTED {...}`. test_vision_contract.py holds the other end
of that contract, where a real session prints them.
"""

from __future__ import annotations

import hashlib
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
STACK = "conv(32) pool conv(64) pool dropout(0.3) linear(256)"
HEAD = "linear(512) dropout(0.5)"
# Tony's own words for the call this hold exists for: one backbone, one head, one depth.
HEADLINE_ASK = "use resnet50 with a head of linear(512) dropout(0.5) and train the last block"


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


def stack_try(
    stack: str = STACK, score: float = 0.90, *, guidance: str | None = None
) -> Experiment:
    """A from-zero fit: `fit(layers=...)` names no backbone, so merge starts it from the
    from-zero reference and the line carries the stack as its canonical text."""
    recipe = {**BASELINE, "backbone": "layers_net", "layers": stack}
    trains = [0.50 + 0.02 * i for i in range(int(BASELINE["epochs"]))]
    cell, payload = fit_cell(recipe, trains, score, start=BASELINE)
    exp = experiment(
        [cell, submit_cell(payload)],
        f"next: layer-stack: train a network from zero through fit(), with layers {stack}",
        score,
        carried=BASELINE,
    )
    if guidance is not None:
        exp.candidate.changes["user_guidance"] = guidance
    return exp


def head_try(head: str = HEAD, score: float = 0.99) -> Experiment:
    """The headline ask as one fit: a backbone, a head and the last block together."""
    recipe = {**RECIPE, "backbone": "resnet50", "unfreeze": "last_block", "head": head}
    cell, payload = fit_cell(recipe, [0.9, 0.95, 0.98], score, start=RECIPE)
    return experiment(
        [cell, submit_cell(payload)],
        f"next: custom-head: fine-tune resnet50 through fit(), the last stage and the head "
        f"(unfreeze last_block), 3 epochs, with head {head}",
        score,
        carried=RECIPE,
    )


def typed(
    text: str, history: list[Experiment] | None = None, carried: Experiment | None = None
) -> tuple[str, str]:
    """One note as a live image run holds it: read whole, then stored clipped, with the
    line the user was told. Every ask test goes through this, because the stored note is
    what the ladder actually reads an iteration later."""
    from iterate.core.interactive import RunController

    said: list[str] = []
    ctrl = RunController(reply=said.append)
    ctrl.note_reader = lambda t: vl.ask_note(t, history or [], vl.recipe_of(carried))
    ctrl.add_brief_note(text)
    (note,) = ctrl.take_brief_notes()
    return note, said[0] if said else ""


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


# ─── the two layer classes, and the user ask that opens them ─────────────────


@pytest.mark.parametrize(
    "ask",
    [
        "never build a network from scratch",
        "don't use dropout 0.5",
        "don't use dropout(0.5) on the head",
        "don't use efficientnet_b0",
        "no custom CNN this run",
        f"stop trying {STACK}",
        f"avoid a head like {HEAD}",
    ],
)
def test_an_ask_that_says_no_opens_nothing(ask: str) -> None:
    """The blocker a skeptic measured: word matching read "never build a network from
    scratch" as a request for one, and put it first. A refusal anywhere shuts the lot."""
    best = first_try()
    assert [r.lever for r in ready_for([best], best, asks=ask)] == [
        r.lever for r in ready_for([best], best)
    ]


@pytest.mark.parametrize(
    "ask",
    [
        "try a CNN 5 layers deep",
        "a CNN (2015) is what the paper used",
        "go back to simple_cnn 64px",
        "run a linear probe first",
        "please aim for better accuracy",
        "conv is fine but watch the budget",
        # A fine-tune-depth ask in plain English: "head with" is two ordinary words.
        "train more than the head with a lower learning rate",
        "unfreeze more than the head with lr 1e-4",
    ],
)
def test_prose_is_not_a_layer_stack(ask: str) -> None:
    """Strict reading for an ask: prose that merely says the word must open nothing, or
    the run trains a one-layer width-5 network and calls the ask spent."""
    best = first_try()
    items = ready_for([best], best, asks=ask)
    assert not [r for r in items if r.lever in vl.LAYER_LEVERS]
    assert vl.ask_note(ask) == ("", "")


def test_an_ask_leads_the_line_only_when_the_user_wrote_the_stack() -> None:
    """An ask with a stack is the user's own choice, so it leads and may become the
    harness's brief. A vague ask only gets the example stack, which nobody chose: it
    steers the model from the back of the line and never becomes the fallback."""
    best = first_try()
    wrote = ready_for([best], best, asks=f"please try {STACK}")
    assert wrote[0].lever == "layer-stack"
    assert wrote[0].stack == STACK
    assert (vl.fallback_move(wrote) or ("", ""))[0] == "evidence: layer-stack"

    vague = ready_for([best], best, asks="build a network from scratch")
    assert vague[-1].lever == "layer-stack"
    assert vague[0].lever != "layer-stack"
    assert (vl.fallback_move(vague) or ("", ""))[0] != "evidence: layer-stack"


def test_an_ask_buys_one_experiment_per_class_whatever_stack_ran() -> None:
    """The ask is a fact, not a pin: once an experiment it steered has spent the class,
    the run's own closed rule decides, even when the model ran a different stack."""
    best = first_try()
    ask = f"please try {STACK}"
    other = stack_try("conv(64) pool conv(128) pool linear(256)", 0.90, guidance=ask)
    assert "layer-stack" in [r.lever for r in ready_for([best], best, asks=ask)]
    assert "layer-stack" not in [r.lever for r in ready_for([best, other], best, asks=ask)]


def test_the_headline_ask_runs_as_one_experiment_and_the_gate_credits_the_head() -> None:
    """resnet50 plus a head plus the last block is three classes, and a brief may name
    one. So it is ONE custom-head entry that carries the backbone and the depth, and the
    fit it asks for has to credit custom-head or the try lands unmeasured."""
    best = first_try()
    note, said = typed(HEADLINE_ASK)
    assert f"read head {HEAD}; it opens the next experiment" == said
    items = ready_for([best], best, asks=note)
    (entry,) = [r for r in items if r.lever in vl.LAYER_LEVERS]
    assert entry.lever == "custom-head"
    assert all(word in entry.move for word in ("resnet50", "last_block", HEAD))
    assert vl.brief_call(f"next: custom-head: {entry.move} (because {entry.reason}).") == (
        'fit(backbone=\'resnet50\', head=[("linear", 512), ("dropout", 0.5)], '
        "unfreeze='last_block')"
    )
    ran = head_try()
    assert "custom-head" in ran.candidate.changes["levers_moved"]


# ─── what the stored note carries, which is all the ladder reads ─────────────

# The two asks a skeptic typed: the refusal and the opener both sit past the clip.
_BANNED = (
    f"{STACK} is the shape the paper uses and I would like to see it tried on this dataset "
    "at some point in the future when we have more compute available on the laptop, but "
    "never build it from scratch now"
)
_VAGUE_LATE = (
    "the backbone swaps keep landing in the same place and the images here are tiny "
    "thumbnails so imagenet features are probably a poor match for them, which makes me "
    "think the right move is to just build a custom cnn from scratch and see what happens"
)


def test_a_no_word_the_clip_ate_still_shuts_every_lever() -> None:
    """The blocker: the reader saw "never", the stored note did not, and the ban opened
    the lever it bans and led the line."""
    best = first_try()
    note, said = typed(_BANNED)
    assert "never" not in note.split("]", 1)[1]
    assert "opens no lever" in said
    assert not [r for r in ready_for([best], best, asks=note) if r.lever in vl.LAYER_LEVERS]


def test_a_refused_ask_names_the_word_that_refused_it() -> None:
    """The stack is read and valid; the "do not" is about the epochs. Refusing is the
    safe direction, but the line said the user's words were a limit on the layers."""
    best = first_try()
    note, said = typed(f"use layers {STACK}, do not go above 10 epochs")
    assert "'do not'" in said
    assert "opens no lever" in said
    assert not [r for r in ready_for([best], best, asks=note) if r.lever in vl.LAYER_LEVERS]


def test_a_vague_ask_the_clip_ate_still_opens_the_example_stack() -> None:
    """The words that opened it sat past the clip, so the promise in the reply was kept
    by nothing."""
    best = first_try()
    note, said = typed(_VAGUE_LATE)
    assert "from scratch" not in note.split("]", 1)[1]
    assert "it opens the next experiment" in said
    (entry,) = [r for r in ready_for([best], best, asks=note) if r.lever in vl.LAYER_LEVERS]
    assert (entry.lever, entry.invented) == ("layer-stack", True)


@pytest.mark.parametrize(
    ("ask", "lever", "stack"),
    [
        (STACK, "layer-stack", STACK),
        ("conv(32) pool conv(64) pool conv(128) pool", "layer-stack", None),
        (f"{HEAD} on resnet50", "custom-head", HEAD),
        (f"{HEAD} head on resnet50, last block", "custom-head", HEAD),
    ],
)
def test_an_ask_that_opens_with_its_stack_opens_that_stack_once(
    ask: str, lever: str, stack: str | None
) -> None:
    """Typing the stack first is how a stack gets typed. Reading the stored mark and the
    words it stands for as one run merged the two and trained a doubled network."""
    best = first_try()
    note, said = typed(ask)
    (entry,) = [r for r in ready_for([best], best, asks=note) if r.lever in vl.LAYER_LEVERS]
    assert (entry.lever, entry.stack) == (lever, stack or ask)
    assert f"read {entry.stack}" in said.replace("layers ", "").replace("head ", "")


def test_a_no_word_in_another_note_does_not_kill_the_ask() -> None:
    """Notes are judged one at a time: the batch reaches the ladder as one joined string,
    and a "no hurry" typed after a stack used to shut the stack."""
    best = first_try()
    ask, _ = typed(f"build a cnn from scratch {STACK}")
    polite, _ = typed("no rush, take your time")
    (entry,) = [
        r for r in ready_for([best], best, asks=f"{ask}; {polite}") if r.lever in vl.LAYER_LEVERS
    ]
    assert entry.stack == STACK


def test_a_short_earlier_note_does_not_make_a_later_ask_dead() -> None:
    """The spend was a substring test, so one earlier "please" killed every later ask
    that held the word, with the reply still promising an experiment."""
    best = first_try()
    earlier = head_try("linear(256) dropout(0.2)")
    earlier.candidate.changes["user_guidance"] = "please"
    note, said = typed(f"use resnet50 with a head of {HEAD} please", [best, earlier], best)
    assert "it opens the next experiment" in said
    assert "custom-head" in [r.lever for r in ready_for([best, earlier], best, asks=note)]


def test_a_stack_the_grammar_refuses_opens_nothing_at_all() -> None:
    """The refusal reached the user and the example stack opened behind it, so the run
    proposed a network nobody had written."""
    best = first_try()
    fifteen = (
        "build from scratch conv(32) pool conv(64) pool conv(96) pool conv(128) pool "
        "conv(160) pool conv(192) pool conv(256) linear(512) linear(256)"
    )
    note, said = typed(fifteen)
    assert "refused it" in said
    assert not [r for r in ready_for([best], best, asks=note) if r.lever in vl.LAYER_LEVERS]


def test_a_head_whose_number_is_out_of_range_gets_a_line_of_its_own() -> None:
    """A concrete head that fails a bound is the case where "why nothing opened" is worth
    most, and it was the one case that printed nothing at all."""
    note, said = typed("put linear(4096) dropout(0.5) on resnet50")
    assert "refused it" in said
    assert "2048" in said
    assert note.startswith("[ask: none]")


def test_an_ask_already_tried_says_so_instead_of_promising_an_experiment() -> None:
    best = first_try()
    ran = head_try()
    note, said = typed(f"use a head of {HEAD}", [best, ran], best)
    assert "already tried" in said
    assert not [r for r in ready_for([best, ran], best, asks=note) if r.lever in vl.LAYER_LEVERS]


def test_a_bare_last_block_ask_opens_the_depth_it_names() -> None:
    """The three plain words an ask may use are "from scratch", "custom head" and "last
    block". The third opened nothing and printed nothing."""
    best = first_try()
    note, said = typed("train only the last block")
    assert "it opens the next experiment" in said
    (entry,) = [r for r in ready_for([best], best, asks=note) if r.lever == "fine-tune-depth"]
    assert "last_block" in entry.move
    brief = f"next: fine-tune-depth: {entry.move} (because {entry.reason})."
    assert vl.missing_value(brief) is None


def test_a_failed_layer_try_that_named_no_stack_is_not_a_layer_repair() -> None:
    """A layer entry with no stack is a layer class the stack guard cannot judge, and it
    would let the next brief name any network at all."""
    best = first_try()
    failed = experiment(
        [{"code": "f = fit()", "stdout": "", "source": "agent", "error": "boom"}],
        "next: layer-stack: build a deeper cnn",
        None,
        carried=RECIPE,
        error="no predictions were written",
    )
    (repair,) = ready_for([best, failed], best)[:1]
    assert repair.lever not in vl.LAYER_LEVERS
    assert repair.stack == ""


def test_a_last_block_best_is_not_described_as_training_the_head_only() -> None:
    """`unfreeze` gained a third value, and the reason read two ways before it."""
    ran = head_try()
    (depth,) = [r for r in ready_for([ran], ran) if r.lever == "fine-tune-depth"]
    assert depth.reason == "the best trains the last stage and the head"


def test_a_from_zero_fit_credits_the_stack_and_leaves_the_backbone_untouched() -> None:
    assert stack_try().candidate.changes["levers_moved"] == ["layer-stack"]


def test_two_losing_stack_tries_close_the_stack_and_leave_the_backbone_open() -> None:
    best = first_try()
    two = [stack_try(STACK, 0.90), stack_try("conv(64) pool conv(128) pool linear(256)", 0.91)]
    levers = [r.lever for r in ready_for([best, *two], best)]
    assert "layer-stack" not in levers
    assert "backbone" in levers


@pytest.mark.parametrize("kind", ["out of memory", "a crash"])
def test_the_repair_for_a_failed_stack_try_keeps_the_stack_and_nothing_else(kind: str) -> None:
    """The carried best is the pretrained network, not the try that failed, so a repair
    built from it would retrain a from-zero net for three epochs at its backbone."""
    best = first_try()
    error = (
        "out of memory on mps at batch_size=64, image_size=64: halve one of them"
        if kind == "out of memory"
        else "RuntimeError: Given groups=1, expected input to have 3 channels"
    )
    failed = experiment(
        [{"code": "f = fit(layers=[...])", "stdout": "", "source": "agent", "error": error}],
        f"next: layer-stack: train a network from zero through fit(), with layers {STACK}",
        None,
        carried=RECIPE,
        error="no predictions were written",
    )
    (repair,) = ready_for([best, failed], best)[:1]
    assert repair.lever == "layer-stack"
    assert "epochs=" not in repair.move
    assert "backbone=" not in repair.move
    assert repair.stack == STACK
    if kind == "out of memory":
        assert "add a pool after the first conv" in repair.move


def test_every_layer_entry_the_ready_line_can_emit_names_a_stack_the_guard_accepts() -> None:
    """A brief copied from an entry has to clear the stack guard, or the harness refuses
    its own line and the entry stays untried and reopens for ever."""
    best = first_try()
    failed = experiment(
        [{"code": "f = fit(head=[...])", "stdout": "", "source": "agent", "error": "boom"}],
        f"next: custom-head: keep the recipe and set head to {HEAD}",
        None,
        carried=RECIPE,
        error="no predictions were written",
    )
    cases = [
        ("a written stack", [best], best, {"asks": f"please try {STACK}"}),
        ("a vague from-zero ask", [best], best, {"asks": "build a network from scratch"}),
        ("a written head", [best], best, {"asks": HEADLINE_ASK}),
        ("a vague head ask", [best], best, {"asks": "give it a bigger head"}),
        ("a failed head try", [best, failed], best, {}),
    ]
    seen = 0
    for label, history, carried, kw in cases:
        items = ready_for(history, carried, **kw)
        layer = [r for r in items if r.lever in vl.LAYER_LEVERS]
        assert layer, f"{label} opens no layer lever"
        for entry in layer:
            seen += 1
            brief = f"next: {entry.lever}: {entry.move} (because {entry.reason})."
            assert vl.classes_named(brief)[:1] == [entry.lever], f"{label}: {brief}"
            assert vl.missing_value(brief) is None, f"{label}: {brief}"
            opens = {r.stack for r in items if r.lever == entry.lever and r.stack}
            assert vl.proposed_value(entry.lever, vl.change_clause(brief)) in opens, label
    assert seen >= 5


# What main 621987d writes for these histories with no ask and no stack anywhere. PR D
# adds classes; it must not move a word of a run that never opens one.
_LINES_ON_MAIN: dict[str, tuple[str, str]] = {
    "experiment 1": (
        "Levers ready now: backbone: fine-tune resnet18 through fit(), all layers, 3 epochs, at "
        "the session image size (because no pretrained model has scored this run); own-model: "
        "write torch code for efficientnet_b0 with pretrained weights, all layers, at the input "
        "size its pretrained_cfg names (because a literature finding names efficientnet_b0 and "
        "vit_base_patch16_224).",
        "Levers tried: none | Levers NOT yet tried: backbone, own-model, image-size, epochs, "
        "augmentation, regularisation, fine-tune-depth",
    ),
    "a fine-tune carried": (
        "Levers ready now: backbone: keep the recipe and swap the backbone to convnext_tiny "
        "(because resnet18 took 12s an epoch, so convnext_tiny fits the budget at about 126s); "
        "image-size: keep the best and train at 128 px (because a pretrained network sees more "
        "detail above 64 px, and 128 px fits the budget at about 144s); own-model: write torch "
        "code for efficientnet_b0 with pretrained weights, all layers, at the input size its "
        "pretrained_cfg names (because a literature finding names efficientnet_b0 and "
        "vit_base_patch16_224).",
        "Levers tried: backbone | Levers NOT yet tried: own-model, image-size, epochs, "
        "augmentation, regularisation, fine-tune-depth",
    ),
    "an own-code best": (
        "Levers ready now: image-size: keep the best and train at 128 px (because a pretrained "
        "network sees more detail above 64 px, and 128 px fits the budget at about 360s); "
        "epochs: keep that code and train 6 epochs (because your own efficientnet_b0 trained "
        "only 3 epochs in 90s); own-model: write torch code for vit_base_patch16_224 with "
        "pretrained weights, all layers, at the input size its pretrained_cfg names (because a "
        "literature finding names vit_base_patch16_224).",
        "Levers tried: own-model | Levers NOT yet tried: backbone, image-size, epochs, "
        "augmentation, regularisation, fine-tune-depth",
    ),
}


@pytest.mark.parametrize("label", sorted(_LINES_ON_MAIN))
def test_a_run_that_opens_no_layer_class_reads_exactly_as_main_read_it(label: str) -> None:
    best, own = first_try(), own_try()
    histories: dict[str, tuple[list[Experiment], Experiment | None, dict[str, Any]]] = {
        "experiment 1": ([], None, {"findings": FINDINGS}),
        "a fine-tune carried": ([best], best, {"findings": FINDINGS, "median_width": 64}),
        "an own-code best": ([own], own, {"findings": FINDINGS}),
    }
    history, carried, kw = histories[label]
    items = ready_for(history, carried, **kw)
    assert (vl.ready_line(items), vl.ledger_line(history)) == _LINES_ON_MAIN[label]


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
    # The same recipe again, kept on the experiment: the delivered notebook hands it to
    # its own Run All, and by then the kernel's copy is gone with the kernel's folder.
    assert one.candidate.changes["started_from"]["backbone"] == "simple_cnn"
    assert two.candidate.changes["started_from"]["backbone"] == "resnet18"


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


# ─── the ask, through the supervisor ─────────────────────────────────────────


# The sha256 of every message main 621987d sends for this image call. An ask is the only
# new thing on the wire, so a run with no guidance must send exactly these bytes.
_IMAGE_CALL_ON_MAIN = "81da841bb08767d2e5f69f9699138f84bcef9aff4f7e020ce04a7c7e582ee312"
_SWAP_BRIEF = (
    "next: backbone: keep the recipe and swap the backbone to convnext_tiny (because resnet18 "
    "took 12s an epoch, so convnext_tiny fits the budget at about 126s)"
)


def test_with_no_guidance_the_image_call_is_the_one_main_sent() -> None:
    best = first_try()
    client = Scripted(_SWAP_BRIEF)
    vision_supervisor(client).decide(
        data_summary="Images: 10",
        baseline=baseline_result(),
        history=[best],
        carried_best=best,
    )
    sent = "\n".join(m.content for m in client.seen[0][0])
    assert hashlib.sha256(sent.encode()).hexdigest() == _IMAGE_CALL_ON_MAIN


def test_a_standing_rule_reaches_the_model_but_never_opens_a_lever() -> None:
    """A rule comes in the shape "never use lightgbm", and it holds for every remaining
    experiment. Read as a fact, a rule would open the lever it bans, for ever."""
    best = first_try()
    rule = f"always build a network from zero with {STACK}"
    client = Scripted(_SWAP_BRIEF)
    vision_supervisor(client).decide(
        data_summary="Images: 10",
        baseline=baseline_result(),
        history=[best],
        carried_best=best,
        standing_rules=(rule,),
    )
    messages = client.seen[0][0]
    assert "layer-stack" not in messages[1].content
    assert rule in messages[-1].content


def test_a_typed_ask_opens_the_class_and_the_line_says_so() -> None:
    best = first_try()
    client = Scripted(f"next: layer-stack: train a network from zero through fit(), layers {STACK}")
    decision = vision_supervisor(client).decide(
        data_summary="Images: 10",
        baseline=baseline_result(),
        history=[best],
        carried_best=best,
        user_guidance=f"please try {STACK}",
    )
    assert "layer-stack: train a network from zero" in client.seen[0][0][1].content
    assert len(client.seen) == 1  # the brief cleared every guard first time
    assert vl.lever_class(decision.brief) == "layer-stack"


def opens_nothing() -> Experiment:
    """A best whose numbers open no lever at all: a 12-epoch fine-tune at 200s an epoch
    has no budget for a stronger backbone, a bigger image or more epochs."""
    cell, payload = fit_cell(
        {**RECIPE, "unfreeze": "all", "epochs": 12}, [0.99] * 12, 0.99, secs=200, start=BENCH
    )
    return experiment(
        [cell, submit_cell(payload)],
        "next: backbone: fine-tune resnet18",
        0.99,
        carried=BASELINE,
    )


def test_a_layer_brief_is_refused_even_when_the_ready_line_is_empty() -> None:
    """Any valid stack clears every other guard, so with nothing ready the run would
    train a network no fact opened. The empty line is exactly when that bites: the nudge
    was sent and the same brief was then accepted on the last attempt."""
    best = opens_nothing()
    assert not ready_for([best], best)
    brief = f"next: layer-stack: train a network from zero through fit(), with layers {STACK}"
    client = Scripted(brief, brief)
    with pytest.raises(sup.SupervisorError):
        vision_supervisor(client).decide(
            data_summary="Images: 10",
            baseline=baseline_result(),
            history=[best],
            carried_best=best,
        )
    assert "opens layer-stack" in client.seen[1][0][-1].content


def test_a_mis_copied_stack_is_refused_when_the_only_entry_is_a_vague_ask() -> None:
    """An entry the harness filled in is not a brief the harness writes, so there is no
    fallback to swap in: the mis-copied stack has to be refused outright."""
    best = opens_nothing()
    note, _ = typed("give it a bigger head")
    wrong = (
        "next: custom-head: fine-tune resnet18 through fit(), all layers (unfreeze all), "
        "3 epochs, with head linear(2048) dropout(0.4)"
    )
    client = Scripted(wrong, wrong)
    with pytest.raises(sup.SupervisorError):
        vision_supervisor(client).decide(
            data_summary="Images: 10",
            baseline=baseline_result(),
            history=[best],
            carried_best=best,
            user_guidance=note,
        )
    assert "copy one of them exactly" in client.seen[1][0][-1].content


def test_a_mis_copied_stack_is_refused_so_the_entry_cannot_reopen_for_ever() -> None:
    best = first_try()
    ask = f"please try {STACK}"
    wrong = "next: layer-stack: train a network from zero through fit(), layers conv(8) linear(8)"
    client = Scripted(wrong, f"next: layer-stack: train from zero with layers {STACK}")
    decision = vision_supervisor(client).decide(
        data_summary="Images: 10",
        baseline=baseline_result(),
        history=[best],
        carried_best=best,
        user_guidance=ask,
    )
    assert "copy one of them exactly" in client.seen[1][0][-1].content
    assert vl.proposed_value("layer-stack", vl.change_clause(decision.brief)) == STACK


def test_a_table_run_and_a_prompt_run_never_ask_what_is_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The lever ladder is the image family's. A table or prompt run calling it would
    read an image recipe out of a tabular experiment."""

    def boom(*_: Any, **__: Any) -> list[vl.Ready]:
        raise AssertionError("ready() is the image family's")

    monkeypatch.setattr(vl, "ready", boom)
    for family in ("tabular", "prompt"):
        client = Scripted("next: model-family: try a gradient boosting model")
        sup.Supervisor(client, metric="f1", family=family).decide(
            data_summary="120 rows", baseline=baseline_result(), history=[]
        )


def test_an_image_run_reads_a_typed_ask_and_a_table_run_leaves_the_note_alone() -> None:
    """The reader is the image family's: it stores a canonical stack in front of the
    note, which a tabular note has no use for and a tabular brief cannot spend."""
    from iterate.core.agent_loop import run_supervised
    from iterate.core.coder import Cell, CodingResult
    from iterate.core.interactive import RunController
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
            return SupervisorDecision(
                False, "t", "next: backbone: fine-tune resnet18 through fit()"
            )

    class Coder:
        def run(self, **kw: Any) -> CodingResult:
            cell, payload = fit_cell(RECIPE, [0.9, 0.95, 0.98], 0.97, start=BENCH)
            cells = [
                Cell(c["code"], c["stdout"], "", None, "agent")
                for c in [cell, submit_cell(payload)]
            ]
            return CodingResult(result=scored(0.97), cells=cells, predictions_sha256="x")

    for family, reads in (("vision", True), ("tabular", False)):
        ctrl = RunController()
        run_supervised(
            target=Target(),
            dataset=object(),  # type: ignore[arg-type]
            supervisor=Supervisor(),  # type: ignore[arg-type]
            make_coder=Coder,  # type: ignore[arg-type]
            terminator=MaxIterations(1),
            memory=InMemoryMemory(),
            data_summary="d",
            family=family,
            controller=ctrl,
        )
        if reads:
            assert ctrl.note_reader is not None
            assert ctrl.note_reader(f"try {STACK}")[0] == f"[ask: layers {STACK}]"
        else:
            assert ctrl.note_reader is None
