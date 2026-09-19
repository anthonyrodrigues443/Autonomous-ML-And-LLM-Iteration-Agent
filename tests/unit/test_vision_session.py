"""The image session inside the kernel, with no torch.

A fake runner stands in for the backbone, as the vision target's own tests do, so CI
covers the fold, what `fit()` starts from, the printed contract, the shape refusals and
the files a submit writes without a GPU or a download.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from iterate.core import codegen, vision_session
from iterate.core.vision_session import Fit, Session, merge
from iterate.targets import dl
from iterate.targets.dl import BASELINE, Recipe, RecipeError
from tests.unit.image_fixtures import FakeVisionRunner
from tests.unit.image_fixtures import vision_session as _session

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit


def _printed(capsys: pytest.CaptureFixture[str], tag: str) -> list[dict[str, Any]]:
    out = capsys.readouterr().out
    return [
        json.loads(line[len(tag) + 1 :]) for line in out.splitlines() if line.startswith(tag + " ")
    ]


# ─── the fold ────────────────────────────────────────────────────────────────


def test_the_fold_holds_back_a_fifth_of_every_class(tmp_path: Path) -> None:
    session = _session(tmp_path, classes=4, per_class=10)
    assert len(session.val_idx) == 8  # 20% of each of the four classes
    assert set(np.asarray(session.labels)[session.val_idx]) == {0, 1, 2, 3}
    assert not set(session.val_idx.tolist()) & set(session.fit_idx.tolist())
    assert len(session.val_idx) + len(session.fit_idx) == len(session.labels)


def test_a_class_with_one_or_two_images_still_reaches_the_fold(tmp_path: Path) -> None:
    """Rounding 20% of one image gives none, and a probability metric then has no
    column for that class."""
    session = _session(tmp_path, classes=3, per_class=1)
    assert set(np.asarray(session.labels)[session.val_idx]) == {0, 1, 2}
    two = _session(tmp_path / "two", classes=3, per_class=2)
    assert set(np.asarray(two.labels)[two.val_idx]) == {0, 1, 2}


def test_the_same_fold_comes_back_in_a_second_session(tmp_path: Path) -> None:
    first = _session(tmp_path, classes=3, per_class=10)
    second = _session(tmp_path, classes=3, per_class=10)
    assert first.val_idx.tolist() == second.val_idx.tolist()


def test_a_number_run_holds_back_a_plain_fifth(tmp_path: Path) -> None:
    session = _session(tmp_path, task="regression", metric="rmse", classes=4, per_class=5)
    assert len(session.val_idx) == 4
    assert not set(session.val_idx.tolist()) & set(session.fit_idx.tolist())


# ─── what a fit starts from ──────────────────────────────────────────────────


def _merge(best: Recipe, changes: dict[str, Any], *, baseline: Recipe | None = None) -> Recipe:
    return merge(
        best,
        changes,
        "classification",
        baseline=baseline or replace(BASELINE, image_size=64),
    )[0]


def test_a_pretrained_backbone_from_the_plain_cnn_starts_as_a_fine_tune() -> None:
    recipe, start = merge(
        BASELINE,
        {"backbone": "resnet18"},
        "classification",
        baseline=replace(BASELINE, image_size=64),
    )
    assert (recipe.backbone, recipe.unfreeze, recipe.epochs) == ("resnet18", "all", 3)
    assert start["backbone"] == "resnet18"  # the reference, not the 20-epoch baseline


def test_the_plain_cnn_from_a_fine_tune_starts_from_the_runs_own_baseline() -> None:
    """The host scored the baseline at 64 px, so `fit(backbone='simple_cnn')` on a
    160 px session must mean that recipe and not a 160 px, 20-epoch CNN."""
    tuned = Recipe(backbone="resnet18", unfreeze="all", epochs=3, image_size=160)
    recipe, start = merge(
        tuned,
        {"backbone": "simple_cnn"},
        "classification",
        baseline=replace(BASELINE, image_size=64),
    )
    assert (recipe.backbone, recipe.epochs, recipe.image_size) == ("simple_cnn", 20, 64)
    assert start["image_size"] == 64


def test_epochs_and_unfreeze_follow_each_other() -> None:
    probe = Recipe(backbone="resnet18", unfreeze="none", epochs=0)
    assert _merge(probe, {"epochs": 5}).unfreeze == "all"
    assert _merge(probe, {"unfreeze": "head"}).epochs == 3
    tuned = Recipe(backbone="resnet18", unfreeze="all", epochs=3)
    assert _merge(tuned, {"unfreeze": "none"}).epochs == 0
    assert _merge(tuned, {"epochs": 0}).unfreeze == "none"


def test_a_recipe_the_runner_refuses_is_still_refused() -> None:
    with pytest.raises(RecipeError, match="unknown recipe keys"):
        _merge(BASELINE, {"dropout": 0.5})
    with pytest.raises(RecipeError, match="label_smoothing"):
        merge(
            Recipe(backbone="resnet18", unfreeze="all", epochs=3),
            {"label_smoothing": 0.1},
            "regression",
            baseline=BASELINE,
        )


def test_the_carried_best_is_what_fit_starts_from(tmp_path: Path, capsys: Any) -> None:
    carried = {"backbone": "resnet18", "unfreeze": "all", "epochs": 3, "image_size": 32}
    session = _session(tmp_path, carried=carried)
    assert session.best.backbone == "resnet18"
    session.fit(epochs=6)
    line = _printed(capsys, "FIT")[-1]
    assert (line["backbone"], line["epochs"]) == ("resnet18", 6)
    assert line["from"]["backbone"] == "resnet18"


def test_a_second_fit_starts_from_the_first_and_a_restart_rebuilds_it(
    tmp_path: Path, capsys: Any
) -> None:
    carried = {"backbone": "resnet18", "unfreeze": "all", "epochs": 3, "image_size": 32}
    session = _session(tmp_path, carried=carried)
    session.fit(epochs=6)
    session.fit(augment="flip_crop")
    second = _printed(capsys, "FIT")[-1]
    assert (second["epochs"], second["augment"]) == (6, "flip_crop")

    restarted = _session(tmp_path, carried=carried)  # the same workdir, a new kernel
    assert asdict(restarted.best)["epochs"] == 6
    assert asdict(restarted.best)["augment"] == "flip_crop"


def test_the_carried_file_may_carry_what_a_recipe_does_not(tmp_path: Path) -> None:
    carried = {
        "backbone": "resnet18",
        "unfreeze": "all",
        "epochs": 3,
        "val": 0.91,
        "epochs_planned": 3,
        "predictions_sha256": "beef",
    }
    assert _session(tmp_path, carried=carried).best.backbone == "resnet18"


def test_with_nothing_carried_a_fit_starts_from_the_runs_baseline(tmp_path: Path) -> None:
    session = _session(tmp_path, size=160)
    assert session.best == replace(BASELINE, image_size=64)


def test_an_own_model_incumbent_is_not_a_fit_recipe(tmp_path: Path, capsys: Any) -> None:
    """The payload an own-code win leaves behind: `epochs` counts loops the agent wrote
    and no backbone is named, so `fit()` departs from this run's baseline instead."""
    carried = {
        "model": "efficientnet_b0",
        "image_size": 64,
        "epochs": 3,
        "seconds": 143,
        "val": 0.9417115871541657,
        "val_accuracy": 0.944,
    }
    session = _session(tmp_path, carried=carried)
    assert session.best == session.baseline
    session.fit(backbone="resnet18", epochs=2)
    assert _printed(capsys, "FIT")[-1]["backbone"] == "resnet18"


def test_a_partial_carry_fills_in_from_this_runs_baseline(tmp_path: Path) -> None:
    """Not from `Recipe`'s own defaults, which would read a carried size alone as a
    resnet18 probe, nothing like the run that produced it."""
    session = _session(tmp_path, carried={"image_size": 32})
    assert (session.best.backbone, session.best.epochs) == (BASELINE.backbone, BASELINE.epochs)


def test_a_carried_recipe_the_runner_refuses_starts_from_the_baseline(
    tmp_path: Path, capsys: Any
) -> None:
    """No file the host wrote may raise out of the preamble: that leaves the cell with no
    helper bound and costs the whole iteration."""
    session = _session(tmp_path, carried={"unfreeze": "none"})
    assert session.best == session.baseline
    assert "starting from the baseline" in capsys.readouterr().out


# ─── the fit itself ──────────────────────────────────────────────────────────


def test_a_fit_trains_on_the_fold_and_predicts_val_and_holdout(tmp_path: Path) -> None:
    runner = FakeVisionRunner()
    session = _session(tmp_path, per_class=10, holdout=6, runner=runner)
    session.begin_cell()
    session.fit(backbone="resnet18", epochs=2)
    job = runner.jobs[-1]
    assert len(job.train) == len(session.fit_idx)
    assert len(job.holdout) == len(session.val_idx) + 6
    assert len(job.labels) == len(session.fit_idx)
    assert job.deadline == pytest.approx(session._tick() + session.budget, abs=1.0)


def test_the_fit_line_carries_the_epochs_planned_and_run_and_a_plain_score(
    tmp_path: Path, capsys: Any
) -> None:
    session = _session(tmp_path, per_class=10)
    session.fit(backbone="resnet18", epochs=4)
    line = _printed(capsys, "FIT")[-1]
    assert (line["epochs_planned"], line["epochs_run"]) == (4, 3)
    assert 0.0 <= line["val_accuracy"] <= 1.0
    assert line["val"] == pytest.approx(line["val_accuracy"], abs=1e-4)  # accuracy here
    assert set(asdict(session.best)) <= set(line)


def test_a_number_run_prints_r2_beside_its_own_metric(tmp_path: Path, capsys: Any) -> None:
    session = _session(tmp_path, task="regression", metric="rmse", per_class=10)
    session.fit(backbone="resnet18", epochs=2)
    line = _printed(capsys, "FIT")[-1]
    assert "val_r2" in line
    assert "val_accuracy" not in line
    assert line["val"] > 0  # rmse in the label's own units


def test_only_a_fit_that_returns_moves_the_best(tmp_path: Path) -> None:
    session = _session(tmp_path, runner=FakeVisionRunner(fail=RuntimeError("device gone")))
    before = session.best
    with pytest.raises(RuntimeError):
        session.fit(backbone="resnet18", epochs=2)
    assert session.best == before


def test_a_second_fit_in_one_cell_is_told_to_start_a_new_cell(tmp_path: Path) -> None:
    session = _session(tmp_path)
    session.begin_cell()
    session.fit(backbone="resnet18", epochs=2)
    session.runner = FakeVisionRunner(fail=RecipeError("the fit budget ran out"))  # type: ignore[assignment]
    with pytest.raises(RecipeError, match="run this fit at the top of a new cell"):
        session.fit(epochs=3)


def test_the_first_fit_of_a_cell_keeps_the_runners_own_refusal(tmp_path: Path) -> None:
    session = _session(tmp_path, runner=FakeVisionRunner(fail=RecipeError("halve image_size")))
    session.begin_cell()
    with pytest.raises(RecipeError, match=r"^halve image_size$"):
        session.fit(backbone="resnet18", epochs=2)


def test_a_fit_at_a_new_size_decodes_that_size_and_keeps_the_sessions(
    tmp_path: Path,
) -> None:
    runner = FakeVisionRunner()
    session = _session(tmp_path, size=32, runner=runner)
    session.pixels(session.size)
    session.fit(backbone="resnet18", epochs=1, image_size=64)
    assert runner.jobs[-1].train.shape[-1] == 64
    assert set(session._pixels) == {32, 64}
    session.fit(image_size=96)
    assert set(session._pixels) == {32, 96}  # one other size at a time, beside the session's


# ─── what a submit writes ────────────────────────────────────────────────────


def test_submitting_a_fit_writes_predictions_probabilities_and_the_recipe(
    tmp_path: Path, capsys: Any
) -> None:
    session = _session(tmp_path, per_class=10, holdout=6)
    fit = session.fit(backbone="resnet18", epochs=2)
    session.submit(fit)
    work = session.workdir
    predictions = (work / codegen.PREDICTIONS_CSV).read_text().splitlines()
    assert len(predictions) == 6
    assert set(predictions) <= {"c0", "c1", "c2"}
    assert len((work / codegen.PROBABILITIES_CSV).read_text().splitlines()) == 6
    recorded = json.loads((work / codegen.RECIPE_JSON).read_text())
    digest = hashlib.sha256((work / codegen.PREDICTIONS_CSV).read_bytes()).hexdigest()
    assert recorded["predictions_sha256"] == digest
    assert recorded["backbone"] == "resnet18"
    submitted = _printed(capsys, "SUBMITTED")[-1]
    assert submitted["backbone"] == "resnet18"
    assert submitted["val"] == fit.val


def test_submitting_needs_what_fit_returned(tmp_path: Path) -> None:
    session = _session(tmp_path)
    with pytest.raises(TypeError, match="submit_probabilities"):
        session.submit(np.zeros((6, 3)))  # type: ignore[arg-type]
    numbers = _session(tmp_path / "n", task="regression", metric="rmse")
    with pytest.raises(TypeError, match="submit_numbers"):
        numbers.submit(np.zeros(6))  # type: ignore[arg-type]


def test_own_probabilities_are_submitted_under_their_model_name(
    tmp_path: Path, capsys: Any
) -> None:
    session = _session(tmp_path, holdout=6)
    probs = np.full((6, 3), 1 / 3)
    session.submit_probabilities(probs, model="efficientnet_b0")
    recorded = json.loads((session.workdir / codegen.RECIPE_JSON).read_text())
    assert recorded["model"] == "efficientnet_b0"
    assert _printed(capsys, "SUBMITTED")[-1]["model"] == "efficientnet_b0"


def test_submitted_probability_rows_sum_to_one_tightly_enough_to_score(tmp_path: Path) -> None:
    """Live on EuroSAT, 2026-09-18: an own-code efficientnet_b0 submitted a float32
    softmax, and scoring it printed "The y_prob values do not sum to one"."""
    session = _session(tmp_path, holdout=6)
    logits = np.linspace(-6, 6, 18, dtype=np.float32).reshape(6, 3)
    raised = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = np.asarray((raised / raised.sum(axis=1, keepdims=True)).astype(np.float32), dtype=float)
    assert np.abs(probs.sum(axis=1) - 1).max() > 5 * np.finfo(float).eps  # what the model hands in
    session.submit_probabilities(probs, model="efficientnet_b0")
    written = np.loadtxt(session.workdir / codegen.PROBABILITIES_CSV, delimiter=",", ndmin=2)
    assert np.abs(written.sum(axis=1) - 1).max() <= 5 * np.finfo(float).eps


def test_a_model_evaluated_first_carries_its_numbers_into_the_submission(
    tmp_path: Path, capsys: Any
) -> None:
    session = _session(tmp_path, per_class=10, holdout=6)
    n_val = len(session.val_idx)
    session.evaluate(np.full((n_val, 3), 1 / 3), model="vit", image_size=64, epochs=3)
    session.submit_probabilities(np.full((6, 3), 1 / 3), model="vit")
    submitted = _printed(capsys, "SUBMITTED")[-1]
    assert (submitted["model"], submitted["image_size"], submitted["epochs"]) == ("vit", 64, 3)
    assert "val" in submitted


def test_own_numbers_are_submitted_in_the_labels_own_units(tmp_path: Path) -> None:
    session = _session(tmp_path, task="regression", metric="rmse", holdout=6)
    session.submit_numbers(np.arange(6, dtype=float), model="own")
    written = (session.workdir / codegen.PREDICTIONS_CSV).read_text().splitlines()
    assert [float(v) for v in written] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert not (session.workdir / codegen.PROBABILITIES_CSV).exists()


@pytest.mark.parametrize(
    ("bad", "message"),
    [
        (np.full((5, 3), 1 / 3), "expected probabilities of shape"),
        (np.full((6, 2), 0.5), "expected probabilities of shape"),
        (np.full((6, 3), np.nan), "finite"),
        (np.full((6, 3), 3.0), "sum to 1"),
    ],
)
def test_probabilities_that_cannot_be_scored_are_refused(
    tmp_path: Path, bad: np.ndarray, message: str
) -> None:
    session = _session(tmp_path, holdout=6)
    with pytest.raises(ValueError, match=message):
        session.submit_probabilities(bad, model="own")


def test_logits_are_refused_with_the_line_that_fixes_them(tmp_path: Path) -> None:
    session = _session(tmp_path, holdout=6)
    with pytest.raises(ValueError, match=r"torch.softmax\(logits, dim=1\)"):
        session.submit_probabilities(np.linspace(-4, 4, 18).reshape(6, 3), model="own")


def test_the_helpers_of_the_other_task_name_the_right_one(tmp_path: Path) -> None:
    session = _session(tmp_path, holdout=6)
    with pytest.raises(ValueError, match=r"submit_probabilities\(probs, model=NAME\)"):
        session.submit_numbers(np.zeros(6), model="own")
    numbers = _session(tmp_path / "n", task="regression", metric="rmse", holdout=6)
    with pytest.raises(ValueError, match=r"submit_numbers\(values, model=NAME\)"):
        numbers.submit_probabilities(np.zeros((6, 3)), model="own")


def test_a_wrong_number_of_numbers_is_refused(tmp_path: Path) -> None:
    session = _session(tmp_path, task="regression", metric="rmse", holdout=6)
    with pytest.raises(ValueError, match="expected 6 numbers"):
        session.submit_numbers(np.zeros(5), model="own")
    session.submit_numbers(np.zeros((6, 1)), model="own")  # a column is accepted


# ─── evaluate ────────────────────────────────────────────────────────────────


def test_evaluate_needs_the_model_name(tmp_path: Path) -> None:
    session = _session(tmp_path, per_class=10)
    n_val = len(session.val_idx)
    with pytest.raises(TypeError, match="model"):
        session.evaluate(np.full((n_val, 3), 1 / 3))  # type: ignore[call-arg]


def test_evaluate_scores_the_fold_with_the_runs_metric(tmp_path: Path, capsys: Any) -> None:
    session = _session(tmp_path, per_class=10)
    truth = np.asarray(session.labels)[session.val_idx]
    perfect = np.zeros((len(truth), 3))
    perfect[np.arange(len(truth)), truth] = 1.0
    assert session.evaluate(perfect, model="own") == pytest.approx(1.0)
    line = _printed(capsys, "MODEL")[-1]
    assert (line["model"], line["val"], line["val_accuracy"]) == ("own", 1.0, 1.0)


def test_evaluate_passes_probabilities_only_where_the_metric_needs_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[Any] = []
    real = vision_session.Session._score

    def spy(self: Session, out: np.ndarray) -> float:
        from iterate.core import scoring

        captured: dict[str, Any] = {}
        original = scoring.score

        def record(*args: Any, **kwargs: Any) -> Any:
            captured["y_proba"] = kwargs.get("y_proba")
            return original(*args, **kwargs)

        monkeypatch.setattr(scoring, "score", record)
        value = real(self, out)
        seen.append(captured.get("y_proba"))
        monkeypatch.setattr(scoring, "score", original)
        return value

    monkeypatch.setattr(vision_session.Session, "_score", spy)
    labels = _session(tmp_path, per_class=10)
    n_val = len(labels.val_idx)
    labels.evaluate(np.full((n_val, 3), 1 / 3), model="own")
    assert seen[-1] is None

    proba = _session(tmp_path / "p", metric="log_loss", per_class=10)
    n_val = len(proba.val_idx)
    proba.evaluate(np.full((n_val, 3), 1 / 3), model="own")
    assert seen[-1] is not None


def test_evaluate_refuses_a_shape_that_is_not_the_fold(tmp_path: Path) -> None:
    session = _session(tmp_path, per_class=10)
    with pytest.raises(ValueError, match="VAL_IDX order"):
        session.evaluate(np.full((2, 3), 1 / 3), model="own")


# ─── the cell prefix ─────────────────────────────────────────────────────────


def test_begin_cell_puts_back_a_name_a_cell_rebound(tmp_path: Path) -> None:
    session = _session(tmp_path)
    names = {"fit": session.fit, "labels": session.labels}
    vision_session._CURRENT = (session, names)
    try:
        namespace: dict[str, Any] = {"fit": "clobbered", "labels": None}
        vision_session.begin_cell(namespace)
        assert namespace["fit"] == session.fit
        assert namespace["labels"] is session.labels
    finally:
        vision_session._CURRENT = None


def test_begin_cell_before_a_session_starts_does_nothing() -> None:
    assert vision_session._CURRENT is None
    namespace: dict[str, Any] = {"X_train": 1}
    vision_session.begin_cell(namespace)
    assert namespace == {"X_train": 1}


def test_begin_cell_restamps_the_fit_clock(tmp_path: Path) -> None:
    session = _session(tmp_path, budget=30.0)
    session.begin_cell()
    first = session.seconds_left()
    session._cell_start = session._cell_start - 10.0 if session._cell_start else None
    assert session.seconds_left() < first - 5
    session.begin_cell()
    assert session.seconds_left() == pytest.approx(first, abs=1.0)


def test_a_cells_own_result_is_let_go_at_the_top_of_the_next_cell() -> None:
    """IPython keeps the last cell's value in `last_execution_result` until the cell
    after next ends. On the device that value is a whole fit."""
    from IPython.core.interactiveshell import InteractiveShell

    shell = InteractiveShell.instance()
    try:
        vision_session.stop_output_cache()
        shell.run_cell(
            "import gc, weakref\n"
            "class Big: pass\n"
            "_holder = [Big()]\n"
            "held = weakref.ref(_holder[0])\n"
            "_holder.pop()\n"  # the object becomes this cell's result and nothing else
        )
        shell.run_cell("gc.collect(); pinned = held() is not None")
        assert shell.user_ns["pinned"]
        shell.run_cell(
            codegen.VISION_CELL_PREFIX.splitlines()[0] + "\ngc.collect()\nfreed = held() is None"
        )
        assert shell.user_ns["freed"]
    finally:
        InteractiveShell.clear_instance()


def test_a_failed_cells_frame_is_let_go_at_the_top_of_the_next_cell(tmp_path: Path) -> None:
    """A traceback IPython keeps holds every tensor of the frame that raised."""
    import gc
    import weakref

    from IPython.core.interactiveshell import InteractiveShell

    shell = InteractiveShell.instance()
    try:
        vision_session.stop_output_cache()
        shell.run_cell("class Big:\n    pass\nkeep = Big()")
        held = weakref.ref(shell.user_ns["keep"])
        shell.run_cell("def train(x):\n    raise RuntimeError('device gone')\ntrain(keep)")
        shell.run_cell("del keep")
        gc.collect()
        assert held() is not None  # pinned by the traceback IPython kept
        _session(tmp_path).begin_cell()
        gc.collect()
        assert held() is None
    finally:
        InteractiveShell.clear_instance()


# ─── RAM ─────────────────────────────────────────────────────────────────────


def test_a_session_too_big_for_ram_lowers_its_size_instead_of_binding_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import inspect

    session = _session(tmp_path, per_class=10, holdout=6, size=160)
    n = len(session.train_paths) + len(session.holdout_paths)
    # A quarter of this RAM holds one and a half times the pixels of a 64 px decode.
    monkeypatch.setattr(dl, "_ram_bytes", lambda: 4 * 1.5 * n * 3 * 64 * 64)
    assert session.largest_size_that_fits(160) == 64
    with pytest.raises(ValueError, match="lower image_size"):
        session.pixels(160)
    # start() must lower the size BEFORE it decodes: a refusal there binds no helpers at
    # all, so the advice to lower image_size could not be taken.
    source = inspect.getsource(vision_session.start)
    assert source.index("largest_size_that_fits") < source.index("session.pixels(")


def test_a_size_that_fits_is_left_alone(tmp_path: Path) -> None:
    session = _session(tmp_path, size=160)
    assert session.largest_size_that_fits(160) == 160


# ─── the pixels a cell reads ─────────────────────────────────────────────────


def test_the_decoded_pixels_cannot_be_written_into(tmp_path: Path) -> None:
    session = _session(tmp_path)
    train_px, _ = session.pixels(session.size)
    with pytest.raises(ValueError, match="read-only"):
        train_px[0, 0, 0, 0] = 1
    with pytest.raises(ValueError, match="read-only"):
        session.labels[0] = 0


def test_a_fit_returns_its_holdout_predictions_only_through_submit(tmp_path: Path) -> None:
    session = _session(tmp_path, holdout=6)
    fit = session.fit(backbone="resnet18", epochs=1)
    assert isinstance(fit, Fit)
    assert "_holdout_out" not in repr(fit)
    assert len(fit._holdout_out) == 6
