"""The real runner on a tiny model, each check in its own child process.

torch and lightgbm each ship their own OpenMP runtime, and on macOS a fit in one
after the other in the same process crashes or hangs, so torch never loads inside
the pytest process. Skipped where torch is absent, so CI without the vision extra
passes; run here before every push.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from typing import Any

import pytest

from evals.config import REPO_ROOT

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is not installed"),
]


def _check(name: str) -> dict[str, Any]:
    out = subprocess.run(
        [sys.executable, "-m", "tests.unit._torch_checks", name],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert out.returncode == 0, out.stderr[-1500:]
    result: dict[str, Any] = json.loads(out.stdout.strip().splitlines()[-1])
    if "skip" in result:
        pytest.skip(result["skip"])
    return result


def test_torch_never_loads_in_the_test_process() -> None:
    assert "torch" not in sys.modules


def test_a_tiny_model_fits_on_cpu_and_prints_its_epochs() -> None:
    result = _check("fit_prints_epochs")
    assert result["shape"] == [9, 3]
    assert result["sums"]
    assert result["epochs"] == [2, 2]
    assert result["heads"] == [["epoch", "1/2"], ["epoch", "2/2"]]


def test_a_trimmed_plan_builds_its_schedule_over_the_epochs_it_runs() -> None:
    """1 s a step and a tenth of margin, 3 steps an epoch, 2 steps kept back to
    predict: 7.5 s left leaves room for 2 whole epochs of the 5 asked for, and the
    schedule spans exactly those."""
    result = _check("trimmed_plan")
    assert result["epochs"] == [2, 2]
    assert result["steps"] == 2 * 3
    assert result["epoch_seconds"] == 3.3
    assert result["left"] == 7.5


def test_an_epoch_the_deadline_cuts_still_predicts() -> None:
    """A clock that moves one second per reading: the plan allows all 5 epochs, each
    epoch reads the clock five times, so the first step of epoch 5 is past the stop."""
    result = _check("cut_epoch")
    assert result["epochs"] == [5, 4]
    assert result["last"] == "stopped in epoch 5/5: the fit budget ran out"
    assert result["shape"] == [9, 3]


def test_a_probe_after_a_fine_tune_gives_the_same_features() -> None:
    assert _check("probe_after_fine_tune")["same"]


def test_a_head_copied_from_the_probe_predicts_what_the_probe_predicts() -> None:
    assert _check("head_copy")["same"]


def test_on_mps_the_default_pool_starts_at_the_capped_ratios() -> None:
    result = _check("mps_starts")
    assert result == {"started": True, "high": "0.7", "low": "0.56"}


def test_on_mps_an_allocation_over_a_small_cap_raises_the_text_oom_kind_reads() -> None:
    assert _check("mps_cap")["kind"] == "oom"


def test_timing_a_step_moves_neither_the_weights_nor_the_batch_norm_statistics() -> None:
    assert _check("timing_moves_nothing") == {"same": True, "lr": 0.01, "state": 0, "timed": True}


def test_a_head_only_fit_keeps_the_backbones_batch_norm_statistics() -> None:
    assert _check("head_only_keeps_batch_norm") == {"head": True, "all": False}


def test_simple_cnn_has_its_parameter_count_for_any_number_of_outputs() -> None:
    result = _check("simple_cnn_parameters")
    assert result["counts"] == {str(k): 93_696 + 129 * k for k in (1, 10, 102)}
    assert result["children"] == ["body", "head"]
    assert result["shapes"] == [[2, 10], [2, 10]]


def test_a_short_fit_learns_a_number_and_prints_train_r2() -> None:
    """The label is each image's mean brightness."""
    result = _check("regression_fit")
    assert result["shape"] == [40]
    assert result["epochs"] == [15, 15]
    assert (result["r2_lines"], result["acc_lines"]) == (15, 0)
    assert result["r2"] > 0.5
    pairs = result["loss_and_r2"]
    assert len(pairs) == 15
    assert all(abs(r2 - (1 - loss)) <= 1.5e-4 for loss, r2 in pairs)
    assert pairs[-1][1] > 0.5


def test_a_ridge_probe_head_copied_into_resnet18_predicts_what_the_probe_predicts() -> None:
    result = _check("ridge_head_copy")
    assert result["shape"][0] == result["shape"][1]
    assert result["gap"] < 1e-4


def test_a_fixed_job_runs_every_epoch_where_the_plan_would_refuse() -> None:
    result = _check("fixed_runs_every_epoch")
    assert result["epochs"] == [3, 3]
    assert result["lines"] == 3
    assert result["plan_calls"] == [0, 1]
    assert result["refused"].startswith("one epoch needs about 3300s")
    assert result["refused"].endswith("are left; halve image_size")


def test_staging_a_fits_weights_moves_none_of_its_outputs() -> None:
    assert _check("staging_moves_no_output") == {"same": True, "staged": True}


@pytest.mark.parametrize(
    ("check", "kind"),
    [
        ("saved_fit_round_trip", ["resnet18", "all", "classification"]),
        ("saved_probe_round_trip", ["resnet18", "none", "classification"]),
        ("saved_simple_cnn_round_trip", ["simple_cnn", "all", "classification"]),
    ],
)
def test_a_submitted_network_opens_again_and_predicts_what_was_submitted(
    check: str, kind: list[str]
) -> None:
    """Opened the way `iterate.vision.load` opens it, from the image files and not the
    session's pixels. A probe's saved head is float32 where the probe was float64."""
    result = _check(check)
    assert result["kind"] == kind
    assert result["same"]
    assert result["gap"] < (1e-4 if kind[1] == "none" else 1e-6)
    assert result["digest"]
    assert result["leftovers"] == []


def test_a_saved_number_network_predicts_in_the_labels_own_units() -> None:
    result = _check("saved_regression_round_trip")
    assert result["kind"] == ["resnet18", "all", "regression"]
    assert result["gap"] < 1e-4
    assert result["digest"]


def test_every_recipe_shape_saves_and_opens_again_with_no_download() -> None:
    from tests.unit._torch_checks import ROUND_TRIP_SHAPES

    result = _check("every_recipe_shape_round_trips")
    assert len(result["gaps"]) == len(ROUND_TRIP_SHAPES)
    assert max(result["gaps"].values()) < 1e-6
    assert result["downloaded"] == []
