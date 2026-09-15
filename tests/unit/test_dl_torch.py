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
    assert 7.0 < result["left"] <= 7.5


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
