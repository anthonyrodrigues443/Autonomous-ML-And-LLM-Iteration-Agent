"""The image session on a real kernel, in its own child process.

Everything else about the session is tested with a fake runner and no torch. This is
the one check that the preamble, the cell prefix, `fit`, `submit`, the own-model
helpers, a restart and the floor all work in a kernel that is confined the way a run
confines it. Skipped where torch is absent, so CI without the vision extra passes.
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
    pytest.mark.slow,
    pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is not installed"),
]


def _check(name: str) -> dict[str, Any]:
    out = subprocess.run(
        [sys.executable, "-m", "tests.unit._vision_kernel_check", name],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    result: dict[str, Any] = json.loads(out.stdout.strip().splitlines()[-1])
    return result


@pytest.fixture(scope="module")
def session() -> dict[str, Any]:
    return _check("session")


@pytest.fixture(scope="module")
def experiment() -> dict[str, Any]:
    return _check("experiment")


@pytest.fixture(scope="module")
def rerun() -> dict[str, Any]:
    if importlib.util.find_spec("nbclient") is None:
        pytest.skip("needs a Jupyter kernel (dev extra: nbclient/ipykernel)")
    return _check("rerun")


def test_torch_never_loads_in_the_test_process() -> None:
    assert "torch" not in sys.modules


def test_the_preamble_decodes_the_images_and_prints_the_worked_example(
    session: dict[str, Any],
) -> None:
    assert session["preamble_error"] is None
    assert session["loaded"]
    assert session["example_printed"]


def test_a_fit_trains_submits_and_records_the_recipe_it_submitted(
    session: dict[str, Any],
) -> None:
    assert session["fit_error"] is None
    (fit,) = session["fit"]
    assert (fit["backbone"], fit["epochs_run"]) == ("simple_cnn", 1)
    assert session["submitted"] == session["fit"]
    assert len(session["predictions"].split()) == 8
    assert set(session["predictions"].split()) <= {"cat", "dog"}
    assert session["probabilities_rows"] == 8
    assert session["recipe"]["predictions_sha256"]


def test_a_confined_cell_stages_a_fit_and_leaves_its_network_beside_the_predictions(
    session: dict[str, Any],
) -> None:
    assert session["staged"]
    assert session["recipe"]["model_sha256"] == session["network_digest"]


def test_an_own_model_submission_leaves_no_network_behind(session: dict[str, Any]) -> None:
    assert not session["network_after_own"]
    assert not session["own_recipe_names_a_network"]


def test_the_own_model_helpers_score_and_submit_under_their_name(
    session: dict[str, Any],
) -> None:
    assert session["own_error"] is None
    (model,) = session["model_lines"]
    assert model["model"] == "tiny_net"
    assert session["own_submitted"][-1]["model"] == "tiny_net"


def test_a_model_that_fixes_its_input_size_gets_pixels_at_that_size(
    session: dict[str, Any],
) -> None:
    assert session["fixed_error"] is None
    assert session["fixed_decoded"]


def test_predict_reads_the_logits_a_hugging_face_model_answers_with(
    session: dict[str, Any],
) -> None:
    """timm answers with a tensor; a Hugging Face image model answers with an object."""
    assert session["logits"] == "[8, 2] 8.0"


def test_a_restart_rebuilds_the_session_and_the_recipe_it_had_reached(
    session: dict[str, Any],
) -> None:
    assert session["restart_error"] is None
    assert session["recipe_after_restart"]["augment"] == "flip"


def test_the_floor_submits_after_a_reset(session: dict[str, Any]) -> None:
    assert session["floor_error"] is None
    assert session["floor_predictions"].split() == ["cat"] * 8


def test_a_cell_cannot_read_outside_the_folders_this_run_named(
    session: dict[str, Any],
) -> None:
    if not session["confined"]:
        pytest.skip("this platform has no sandbox")
    assert session["outside_blocked"] is False  # refused as a path, not as a program


def test_one_image_experiment_runs_the_way_a_run_runs_it(experiment: dict[str, Any]) -> None:
    """The seam the CLI lane could only stub: the real CodingAgent with the vision
    family, a confined kernel, and the host scoring what the session submitted."""
    assert experiment["error"] is None
    assert experiment["score"] == 1.0
    assert experiment["sources"] == ["preamble", "agent"]  # no floor was needed
    assert experiment["stdout_has_fit"]
    assert experiment["artifacts"] == ["recipe.json"]
    assert experiment["recipe"]["backbone"] == "simple_cnn"
    assert experiment["recipe"]["predictions_sha256"]


def test_the_network_outlives_the_kernel_and_predicts_what_the_session_submitted(
    experiment: dict[str, Any],
) -> None:
    """The three hops end to end: the confined cell writes the file, the coder copies it
    out before `close()` deletes the folder, and `iterate.vision.load` opens it."""
    assert experiment["kernel_folder_gone"]
    assert experiment["network_kept"]
    assert experiment["network_digest_matches"]
    assert len(experiment["written"]) == 8
    assert experiment["predicted"] == experiment["written"]
    assert experiment["probability_gap"] < 1e-4


def test_run_all_on_a_delivered_session_carries_past_its_dead_ends(
    rerun: dict[str, Any],
) -> None:
    """Run All has failed at cell 1 since v0.2, and most 12B sessions have a dead end in
    them. The inputs are beside the notebook now and an errored cell is tagged, so the
    whole session replays."""
    assert rerun["errored_cells"] == 1
    assert rerun["tagged_cells"] == 1
    assert rerun["first_reached_the_end"]
    assert rerun["first_submitted"] == 1
    assert rerun["network_written"]
    assert rerun["untagged_stops"]  # the same notebook without the tag stops at it


def test_a_second_run_all_behaves_like_the_first_and_never_touches_the_delivered_model(
    rerun: dict[str, Any],
) -> None:
    """The setup cell puts the folder back: without it the keep-best guard would read
    the first pass's submission and hold the second pass's fit against it."""
    assert rerun["delivered_is_read_only"]
    assert rerun["second_reached_the_end"]
    assert (rerun["second_submitted"], rerun["second_kept"]) == (1, 0)
    assert (rerun["first_submitted"], rerun["first_kept"]) == (1, 0)
    assert rerun["best_model_unchanged"]
    assert rerun["best_model_unchanged_twice"]
    assert rerun["incumbent"] == "simple_cnn"  # the recipe the session started from


def test_a_stack_typed_into_a_cell_builds_trains_and_submits() -> None:
    result = _check("layers_cell")
    assert result["preamble_error"] is None
    assert result["error"] is None
    (fit,) = result["fit"]
    assert fit["backbone"] == "layers_net"
    assert fit["layers"] == "conv(16) pool conv(32) pool"
    assert fit["epochs_run"] == 1
    assert result["submitted"] == result["fit"]
    assert result["said"] == [
        line for line in result["said"] if "(layers conv(16) pool conv(32) pool 32px," in line
    ]
    assert result["said"]
    assert result["predictions"] == 8
    assert result["recipe"]["predictions_sha256"]
    assert result["network"]
