"""One contract, checked from both sides.

The session prints `FIT`, `MODEL` and `SUBMITTED` lines; the lever reader judges a
session from those lines alone. The worked cells the session PRINTS are the shape a
model copies, so they are what runs here: if a helper's signature and the printed
example drift apart, a live cell dies after training.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from iterate.core import codegen
from tests.unit.image_fixtures import session_names, vision_session

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit


def worked_cells(task: str) -> list[str]:
    """The indented blocks of the printed example, as cells a model could paste."""
    blocks: list[list[str]] = []
    for line in codegen.vision_worked_example(task):
        if line.startswith("  "):
            if not blocks or not blocks[-1]:
                blocks.append([])
            blocks[-1].append(line[2:])
        elif blocks and blocks[-1]:
            blocks.append([])
    return ["\n".join(block) for block in blocks if block]


def _run(code: str, namespace: dict[str, Any]) -> None:
    exec(compile(code, "<example>", "exec"), namespace)


def _cell(code: str, capsys: pytest.CaptureFixture[str]) -> list[dict[str, str]]:
    return [{"source": "agent", "code": code, "stdout": capsys.readouterr().out}]


def _levers() -> Any:
    """The roles lane's reader of these lines. Skipped until it lands on this branch."""
    return pytest.importorskip("iterate.core.vision_levers")


# ─── the printed example runs ────────────────────────────────────────────────


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_the_example_is_two_cells_of_valid_python(task: str) -> None:
    cells = worked_cells(task)
    assert len(cells) == 2
    for cell in cells:
        compile(cell, "<example>", "exec")


def test_the_number_example_never_reads_the_classes() -> None:
    """CLASSES is None on a number run, so `num_classes=len(CLASSES)` would raise on
    line one of the cell a model copies."""
    numbers = "\n".join(codegen.vision_worked_example("regression"))
    assert "CLASSES" not in numbers
    assert "cross_entropy" not in numbers
    assert "submit_numbers" in numbers
    classes = "\n".join(codegen.vision_worked_example("classification"))
    assert "len(CLASSES)" in classes
    assert "submit_probabilities" in classes


def test_the_fit_example_runs_on_a_session_and_submits(tmp_path: Path, capsys: Any) -> None:
    session = vision_session(tmp_path, per_class=10, holdout=6, size=32)
    _run(worked_cells("classification")[0], session_names(session))

    out = capsys.readouterr().out
    fits = [json.loads(line[4:]) for line in out.splitlines() if line.startswith("FIT ")]
    submitted = [
        json.loads(line[10:]) for line in out.splitlines() if line.startswith("SUBMITTED ")
    ]
    assert [f["backbone"] for f in fits] == ["resnet18", "resnet18"]
    assert [f["image_size"] for f in fits] == [32, 128]
    assert len(submitted) == 1
    predictions = (session.workdir / codegen.PREDICTIONS_CSV).read_text().splitlines()
    assert len(predictions) == 6


# ─── the lever reader understands what the session printed ───────────────────


def test_the_lever_reader_sees_the_backbone_a_fit_moved(tmp_path: Path, capsys: Any) -> None:
    vl = _levers()
    session = vision_session(tmp_path, per_class=10, holdout=6, size=32)
    carried = dict(vars(session.best))
    code = worked_cells("classification")[0]
    _run(code, session_names(session))

    cells = _cell(code, capsys)
    assert "backbone" in vl.moved_levers(cells, carried)
    assert vl.submitted(cells)["backbone"] == "resnet18"
    assert f"{vl.submitted(cells)['val']:.4f}" in vl.evidence(cells, session.metric)


def test_the_lever_reader_sees_an_own_model_try(tmp_path: Path, capsys: Any) -> None:
    vl = _levers()
    session = vision_session(tmp_path, per_class=10, holdout=6, size=32)
    carried = dict(vars(session.best))
    n_val = len(session.val_idx)
    code = (
        "evaluate(probs, model='vit_base_patch16_224', image_size=224, epochs=3)\n"
        "submit_probabilities(held, model='vit_base_patch16_224')\n"
    )
    _run(
        code,
        {
            **session_names(session),
            "probs": np.full((n_val, 3), 1 / 3),
            "held": np.full((6, 3), 1 / 3),
        },
    )
    cells = _cell(code, capsys)
    assert "own-model" in vl.moved_levers(cells, carried)
    assert vl.submitted(cells)["model"] == "vit_base_patch16_224"


def test_an_own_model_payload_carried_back_still_opens_a_session(
    tmp_path: Path, capsys: Any
) -> None:
    """The round trip an own-code win takes: what the helpers print becomes what the host
    writes into the next session, and the next session's preamble has to survive it."""
    vl = _levers()
    session = vision_session(tmp_path, per_class=10, holdout=6, size=32)
    n_val = len(session.val_idx)
    code = (
        "evaluate(probs, model='efficientnet_b0', image_size=64, epochs=3)\n"
        "submit_probabilities(held, model='efficientnet_b0')\n"
    )
    _run(
        code,
        {
            **session_names(session),
            "probs": np.full((n_val, 3), 1 / 3),
            "held": np.full((6, 3), 1 / 3),
        },
    )
    payload = vl.submitted(_cell(code, capsys))
    assert (payload["model"], payload["epochs"]) == ("efficientnet_b0", 3)

    after = vision_session(tmp_path / "next", per_class=10, holdout=6, size=32, carried=payload)
    assert after.best == after.baseline
    after.fit(backbone="resnet18", epochs=2)
    assert json.loads(capsys.readouterr().out.splitlines()[-2][4:])["backbone"] == "resnet18"


def test_a_fit_the_session_kept_out_never_becomes_the_recipe_the_host_carries(
    tmp_path: Path, capsys: Any
) -> None:
    """The host carries the last SUBMITTED payload, and keeps recipe.json only while it
    describes predictions.csv. A worse fit submitted second must leave both on the first."""
    from iterate.core.coder import _recipe_describes
    from iterate.core.vision_session import Fit

    vl = _levers()
    session = vision_session(tmp_path, holdout=6, size=32)

    def scored(val: float, lr: float, column: int) -> Fit:
        out = np.full((6, 3), 0.1)
        out[:, column] = 0.8
        return Fit({"backbone": "resnet18", "lr": lr, "val": val}, out)

    names = {**session_names(session), "a": scored(0.9573, 0.001, 1), "b": scored(0.9313, 0.002, 2)}
    cells = []
    for code in ("submit(a)", "submit(b)"):
        _run(code, names)
        cells += _cell(code, capsys)

    assert cells[0]["stdout"].startswith("SUBMITTED ")
    assert cells[1]["stdout"].startswith("KEPT ")
    assert "SUBMITTED" not in cells[1]["stdout"]
    assert vl.submitted(cells) == names["a"].line
    work = session.workdir
    recorded = (work / codegen.RECIPE_JSON).read_bytes()
    assert _recipe_describes(recorded, (work / codegen.PREDICTIONS_CSV).read_bytes())
    assert json.loads(recorded)["lr"] == 0.001
    assert set((work / codegen.PREDICTIONS_CSV).read_text().split()) == {"c1"}


def test_an_explicit_size_on_the_first_fine_tune_reads_as_a_move(
    tmp_path: Path, capsys: Any
) -> None:
    """Leaving the plain CNN builds the fine-tune reference, which carries no size of its
    own. The size it would have run at is the session's, not the one this fit asked for,
    or an `image_size=` the agent typed reads as no move."""
    vl = _levers()
    session = vision_session(tmp_path, per_class=10, holdout=6, size=64)
    carried = dict(vars(session.best))
    code = "fit(backbone='resnet18', image_size=32)"
    _run(code, session_names(session))

    cells = _cell(code, capsys)
    assert vl.submitted_try(cells) is None  # nothing was submitted; the fit line is enough
    assert vl.tries(cells)[-1].recipe["image_size"] == 32
    assert vl.tries(cells)[-1].start["image_size"] == 64
    assert vl.moved_levers(cells, carried) == ["backbone", "image-size"]
