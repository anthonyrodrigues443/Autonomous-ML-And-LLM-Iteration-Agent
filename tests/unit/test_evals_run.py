"""`evals.run ceilings` writing to the store: a vision ceiling replaces an older vision
sweep's record, and a tabular one still keeps the better of what is stored."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from evals import run
from evals.corpus import Dataset
from evals.store import Ceiling, Store

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit


def _ceiling(dataset: str, ceiling: float, baseline: float, method: str, detail: str) -> Ceiling:
    return Ceiling(
        dataset=dataset,
        dataset_hash=f"hash-{dataset}",
        metric="accuracy",
        ceiling=ceiling,
        direction="maximize",
        baseline=baseline,
        method=method,
        measured_at="2026-09-16T00:00:00+00:00",
        detail=detail,
    )


def _datasets(tmp_path: Path) -> list[Dataset]:
    rows = []
    for name, family in (("flowers", "vision"), ("heart", "")):
        path = tmp_path / f"{name}.csv"
        path.write_text("a,label\n1,x\n", encoding="utf-8")
        rows.append(Dataset(name, path, "label", "accuracy", family=family))
    return rows


@pytest.fixture
def ceilings_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, bool]]:
    """Both sweeps faked; returns every put_ceiling call as (dataset, replace)."""
    monkeypatch.setattr(run.corpus, "select", lambda _names: _datasets(tmp_path))
    monkeypatch.setattr(Dataset, "content_hash", lambda self: f"hash-{self.name}")
    new_vision = _ceiling(
        "flowers",
        0.95,
        0.55,
        "vision_recipe_sweep_v2 (13 of 13 recipes on mps, best: x)",
        '{"sweep": 2}',
    )
    new_tabular = _ceiling("heart", 0.90, 0.89, "brute_force_sweep_v1 (9 models)", "[]")
    monkeypatch.setattr(run.vision_ceilings, "sweep_in_child", lambda *_a, **_k: new_vision)
    monkeypatch.setattr(run.ceilings_mod, "sweep", lambda *_a, **_k: (new_tabular, []))

    calls: list[tuple[str, bool]] = []
    put = Store.put_ceiling

    def spy(self: Store, ceiling: Ceiling, *, replace: bool = False) -> Ceiling:
        calls.append((ceiling.dataset, replace))
        return put(self, ceiling, replace=replace)

    monkeypatch.setattr(Store, "put_ceiling", spy)
    return calls


def _run(
    tmp_path: Path, stored: list[Ceiling], calls: list[tuple[str, bool]]
) -> dict[str, Ceiling]:
    db = tmp_path / "results.db"
    with Store(db) as store:
        for ceiling in stored:
            store.put_ceiling(ceiling)
    calls.clear()
    assert run.main(["ceilings", "--store", str(db), "--force"]) == 0
    with Store(db) as store:
        return {c.dataset: c for c in store.ceilings()}


def test_a_vision_ceiling_replaces_an_older_sweep_and_carries_its_better_ceiling(
    tmp_path: Path, ceilings_run: list[tuple[str, bool]], capsys: pytest.CaptureFixture[str]
) -> None:
    old = "vision_recipe_sweep_v1 (12 of 12 recipes on mps, best: y)"
    stored = _run(tmp_path, [_ceiling("flowers", 0.97, 0.89, old, '{"sweep": 1}')], ceilings_run)

    assert ceilings_run == [("flowers", True), ("heart", False)]
    flowers = stored["flowers"]
    assert (flowers.ceiling, flowers.baseline, flowers.detail) == (0.97, 0.55, '{"sweep": 2}')
    assert flowers.method.startswith("vision_recipe_sweep_v2 ")
    assert flowers.method.endswith("ceiling carried from vision_recipe_sweep_v1")
    out = capsys.readouterr().out
    assert "ceiling 0.9700, baseline 0.5500" in out
    assert f"stored {flowers.method}" in out


def test_a_vision_ceiling_over_a_worse_older_sweep_is_the_new_record(
    tmp_path: Path, ceilings_run: list[tuple[str, bool]]
) -> None:
    old = "vision_recipe_sweep_v1 (12 of 12 recipes on mps, best: y)"
    stored = _run(tmp_path, [_ceiling("flowers", 0.90, 0.89, old, '{"sweep": 1}')], ceilings_run)

    assert ceilings_run == [("flowers", True), ("heart", False)]
    flowers = stored["flowers"]
    assert (flowers.ceiling, flowers.baseline) == (0.95, 0.55)
    assert flowers.method == "vision_recipe_sweep_v2 (13 of 13 recipes on mps, best: x)"


def test_a_tabular_ceiling_still_keeps_the_better_stored_record(
    tmp_path: Path, ceilings_run: list[tuple[str, bool]]
) -> None:
    treatments = "feature_treatment_sweep_v2 (best: calibrated, 8 treatments)"
    stored = _run(tmp_path, [_ceiling("heart", 0.92, 0.89, treatments, "[]")], ceilings_run)

    assert ceilings_run == [("flowers", False), ("heart", False)]
    assert (stored["heart"].ceiling, stored["heart"].method) == (0.92, treatments)


def test_a_vision_rerun_of_the_same_sweep_keeps_and_prints_the_better_stored_record(
    tmp_path: Path, ceilings_run: list[tuple[str, bool]], capsys: pytest.CaptureFixture[str]
) -> None:
    same = "vision_recipe_sweep_v2 (13 of 13 recipes on mps, best: z)"
    stored = _run(tmp_path, [_ceiling("flowers", 0.99, 0.60, same, '{"sweep": 2}')], ceilings_run)

    assert ceilings_run == [("flowers", False), ("heart", False)]
    assert (stored["flowers"].ceiling, stored["flowers"].method) == (0.99, same)
    out = capsys.readouterr().out
    assert "ceiling 0.9900, baseline 0.6000" in out
    assert f"stored {same}" in out
