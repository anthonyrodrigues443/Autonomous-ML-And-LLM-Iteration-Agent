"""Measuring a vision ceiling: the best a fixed ladder of transfer-learning recipes
reaches, no LLM.

Same two rules as the model and prompt sweeps. The ceiling is measured through the
product's own machinery: the `load_csv` call the CLI makes, the `prepare_images`
step, one `DLModelTarget`, the `core.scoring` ruler. And it is a LOWER BOUND,
labelled as one: every row is a `Recipe` the agent could have submitted, so an agent
going past it is a real result and nothing clamps.

Each recipe runs once, and runs on MPS are not bitwise repeatable, so the detail
records the holdout's standard error: two rows within it are a tie.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from evals import corpus
from evals.config import REPO_ROOT
from evals.store import Ceiling
from iterate.adapters.data.images import prepare_images
from iterate.adapters.data.tabular import load_csv
from iterate.core.scoring import direction as metric_direction
from iterate.core.scoring import task_for_metric
from iterate.schemas.experiment import Candidate
from iterate.targets.dl import RECIPE_JSON, DLModelTarget, Recipe

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from evals.corpus import Dataset
    from iterate.targets.dl import Runner

# Bumped when the ladder changes, so a stored ceiling says which sweep produced it.
METHOD = "vision_recipe_sweep_v1"
_LARGEST = 224
_RESULT = "ceiling-json: "
_BENCH = Recipe(unfreeze="all", epochs=3)


@dataclass(frozen=True)
class RecipeResult:
    """One recipe in the sweep."""

    label: str
    score: float | None
    seconds: float
    error: str = ""
    epochs_planned: int = 0
    epochs_run: int = 0


def larger_size(base: int) -> int:
    return min(_LARGEST, base * 2)


def recipes_for(base: int) -> list[Recipe]:
    """The ladder, grouped by size so each size decodes once. The first is the probe
    `baseline()` measures; a fine-tune with nothing else set is the Day 1 bench."""
    bench = replace(_BENCH, image_size=base)
    large = larger_size(base)
    return [
        Recipe(image_size=base),
        Recipe(backbone="convnext_tiny", image_size=base),
        bench,
        replace(bench, epochs=5),
        replace(bench, unfreeze="head", lr=1e-2),
        replace(bench, head_init="probe", optimizer="sgd", lr=3e-3),
        replace(bench, label_smoothing=0.1),
        replace(bench, augment="flip_crop"),
        replace(bench, backbone="resnet50"),
        replace(bench, backbone="convnext_tiny"),
        Recipe(image_size=large),
        replace(bench, image_size=large),
    ]


def label(recipe: Recipe) -> str:
    if recipe.unfreeze == "none":
        return f"probe {recipe.backbone} {recipe.image_size}px"
    kind = "head" if recipe.unfreeze == "head" else "fine-tune"
    parts = [f"{kind} {recipe.backbone} {recipe.image_size}px {recipe.epochs}ep"]
    for name in ("optimizer", "lr", "schedule", "augment", "label_smoothing", "head_init"):
        if (value := getattr(recipe, name)) != getattr(_BENCH, name):
            parts.append(f"{name}={value}")
    return " ".join(parts)


def _better(candidate: float, incumbent: float, direction: str) -> bool:
    return candidate > incumbent if direction == "maximize" else candidate < incumbent


def sweep(
    dataset: Dataset,
    *,
    runner: Runner | None = None,
    into: Path | None = None,
    on_progress: Callable[[RecipeResult], None] | None = None,
) -> tuple[Ceiling, list[RecipeResult]]:
    """Run the ladder on this dataset and return the best as its ceiling. The first
    row is the probe, the same number `baseline()` gives."""
    loaded = load_csv(dataset.path, target=dataset.target, task=task_for_metric(dataset.metric))
    prepared = prepare_images(loaded, dataset.path, into=into)
    target = DLModelTarget(
        prepared.dataset,
        column=prepared.column.column,
        metric=dataset.metric,
        name=dataset.name,
        image_size=prepared.image_size,
        profile=prepared.profile,
        runner=runner,
    )
    direction = metric_direction(dataset.metric)

    results: list[RecipeResult] = []
    best: RecipeResult | None = None
    for recipe in recipes_for(prepared.image_size):
        started = time.monotonic()
        ran: dict[str, Any] = {}
        try:
            outcome = target.run(
                Candidate(description=label(recipe), changes=asdict(recipe), rationale="sweep")
            )
            value = outcome.metrics.primary_value if outcome.metrics is not None else None
            error = outcome.error or ""
            ran = json.loads(outcome.artifacts.get(RECIPE_JSON, "{}"))
        except Exception as exc:
            value, error = None, f"{type(exc).__name__}: {exc}"
        result = RecipeResult(
            label=label(recipe),
            score=value,
            seconds=time.monotonic() - started,
            error=error,
            epochs_planned=int(ran.get("epochs_planned", 0)),
            epochs_run=int(ran.get("epochs_run", 0)),
        )
        results.append(result)
        if on_progress:
            on_progress(result)
        if value is not None and (
            best is None or best.score is None or _better(value, best.score, direction)
        ):
            best = result

    if best is None or best.score is None:
        raise RuntimeError(f"{dataset.name}: no recipe in the sweep produced a score")

    n = prepared.dataset.n_test
    standard_error = (
        math.sqrt(best.score * (1 - best.score) / n)
        if dataset.metric == "accuracy" and 0.0 <= best.score <= 1.0 and n
        else None
    )
    device = target.device
    detail = json.dumps(
        {
            "device": device,
            "image_version": prepared.dataset.data_hash,
            "image_size": prepared.image_size,
            "holdout": n,
            "left_out": len(prepared.dropped),
            "standard_error": standard_error,
            "recipes": [
                {
                    "recipe": r.label,
                    "score": r.score,
                    "seconds": round(r.seconds, 1),
                    "epochs_planned": r.epochs_planned,
                    "epochs_run": r.epochs_run,
                    "error": r.error,
                }
                for r in results
            ],
        }
    )
    scored = sum(1 for r in results if r.score is not None)
    ceiling = Ceiling(
        dataset=dataset.name,
        dataset_hash=dataset.content_hash(),
        metric=dataset.metric,
        ceiling=best.score,
        direction=direction,
        baseline=results[0].score,
        method=f"{METHOD} ({scored} of {len(results)} recipes on {device}, best: {best.label})",
        measured_at=datetime.now(UTC).isoformat(),
        detail=detail,
    )
    return ceiling, results


def sweep_in_child(dataset: Dataset, *, on_line: Callable[[str], None] | None = None) -> Ceiling:
    """The sweep in a child process of its own. torch and lightgbm each ship an OpenMP
    runtime, and on macOS a fit in one after the other in a single process crashes or
    hangs; a ceilings run over the whole corpus fits lightgbm for its tabular rows."""
    child = subprocess.Popen(
        [sys.executable, "-m", "evals.vision_ceilings", dataset.name],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert child.stdout is not None
    found: str | None = None
    for line in child.stdout:
        if line.startswith(_RESULT):
            found = line[len(_RESULT) :]
        elif on_line is not None:
            on_line(line.rstrip("\n"))
    code = child.wait()
    if code != 0 or found is None:
        raise RuntimeError(f"{dataset.name}: the vision sweep's child exited with {code}")
    return Ceiling(**json.loads(found))


def _show(result: RecipeResult) -> None:
    value = f"{result.score:.4f}" if result.score is not None else f"skip ({result.error[:60]})"
    print(f"    {result.label:<44} {value}  {result.seconds:.0f}s", flush=True)


def main(argv: list[str] | None = None) -> int:
    names = argv if argv is not None else sys.argv[1:]
    if len(names) != 1:
        print("usage: python -m evals.vision_ceilings <dataset>", file=sys.stderr)
        return 2
    (dataset,) = corpus.select(names)
    ceiling, _ = sweep(dataset, on_progress=_show)
    print(_RESULT + json.dumps(asdict(ceiling)), flush=True)
    return 0


__all__ = [
    "METHOD",
    "RecipeResult",
    "label",
    "larger_size",
    "main",
    "recipes_for",
    "sweep",
    "sweep_in_child",
]


if __name__ == "__main__":
    raise SystemExit(main())
