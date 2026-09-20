"""Measuring a vision ceiling: the best a fixed ladder of recipes reaches, no LLM.

Same two rules as the model and prompt sweeps. The ceiling is measured through the
product's own machinery: the loader the CLI calls, the `prepare_images` step, one
`DLModelTarget`, the `core.scoring` ruler. And it is a LOWER BOUND, labelled as one:
the first row is the target's own `baseline()` and every other row is a `Recipe` the
agent could have submitted, so an agent going past it is a real result and nothing
clamps.

Each recipe runs once, and runs on MPS are not bitwise repeatable, so for accuracy the
detail records the holdout's standard error, and two rows within it are a tie; a number
label records why it has none.
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
from iterate.core.scoring import direction as metric_direction
from iterate.core.scoring import task_for_metric
from iterate.schemas.experiment import Candidate
from iterate.targets import layers as arch
from iterate.targets.dl import BASELINE, BASELINE_SIZE, RECIPE_JSON, SCRATCH, DLModelTarget, Recipe

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from evals.corpus import Dataset
    from iterate.targets.dl import Runner

# Bumped when the ladder changes, so a stored ceiling says which sweep produced it.
METHOD = "vision_recipe_sweep_v2"
_METHOD_STEM = "vision_recipe_sweep_v"
# Rows can come in groups the sweep cannot see, so a per-row formula would understate the error.
NO_ERROR_BAR = "none: holdout rows from one group move together and the sweep cannot see groups"
_CARRIED = ", ceiling carried from "
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
    pearson: float | None = None
    spearman: float | None = None


def larger_size(base: int) -> int:
    return min(_LARGEST, base * 2)


def recipes_for(base: int, task: str = "classification") -> list[Recipe]:
    """The ladder, grouped by size so each size decodes once. The first is the fixed
    recipe `baseline()` runs; a fine-tune with nothing else set is the Day 1 bench. A
    number has no label smoothing, so its ladder leaves that rung out."""
    bench = replace(_BENCH, image_size=base)
    large = larger_size(base)
    rungs = [
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
    if task == "regression":
        rungs = [r for r in rungs if r.label_smoothing == 0]
    return [replace(BASELINE, image_size=min(BASELINE_SIZE, base)), *rungs]


def label(recipe: Recipe) -> str:
    """What differs from the bench, or for a model trained from zero, from the baseline."""
    if recipe.unfreeze == "none":
        return f"probe {recipe.backbone} {recipe.image_size}px"
    scratch = recipe.backbone in SCRATCH
    kind = (
        "from zero"
        if scratch
        else {"head": "head", "last_block": "last-block"}.get(recipe.unfreeze, "fine-tune")
    )
    reference = BASELINE if scratch else _BENCH
    parts = [f"{kind} {recipe.backbone} {recipe.image_size}px {recipe.epochs}ep"]
    for name in ("optimizer", "lr", "schedule", "augment", "label_smoothing", "head_init"):
        if (value := getattr(recipe, name)) != getattr(reference, name):
            parts.append(f"{name}={value}")
    if recipe.drop_stages:
        parts.append(f"drop_stages={recipe.drop_stages}")
    for name in ("layers", "head"):
        if (spec := getattr(recipe, name)) is not None:
            parts.append(f"{name}={arch.text(spec)}")
    return " ".join(parts)


def standard_error(metric: str, value: float, n: int) -> tuple[float | None, str]:
    """The holdout's standard error where one exists, or why a number label gets none."""
    if not n:
        return None, ""
    if metric == "accuracy" and 0.0 <= value <= 1.0:
        return math.sqrt(value * (1 - value) / n), ""
    if task_for_metric(metric) == "regression":
        return None, NO_ERROR_BAR
    return None, ""


def carry_stored(stored: Ceiling | None, new: Ceiling) -> tuple[Ceiling, bool]:
    """The record to write, and whether it replaces the stored one. An older vision sweep
    is replaced, with its ceiling carried when it was the better one; everything else,
    tabular sweeps above all, keeps the store's keep-the-better rule."""
    old = _version(stored.method) if stored is not None else None
    ours = _version(new.method)
    if stored is None or old is None or ours is None or old >= ours:
        return new, False
    if not _better(stored.ceiling, new.ceiling, new.direction):
        return new, True
    _, was_carried, source = stored.method.rpartition(_CARRIED)
    origin = source if was_carried else stored.method.split()[0]
    return replace(new, ceiling=stored.ceiling, method=f"{new.method}{_CARRIED}{origin}"), True


def _version(method: str) -> int | None:
    token = next(iter(method.split()), "")
    number = token.removeprefix(_METHOD_STEM)
    return int(number) if token.startswith(_METHOD_STEM) and number.isdigit() else None


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
    row is measured through `baseline()`, fixed and uncapped, and is the baseline."""
    task = task_for_metric(dataset.metric)
    loaded = corpus.load_data(dataset, task=task)
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
    for index, recipe in enumerate(recipes_for(prepared.image_size, task)):
        started = time.monotonic()
        ran: dict[str, Any] = {}
        values: dict[str, float] = {}
        try:
            outcome = (
                target.baseline()
                if index == 0
                else target.run(
                    Candidate(description=label(recipe), changes=asdict(recipe), rationale="sweep")
                )
            )
            value = outcome.metrics.primary_value if outcome.metrics is not None else None
            values = outcome.metrics.values if outcome.metrics is not None else {}
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
            pearson=values.get("pearson"),
            spearman=values.get("spearman"),
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
    error_bar, assumes = standard_error(dataset.metric, best.score, n)
    correlations = task == "regression"
    device = target.device
    detail = json.dumps(
        {
            "device": device,
            "image_version": prepared.dataset.data_hash,
            "image_size": prepared.image_size,
            "holdout": n,
            "left_out": len(prepared.dropped),
            "standard_error": error_bar,
            "standard_error_note": assumes,
            "recipes": [
                {
                    "recipe": r.label,
                    "score": r.score,
                    **({"pearson": r.pearson, "spearman": r.spearman} if correlations else {}),
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
    found: str | None = None
    with subprocess.Popen(
        [sys.executable, "-m", "evals.vision_ceilings", dataset.name],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
    ) as child:
        assert child.stdout is not None
        try:
            for line in child.stdout:
                if line.startswith(_RESULT):
                    found = line[len(_RESULT) :]
                elif on_line is not None:
                    on_line(line.rstrip("\n"))
        except BaseException:
            child.kill()
            child.wait()
            raise
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
    "NO_ERROR_BAR",
    "RecipeResult",
    "carry_stored",
    "label",
    "larger_size",
    "main",
    "recipes_for",
    "standard_error",
    "sweep",
    "sweep_in_child",
]


if __name__ == "__main__":
    raise SystemExit(main())
