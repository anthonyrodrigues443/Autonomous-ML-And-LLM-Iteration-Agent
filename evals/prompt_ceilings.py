"""Measuring a prompt ceiling: the best a fixed list of standard techniques reaches.

The tabular ceiling sweeps model families with no LLM deciding anything, so a flat
run can be read as "the agent missed" or "there was nothing to find". The prompt
path had no equivalent, and it showed immediately: a toxicity run went 0.8611 to
0.8824 and nobody could say whether that was most of the available gain or a
fraction of it. A CLINC run then scored 0.9890 against a 0.9890 baseline, which
looks like failure and is actually a dataset with one error in a hundred.

**The list is generic in form, dataset-specific only in what the harness already
knows.** Model families transfer between tabular datasets; prompt WORDING does not.
So the sweep applies standard prompt-engineering techniques — define the labels,
show examples, ask for reasoning, frame a role — built mechanically from the task
line, the label set and rows sampled from training data. No prompt here was
hand-written for a particular dataset, which is what keeps it a fair floor rather
than a hand-tuned target.

Same two rules as the tabular sweep. It is a LOWER BOUND, so an agent going past it
is a real result and nothing clamps. And every technique is measured on the same
records, so the comparison between them is paired.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from evals.store import Ceiling
from iterate.adapters.data.tabular import load_csv, with_smaller_holdout
from iterate.core.prompting import Prompt
from iterate.core.scoring import direction as metric_direction
from iterate.targets.prompt import PromptTarget, label_set

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from evals.corpus import Dataset

# Bumped when the technique list changes, so a stored ceiling says which sweep
# produced it and an old one can be spotted.
METHOD = "prompt_technique_sweep_v1"

# One model call per record per technique, so the whole sweep is
# len(TECHNIQUES) x RECORDS calls. Enough records to rank techniques against each
# other; the number that gets quoted is the agent's, measured on the full holdout.
DEFAULT_RECORDS = 200
_FEW_SHOT = 6


@dataclass(frozen=True)
class TechniqueResult:
    """One technique in the sweep."""

    name: str
    score: float | None
    seconds: float
    error: str = ""


def _answer_line(labels: Sequence[str] | None) -> str:
    if not labels:
        return ""
    return "\n\nAnswer with exactly one of: " + ", ".join(str(label) for label in labels)


def _examples(dataset: object, columns: Sequence[str], k: int) -> str:
    """k worked examples drawn from TRAINING rows.

    Training only, and stated plainly here because it is the one place a ceiling
    sweep could cheat: examples taken from the holdout would raise the bar using
    answers the agent is never allowed to see, and every capture fraction measured
    against it would be wrong.
    """
    from iterate.core.prompting import render_all

    train_x = dataset.train_features.head(k)  # type: ignore[attr-defined]
    train_y = dataset.train_target.head(k)  # type: ignore[attr-defined]
    lines = []
    for (_, row), answer in zip(train_x.iterrows(), train_y, strict=False):
        lines.append(f"Input:\n{render_all(row.to_dict(), columns)}\nAnswer: {answer}")
    return "\n\n".join(lines)


def techniques(task: str, dataset: object, columns: Sequence[str]) -> dict[str, Prompt]:
    """The fixed list, built from what the harness already knows.

    Deliberately the moves a competent person tries first, in the order they would
    try them — not a search over wordings. A ceiling that took a hundred variants
    would measure patience, not achievability.
    """
    labels = label_set(dataset)  # type: ignore[arg-type]
    answers = _answer_line(labels)
    base = task.strip()

    return {
        "minimal": Prompt(system=base + answers, user_template="{input}"),
        "define-the-labels": Prompt(
            system=base
            + answers
            + "\n\nBefore deciding, be precise about what each answer means and where "
            "the boundary between them falls. Borderline cases matter more than clear "
            "ones.",
            user_template="{input}",
        ),
        "few-shot": Prompt(
            system=base
            + answers
            + "\n\nWorked examples:\n\n"
            + _examples(dataset, columns, _FEW_SHOT),
            user_template="{input}",
        ),
        "reasoning": Prompt(
            system=base
            + answers
            + "\n\nWork out what the record is actually saying before you answer. "
            "Consider the most likely reading, not the most literal one.",
            user_template="{input}",
        ),
        "expert-role": Prompt(
            system="You are an experienced annotator who has labelled thousands of "
            "these and follows the guidelines exactly.\n\n" + base + answers,
            user_template="{input}",
        ),
        "define-plus-few-shot": Prompt(
            system=base
            + answers
            + "\n\nBe precise about where the boundary between answers falls; "
            "borderline cases matter more than clear ones.\n\nWorked examples:\n\n"
            + _examples(dataset, columns, _FEW_SHOT),
            user_template="{input}",
        ),
    }


def _better(candidate: float, incumbent: float, direction: str) -> bool:
    return candidate > incumbent if direction == "maximize" else candidate < incumbent


def sweep(
    dataset: Dataset,
    *,
    task: str,
    target_backend: str,
    target_model: str,
    records: int = DEFAULT_RECORDS,
    cache_path: str | None = None,
    on_progress: Callable[[TechniqueResult], None] | None = None,
) -> tuple[Ceiling, list[TechniqueResult]]:
    """Score every technique on the same records and return the best as the ceiling."""
    loaded = with_smaller_holdout(load_csv(dataset.path, target=dataset.target), records)
    direction = metric_direction(dataset.metric)
    columns = list(loaded.features)

    results: list[TechniqueResult] = []
    best: float | None = None
    best_name = ""

    for name, prompt in techniques(task, loaded, columns).items():
        started = time.monotonic()
        target = PromptTarget(
            loaded,
            metric=dataset.metric,
            task=task,
            target_backend=target_backend,
            target_model=target_model,
            cache_path=cache_path,
            starting_prompt=prompt,
        )
        try:
            outcome = target.baseline()
            value = outcome.metrics.primary_value if outcome.metrics is not None else None
            error = outcome.error or ""
        except Exception as exc:
            value, error = None, f"{type(exc).__name__}: {exc}"

        result = TechniqueResult(name, value, time.monotonic() - started, error)
        results.append(result)
        if on_progress:
            on_progress(result)
        if value is not None and (best is None or _better(value, best, direction)):
            best, best_name = value, name

    if best is None:
        raise RuntimeError(f"{dataset.name}: no technique in the sweep produced a score")

    baseline = next((r.score for r in results if r.name == "minimal"), None)

    return (
        Ceiling(
            dataset=dataset.name,
            dataset_hash=dataset.content_hash(),
            metric=dataset.metric,
            ceiling=best,
            direction=direction,
            baseline=baseline,
            method=f"{METHOD} (best: {best_name}, {records} records)",
            measured_at=datetime.now(UTC).isoformat(),
            detail=json.dumps(
                [
                    {"technique": r.name, "score": r.score, "seconds": round(r.seconds, 1),
                     "error": r.error}
                    for r in results
                ]
            ),
        ),
        results,
    )


__all__ = ["DEFAULT_RECORDS", "METHOD", "TechniqueResult", "sweep", "techniques"]
