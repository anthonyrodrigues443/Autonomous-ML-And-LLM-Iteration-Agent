"""`PromptTarget`: the second problem type, on the same machine as the first.

A prompt eval set is tabular, so this reuses `TabularDataset`, the same split, the
same sealed holdout and the same scoring. The only difference is one LLM call per
record. Holdout answers never enter the session, and the floor is the majority
training answer, computed with no model involved.
"""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING, Any

import pandas as pd

from iterate.adapters.compute.base import CodeJob
from iterate.core import codegen
from iterate.core.prompt_runtime import UNPARSEABLE, AskStats, make_ask
from iterate.core.prompting import Prompt, baseline_prompt
from iterate.core.scoring import score, task_for_metric
from iterate.schemas.experiment import ExperimentResult, Metrics

if TYPE_CHECKING:
    from pathlib import Path

    from iterate.adapters.compute.runner import RunResult
    from iterate.adapters.data.tabular import TabularDataset
    from iterate.schemas.experiment import Candidate

log = logging.getLogger(__name__)

# Above this many distinct answers the target is treated as free text rather than a
# closed label set, so the answer tool stops constraining and parsing takes over.
# The enum the answer tool can carry before a prompt is mostly a list of options.
_MAX_LABELS = 50
# Above this share of rows carrying their own distinct answer, the column is one-off
# text rather than a set of categories — 48 unique summaries in 60 rows is not a
# 48-class problem, and a count alone cannot tell the two apart.
_MOSTLY_UNIQUE = 0.5
_OUTPUT_TAIL_CHARS = 2000


class UnscorableTargetError(Exception):
    """The answer column is something this target has no honest way to score."""


def target_kind(dataset: TabularDataset) -> str:
    """`closed_set`, `numeric`, or `free_text`: what the answer column IS.

    Not "classification or regression", which the metric settles. Free text needs
    both signals, more distinct answers than the enum can carry AND nearly every
    row carrying its own, because a count alone cannot separate 48 intents from
    48 one-off sentences.
    """
    from iterate.adapters.data.tabular import looks_like_classification

    target = dataset.train_target.dropna()
    distinct = int(target.nunique())
    rows = max(1, len(target))

    if pd.api.types.is_numeric_dtype(target) and not looks_like_classification(target):
        return "numeric"
    # The RATIO decides this, not the count. Conflating the two refused full
    # CLINC150 — 150 intents across 2400 rows, ratio 0.06 — as free text, which is
    # plainly wrong: it is a closed set, just a big one. Whether those answers FIT
    # in an enum is a separate question, answered by `label_set`.
    if distinct / rows <= _MOSTLY_UNIQUE:
        return "closed_set"
    return "numeric" if pd.api.types.is_numeric_dtype(target) else "free_text"


def label_set(dataset: TabularDataset) -> list[str] | None:
    """The allowed answers, taken from the TRAINING column only.

    Reading the holdout column to build the enum would leak the answer key's shape
    into every call, including labels that appear nowhere in training. A label the
    training data never shows is one the agent has no way to know about, and the
    model should not be handed it either.

    None when the target is not a closed set — the answer tool then stops
    constraining and coercion does the work instead.
    """
    if target_kind(dataset) != "closed_set":
        return None
    values = sorted({str(v) for v in dataset.train_target.dropna().unique()})
    if len(values) > _MAX_LABELS:
        # A real closed set, too large to put in every call. Dropping the enum is
        # the right call — the prompt would otherwise be mostly a list of options —
        # but it removes the guarantee that an answer is IN the set, so coercion
        # takes over and the user is told rather than left to find out.
        log.warning(
            "%d distinct answers is more than the %d-item answer tool can carry, so "
            "replies are matched by text instead of constrained. Expect more "
            "unparseable answers; narrowing the label set usually scores better.",
            len(values),
            _MAX_LABELS,
        )
        return None
    return values


def majority_answer(dataset: TabularDataset) -> str:
    """The most common training answer. The safety net, and it cannot time out."""
    counts = dataset.train_target.astype(str).value_counts()
    return str(counts.index[0]) if len(counts) else ""


def numeric_range(dataset: TabularDataset) -> tuple[float | None, float | None]:
    """The range the TRAINING answers actually span."""
    values = pd.to_numeric(dataset.train_target, errors="coerce").dropna()
    return (float(values.min()), float(values.max())) if len(values) else (None, None)


def median_answer(dataset: TabularDataset) -> float:
    """The floor for a numeric target, and what an unusable answer becomes.

    Measured before choosing, the same way the label case was. Of four options on a
    300-row rating task with 10% refusals: DROPPING the unparseable rows scored
    BETTER than answering them honestly (rmse 0.691 against 0.701, pearson 0.894
    against 0.890) — so refusing the hard rows would be a winning strategy.
    Substituting the median costs 0.10 rmse against an honest answer, and unlike a
    worst-case substitution it cannot let a handful of refusals swamp the score.
    """
    values = pd.to_numeric(dataset.train_target, errors="coerce").dropna()
    return float(values.median()) if len(values) else 0.0


class PromptTarget:
    """An LLM prompt scored against a labelled eval set on a sealed holdout."""

    def __init__(
        self,
        dataset: TabularDataset,
        *,
        metric: str,
        task: str,
        target_backend: str,
        target_model: str,
        target_base_url: str | None = None,
        average: str | None = None,
        name: str = "prompt",
        cache_path: Path | str | None = None,
        max_workers: int = 8,
        starting_prompt: Prompt | None = None,
    ) -> None:
        self.name = name
        self._dataset = dataset
        self._metric = metric
        self._average = average
        self._task = task
        self._task_kind = task_for_metric(metric)
        self._labels = label_set(dataset) if self._task_kind == "classification" else None
        # Bounds come from the TRAINING answers, so the tool cannot express a rating
        # outside the range the data actually uses — the numeric counterpart of the
        # label enum. Holdout values are never consulted; a range widened by the
        # answer key would be a leak.
        self._numeric_range = numeric_range(dataset) if self._task_kind == "regression" else None
        self._target_backend = target_backend
        self._target_model = target_model
        self._target_base_url = target_base_url
        self._cache_path = cache_path
        self._max_workers = max_workers
        # The user's production prompt when they have one, otherwise the minimal
        # honest prompt built from their one-line task. Either way it is measured
        # rather than assumed: `baseline()` runs it like any other candidate.
        self._baseline_prompt = starting_prompt or baseline_prompt(task, self._labels)

    @property
    def baseline_prompt(self) -> Prompt:
        return self._baseline_prompt

    @property
    def labels(self) -> list[str] | None:
        return self._labels

    @property
    def model_under_test(self) -> str:
        return self._target_model

    # ─── BenchmarkTarget ───────────────────────────────────────────────────
    def baseline(self) -> ExperimentResult:
        return self._evaluate(self._baseline_prompt, experiment_id="baseline")

    def run(self, candidate: Candidate) -> ExperimentResult:
        prompt = Prompt.from_dict(candidate.changes.get("prompt") or {})
        return self._evaluate(prompt, experiment_id=candidate.id)

    # ─── SupportsCodeGen ───────────────────────────────────────────────────
    def build_code_job(self, candidate: Candidate) -> CodeJob:
        code = str(candidate.changes["code"])
        inputs = codegen.build_inputs(self._dataset)
        inputs[codegen.META_JSON] = self.meta_json()
        return CodeJob(
            script=codegen.assemble_script(code),
            inputs=inputs,
            outputs=[codegen.PREDICTIONS_CSV, codegen.PROMPT_JSON],
            packages=codegen.required_imports(code),
        )

    def score_code_job(self, run_result: RunResult, experiment_id: str) -> ExperimentResult:
        stdout_tail = _tail(run_result.stdout) or None
        if not run_result.succeeded:
            reason = "timed out" if run_result.timed_out else "script failed"
            detail = _tail(run_result.stderr) or "(no stderr)"
            return ExperimentResult(
                experiment_id=experiment_id, error=f"code {reason}:\n{detail}", logs=stdout_tail
            )
        result = codegen.score_predictions(
            self._dataset,
            run_result.outputs.get(codegen.PREDICTIONS_CSV),
            metric=self._metric,
            experiment_id=experiment_id,
            open_vocabulary=True,
        )
        artifacts = dict(result.artifacts)
        if (submitted := run_result.outputs.get(codegen.PROMPT_JSON)) is not None:
            artifacts[codegen.PROMPT_JSON] = submitted.decode(errors="replace")
        return result.model_copy(update={"logs": stdout_tail, "artifacts": artifacts})

    # ─── session wiring ────────────────────────────────────────────────────
    def session_preamble(self) -> str:
        """The prompt path's opening cell: the data, plus `ask`."""
        return codegen.prompt_session_preamble()

    def meta_json(self) -> bytes:
        """Everything the in-kernel `ask` needs to reach the model under test.

        The api key is deliberately absent — it travels in the environment, so it
        never lands on disk beside data the generated code reads.
        """
        payload = {
            "target": self._dataset.target,
            "task": self._task,
            "features": list(self._dataset.features),
            "labels": self._labels,
            "metric": self._metric,
            "family": "prompt",
            # The session must ask and score the SAME way the host does. Without
            # these the in-kernel helpers fell back to the label tool and the
            # classification scorer, so on a regression run the agent iterated
            # against a different ruler than the one deciding its score.
            "task_kind": self._task_kind,
            "numeric_range": list(self._numeric_range) if self._numeric_range else None,
            "median": median_answer(self._dataset) if self._task_kind == "regression" else None,
            "target_backend": self._target_backend,
            "target_model": self._target_model,
            "target_base_url": self._target_base_url,
            "cache_path": str(self._cache_path) if self._cache_path else None,
            "max_workers": self._max_workers,
            "baseline_prompt": self._baseline_prompt.as_dict(),
        }
        return json.dumps(payload).encode()

    _warned_about_parallelism = False

    def _warn_if_serialised(self) -> None:
        """Say so, once, when the concurrency is probably not real.

        Measured on a default local Ollama: 8 workers gave a 1.10x speedup — 3.44s
        per call sequential against 3.14s at 8-wide. The server answers one request
        at a time unless OLLAMA_NUM_PARALLEL is set, so the worker pool is only a
        queue and a 300-record pass costs 300 sequential calls whatever the pool
        size. Not a bug in iterate, but it is the difference between a four-minute
        pass and a thirty-second one, and a user cannot fix what they cannot see.
        """
        if PromptTarget._warned_about_parallelism or self._target_backend != "ollama":
            return
        if os.environ.get("OLLAMA_NUM_PARALLEL"):
            return
        PromptTarget._warned_about_parallelism = True
        log.info(
            "OLLAMA_NUM_PARALLEL is not set, so the server answers one request at a "
            "time and the %d concurrent workers are only a queue. Restarting ollama "
            "with OLLAMA_NUM_PARALLEL=%d makes a pass roughly that many times "
            "faster (measured without it: 1.10x from 8 workers).",
            self._max_workers,
            self._max_workers,
        )

    def _evaluate(self, prompt: Prompt, *, experiment_id: str) -> ExperimentResult:
        """Run one prompt over the sealed holdout, host-side.

        The direct path, used for the baseline and for a typed prompt candidate. The
        code path goes through the session instead, but lands on the same scorer.
        """
        ask = make_ask(
            columns=list(self._dataset.features),
            labels=self._labels,
            numeric_range=self._numeric_range,
            backend=self._target_backend,
            model=self._target_model,
            base_url=self._target_base_url,
            cache_path=self._cache_path,
            max_workers=self._max_workers,
        )
        rows = self._dataset.test_features.to_dict(orient="records")
        log.info(
            "%s: running the prompt over %d holdout records (one model call each, %d at a time)",
            experiment_id,
            len(rows),
            self._max_workers,
        )
        self._warn_if_serialised()
        stats = AskStats()
        try:
            answers = ask(prompt, rows, stats=stats)
        except Exception as exc:
            return ExperimentResult(
                experiment_id=experiment_id, error=f"prompt run failed: {type(exc).__name__}: {exc}"
            )

        if all(answer == UNPARSEABLE for answer in answers) and answers:
            # Every row unusable is not a score of zero, it is a broken run: the
            # endpoint is down, the model is missing, or the prompt provokes nothing.
            # Reporting 0.0 would bank a number and teach the next iteration a lie.
            detail = stats.errors[0] if stats.errors else "no usable answers"
            return ExperimentResult(
                experiment_id=experiment_id,
                error=f"every record came back unusable ({detail})",
                logs=stats.summary(),
            )

        if self._task_kind == "regression":
            # An unusable answer becomes the training median. Measured across four
            # options: DROPPING the unparseable rows scores BETTER than answering
            # them honestly, so refusing the hard ones would be a winning strategy.
            # The median penalises without letting a few refusals swamp the score.
            floor = median_answer(self._dataset)
            numeric = [floor if a == UNPARSEABLE else float(a) for a in answers]
            values = score(
                "regression",
                pd.to_numeric(self._dataset.test_target, errors="coerce").to_numpy(),
                numeric,
                include=(self._metric,),
            )
        else:
            values = score(
                "classification",
                self._dataset.test_target.astype(str),
                answers,
                average=self._average,
                include=(self._metric,),
                # A model that will not answer usably yields a sentinel, which is a
                # value the target column never contains. It has to count as wrong
                # rather than register as a new class — untagged, ONE such answer in
                # 300 turns a binary target multiclass and the metric refuses to
                # score at all. Measured on the first live run.
                open_vocabulary=True,
            )
        return ExperimentResult(
            experiment_id=experiment_id,
            metrics=Metrics(
                values=values,
                primary=self._metric,
                direction=_direction(self._metric),
                n_samples=len(answers),
            ),
            logs=stats.summary(),
        )


def _direction(metric: str) -> Any:
    from iterate.core.scoring import direction

    return direction(metric)


def _tail(text: str | None) -> str:
    if not text:
        return ""
    return text[-_OUTPUT_TAIL_CHARS:]


__all__ = ["PromptTarget", "label_set", "majority_answer"]
