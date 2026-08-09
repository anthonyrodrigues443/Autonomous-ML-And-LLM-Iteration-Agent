"""`PromptTarget` — the second problem type, on the same machine as the first.

A prompt eval set IS tabular: input columns plus a column holding the right answer.
So this target reuses `TabularDataset`, the same deterministic split, the same
sealed holdout, the same `core.scoring` ruler and the same predictions contract that
`ModelTarget` uses. The supervised loop, the Supervisor, the Researcher, the Critic
and the Summarizer all work unchanged, because none of them ever knew what a target
was made of.

**The only difference is what happens per record.** Tabular fits a model once and
predicts many rows. This makes one LLM call per record, with the record substituted
into the prompt, and stores the structured reply as that row's prediction.

Two consequences worth naming:

*The sealed holdout is already safe, structurally.* `codegen.build_inputs` writes
train.csv WITH answers and holdout.csv WITHOUT them, so holdout answers never enter
the session. The prompt-path version of fitting on the test set — mining the answer
key for few-shot examples — is therefore not something the agent is prevented from
doing, it is something it cannot see how to do.

*The safety net must not be an LLM pass.* v0.4 learned that a fallback slower than
the thing it catches produces nothing exactly when it is needed, which is why the
tabular floor became logistic regression. Here the floor is the majority answer,
computed instantly from the training column with no model involved at all.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from iterate.adapters.compute.base import CodeJob
from iterate.core import codegen
from iterate.core.prompt_runtime import UNPARSEABLE, AskStats, make_ask
from iterate.core.prompting import Prompt, baseline_prompt
from iterate.core.scoring import score
from iterate.schemas.experiment import ExperimentResult, Metrics

if TYPE_CHECKING:
    from pathlib import Path

    from iterate.adapters.compute.runner import RunResult
    from iterate.adapters.data.tabular import TabularDataset
    from iterate.schemas.experiment import Candidate

# Above this many distinct answers the target is treated as free text rather than a
# closed label set, so the answer tool stops constraining and parsing takes over.
_MAX_LABELS = 50
_OUTPUT_TAIL_CHARS = 2000


def label_set(dataset: TabularDataset) -> list[str] | None:
    """The allowed answers, taken from the TRAINING column only.

    Reading the holdout column to build the enum would leak the answer key's shape
    into every call, including labels that appear nowhere in training. A label the
    training data never shows is one the agent has no way to know about, and the
    model should not be handed it either.
    """
    values = sorted({str(v) for v in dataset.train_target.dropna().unique()})
    return values if 0 < len(values) <= _MAX_LABELS else None


def majority_answer(dataset: TabularDataset) -> str:
    """The most common training answer. The safety net, and it cannot time out."""
    counts = dataset.train_target.astype(str).value_counts()
    return str(counts.index[0]) if len(counts) else ""


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
        self._labels = label_set(dataset)
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
            "target_backend": self._target_backend,
            "target_model": self._target_model,
            "target_base_url": self._target_base_url,
            "cache_path": str(self._cache_path) if self._cache_path else None,
            "max_workers": self._max_workers,
            "baseline_prompt": self._baseline_prompt.as_dict(),
        }
        return json.dumps(payload).encode()

    def _evaluate(self, prompt: Prompt, *, experiment_id: str) -> ExperimentResult:
        """Run one prompt over the sealed holdout, host-side.

        The direct path, used for the baseline and for a typed prompt candidate. The
        code path goes through the session instead, but lands on the same scorer.
        """
        ask = make_ask(
            columns=list(self._dataset.features),
            labels=self._labels,
            backend=self._target_backend,
            model=self._target_model,
            base_url=self._target_base_url,
            cache_path=self._cache_path,
            max_workers=self._max_workers,
        )
        rows = self._dataset.test_features.to_dict(orient="records")
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

        values = score(
            "classification",
            self._dataset.test_target.astype(str),
            answers,
            average=self._average,
            include=(self._metric,),
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
