"""The Critic specialist — can this experiment's score be believed?

Runs once per finished experiment, after scoring and before the result is allowed
to bank as the run's best. Two questions, with deliberately different consequences:

* **leak** — a verifiable defect in the submitted code (a transform fitted on the
  holdout, the target used to build a feature). This VETOES banking, because a
  number produced by cheating is not a score.
* **mirage** — a statistical suspicion about the gain, most often a holdout score
  far above the validation trail the session printed. This FLAGS only.

That asymmetry is the design. A leak is visible in the code and checkable, so
acting on it is safe. Whether a gain is "real" is a probabilistic judgement, and
the sealed holdout is already this project's ruler — letting a 12B overrule it
would put a model back into the control flow that direction, the guard stack and
duplicate-hashing were all deliberately kept out of. So the Critic can subtract a
win it can prove was cheated, and can raise a hand about one it merely doubts.

Precedence, unchanged from every other agent here: it can never unseal the holdout,
never overturn a deterministic guard, and never promote an experiment that did not
improve. It only ever takes away. Like the Summarizer and the Researcher it never
raises — a failed review accepts the experiment, because a review is a
nice-to-have and losing a real result to a flaky LLM call is not.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from iterate.core import dossier
from iterate.core.supervisor import submit_path_code
from iterate.prompts import PROMPTS
from iterate.schemas.llm import Message, ToolSpec

if TYPE_CHECKING:
    from iterate.llm.base import LLMClient
    from iterate.schemas.experiment import Experiment

log = logging.getLogger(__name__)

_PROMPTS = PROMPTS["critic"]
_CODE_CAP = 6000
_REASON_CAP = 240
# The stamp key. Joins `duplicate_submission` and `lever_unmeasured`, which the
# supervisor's technique table already excludes from crediting their techniques —
# so a vetoed experiment stops teaching a false lesson with no new prompt context.
REJECTED = "critic_rejected"
FLAGGED = "critic_flagged"


@dataclass(frozen=True)
class Verdict:
    """One review. ``leak`` vetoes banking; ``mirage`` only annotates."""

    leak: bool = False
    mirage: bool = False
    reason: str = ""

    @property
    def rejected(self) -> bool:
        """Whether this experiment must not bank as the run's best."""
        return self.leak

    @property
    def clean(self) -> bool:
        return not self.leak and not self.mirage

    def render(self) -> str:
        if self.leak:
            return f"leak: {self.reason}" if self.reason else "leak"
        if self.mirage:
            return f"suspicious gain: {self.reason}" if self.reason else "suspicious gain"
        return ""


def _build_tool() -> ToolSpec:
    spec = _PROMPTS["tool"]
    fields = spec["fields"]
    return ToolSpec(
        name=spec["name"],
        description=spec["description"],
        parameters={
            "type": "object",
            "properties": {
                "leak": {"type": "boolean", "description": fields["leak"]},
                "mirage": {"type": "boolean", "description": fields["mirage"]},
                "reason": {"type": "string", "description": fields["reason"]},
            },
            "required": ["leak", "mirage"],
        },
    )


REVIEW_EXPERIMENT = _build_tool()


def _coerce_bool(value: Any) -> bool:
    """Weak models emit booleans as the STRINGS "true"/"false", and bool("false")
    is True. The supervisor hit this live on a groq backend; the same coercion
    applies here because the same models drive both."""
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    return bool(value)


class Critic:
    """Reviews a finished experiment for leakage and for luck."""

    def __init__(
        self,
        client: LLMClient,
        *,
        metric: str,
        direction: str,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> None:
        self._client = client
        self._metric = metric
        self._direction = direction
        self._temperature = temperature
        self._max_tokens = max_tokens

    def review(self, experiment: Experiment, *, previous_best: float | None = None) -> Verdict:
        """Review one finished experiment. Never raises: a failed review returns a
        clean verdict, because a flaky call must not cost a real result."""
        result = experiment.result
        if result is None or not result.succeeded or result.metrics is None:
            # A failed experiment has no score to disbelieve.
            return Verdict()
        code = experiment.candidate.changes.get("code")
        if not isinstance(code, str) or not code.strip():
            return Verdict()

        record = dossier.build(experiment)
        try:
            args = self._call(
                score=f"{result.metrics.primary_value:.4f}",
                previous_best=("none" if previous_best is None else f"{previous_best:.4f}"),
                val_trail=(" -> ".join(f"{v:.4f}" for v in record.val_trail) or "none printed"),
                code=_tail(submit_path_code(code), _CODE_CAP),
            )
        except Exception as exc:
            log.info("critic: review failed (%s: %s)", type(exc).__name__, exc)
            return Verdict()
        if args is None:
            return Verdict()

        leak = _coerce_bool(args.get("leak"))
        mirage = _coerce_bool(args.get("mirage"))
        reason = str(args.get("reason") or "").strip()[:_REASON_CAP]
        if leak or mirage:
            log.info("critic: %s (%s)", "LEAK" if leak else "suspicious gain", reason or "no reason given")
        return Verdict(leak=leak, mirage=mirage, reason=reason)

    def _call(self, **fields: str) -> dict[str, Any] | None:
        messages = [
            Message(
                role="system",
                content=_PROMPTS["system"].format(
                    metric=self._metric, direction=self._direction
                ),
            ),
            Message(
                role="user",
                # Concatenated field-by-field rather than one .format over the code:
                # generated code contains braces and would break str.format.
                content=_PROMPTS["user_template"].replace("{metric}", self._metric).replace(
                    "{score}", fields["score"]
                ).replace("{previous_best}", fields["previous_best"]).replace(
                    "{val_trail}", fields["val_trail"]
                ).replace("{code}", fields["code"]),
            ),
        ]
        for attempt in range(2):
            response = self._client.chat(
                messages,
                tools=[REVIEW_EXPERIMENT],
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
            call = next(
                (c for c in response.tool_calls if c.name == REVIEW_EXPERIMENT.name), None
            )
            if call is not None:
                return dict(call.arguments)
            if attempt == 0:
                messages = [*messages, Message(role="user", content=_PROMPTS["retry_nudge"])]
        return None


def _tail(text: str, limit: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else "...(truncated)\n" + text[-limit:]


def stamp(experiment: Experiment, verdict: Verdict) -> None:
    """Record the verdict on the candidate, using the same `changes` convention as
    `duplicate_submission` and `lever_unmeasured`.

    Deliberately not a new field on the supervisor's prompt: the technique table
    already skips stamped experiments when crediting techniques, so a vetoed
    experiment stops teaching its lesson without a single extra line of context.
    """
    if verdict.leak:
        experiment.candidate.changes[REJECTED] = verdict.render()
    elif verdict.mirage:
        experiment.candidate.changes[FLAGGED] = verdict.render()


def was_rejected(experiment: Experiment) -> bool:
    return bool(experiment.candidate.changes.get(REJECTED))


__all__ = ["FLAGGED", "REJECTED", "Critic", "Verdict", "stamp", "was_rejected"]
