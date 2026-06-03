"""The Reconstructor — read a user-supplied source (md / txt / notebook / .py) as
TEXT ONLY and WRITE the baseline approach as a runnable `train_and_predict` function.

Sibling of `Proposer` / `CodeProposer`: same LLM/tool-calling/retry machinery,
different intent and prompt. The Proposer picks the *next* experiment; the
Reconstructor reproduces the *user's existing* approach from their source so we can
run it through our own eval as a comparable baseline.

**Hard security boundary:** user-provided source is *never* executed — not by us,
not in the e2b sandbox, not ever. The LLM reads it as text and WRITES new code (the
agent's own, run on the code path like any other candidate) that reproduces the
approach. Because it emits code rather than an allow-listed spec, it can reproduce
the source faithfully — real CatBoost, a custom architecture — with no "closest
equivalent" approximation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from iterate.core.codegen import validate_train_and_predict
from iterate.prompts import PROMPTS
from iterate.schemas.experiment import Candidate
from iterate.schemas.llm import Message, ToolSpec

if TYPE_CHECKING:
    from iterate.llm.base import LLMClient


class ReconstructorError(RuntimeError):
    """The model failed to return a usable reconstruction after retries."""


_PROMPTS = PROMPTS["reconstructor"]


def _build_tool() -> ToolSpec:
    tool = _PROMPTS["tool"]
    fields = tool["fields"]
    return ToolSpec(
        name=tool["name"],
        description=tool["description"],
        parameters={
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": fields["code"]},
                "description": {"type": "string", "description": fields["description"]},
                "rationale": {"type": "string", "description": fields["rationale"]},
            },
            "required": ["code", "description", "rationale"],
        },
    )


RECONSTRUCT_BASELINE = _build_tool()


class Reconstructor:
    """Asks the LLM to reconstruct the user's source as a runnable `Candidate`."""

    def __init__(
        self,
        client: LLMClient,
        *,
        temperature: float = 0.2,  # low — we want fidelity, not creativity
        max_tokens: int = 2048,
        max_retries: int = 1,
    ) -> None:
        self._client = client
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_retries = max_retries

    def reconstruct(
        self,
        *,
        data_summary: str,
        source_text: str,
        metric: str,
        direction: str,
    ) -> Candidate:
        messages = _build_messages(
            data_summary=data_summary,
            source_text=source_text,
            metric=metric,
            direction=direction,
        )
        detail = ""
        for _ in range(self._max_retries + 1):
            response = self._client.chat(
                messages,
                tools=[RECONSTRUCT_BASELINE],
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
            call = next(
                (c for c in response.tool_calls if c.name == RECONSTRUCT_BASELINE.name),
                None,
            )
            if call is None:
                detail = "model replied without calling reconstruct_baseline"
                messages.append(Message(role="user", content=_PROMPTS["retry_nudge"]))
                continue
            code = call.arguments.get("code")
            if not isinstance(code, str) or not code.strip():
                detail = "reconstruction carried no 'code'"
                messages.append(Message(role="user", content=_PROMPTS["retry_nudge"]))
                continue
            reason = validate_train_and_predict(code)
            if reason is not None:
                detail = reason
                messages.append(Message(role="user", content=_PROMPTS["retry_nudge"]))
                continue
            return _to_candidate(call.arguments, code)
        raise ReconstructorError(
            f"no reconstruction after {self._max_retries + 1} attempt(s): {detail}"
        )


def _to_candidate(args: dict[str, object], code: str) -> Candidate:
    description = str(args.get("description") or "").strip() or "reconstructed baseline"
    rationale = str(args.get("rationale") or "").strip() or "(no rationale given)"
    return Candidate(
        description=description,
        changes={"code": code.strip()},
        rationale=rationale,
        source="human",  # the user provided the source; the LLM only restructured it
    )


def _build_messages(
    *,
    data_summary: str,
    source_text: str,
    metric: str,
    direction: str,
) -> list[Message]:
    system = _PROMPTS["system"].format(metric=metric, direction=direction)
    user = _PROMPTS["user_template"].format(
        data_summary=data_summary,
        metric=metric,
        direction=direction,
        source_text=source_text,
    )
    return [Message(role="system", content=system), Message(role="user", content=user)]


__all__ = ["RECONSTRUCT_BASELINE", "Reconstructor", "ReconstructorError"]
