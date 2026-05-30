"""The Reconstructor — read a user-supplied source (md / txt / notebook / .py) as
TEXT ONLY and emit the baseline approach as a runnable `{"model","params"}` spec.

Sibling of `Proposer`: same LLM/tool-calling/retry machinery, different intent
and prompt. The Proposer picks the *next* experiment; the Reconstructor extracts
the *user's existing* approach from their source so we can run it through our
own eval as a comparable baseline.

**Hard security boundary:** user-provided source is *never* executed — not by us,
not in the v0.2 e2b sandbox, not ever. The LLM sees it as text only and emits a
spec the factory builds from an allow-listed library. If the source uses a model
outside our allow-list (CatBoost, a custom architecture, …), the LLM picks the
closest faithful equivalent and flags the approximation in its rationale.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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
                "model": {"type": "string", "description": fields["model"]},
                "params": {
                    "type": "object",
                    "description": fields["params"],
                    "additionalProperties": True,
                },
                "description": {"type": "string", "description": fields["description"]},
                "rationale": {"type": "string", "description": fields["rationale"]},
            },
            "required": ["model", "description", "rationale"],
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
            if call is not None:
                return _to_candidate(call.arguments)
            detail = "model replied without calling reconstruct_baseline"
            messages.append(Message(role="user", content=_PROMPTS["retry_nudge"]))
        raise ReconstructorError(
            f"no reconstruction after {self._max_retries + 1} attempt(s): {detail}"
        )


def _to_candidate(args: dict[str, Any]) -> Candidate:
    model = args.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ReconstructorError("reconstruction is missing a 'model'")
    changes: dict[str, Any] = {"model": model.strip()}
    params = args.get("params")
    if isinstance(params, dict) and params:
        changes["params"] = params
    description = str(args.get("description") or "").strip() or f"reconstructed {model.strip()}"
    rationale = str(args.get("rationale") or "").strip() or "(no rationale given)"
    return Candidate(
        description=description,
        changes=changes,
        rationale=rationale,
        source="human",  # the user provided the source; LLM only structured it
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
