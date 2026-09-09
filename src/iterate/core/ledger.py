"""The tried/untried ledger: what this run has already spent itself on.

Two deterministic dimensions: lever classes, detected by marker strings in
submitted code, and components, the classes a session instantiated, AST-extracted.
An input to guards, deliberately not rendered into the supervisor's planning
prompt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from iterate.core import codegen

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class Ledger:
    """What has and has not been attempted this run."""

    tried_levers: frozenset[str]
    untried_levers: tuple[str, ...]
    tried_components: tuple[str, ...]
    experiments: int

    def first_untried(self) -> str | None:
        """The next lever class never attempted this run, in canonical order, or
        None when the run has exhausted every class."""
        return self.untried_levers[0] if self.untried_levers else None

    def component_tried(self, name: str) -> bool:
        """Whether a specific class was ever instantiated, case-insensitively.

        Exact on the component name rather than substring on the code, so
        `GradientBoostingClassifier` does not read as tried because
        `HistGradientBoostingClassifier` was built.
        """
        lowered = name.lower()
        return any(c.lower() == lowered for c in self.tried_components)


def build(
    history: Sequence[Any],
    *,
    lever_markers: Mapping[str, Iterable[str]],
    lever_order: Iterable[str],
    normalize: Callable[[str], str] | None = None,
) -> Ledger:
    """Read the run's submitted code into a ledger.

    `normalize` is the caller's marker-safety pass (the supervisor neutralises
    "histgradientboosting" so it cannot satisfy a "gradientboosting" marker). Kept
    injectable rather than imported so this module owns no marker vocabulary.
    """
    normalizer = normalize or (lambda text: text)
    tried: set[str] = set()
    components: list[str] = []
    counted = 0

    for exp in history:
        code = exp.candidate.changes.get("code")
        if not isinstance(code, str):
            continue
        counted += 1
        low = normalizer(code.lower())
        for lever, markers in lever_markers.items():
            if lever not in tried and any(m in low for m in markers):
                tried.add(lever)
        for component in codegen.components_used(code):
            if component not in components:
                components.append(component)

    return Ledger(
        tried_levers=frozenset(tried),
        untried_levers=tuple(lever for lever in lever_order if lever not in tried),
        tried_components=tuple(components),
        experiments=counted,
    )


__all__ = ["Ledger", "build"]
