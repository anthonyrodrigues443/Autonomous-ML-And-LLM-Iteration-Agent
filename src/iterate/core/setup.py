"""Resolving how a run will be measured, when the user did not say.

The first input dial to turn since v0.1: `--metric` becomes optional. What replaces
it is not "the agent decides" but a narrow, guarded sequence:

1. an explicit `--metric` always wins, unchanged;
2. otherwise the Researcher proposes a metric and a starting model, judged over the
   same papers its technique suggestions came from;
3. whatever it proposes is validated against the registry and against the target
   column before anything is built;
4. anything that fails validation falls back to the deterministic default, which is
   exactly what v0.3 would have done.

So the agent can only ever UPGRADE the default. It cannot produce a run that fails
to start, and it cannot pick a metric that does not apply to the target it was
given. The whole point of the dial is to remove an input, not to add a way to be
wrong.

The resolved metric is then FIXED for the run. Every score in Memory and every
cross-run baseline carry-over is measured with one ruler; a metric that changed
mid-run would leave the history silently incomparable and break the prior-best
inheritance a later run depends on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from iterate.core.scoring import REGISTRY, task_for_metric

if TYPE_CHECKING:
    from iterate.adapters.data.tabular import TabularDataset
    from iterate.core.researcher import Setup

# What v0.3 did, kept as the floor every failure path lands on.
_BINARY_DEFAULT = "f1"
_MULTICLASS_DEFAULT = "f1_macro"
_REGRESSION_DEFAULT = "rmse"
_CLASSIFIER = "sklearn.ensemble.HistGradientBoostingClassifier"
_REGRESSOR = "sklearn.ensemble.HistGradientBoostingRegressor"


@dataclass(frozen=True)
class RunSetup:
    """The frozen answer to "how is this run measured, and from what"."""

    metric: str
    starting_model: str
    why: str = ""
    chosen_by_agent: bool = False

    def render(self) -> str:
        """The line printed before the first experiment. A tool that silently picks
        your evaluation metric is worse than one that asks, so when the agent chose
        it says so and says why."""
        if not self.chosen_by_agent:
            return ""
        line = f"metric: {self.metric}"
        if self.why:
            line += f" — {self.why}"
        return line


def target_task(dataset: TabularDataset) -> str:
    """classification or regression, from the target column alone. No LLM.

    DELEGATES to the loader's own heuristic rather than reimplementing one. The
    loader already decides this to choose whether to stratify the split, and two
    definitions of "is this classification" would eventually disagree — which is
    exactly how a metric gets picked for a task the data was never split for.
    """
    from iterate.adapters.data.tabular import looks_like_classification

    return "classification" if looks_like_classification(dataset.train_target) else "regression"


def _n_classes(target: Any) -> int:
    try:
        return int(target.nunique())
    except Exception:
        return len(set(target))


def default_setup(dataset: TabularDataset) -> RunSetup:
    """What v0.3 would have run. Every failure path lands here."""
    task = target_task(dataset)
    if task == "regression":
        return RunSetup(metric=_REGRESSION_DEFAULT, starting_model=_REGRESSOR)
    binary = _n_classes(dataset.train_target) <= 2
    return RunSetup(
        metric=_BINARY_DEFAULT if binary else _MULTICLASS_DEFAULT,
        starting_model=_CLASSIFIER,
    )


def validate(metric: str, dataset: TabularDataset) -> str | None:
    """Why this metric cannot be used for this dataset, or None if it can."""
    if metric not in REGISTRY:
        return f"{metric!r} is not a known metric"
    wanted = task_for_metric(metric)
    actual = target_task(dataset)
    if wanted != actual:
        return f"{metric!r} is a {wanted} metric but the target looks like {actual}"
    return None


def resolve(
    dataset: TabularDataset,
    *,
    explicit: str | None = None,
    proposed: Setup | None = None,
) -> RunSetup:
    """The single place a run's ruler is decided. Never raises."""
    fallback = default_setup(dataset)

    if explicit:
        # The user's choice is not second-guessed. It was already validated at the
        # CLI boundary, where a bad name gets a proper error rather than a silent
        # substitution.
        return RunSetup(metric=explicit, starting_model=fallback.starting_model)

    if proposed is None or not proposed.metric:
        return fallback

    if (reason := validate(proposed.metric, dataset)) is not None:
        # Rejected proposals are logged by the caller and the run continues on the
        # default — an agent that picks badly costs nothing.
        return RunSetup(
            metric=fallback.metric,
            starting_model=fallback.starting_model,
            why=f"agent proposed {proposed.metric!r} but {reason}; using the default",
            chosen_by_agent=False,
        )

    return RunSetup(
        metric=proposed.metric,
        starting_model=_valid_model(proposed.starting_model, fallback.starting_model),
        why=proposed.why,
        chosen_by_agent=True,
    )


def _valid_model(proposed: str, fallback: str) -> str:
    """A starting model is a weaker claim than a metric — the agent rewrites the
    model every iteration anyway, so a bad one costs one experiment rather than the
    whole run's ruler. Accepted if it looks like an estimator name at all."""
    name = (proposed or "").strip()
    if not name or not name.replace(".", "").replace("_", "").isalnum():
        return fallback
    return name


__all__ = ["RunSetup", "default_setup", "resolve", "target_task", "validate"]
