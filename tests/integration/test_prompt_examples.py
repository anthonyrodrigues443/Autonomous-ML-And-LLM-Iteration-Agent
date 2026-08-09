"""The two shipped prompt examples, exercised end to end without a live model.

Marked `integration` because both need their dataset present, and both are
gitignored — a fresh clone runs `prepare.py` first. What they check is the part a
unit test cannot: that the tracked `prepare.py` produces a file the real loader,
the real target and the real session preamble all accept, for a BINARY example and
a MULTICLASS one.

The model itself is scripted. This is about the plumbing between the example and
the harness, not about whether a 12B is any good at toxicity.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from iterate.adapters.data.tabular import load_csv
from iterate.core import codegen
from iterate.schemas.experiment import Candidate
from iterate.schemas.llm import ChatResponse, ToolCall
from iterate.targets.prompt import PromptTarget, label_set, target_kind

if TYPE_CHECKING:
    from pathlib import Path

    from iterate.schemas.llm import Message

pytestmark = pytest.mark.integration

EXAMPLES = [
    ("toxicity_jigsaw", "label", "f1", 2),
    ("intent_clinc150", "intent", "f1_macro", 20),
]


def _dataset(repo_root: Path, name: str, target: str) -> Any:
    path = repo_root / "examples" / name / "data.csv"
    if not path.exists():
        pytest.skip(f"{path} not present — run examples/{name}/prepare.py")
    return load_csv(path, target=target)


class Scripted:
    """Answers with the first allowed label, whatever it is asked."""

    def __init__(self, answer: str) -> None:
        self._answer = answer

    @property
    def model(self) -> str:
        return "scripted"

    def chat(self, messages: list[Message], **kwargs: Any) -> ChatResponse:
        return ChatResponse(
            model="scripted",
            tool_calls=[ToolCall(id="1", name="answer", arguments={"value": self._answer})],
        )


@pytest.mark.parametrize(("name", "target", "metric", "labels"), EXAMPLES, ids=[e[0] for e in EXAMPLES])
def test_the_example_is_shaped_the_way_the_target_expects(
    name: str, target: str, metric: str, labels: int, repo_root: Path
) -> None:
    dataset = _dataset(repo_root, name, target)

    assert target_kind(dataset) == "closed_set"
    assert len(label_set(dataset) or []) == labels
    assert dataset.n_train > 0
    assert dataset.n_test > 0
    assert target not in dataset.features  # the answer column is not also an input


@pytest.mark.parametrize(("name", "target", "metric", "labels"), EXAMPLES, ids=[e[0] for e in EXAMPLES])
def test_a_session_runs_and_submits_on_the_example(
    name: str,
    target: str,
    metric: str,
    labels: int,
    repo_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole contract on real data: preamble executes, `ask` answers every row,
    `evaluate` scores with the run's metric, `submit` writes both artifacts, and the
    fingerprint proves the predictions came from the model."""
    dataset = _dataset(repo_root, name, target)
    answer = (label_set(dataset) or ["x"])[0]
    monkeypatch.setattr("iterate.llm.factory.build_client", lambda *a, **k: Scripted(answer))

    prompt_target = PromptTarget(
        dataset,
        metric=metric,
        task=f"decide the {target}",
        target_backend="ollama",
        target_model="scripted",
        cache_path=tmp_path / "answers.db",
    )
    job = prompt_target.build_code_job(
        Candidate(description="x", changes={"code": "pass"}, rationale="r")
    )
    for filename, blob in job.inputs.items():
        (tmp_path / filename).write_bytes(blob)
    monkeypatch.chdir(tmp_path)

    namespace: dict[str, Any] = {}
    exec(codegen.prompt_session_preamble(), namespace)

    sample = namespace["X_train"].head(5)
    answers = namespace["ask"](namespace["BASE"], sample)
    assert len(answers) == 5
    assert 0.0 <= namespace["evaluate"](answers, namespace["y_train"].head(5)) <= 1.0

    namespace["submit"](namespace["BASE"])
    predictions = (tmp_path / codegen.PREDICTIONS_CSV).read_bytes()
    prompt_json = (tmp_path / codegen.PROMPT_JSON).read_bytes()

    assert len(predictions.decode().strip().splitlines()) == dataset.n_test
    assert codegen.submission_was_swapped(prompt_json, predictions) is None


def test_the_holdout_answers_are_absent_from_both_examples(repo_root: Path) -> None:
    """The sealed holdout is what every score rests on, so it is asserted against
    the real files rather than a fixture."""
    for name, target, _, _ in EXAMPLES:
        dataset = _dataset(repo_root, name, target)
        job = PromptTarget(
            dataset,
            metric="f1",
            task="t",
            target_backend="ollama",
            target_model="scripted",
        ).build_code_job(Candidate(description="x", changes={"code": "pass"}, rationale="r"))

        header = job.inputs[codegen.HOLDOUT_CSV].decode().splitlines()[0]
        assert target not in header, f"{name}: holdout answers reached the session"
