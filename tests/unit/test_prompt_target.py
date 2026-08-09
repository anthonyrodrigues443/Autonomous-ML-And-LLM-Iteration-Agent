"""`PromptTarget`: the label set, the safety net, and what a dead endpoint means."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from iterate.adapters.data.tabular import load_csv
from iterate.core import codegen
from iterate.schemas.experiment import Candidate
from iterate.schemas.llm import ChatResponse, ToolCall, Usage
from iterate.targets.base import BenchmarkTarget
from iterate.targets.prompt import PromptTarget, label_set, majority_answer

if TYPE_CHECKING:
    from pathlib import Path

    from iterate.adapters.data.tabular import TabularDataset
    from iterate.schemas.llm import Message, ToolSpec

pytestmark = pytest.mark.unit

_ROWS = 40


@pytest.fixture
def dataset(tmp_path: Path) -> TabularDataset:
    lines = ["text,label"]
    for i in range(_ROWS):
        lines.append(f"comment number {i},{'toxic' if i % 4 == 0 else 'not toxic'}")
    path = tmp_path / "eval.csv"
    path.write_text("\n".join(lines), encoding="utf-8")
    return load_csv(path, target="label")


class ScriptedClient:
    def __init__(self, answer: str | Exception) -> None:
        self._answer = answer
        self.calls = 0

    @property
    def model(self) -> str:
        return "fake-12b"

    def chat(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        self.calls += 1
        if isinstance(self._answer, Exception):
            raise self._answer
        return ChatResponse(
            model="fake-12b",
            tool_calls=[ToolCall(id="1", name="answer", arguments={"value": self._answer})],
            usage=Usage(prompt_tokens=5, completion_tokens=1),
        )


@pytest.fixture
def scripted(monkeypatch: pytest.MonkeyPatch) -> Any:
    def install(answer: str | Exception) -> ScriptedClient:
        client = ScriptedClient(answer)
        monkeypatch.setattr("iterate.llm.factory.build_client", lambda *a, **k: client)
        return client

    return install


def _target(dataset: TabularDataset, **kwargs: Any) -> PromptTarget:
    defaults: dict[str, Any] = {
        "metric": "accuracy",
        "task": "Say whether the comment is toxic.",
        "target_backend": "ollama",
        "target_model": "fake-12b",
    }
    return PromptTarget(dataset, **{**defaults, **kwargs})


def test_it_satisfies_the_benchmark_target_contract(dataset: TabularDataset) -> None:
    assert isinstance(_target(dataset), BenchmarkTarget)


def test_labels_come_from_the_training_column_only(dataset: TabularDataset) -> None:
    """Reading the holdout column would put labels the agent has no way to know
    about into every call."""
    assert label_set(dataset) == ["not toxic", "toxic"]


def test_a_high_cardinality_target_is_treated_as_free_text(tmp_path: Path) -> None:
    lines = ["text,answer"] + [f"row {i},answer number {i}" for i in range(120)]
    path = tmp_path / "free.csv"
    path.write_text("\n".join(lines), encoding="utf-8")

    assert label_set(load_csv(path, target="answer", stratify=False)) is None


def test_the_baseline_prompt_is_built_from_the_task(dataset: TabularDataset) -> None:
    target = _target(dataset)

    assert "Say whether the comment is toxic." in target.baseline_prompt.system


def test_a_supplied_production_prompt_becomes_the_baseline(dataset: TabularDataset) -> None:
    """The user's current prompt is measured, not assumed — it runs like any other
    candidate so the improvement is against what they actually have today."""
    from iterate.core.prompting import Prompt

    mine = Prompt(system="you are a moderator", user_template="{text}")

    assert _target(dataset, starting_prompt=mine).baseline_prompt == mine


def test_the_baseline_scores_against_the_sealed_holdout(
    dataset: TabularDataset, scripted: Any
) -> None:
    scripted("not toxic")
    result = _target(dataset).baseline()

    assert result.succeeded
    assert result.metrics is not None
    assert result.metrics.n_samples == dataset.n_test
    assert result.metrics.primary == "accuracy"


def test_a_dead_endpoint_is_an_error_not_a_score_of_zero(
    dataset: TabularDataset, scripted: Any
) -> None:
    """Reporting 0.0 would bank a number and teach the next iteration a lie."""
    scripted(RuntimeError("connection refused"))

    result = _target(dataset).baseline()

    assert not result.succeeded
    assert "unusable" in str(result.error)


def test_a_typed_prompt_candidate_runs(dataset: TabularDataset, scripted: Any) -> None:
    scripted("toxic")
    candidate = Candidate(
        description="blunter instruction",
        changes={"prompt": {"system": "toxic or not toxic?", "user_template": "{text}"}},
        rationale="test",
    )

    assert _target(dataset).run(candidate).succeeded


def test_the_majority_answer_is_the_safety_net(dataset: TabularDataset) -> None:
    """The floor must not be an LLM pass: it would be slowest in exactly the
    situation that triggers it."""
    assert majority_answer(dataset) == "not toxic"


def test_the_floor_cell_calls_no_model() -> None:
    cell = codegen.prompt_fallback_baseline()

    assert "ask(" not in cell
    assert "value_counts" in cell
    assert codegen.PROMPT_JSON in cell


def test_the_holdout_answers_never_reach_the_session(dataset: TabularDataset) -> None:
    """The prompt-path version of fitting on the test set is not prevented, it is
    invisible: the answers are simply not in the working directory."""
    target = _target(dataset)
    job = target.build_code_job(Candidate(description="x", changes={"code": "pass"}, rationale="r"))

    holdout = job.inputs[codegen.HOLDOUT_CSV].decode()

    assert dataset.target not in holdout.splitlines()[0]


def test_the_api_key_is_not_written_beside_the_data(dataset: TabularDataset) -> None:
    target = _target(dataset)
    job = target.build_code_job(Candidate(description="x", changes={"code": "pass"}, rationale="r"))

    meta = json.loads(job.inputs[codegen.META_JSON])

    assert "api_key" not in json.dumps(meta).casefold()
    assert meta["target_model"] == "fake-12b"
    assert meta["labels"] == ["not toxic", "toxic"]


def test_the_submitted_prompt_is_captured_as_an_artifact(dataset: TabularDataset) -> None:
    from iterate.adapters.compute.runner import RunResult

    target = _target(dataset)
    preds = "\n".join(["not toxic"] * dataset.n_test).encode()
    run_result = RunResult(
        stdout="",
        stderr="",
        exit_code=0,
        outputs={
            codegen.PREDICTIONS_CSV: preds,
            codegen.PROMPT_JSON: b'{"system": "be terse", "user_template": "{text}"}',
        },
    )

    result = target.score_code_job(run_result, "iter-01")

    assert result.succeeded
    assert "be terse" in result.artifacts[codegen.PROMPT_JSON]


def test_the_session_preamble_exposes_the_three_helpers(dataset: TabularDataset) -> None:
    preamble = _target(dataset).session_preamble()

    for helper in ("def ask(", "def evaluate(", "def submit("):
        assert helper in preamble


def test_the_preamble_is_valid_python() -> None:
    """It is a large generated code string; a syntax error would only surface in a
    live run, several minutes into a session."""
    compile(codegen.prompt_session_preamble(), "<preamble>", "exec")
    compile(codegen.prompt_fallback_baseline(), "<floor>", "exec")


def test_the_session_runs_end_to_end_in_a_real_directory(
    dataset: TabularDataset, scripted: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Executes the preamble the way a kernel would, then submits.

    The preamble, `ask`, `evaluate` and `submit` are only ever exercised together
    inside a live session, so this is the one test that proves the contract holds:
    predictions and prompt.json land together, with one answer per holdout row.
    """
    scripted("not toxic")
    target = _target(dataset)
    job = target.build_code_job(Candidate(description="x", changes={"code": "pass"}, rationale="r"))
    for name, blob in job.inputs.items():
        (tmp_path / name).write_bytes(blob)
    monkeypatch.chdir(tmp_path)

    namespace: dict[str, Any] = {}
    exec(codegen.prompt_session_preamble(), namespace)

    assert namespace["BASELINE_PROMPT"].system == target.baseline_prompt.system
    assert namespace["TASK"] == "Say whether the comment is toxic."

    train_rows = namespace["X_train"].head(3)
    answers = namespace["ask"](namespace["BASELINE_PROMPT"], train_rows)
    assert answers == ["not toxic"] * 3

    scored = namespace["evaluate"](answers, namespace["y_train"].head(3))
    assert 0.0 <= scored <= 1.0

    namespace["submit"](namespace["BASELINE_PROMPT"])

    predictions = (tmp_path / codegen.PREDICTIONS_CSV).read_text().strip().splitlines()
    submitted = json.loads((tmp_path / codegen.PROMPT_JSON).read_text())
    assert len(predictions) == dataset.n_test
    assert submitted["system"] == target.baseline_prompt.system


def test_the_floor_cell_runs_and_submits(
    dataset: TabularDataset, scripted: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The safety net has to work in the situation that triggers it, so it is
    executed rather than merely inspected."""
    scripted(RuntimeError("the endpoint is down"))
    target = _target(dataset)
    job = target.build_code_job(Candidate(description="x", changes={"code": "pass"}, rationale="r"))
    for name, blob in job.inputs.items():
        (tmp_path / name).write_bytes(blob)
    monkeypatch.chdir(tmp_path)

    namespace: dict[str, Any] = {}
    exec(codegen.prompt_session_preamble(), namespace)
    exec(codegen.prompt_fallback_baseline(), namespace)

    predictions = (tmp_path / codegen.PREDICTIONS_CSV).read_text().strip().splitlines()
    assert predictions == ["not toxic"] * dataset.n_test
    assert (tmp_path / codegen.PROMPT_JSON).exists()


def test_the_prompt_family_gets_its_own_instructions() -> None:
    """The tabular system message is dense with advice about dtype splits and
    encoders. On a floor model that advice is not merely irrelevant to prompt work,
    it gets followed."""
    from iterate.core.coder import _build_messages

    common = {
        "data_summary": "40 rows",
        "metric": "accuracy",
        "direction": "maximize",
        "brief": "next: sharpen the instruction",
        "preamble_output": "loaded",
    }
    prompt_system = _build_messages(**common, family="prompt")[0].content or ""
    tabular_system = _build_messages(**common)[0].content or ""

    assert "prompt engineer" in prompt_system
    assert "submit(prompt)" in prompt_system
    assert "num_cols" not in prompt_system
    assert "num_cols" in tabular_system


def test_the_prompt_instructions_survive_placeholder_rendering() -> None:
    """The system message documents {input} to the agent, which has to survive the
    .format() pass that fills in the metric."""
    from iterate.core.coder import _build_messages

    system = (
        _build_messages(
            data_summary="x",
            metric="f1",
            direction="maximize",
            brief="b",
            preamble_output="o",
            family="prompt",
        )[0].content
        or ""
    )

    assert "{input}" in system
    assert "{{input}}" not in system
    assert "'f1'" in system


def test_a_single_unusable_answer_does_not_break_scoring(
    dataset: TabularDataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The live-run bug, as a regression test.

    Two unparseable answers out of 300 made y_pred carry a third value, sklearn read
    the union of true and predicted as multiclass, average='binary' raised, and the
    whole run aborted at the baseline. One bad answer must cost one row, not the run.
    """
    calls = {"n": 0}

    class Flaky(ScriptedClient):
        def chat(self, messages: list[Message], **kwargs: Any) -> ChatResponse:
            calls["n"] += 1
            if calls["n"] == 1:  # the first row answers with unusable prose
                return ChatResponse(model="fake-12b", content="hmm, could be either")
            return ChatResponse(
                model="fake-12b",
                tool_calls=[ToolCall(id="1", name="answer", arguments={"value": "not toxic"})],
            )

    monkeypatch.setattr("iterate.llm.factory.build_client", lambda *a, **k: Flaky("not toxic"))

    result = _target(dataset, metric="f1").baseline()

    assert result.succeeded, result.error
    assert result.metrics is not None
    assert 0.0 <= result.metrics.primary_value <= 1.0


def test_an_unusable_answer_can_never_improve_the_score() -> None:
    """Measured before choosing the fix: mapping the sentinel to the POSITIVE class
    scored 0.857 where the honest answer is 0.800, so garbage would have paid."""
    from iterate.core.scoring import score

    truth = ["not toxic", "toxic", "not toxic", "toxic", "toxic", "not toxic"]
    perfect = list(truth)
    with_junk = ["not toxic", "toxic", "__unparseable__", "__unparseable__", "toxic", "not toxic"]

    clean = score("classification", truth, perfect, include=("f1",), open_vocabulary=True)["f1"]
    junked = score("classification", truth, with_junk, include=("f1",), open_vocabulary=True)["f1"]

    assert clean == pytest.approx(1.0)
    assert junked == pytest.approx(0.8)
    assert junked < clean


def test_open_vocabulary_leaves_clean_predictions_identical() -> None:
    """The flag must be a no-op when every prediction is in the label set, or it
    would silently re-rule every tabular score."""
    from iterate.core.scoring import score

    truth = ["a", "b", "a", "b", "b"]
    preds = ["a", "b", "b", "b", "a"]

    assert score("classification", truth, preds, include=("f1",)) == pytest.approx(
        score("classification", truth, preds, include=("f1",), open_vocabulary=True)
    )


def test_a_sentinel_does_not_pollute_a_macro_average() -> None:
    """Without naming the labels the sentinel becomes a class with an f1 of zero and
    drags macro down, penalising the run twice for one bad answer."""
    from iterate.core.scoring import score

    truth = ["a", "b", "c", "a", "b", "c"]
    preds = ["a", "b", "c", "a", "b", "__unparseable__"]

    naive = score("classification", truth, preds, average="macro", include=("f1",))["f1"]
    fixed = score(
        "classification", truth, preds, average="macro", include=("f1",), open_vocabulary=True
    )["f1"]

    # Only class "c" is hurt, and only by the one row it lost: a=1.0, b=1.0,
    # c=0.667 (one false negative), macro 0.889. The naive number is 0.667 because
    # the sentinel is averaged in as a fourth class scoring zero.
    assert naive == pytest.approx(0.6667, abs=1e-4)
    assert fixed == pytest.approx(0.8889, abs=1e-4)


def test_the_supervisor_walks_a_prompt_ladder_not_a_tabular_one() -> None:
    """The first live run's actual failure: the tabular supervisor read a column of
    comment text as a high-cardinality categorical and briefed "one-hot encode
    categorical 'comment', fit HistGradientBoostingClassifier". The coder obeyed and
    never submitted a prompt."""
    from iterate.core.supervisor import _build_messages

    common = {
        "data_summary": "1200 rows, one input column 'comment'",
        "metric": "f1",
        "direction": "maximize",
        "score": 0.8611,
        "history": [],
    }
    prompt_side = _build_messages(**common, family="prompt")[0].content or ""
    tabular_side = _build_messages(**common)[0].content or ""

    assert "THERE IS NO MODEL TO TRAIN" in prompt_side
    for tabular_move in ("HistGradientBoosting", "one-hot", "median-impute", "class_weight"):
        assert tabular_move not in prompt_side, f"{tabular_move} leaked into the prompt ladder"
    assert "HistGradientBoosting" in tabular_side


def test_the_prompt_ladder_names_moves_that_exist() -> None:
    from iterate.core.supervisor import _build_messages

    system = (
        _build_messages(
            data_summary="x",
            metric="f1",
            direction="maximize",
            score=0.5,
            history=[],
            family="prompt",
        )[0].content
        or ""
    )

    for rung in ("READ THE MISTAKES", "DEFINE THE HARD CASE", "FEW-SHOT", "OUTPUT DISCIPLINE"):
        assert rung in system
    assert "100 to 150" in system  # sample, do not sweep


def test_the_preamble_shows_a_worked_example() -> None:
    """Describing the objects was not enough: the live session called .split() on a
    Prompt and passed one to re.sub."""
    preamble = codegen.prompt_session_preamble()

    assert "HOW TO WORK" in preamble
    assert "ask(BASE, sample)" in preamble
    assert "BASE = BASELINE_PROMPT" in preamble  # the alias the example leans on
    assert "is an OBJECT, not a string" in preamble


def test_the_coder_captures_the_submitted_prompt_from_the_kernel() -> None:
    """The live path reads its outputs in `CodingAgent.run`, not through
    `score_code_job`. Without reading prompt.json there, a run that improved f1
    three times shipped a prompts.yaml containing only the baseline — measured.
    """
    import inspect

    from iterate.core import coder

    source = inspect.getsource(coder.CodingAgent.run)

    assert "PROMPT_JSON" in source, "the live path never reads the submitted prompt"
    assert "artifacts" in source


def test_the_record_carries_a_prompt_the_coder_captured() -> None:
    """End to end from a captured artifact to the delivered file."""
    from iterate.core.prompting import Prompt
    from iterate.deliver import prompt_record
    from iterate.schemas.experiment import Experiment, ExperimentResult, Metrics

    winner = Experiment(
        candidate=Candidate(
            description="define the hard case", changes={"code": "..."}, rationale="r"
        ),
        target="prompt",
        hypothesis="h",
        status="completed",
        result=ExperimentResult(
            experiment_id="iter-01",
            metrics=Metrics(values={"f1": 0.8822}, primary="f1", direction="maximize"),
            artifacts={
                codegen.PROMPT_JSON: json.dumps(
                    {"system": "toxicity is personal attacks", "user_template": "{input}"}
                )
            },
        ),
    )

    document = prompt_record.build(
        task="decide whether this comment is toxic",
        metric="f1",
        direction="maximize",
        model_under_test="gemma4:12b",
        baseline_prompt=Prompt(system="decide", user_template="{input}"),
        baseline_score=0.8611,
        history=[winner],
    )

    assert "toxicity is personal attacks" in document
    assert "v1" in document


def test_the_prompt_fallback_never_reaches_for_a_tabular_lever() -> None:
    """A live prompt run was handed "untried lever: categorical-encoding" as its
    third experiment. The fallback exists to be novel BY CONSTRUCTION; novel
    nonsense is not the deal. It scored only because the coder ignored the name."""
    from iterate.core.supervisor import _prompt_fallback_move

    result = _prompt_fallback_move([])

    assert result is not None
    title, move = result
    for tabular in ("categorical-encoding", "imbalance", "model-swap", "hyperparameter"):
        assert tabular not in title, f"{tabular} leaked into a prompt fallback"
        assert tabular not in move


def test_the_prompt_fallback_skips_moves_already_briefed() -> None:
    from iterate.core.supervisor import _prompt_fallback_move
    from iterate.schemas.experiment import Experiment

    already = Experiment(
        candidate=Candidate(
            description="define-the-hard-case: quoted insults",
            changes={"code": "..."},
            rationale="r",
        ),
        target="prompt",
        hypothesis="next: define-the-hard-case: ...",
    )

    result = _prompt_fallback_move([already])

    assert result is not None
    assert "define-the-hard-case" not in result[0]


def test_the_rebrief_nudge_points_at_the_prompt_ladder() -> None:
    """The tabular wording sends the supervisor to "Levers NOT yet tried" and asks
    it to name a lever class, neither of which exists on this path."""
    from iterate.prompts import PROMPTS

    nudge = PROMPTS["supervisor"]["prompt_baseline_rebrief_nudge"]

    assert "READ THE MISTAKES" in nudge
    assert "lever" not in nudge.casefold()
    assert "few-shot" in nudge


# ─── Day 4: what the second example broke ──────────────────────────────────


def test_a_label_containing_another_label_still_resolves() -> None:
    """Measured on the real CLINC intents: "time" is a substring of "timer", so the
    clear reply "the intent is timer" matched two labels and was thrown away. Two
    answers were binary-only luck — no two-label set collides, so this could not
    surface until the second example."""
    from iterate.core.prompt_runtime import coerce

    labels = ["time", "timer", "transfer", "balance"]

    assert coerce("the intent is timer", labels) == "timer"
    assert coerce("this is a time question", labels) == "time"


def test_a_reply_naming_two_different_labels_is_still_ambiguous() -> None:
    """The longest-match rule must not turn genuine ambiguity into a confident
    answer — that would be worse than unparseable, because it is invisible."""
    from iterate.core.prompt_runtime import UNPARSEABLE, coerce

    labels = ["time", "timer", "transfer", "balance"]

    assert coerce("either transfer or timer", labels) == UNPARSEABLE
    assert coerce("balance or transfer, unclear", labels) == UNPARSEABLE


def test_a_label_inside_a_longer_word_does_not_match() -> None:
    from iterate.core.prompt_runtime import UNPARSEABLE, coerce

    assert coerce("timers", ["time", "timer"]) == UNPARSEABLE
    assert coerce("the timekeeper", ["time"]) == UNPARSEABLE


@pytest.mark.parametrize(
    ("answers", "expected"),
    [
        (["toxic", "clean"] * 30, "closed_set"),
        ([f"intent_{i % 20}" for i in range(200)], "closed_set"),
        ([i % 10 + 1 for i in range(60)], "closed_set"),
        ([i * 0.37 for i in range(60)], "numeric"),
        ([f"a unique summary {i}" for i in range(60)], "free_text"),
        ([f"intent_{i % 150}" for i in range(3000)], "closed_set"),
    ],
    ids=["binary", "20-intents", "rating-1-10", "continuous", "free-text", "150-intents"],
)
def test_the_answer_column_is_read_correctly(
    answers: list[Any], expected: str, tmp_path: Path
) -> None:
    """A count alone cannot separate 48 intents from 48 one-off sentences, so the
    share of rows carrying their own answer decides it too."""
    import pandas as pd

    from iterate.targets.prompt import target_kind

    path = tmp_path / "d.csv"
    pd.DataFrame({"t": [f"c{i}" for i in range(len(answers))], "y": answers}).to_csv(
        path, index=False
    )

    assert target_kind(load_csv(path, target="y", stratify=False)) == expected


def test_a_rating_column_leaves_the_task_to_the_metric() -> None:
    """A 1-to-10 rating is honestly ten ordered classes or a score, depending on how
    you measure it. Deciding from the dtype here would overrule the user — and for a
    rating, classification scores 9-vs-10 the same as 1-vs-10."""
    from iterate.core.scoring import task_for_metric

    assert task_for_metric("f1_macro") == "classification"
    assert task_for_metric("rmse") == "regression"


def test_the_submission_must_come_from_the_model_under_test() -> None:
    """Nothing stopped a later cell overwriting predictions.csv with a hardcoded
    rule while prompt.json still sat there. That would score well and would not be
    prompt engineering at all. Verifiable, so it is a hard rejection rather than a
    Critic opinion — the same rule as "a leak vetoes, a mirage only flags".
    """
    import hashlib

    answers = ["toxic", "not toxic", "toxic"]
    honest = "\n".join(answers).encode()
    digest = hashlib.sha256("\n".join(answers).encode()).hexdigest()
    prompt_json = json.dumps(
        {"system": "s", "user_template": "{input}", "answers_sha256": digest}
    ).encode()

    assert codegen.submission_was_swapped(prompt_json, honest) is None
    swapped = "\n".join(["toxic"] * 3).encode()
    assert codegen.submission_was_swapped(prompt_json, swapped) is not None


def test_a_submission_with_no_fingerprint_is_not_accused() -> None:
    """Absence of evidence is not evidence. An older run, or a hand-written
    submission, must pass rather than be rejected on a missing field."""
    plain = json.dumps({"system": "s", "user_template": "{input}"}).encode()

    assert codegen.submission_was_swapped(plain, b"toxic\nnot toxic") is None
    assert codegen.submission_was_swapped(None, b"toxic") is None
    assert codegen.submission_was_swapped(b"not json at all", b"toxic") is None


def test_submit_records_what_the_model_answered() -> None:
    preamble = codegen.prompt_session_preamble()

    assert "answers_sha256" in preamble
    assert "sha256" in preamble


def test_the_critic_asks_prompt_questions_on_a_prompt_run() -> None:
    """It ran on all three experiments of a live run and returned clean verdicts —
    correctly, but while asking about scalers and target encoding, which cannot
    happen on this path."""
    from iterate.prompts import PROMPTS

    prompt_side = PROMPTS["critic"]["prompt_system"]
    tabular_side = PROMPTS["critic"]["system"]

    assert "lookup" in prompt_side
    assert "keyword match" in prompt_side
    for tabular_only in ("scaler", "target encoding", "fit_transform"):
        assert tabular_only not in prompt_side
    assert "scaler" in tabular_side


def test_the_researcher_searches_for_prompting_literature() -> None:
    from iterate.prompts import PROMPTS

    queries = PROMPTS["researcher"]["prompt_queries_system"]

    assert "TABULAR" not in queries
    assert "few-shot" in queries
    assert "fine-tuning" in queries  # named as a NON-lever, so it is not searched for


def test_a_large_closed_set_keeps_its_kind_but_loses_the_enum(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The bug my own check found: conflating "too many for an enum" with "not a
    closed set" refused full CLINC150 — 150 intents in 2400 rows — as free text.
    The ratio decides the kind; the cap only decides whether an enum is feasible,
    and dropping the enum has to be said out loud because it removes the guarantee
    that an answer is in the set.
    """
    import pandas as pd

    from iterate.targets.prompt import label_set, target_kind

    path = tmp_path / "many.csv"
    pd.DataFrame(
        {"t": [f"c{i}" for i in range(3000)], "y": [f"intent_{i % 150}" for i in range(3000)]}
    ).to_csv(path, index=False)
    dataset = load_csv(path, target="y", stratify=False)

    assert target_kind(dataset) == "closed_set"
    with caplog.at_level("WARNING"):
        assert label_set(dataset) is None
    assert "more than the 50-item answer tool" in caplog.text
