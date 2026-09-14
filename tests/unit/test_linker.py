"""Tests for the Linker specialist.

The load-bearing property is that nothing it says becomes a plan until the harness
rebuilt the frame from its picks and measured the coverage: a table it was not
shown, a column the header does not have, a key that resolves too few rows, prose
instead of a tool call, all end as no plan and a reason. The LLM is a scripted fake.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
import pytest
from PIL import Image

from iterate.adapters.data.linking import LinkError, inventory, plan
from iterate.core.linker import (
    MAX_LISTING_CHARS,
    PROPOSE_LINK,
    Linker,
    Proposal,
    describe,
    listing,
)
from iterate.schemas.llm import ChatResponse, Message, ToolCall

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit


class _FakeLLM:
    """Replies in order. A plain string means 'no tool call' (the retry path)."""

    def __init__(self, replies: list[Any], *, boom: bool = False) -> None:
        self._replies = list(replies)
        self._boom = boom
        self.seen: list[list[Message]] = []

    @property
    def model(self) -> str:
        return "fake"

    def chat(self, messages, *, tools=None, temperature=None, max_tokens=None) -> ChatResponse:  # type: ignore[no-untyped-def]
        self.seen.append(list(messages))
        if self._boom:
            raise RuntimeError("connection refused")
        reply = self._replies.pop(0) if self._replies else "nothing"
        if isinstance(reply, str):
            return ChatResponse(model="fake", content=reply, tool_calls=[])
        return ChatResponse(
            model="fake",
            content="",
            tool_calls=[ToolCall(id="call-1", name="propose_link", arguments=reply)],
        )


def _png(path: Path, seed: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 8), (seed * 9 % 255, 60, 120)).save(path)


def _ambiguous(root: Path, *, n: int = 24) -> Path:
    """Images and a table with a key that resolves but no column the rules can call
    the label: three other columns, none named like a target."""
    for i in range(n):
        _png(root / "img" / f"{i:03d}.jpg", i)
    pd.DataFrame(
        {
            "file": [f"{i:03d}.jpg" for i in range(n)],
            "region": ["north", "south", "east"] * (n // 3),
            "species_code": ["x", "y"] * (n // 2),
            "photographer": [f"person {i}" for i in range(n)],
        }
    ).to_csv(root / "meta.csv", index=False)
    return root


def _pick(**overrides: Any) -> dict[str, Any]:
    args: dict[str, Any] = {
        "labels_table": 1,
        "key_column": "file",
        "key_to_file": "basename",
        "target_column": "species_code",
        "why": "file finds an image for every row",
    }
    args.update(overrides)
    return args


def test_the_fixture_is_one_the_rules_refuse_as_a_choice(tmp_path: Path) -> None:
    with pytest.raises(LinkError, match="no rule links them") as caught:
        plan(inventory(_ambiguous(tmp_path)))
    assert caught.value.ambiguous is True


def test_a_valid_pick_becomes_a_measured_plan(tmp_path: Path) -> None:
    inv = inventory(_ambiguous(tmp_path))
    llm = _FakeLLM([_pick()])
    proposal = Linker(llm).propose(inv, refusal="no rule links them")
    assert proposal.plan is not None
    assert proposal.plan.source == "agent"
    assert proposal.plan.target_column == "species_code"
    assert proposal.plan.key_to_file == "basename"
    assert proposal.plan.coverage == 1.0
    assert proposal.plan.task == "classification"
    assert proposal.reason == "file finds an image for every row"


def test_the_harness_picks_the_task_not_the_model(tmp_path: Path) -> None:
    root = _ambiguous(tmp_path)
    frame = pd.read_csv(root / "meta.csv")
    frame["species_code"] = [1.0 + i * 0.37 for i in range(len(frame))]
    frame.to_csv(root / "meta.csv", index=False)
    proposal = Linker(_FakeLLM([_pick()])).propose(inventory(root))
    assert proposal.plan is not None
    assert proposal.plan.task == "regression"


# ─── the guarantee: every pick is checked against what was shown ─────────────


@pytest.mark.parametrize("number", [2, 0, -1, "two", True, None, 1.5])
def test_a_table_number_not_shown_is_no_plan(tmp_path: Path, number: object) -> None:
    inv = inventory(_ambiguous(tmp_path))
    proposal = Linker(_FakeLLM([_pick(labels_table=number)])).propose(inv)
    assert proposal.plan is None
    assert "was not listed" in proposal.reason


def test_a_table_number_as_text_still_counts(tmp_path: Path) -> None:
    inv = inventory(_ambiguous(tmp_path))
    proposal = Linker(_FakeLLM([_pick(labels_table="1")])).propose(inv)
    assert proposal.plan is not None


def test_a_column_the_header_does_not_have_is_no_plan(tmp_path: Path) -> None:
    inv = inventory(_ambiguous(tmp_path))
    proposal = Linker(_FakeLLM([_pick(target_column="label")])).propose(inv)
    assert proposal.plan is None
    assert "has no column 'label'" in proposal.reason


def test_a_missing_field_is_no_plan(tmp_path: Path) -> None:
    inv = inventory(_ambiguous(tmp_path))
    args = _pick()
    del args["key_to_file"]
    proposal = Linker(_FakeLLM([args])).propose(inv)
    assert proposal.plan is None
    assert "not a way to read a key" in proposal.reason


def test_key_and_target_cannot_be_one_column(tmp_path: Path) -> None:
    inv = inventory(_ambiguous(tmp_path))
    proposal = Linker(_FakeLLM([_pick(target_column="file")])).propose(inv)
    assert proposal.plan is None
    assert "both the key and the target" in proposal.reason


def test_a_key_under_the_floor_is_refused_with_the_number(tmp_path: Path) -> None:
    inv = inventory(_ambiguous(tmp_path))
    proposal = Linker(_FakeLLM([_pick(key_column="photographer")])).propose(inv)
    assert proposal.plan is None
    assert "resolves 0.0% of rows" in proposal.reason
    assert "under the 90% floor" in proposal.reason


def test_prose_twice_is_no_plan_and_the_nudge_was_sent(tmp_path: Path) -> None:
    inv = inventory(_ambiguous(tmp_path))
    llm = _FakeLLM(["Sure! The labels are in meta.csv.", "Column species_code."])
    proposal = Linker(llm).propose(inv)
    assert proposal.plan is None
    assert "no answer in the tool's shape" in proposal.reason
    assert len(llm.seen) == 2
    assert llm.seen[1][-1].content == "Respond ONLY with the propose_link tool call. No prose."


def test_json_in_prose_is_not_a_tool_call(tmp_path: Path) -> None:
    """Only a structured call counts; a JSON blob in the text is never parsed."""
    inv = inventory(_ambiguous(tmp_path))
    text = '{"labels_table": 1, "key_column": "file", "key_to_file": "basename", '
    text += '"target_column": "species_code", "why": "x"}'
    proposal = Linker(_FakeLLM([text, text])).propose(inv)
    assert proposal.plan is None


def test_the_call_failing_is_no_plan_not_a_crash(tmp_path: Path) -> None:
    inv = inventory(_ambiguous(tmp_path))
    proposal = Linker(_FakeLLM([], boom=True)).propose(inv)
    assert proposal == Proposal(None, "the model call failed (RuntimeError)")


# ─── what the model is told ─────────────────────────────────────────────────


def test_notes_previous_and_refusal_reach_the_prompt(tmp_path: Path) -> None:
    inv = inventory(_ambiguous(tmp_path))
    llm = _FakeLLM([_pick(target_column="region"), _pick()])
    linker = Linker(llm)
    first = linker.propose(inv, refusal="24 images and 1 table(s), and no rule links them")
    assert first.plan is not None
    second = linker.propose(
        inv,
        refusal="24 images and 1 table(s), and no rule links them",
        notes=["the label is species_code, not region"],
        previous=first.plan,
    )
    assert second.plan is not None
    user = llm.seen[1][1].content or ""
    assert "Why the rules stopped: 24 images and 1 table(s), and no rule links them" in user
    assert "- the label is species_code, not region" in user
    assert "Your previous answer: table meta.csv, key 'file' read as the file name" in user
    assert "target 'region'" in user
    assert llm.seen[1][0].role == "system"


def test_the_listing_numbers_tables_and_shows_measured_shares(tmp_path: Path) -> None:
    inv = inventory(_ambiguous(tmp_path))
    shown = listing(inv)
    assert shown.tables == tuple(inv.tables)
    text = shown.text
    assert "layout:" in text
    assert "img/  24 images" in text
    assert "1. meta.csv  (24 rows, 4 columns)" in text
    assert (
        "- 'file': text, 24 distinct; read as the file name it finds an image for 100% of rows"
        in text
    )
    assert "- 'species_code': text, 2 distinct" in text
    assert "- 'photographer': text, 24 distinct\n" in text  # no share: it names no file
    assert "first 5 rows:" in text
    assert "000.jpg,north,x,person 0" in text


def test_the_listing_names_the_split_folders_and_the_readme(tmp_path: Path) -> None:
    _ambiguous(tmp_path / "train")
    _ambiguous(tmp_path / "test", n=12)
    (tmp_path / "README.md").write_text("x" * 3000, encoding="utf-8")
    text = listing(inventory(tmp_path)).text
    assert "split folders: train/ is train, test/ is holdout" in text
    assert "1. test/meta.csv" in text  # tables in inventory order, which is sorted
    assert "2. train/meta.csv" in text
    assert "notes found in the folder" in text
    assert text.endswith("x" * 100 + " [cut]")


def test_the_listing_stays_inside_its_budget(tmp_path: Path) -> None:
    root = _ambiguous(tmp_path)
    for k in range(40):
        pd.DataFrame({f"column_{j}_{'z' * 30}": ["v" * 40] * 8 for j in range(12)}).to_csv(
            root / f"extra_{k:02d}.csv", index=False
        )
    text = listing(inventory(root)).text
    assert len(text) <= MAX_LISTING_CHARS + len("\n[listing cut at its budget]")
    assert text.endswith("[listing cut at its budget]")


def test_a_budget_squeeze_drops_sample_rows_before_cutting(tmp_path: Path) -> None:
    root = _ambiguous(tmp_path)
    for k in range(12):
        pd.DataFrame({f"c{j}": ["v" * 60] * 8 for j in range(12)}).to_csv(
            root / f"extra_{k:02d}.csv", index=False
        )
    text = listing(inventory(root)).text
    assert "first 5 rows:" not in text
    assert "[listing cut" not in text


# ─── the split ───────────────────────────────────────────────────────────────


def _with_subset(root: Path) -> Path:
    _ambiguous(root)
    frame = pd.read_csv(root / "meta.csv")
    frame["subset"] = ["train"] * 18 + ["test"] * 6
    frame.to_csv(root / "meta.csv", index=False)
    return root


def test_a_split_column_the_model_names_is_the_users_split(tmp_path: Path) -> None:
    inv = inventory(_with_subset(tmp_path))
    proposal = Linker(_FakeLLM([_pick(split_column="subset")])).propose(inv)
    assert proposal.plan is not None
    assert (proposal.plan.split, proposal.plan.split_column) == ("column", "subset")


def test_a_split_column_is_found_by_name_when_the_model_omits_it(tmp_path: Path) -> None:
    inv = inventory(_with_subset(tmp_path))
    proposal = Linker(_FakeLLM([_pick()])).propose(inv)
    assert proposal.plan is not None
    assert proposal.plan.split_column == "subset"


def test_a_split_column_without_split_values_is_refused(tmp_path: Path) -> None:
    inv = inventory(_ambiguous(tmp_path))
    proposal = Linker(_FakeLLM([_pick(split_column="region")])).propose(inv)
    assert proposal.plan is None
    assert "does not hold train and test values" in proposal.reason


def test_a_tie_is_settled_by_picking_the_table(tmp_path: Path) -> None:
    root = tmp_path
    for i in range(20):
        _png(root / "images" / f"{i}.jpg", i)
    pd.DataFrame({"id": [str(i) for i in range(20)], "label": ["a", "b"] * 10}).to_csv(
        root / "labels.csv", index=False
    )
    pd.DataFrame({"id": [str(i) for i in range(20)], "label": ["a"] * 20}).to_csv(
        root / "sample_submission.csv", index=False
    )
    inv = inventory(root)
    with pytest.raises(LinkError, match="equally well") as caught:
        plan(inv)
    assert caught.value.ambiguous is True
    proposal = Linker(
        _FakeLLM(
            [_pick(labels_table=1, key_column="id", key_to_file="stem", target_column="label")]
        )
    ).propose(inv, refusal=str(caught.value))
    assert proposal.plan is not None
    assert proposal.plan.labels_file.endswith("labels.csv")


def test_describe_reads_as_one_line(tmp_path: Path) -> None:
    inv = inventory(_with_subset(tmp_path))
    proposal = Linker(_FakeLLM([_pick(split_column="subset")])).propose(inv)
    assert proposal.plan is not None
    assert describe(proposal.plan) == (
        "table meta.csv, key 'file' read as the file name, target 'species_code', "
        "split column 'subset'"
    )


def test_the_tool_has_no_task_and_no_free_text_choice() -> None:
    fields = PROPOSE_LINK.parameters["properties"]
    assert PROPOSE_LINK.name == "propose_link"
    assert "task" not in fields
    assert fields["key_to_file"]["enum"] == ["path", "basename", "stem", "stem_int"]
    assert set(PROPOSE_LINK.parameters["required"]) == {
        "labels_table",
        "key_column",
        "key_to_file",
        "target_column",
        "why",
    }
