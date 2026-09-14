"""The Linker: one look at a folder the rules could not settle.

The rules ladder refuses a folder it cannot prove and says why. When the refusal is
a choice rather than a fault in the data (no rule fits, two tables tie, two key
columns disagree, a label table no key resolves) the Linker is shown a capped
listing of the folder: the layout, the tables numbered, each column with the share
of rows the harness could match to an image file, a few rows, and any README. It
answers with one tool call that picks a table by number and columns by name.

Nothing it says is a plan until `plan_from_choice` rebuilt the frame from it and
measured the coverage; the task comes from the label values, never from the model;
below the floor its answer is a refusal like any other. A person then says yes, no,
or what to change, and it proposes again with that note, `MAX_ROUNDS` times at most.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd

from iterate.adapters.data import linking
from iterate.adapters.data.linking import HOW, KEY_METHODS, LinkError, plan_from_choice
from iterate.prompts import PROMPTS
from iterate.schemas.llm import Message, ToolSpec

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from iterate.adapters.data.linking import Inventory
    from iterate.llm.base import LLMClient
    from iterate.schemas.link import LinkPlan

log = logging.getLogger(__name__)

_PROMPTS = PROMPTS["linker"]
MAX_ROUNDS = 5
MAX_LISTING_CHARS = 10_000
SAMPLE_ROWS = 5
COLUMNS_SHOWN = 12
TREE_DEPTH = 4
SIBLINGS_SHOWN = 12
README_CHARS = 1_500
README_NAMES = frozenset({"readme", "readme.md", "readme.txt", "about.txt", "description.txt"})


@dataclass(frozen=True)
class Listing:
    """What the Linker is shown, and the tables in the order they were numbered."""

    text: str
    tables: tuple[Path, ...]

    def table_at(self, index: object) -> Path | None:
        """The table the model's 1-based number names, or None. This is the guarantee:
        a number that is not a position in the list it was shown names nothing."""
        if isinstance(index, bool):
            return None
        if isinstance(index, str) and index.strip().isdigit():
            index = int(index.strip())
        if not isinstance(index, int):
            return None
        return self.tables[index - 1] if 1 <= index <= len(self.tables) else None


@dataclass(frozen=True)
class Proposal:
    """One answer: a measured plan or None, and the reason in either case. With a
    plan the reason is the model's one sentence; without one it is what refused it."""

    plan: LinkPlan | None
    reason: str


# ─── the listing ──────────────────────────────────────────────────────────


def _tree(inv: Inventory) -> list[str]:
    counts: dict[tuple[str, ...], list[int]] = {(): [0, 0]}
    for slot, files in ((0, inv.images), (1, inv.tables)):
        for p in files:
            parts = p.relative_to(inv.root).parent.parts
            for depth in range(len(parts) + 1):
                counts.setdefault(parts[:depth], [0, 0])[slot] += 1

    def describe(key: tuple[str, ...]) -> str:
        images, tables = counts[key]
        bits = []
        if images:
            bits.append(f"{images} images")
        if tables:
            bits.append(f"{tables} table" + ("s" if tables != 1 else ""))
        return ", ".join(bits) or "nothing the rules read"

    lines = [f"{inv.root.name}/  {describe(())}"]

    def walk(key: tuple[str, ...], depth: int) -> None:
        if depth >= TREE_DEPTH:
            return
        children = sorted(k for k in counts if len(k) == len(key) + 1 and k[: len(key)] == key)
        for child in children[:SIBLINGS_SHOWN]:
            lines.append(f"{'  ' * (depth + 1)}{child[-1]}/  {describe(child)}")
            walk(child, depth + 1)
        if len(children) > SIBLINGS_SHOWN:
            lines.append(
                f"{'  ' * (depth + 1)}... and {len(children) - SIBLINGS_SHOWN} more folders"
            )

    walk((), 0)
    return lines


def _kind(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "true/false"
    if pd.api.types.is_integer_dtype(series):
        return "integer"
    if pd.api.types.is_float_dtype(series):
        return "decimal"
    if pd.api.types.is_string_dtype(series):
        return "text"
    return "mixed"


def _table_block(inv: Inventory, number: int, table: Path, rows: int) -> str:
    rel = table.relative_to(inv.root).as_posix()
    try:
        frame = linking.read_table(table)
        rates = linking.join_rates(inv, table)
    except (OSError, ValueError) as exc:
        return f"{number}. {rel}  (could not be read: {type(exc).__name__})"
    lines = [f"{number}. {rel}  ({len(frame)} rows, {len(frame.columns)} columns)", "   columns:"]
    for c in list(frame.columns)[:COLUMNS_SHOWN]:
        series = frame[c]
        line = f"   - {str(c)!r}: {_kind(series)}, {series.nunique(dropna=True)} distinct"
        blanks = int(series.isna().sum())
        if blanks:
            line += f", {blanks} blank"
        if str(c) in rates:
            method, rate = rates[str(c)]
            line += f"; read as {HOW[method]} it finds an image for {rate:.0%} of rows"
        lines.append(line)
    if len(frame.columns) > COLUMNS_SHOWN:
        lines.append(f"   ... and {len(frame.columns) - COLUMNS_SHOWN} more columns")
    if rows:
        shown = frame.iloc[:rows, :COLUMNS_SHOWN]
        lines.append(f"   first {len(shown)} rows:")
        lines.append("   " + shown.to_csv(index=False).strip().replace("\n", "\n   "))
    return "\n".join(lines)


def _readme(root: Path) -> str | None:
    for p in sorted(root.iterdir()):
        if not (p.is_file() and p.name.lower() in README_NAMES):
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            continue
        if text:
            return text[:README_CHARS] + (" [cut]" if len(text) > README_CHARS else "")
    return None


def listing(inv: Inventory) -> Listing:
    """The folder as the Linker sees it, within `MAX_LISTING_CHARS`: fewer sample rows
    first, then a hard cut."""
    where = str(inv.root) + (f"  (inside {'/'.join(inv.collapsed)})" if inv.collapsed else "")
    head = [f"folder: {where}", "", "layout:", *_tree(inv)]
    if inv.split_pair:
        head += [
            "",
            f"split folders: {inv.split_pair[0].name}/ is train, "
            f"{inv.split_pair[1].name}/ is holdout",
        ]
    readme = _readme(inv.root)
    text = ""
    for rows in (SAMPLE_ROWS, 2, 0):
        parts = list(head)
        if inv.tables:
            parts += ["", "tables (pick one by its number):"]
            parts += [_table_block(inv, i + 1, t, rows) for i, t in enumerate(inv.tables)]
        else:
            parts += ["", "tables: none"]
        if readme:
            parts += ["", "notes found in the folder (may be wrong; the shares are measured):"]
            parts.append(readme)
        text = "\n".join(parts)
        if len(text) <= MAX_LISTING_CHARS:
            break
    if len(text) > MAX_LISTING_CHARS:
        text = text[:MAX_LISTING_CHARS] + "\n[listing cut at its budget]"
    return Listing(text=text, tables=tuple(inv.tables))


def describe(plan_: LinkPlan) -> str:
    """A plan in one line, for the person and for the model's next round."""
    if plan_.shape == "class_folders":
        return "labels from the folder names"
    table = plan_.labels_file.rsplit("/", 1)[-1] if plan_.labels_file else "?"
    how = HOW[plan_.key_to_file or "path"]
    out = f"table {table}, key {plan_.key_column!r} read as {how}, target {plan_.target_column!r}"
    if plan_.split_column:
        out += f", split column {plan_.split_column!r}"
    return out


# ─── the tool ─────────────────────────────────────────────────────────────


def _tool() -> ToolSpec:
    spec = _PROMPTS["tool"]
    fields = spec["fields"]
    return ToolSpec(
        name=spec["name"],
        description=spec["description"],
        parameters={
            "type": "object",
            "properties": {
                "labels_table": {"type": "integer", "description": fields["labels_table"]},
                "key_column": {"type": "string", "description": fields["key_column"]},
                "key_to_file": {
                    "type": "string",
                    "enum": list(KEY_METHODS),
                    "description": fields["key_to_file"],
                },
                "target_column": {"type": "string", "description": fields["target_column"]},
                "split_column": {"type": "string", "description": fields["split_column"]},
                "why": {"type": "string", "description": fields["why"]},
            },
            "required": ["labels_table", "key_column", "key_to_file", "target_column", "why"],
        },
    )


PROPOSE_LINK = _tool()


def _text(value: object) -> str:
    return "" if value is None else str(value).strip()


class Linker:
    """Proposes how a folder links, by picking from what it was shown. Never decides:
    every proposal is measured before a person sees it."""

    def __init__(
        self, client: LLMClient, *, temperature: float = 0.0, max_tokens: int = 512
    ) -> None:
        self._client = client
        self._temperature = temperature
        self._max_tokens = max_tokens

    def propose(
        self,
        inv: Inventory,
        *,
        refusal: str = "",
        notes: Sequence[str] = (),
        previous: LinkPlan | None = None,
    ) -> Proposal:
        """One proposal for this folder. Never raises into the run."""
        shown = listing(inv)
        messages = [
            Message(role="system", content=_PROMPTS["system"]),
            Message(
                role="user",
                content=_PROMPTS["user_template"]
                .replace("{listing}", shown.text)
                .replace("{refusal}", refusal or "(not given)")
                .replace("{previous}", describe(previous) if previous else "(none)")
                .replace("{notes}", "\n".join(f"- {n}" for n in notes) or "(none)"),
            ),
        ]
        try:
            args = self._call(messages, PROPOSE_LINK)
        except Exception as exc:
            log.info("linker: the call failed (%s: %s)", type(exc).__name__, exc)
            return Proposal(None, f"the model call failed ({type(exc).__name__})")
        if not args:
            return Proposal(None, "the model gave no answer in the tool's shape")
        table = shown.table_at(args.get("labels_table"))
        if table is None:
            return Proposal(
                None, f"the model picked table {args.get('labels_table')!r}, which was not listed"
            )
        try:
            plan_ = plan_from_choice(
                inv,
                table=table,
                key_column=_text(args.get("key_column")),
                key_to_file=_text(args.get("key_to_file")),
                target_column=_text(args.get("target_column")),
                split_column=_text(args.get("split_column")) or None,
                source="agent",
            )
        except (LinkError, OSError, ValueError) as exc:
            log.info("linker: proposal refused (%s)", exc)
            return Proposal(None, str(exc))
        return Proposal(plan_, " ".join(_text(args.get("why")).split())[:240])

    def _call(self, messages: list[Message], tool: ToolSpec) -> dict[str, Any] | None:
        """One structured call with a single retry nudge, the Researcher's shape."""
        for attempt in range(2):
            response = self._client.chat(
                messages,
                tools=[tool],
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
            call = next((c for c in response.tool_calls if c.name == tool.name), None)
            if call is not None:
                return dict(call.arguments)
            if attempt == 0:
                messages = [*messages, Message(role="user", content=_PROMPTS["retry_nudge"])]
        return None


__all__ = ["MAX_ROUNDS", "PROPOSE_LINK", "Linker", "Listing", "Proposal", "describe", "listing"]
