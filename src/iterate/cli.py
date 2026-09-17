"""CLI entry point.

`iterate version` / `iterate config` / `iterate setup` / `iterate run`. The `run`
command wires the agent end-to-end: load data → build target → build LLM client →
reconstruct baseline from `--source` if given → loop via the Orchestrator → render
a summary. By default the agent WRITES model code (`--code`) and runs it locally
(`--compute local`); `iterate setup` saves a user's preferred defaults so they
don't repeat flags.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import signal
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.markup import escape
from rich.table import Table

from iterate import __version__, userconfig
from iterate.config import get_settings

# NOTE: the heavy stack (pandas, scikit-learn, joblib, the orchestrator/model
# chain) is imported lazily inside `run()` — not at module load — so `iterate
# version` / `config` / `--help` stay instant instead of paying the ~2-3s
# sklearn+pandas import cost on every invocation.

if TYPE_CHECKING:
    from collections.abc import Callable

    from iterate.adapters.data.images import OutsideDataError
    from iterate.adapters.data.linking import Inventory, LinkedFrames, LinkError
    from iterate.adapters.data.workspace import Sides
    from iterate.core.interactive import RunController
    from iterate.core.linker import Linker
    from iterate.core.memory import Memory
    from iterate.core.orchestrator import RunResult
    from iterate.schemas.experiment import Candidate, Experiment, ExperimentResult
    from iterate.schemas.link import LinkPlan
    from iterate.schemas.monitor import DataReport
    from iterate.targets.model import ModelTarget

app = typer.Typer(
    name="iterate",
    help=(
        "Autonomous research-aware iteration agent for ML models and LLM prompts. "
        "See https://github.com/anthonyrodrigues443/iterate"
    ),
    no_args_is_help=True,
)

console = Console()


@app.callback()
def _root() -> None:
    """Root callback — its presence keeps subcommands as real subcommands."""


@app.command()
def version() -> None:
    """Print the iterate version."""
    typer.echo(f"iterate {__version__}")


@app.command()
def config() -> None:
    """Show the resolved configuration (the backend api-key is masked)."""
    settings = get_settings()
    typer.echo(f"model:        {settings.iterate_model}")
    typer.echo(f"backend_url:  {settings.iterate_backend_url}")
    typer.echo(f"api_key:      {_mask(settings.iterate_backend_api_key)}")
    typer.echo(f"timeout:      {settings.iterate_backend_timeout}s")
    typer.echo(f"ollama_host:  {settings.ollama_host}")
    typer.echo(f"memory_db:    {settings.iterate_memory_db}")


@app.command()
def setup() -> None:
    """Save your default backend, model, keys, compute venue, and install consent.

    Writes ~/.config/iterate/config.toml. Override any of it per run with a flag.
    """
    console.print(
        "[bold]iterate setup[/bold] — your saved defaults "
        "(override any of these per run with a flag).\n"
    )
    backend = typer.prompt(
        "LLM backend (ollama / openai / groq / together / deepseek)", default="ollama"
    ).strip()
    model = typer.prompt("Model name (blank = backend default)", default="", show_default=False)
    api_key = ""
    if backend != "ollama":
        api_key = typer.prompt(
            f"API key for {backend}", default="", hide_input=True, show_default=False
        ).strip()

    compute = typer.prompt("Run generated code on (local / e2b)", default="local").strip().lower()
    e2b_api_key = ""
    install = False
    if compute == "e2b":
        e2b_api_key = typer.prompt(
            "E2B API key", default="", hide_input=True, show_default=False
        ).strip()
    else:
        console.print(
            "[dim]On local, generated code runs on THIS machine with your permissions.[/dim]"
        )
        install = typer.confirm(
            "May iterate install packages your generated code imports into iterate's "
            "environment? It never changes torch, and an install that would change what "
            "iterate is running on waits for the next run.",
            default=False,
        )

    values: dict[str, object] = {"backend": backend, "compute": compute, "install": install}
    if model.strip():
        values["model"] = model.strip()
    if api_key:
        values["api_key"] = api_key
    if e2b_api_key:
        values["e2b_api_key"] = e2b_api_key
    path = userconfig.save_user_config(values)
    console.print(f"\n[green]saved[/green] → {path}")


class _NoPlanError(Exception):
    """A correction the Linker could not turn into a plan; the message says why."""


def _transfer(hint: LinkPlan, inv: Inventory) -> LinkPlan | None:
    """The other folder's choice tried on this one: the table with the same name, or
    the only table, read with the same columns. Measured like any plan; None if it
    does not hold here."""
    from iterate.adapters.data import linking

    if hint.shape == "class_folders" or not (hint.labels_file and hint.target_column):
        return None
    name = Path(hint.labels_file).name
    table = next((t for t in inv.tables if t.name == name), None)
    if table is None and len(inv.tables) == 1:
        table = inv.tables[0]
    if table is None:
        return None
    try:
        moved = linking.plan_from_choice(
            inv,
            table=table,
            key_column=str(hint.key_column),
            key_to_file=str(hint.key_to_file),
            target_column=hint.target_column,
            split_column=hint.split_column,
            source=hint.source,
        )
    except linking.LinkError:
        return None
    note = f"columns taken from the other folder's {name}"
    return moved.model_copy(update={"notes": [*moved.notes, note]})


def _refused(exc: LinkError | OutsideDataError) -> typer.BadParameter:
    """A refusal as a usage error. A copy command goes first, on a line of its own:
    the error panel wraps at the terminal width, and a wrapped command does not paste."""
    if exc.command is None:
        return typer.BadParameter(str(exc))
    console.print(exc.command, soft_wrap=True, highlight=False, markup=False)
    return typer.BadParameter(f"{exc.reason}. The copy command is printed above")


def _side(
    inv: Inventory,
    *,
    labels: Path | None,
    key: str | None,
    target: str | None,
    linker: Callable[[], Linker],
    hint: LinkPlan | None,
) -> tuple[LinkPlan, str | None, str]:
    """One folder's first plan: rules, then the other folder's choice, then the
    Linker, and only on a refusal a choice would settle. Returns the plan, the
    Linker's sentence when it spoke, and the rules' refusal when they did."""
    from iterate.adapters.data import linking

    try:
        return linking.plan(inv, labels=labels, key=key, target=target), None, ""
    except linking.LinkError as exc:
        if not exc.ambiguous or labels is not None:
            raise _refused(exc) from exc
        refusal = str(exc)
    if hint is not None:
        moved = _transfer(hint, inv)
        if moved is not None:
            return moved, None, refusal
    proposal = linker().propose(inv, refusal=refusal, previous=hint)
    if proposal.plan is None:
        raise typer.BadParameter(
            f"{refusal}\nthe Linker could not settle it either: {proposal.reason}"
        )
    return proposal.plan, proposal.reason, refusal


def _corrected(
    inventories: list[Inventory],
    plans: list[LinkPlan],
    refusals: list[str],
    *,
    notes: list[str],
    linker: Callable[[], Linker],
) -> tuple[list[LinkPlan], list[str | None]]:
    """The plans after a correction. Only a folder the rules refused is the model's to
    change: the first such folder gets the note; a second one takes that choice when
    it holds there, or gets its own proposal with the same note."""
    out = list(plans)
    whys: list[str | None] = [None] * len(plans)
    changed_first = False
    for i, (inv, refusal) in enumerate(zip(inventories, refusals, strict=True)):
        if not refusal:
            continue
        follows_first = i == 1 and changed_first
        if follows_first:
            moved = _transfer(out[0], inv)
            if moved is not None:
                out[i] = moved
                continue
        hint = out[0] if follows_first else plans[i]
        proposal = linker().propose(inv, refusal=refusal, notes=notes, previous=hint)
        if proposal.plan is None:
            raise _NoPlanError(proposal.reason)
        out[i], whys[i] = proposal.plan, proposal.reason
        changed_first = changed_first or i == 0
    return out, whys


def _combined(plans: list[LinkPlan], inventories: list[Inventory]) -> tuple[LinkedFrames, LinkPlan]:
    """The frames the plans build, and the one plan shown for them: for two folders
    the first plan carrying the user's split and the lower coverage."""
    from iterate.adapters.data import linking

    if len(plans) == 1:
        return linking.apply(plans[0], inventories[0]), plans[0]
    if plans[0].task != plans[1].task:
        raise linking.LinkError(
            f"--train reads as {plans[0].task} and --holdout as {plans[1].task}"
        )
    for plan_, flag in zip(plans, ("--train", "--holdout"), strict=True):
        if plan_.split == "column":
            raise linking.LinkError(
                f"the folders are your split, but the table under {flag} also names one in "
                f"column {plan_.split_column!r}; drop that column, or pass the parent folder as "
                "--data and let the column decide"
            )
    frames = linking.LinkedFrames(
        linking.apply(plans[0], inventories[0]).train,
        linking.apply(plans[1], inventories[1]).train,
    )
    notes = [*plans[0].notes, *(n for n in plans[1].notes if n not in plans[0].notes)]
    shown = plans[0].model_copy(
        update={
            "split": "folders",
            "coverage": min(p.coverage for p in plans),
            "source": "agent" if any(p.source == "agent" for p in plans) else plans[0].source,
            "notes": notes,
        }
    )
    return frames, shown


def _revalidated(remembered: list[LinkPlan], inventories: list[Inventory]) -> list[LinkPlan] | None:
    """Remembered plans measured again on the folder as it is now; None if any no
    longer holds, so a stale memory can never skip the rules."""
    from iterate.adapters.data import linking

    out: list[LinkPlan] = []
    for plan_, inv in zip(remembered, inventories, strict=True):
        try:
            if plan_.source != "agent":
                fresh = linking.plan(inv)
                if fresh != plan_:
                    return None
                out.append(fresh)
                continue
            if not (plan_.labels_file and plan_.target_column):
                return None
            out.append(
                linking.plan_from_choice(
                    inv,
                    table=Path(plan_.labels_file),
                    key_column=str(plan_.key_column),
                    key_to_file=str(plan_.key_to_file),
                    target_column=plan_.target_column,
                    split_column=plan_.split_column,
                    source=plan_.source,
                )
            )
        except linking.LinkError:
            return None
    return out


def _link_folder(
    *,
    data: Path | None,
    train: Path | None,
    holdout: Path | None,
    labels: Path | None,
    key: str | None,
    target: str | None,
    yes: bool,
    make_linker: Callable[[], Linker],
) -> Any:
    """Link a folder of images to its labels before the run: rules first, a plan
    remembered from an earlier yes, the Linker where the rules could only say "one
    of these", then the block a person reads with the monitor's checks under it,
    on the rows the folder will be written with. The pause is a conversation: yes,
    no, drop to take byte twins out of a holdout the person gave, or what to change
    in plain English, `MAX_ROUNDS` corrections at most. Nothing the Linker says is
    shown before it was measured. Returns the workspace."""
    from iterate.adapters.data import linking, monitor, workspace
    from iterate.core.linker import MAX_ROUNDS

    if (key or target) and labels is None:
        raise typer.BadParameter("--key and --target describe --labels; pass --labels too")
    if labels is not None and data is None:
        raise typer.BadParameter(
            "--labels goes with --data: put both folders under one folder as train/ and "
            "test/ with the table beside them, or give each folder its own table"
        )
    sources = [data] if data is not None else [cast("Path", train), cast("Path", holdout)]
    out_root = Path(get_settings().iterate_runs_dir).parent / "data"
    built: list[Linker] = []
    checker = monitor.Monitor()

    def linker() -> Linker:
        if not built:
            built.append(make_linker())
        return built[0]

    def split(shown_: LinkPlan, frames_: LinkedFrames) -> tuple[Sides | None, str]:
        """The split the pause checks and the writer lays out, or why there is none."""
        try:
            hashes = (
                None
                if frames_.holdout is not None
                else checker.hashes(map(str, frames_.train["image"]))
            )
            return workspace.sides(shown_, frames_, hashes=hashes), ""
        except linking.LinkError as exc:
            return None, str(exc)

    try:
        inventories = [linking.inventory(s) for s in sources]
    except linking.LinkError as exc:
        raise _refused(exc) from exc

    remembered = None if labels is not None else workspace.recall_plan(sources, out=out_root)
    plans = _revalidated(remembered, inventories) if remembered else None
    recalled = plans is not None
    whys: list[str | None] = []
    refusals: list[str] = []
    if plans is None:
        if remembered:
            workspace.forget_plan(sources, out=out_root)
        plans = []
        for inv in inventories:
            plan_, why, refusal = _side(
                inv,
                labels=labels,
                key=key,
                target=target,
                linker=linker,
                hint=plans[0] if plans else None,
            )
            plans.append(plan_)
            whys.append(why)
            refusals.append(refusal)
    try:
        frames, shown = _combined(plans, inventories)
    except linking.LinkError as exc:
        raise _refused(exc) from exc

    both, refused_split = split(shown, frames)
    if recalled and both is not None and workspace.recall_drop(sources, out=out_root):
        # The earlier yes came with a drop of the twins; the same folders get the same.
        holdout = both.frames.holdout
        assert holdout is not None  # sides
        paths = map(str, (*both.frames.train["image"], *holdout["image"]))
        with contextlib.suppress(linking.LinkError):
            kept, gone, gone_labels = monitor.drop_twins(both.frames, checker.hashes(paths))
            if gone:
                both = workspace.Sides(kept, gone, gone_labels)
                console.print(
                    f"[dim]dropped {len(gone)} holdout images that are byte copies of a "
                    "training image, as before[/dim]"
                )
    report: DataReport | None = None
    rounds = 0
    notes: list[str] = []
    while True:
        console.print(
            escape(
                linking.render(shown, inventories[0], frames)
                if both is None
                else linking.render(shown, inventories[0], both.frames, dropped=len(both.dropped))
            )
        )
        for why in whys:
            if why:
                console.print(f"[dim]the Linker: {escape(why)}[/dim]")
        if both is None:
            report = None
            console.print(f"[dim]{escape(refused_split)}[/dim]")
        else:
            report = checker.check(plans, inventories, both)
            console.print(escape(report.render()))
        if recalled:
            console.print(
                "[dim]remembered from an earlier yes; delete "
                f"{escape(str(workspace.plan_path(sources, out=out_root)))} to link afresh[/dim]"
            )
        if both is None and (recalled or not any(refusals)):
            # Nothing a person could say here would change the split; only the data can.
            raise typer.BadParameter(refused_split)
        proven = shown.source != "agent" and shown.coverage >= linking.ACCEPT and not any(refusals)
        settled = (proven or recalled) and report is not None and not report.needs_a_look
        if yes or settled:
            if both is None:
                raise typer.BadParameter(refused_split)
            break
        if not _stdin_owns_tty():
            if both is None:
                raise typer.BadParameter(refused_split)
            if any(refusals):
                why_ask = (
                    "the rules could not settle this folder, so the plan shown was proposed for it"
                )
            elif shown.coverage < linking.ACCEPT:
                why_ask = f"coverage is {shown.coverage:.1%}, under {linking.ACCEPT:.0%}"
            else:
                why_ask = "the checks found something worth a look"
            raise typer.BadParameter(
                f"{why_ask}; re-run with --yes to accept it, or settle it with --labels, --key "
                "and --target"
            )
        twins = None
        if report is not None:
            twins = next(
                (f for f in report.findings if f.check == "twins" and f.severity == "warn"), None
            )
        ask = "yes to continue, no to stop, or say what to change"
        if twins is not None:
            ask = (
                "yes to continue, no to stop, drop to take the twins out of your holdout, or say "
                "what to change"
            )
        answer = ""
        while not answer:
            answer = typer.prompt(ask, default="", show_default=False).strip()
        word = answer.lower().rstrip(" .!")
        if not word:
            continue
        if word in ("y", "yes"):
            if both is None:
                raise typer.BadParameter(refused_split)
            break
        if word in ("n", "no"):
            raise typer.Exit(code=1)
        if re.fullmatch(r"(drop|remove)( (them|it|twins|the twins|copies|the copies))?", word):
            if both is None or twins is None:
                console.print(
                    "[dim]nothing to drop: no holdout image is a byte copy of a training image[/dim]"
                )
                continue
            holdout = both.frames.holdout
            assert holdout is not None  # sides
            paths = map(str, (*both.frames.train["image"], *holdout["image"]))
            try:
                kept, gone, gone_labels = monitor.drop_twins(both.frames, checker.hashes(paths))
            except linking.LinkError as exc:
                console.print(f"[dim]{escape(str(exc))}[/dim]")
                continue
            both = workspace.Sides(
                kept, [*both.dropped, *gone], [*both.dropped_labels, *gone_labels]
            )
            continue
        if not any(refusals) or recalled:
            # The rules proved every folder here; a change of mind is a job for the flags,
            # not for the model.
            console.print(
                "[dim]this plan came from the rules; answer yes or no, or settle it with "
                "--labels, --key and --target[/dim]"
            )
            continue
        if rounds >= MAX_ROUNDS:
            raise typer.BadParameter(
                f"{MAX_ROUNDS} corrections and no plan you accepted; settle it with --labels, "
                "--key and --target"
            )
        rounds += 1
        notes.append(answer)
        try:
            new_plans, new_whys = _corrected(
                inventories, plans, refusals, notes=notes, linker=linker
            )
            new_frames, new_shown = _combined(new_plans, inventories)
        except (_NoPlanError, linking.LinkError) as exc:
            console.print(
                f"[dim]the Linker could not turn that into a plan: {escape(str(exc))}; say it "
                "another way, or stop with no[/dim]"
            )
            continue
        frames, shown, plans, whys = new_frames, new_shown, new_plans, new_whys
        both, refused_split = split(shown, frames)

    assert both is not None  # every break above has a split
    assert report is not None
    try:
        ws = workspace.write(shown, frames, sources=sources, out=out_root, both=both)
    except linking.LinkError as exc:
        raise _refused(exc) from exc
    monitor.save(report, ws.root)
    if any(p.source == "agent" for p in plans) and not recalled:
        asked_to_drop = bool(both.dropped) and shown.split != "ours"
        workspace.remember_plan(plans, sources=sources, out=out_root, drop_twins=asked_to_drop)
    n_train = ws.train_csv.read_text(encoding="utf-8").count("\n") - 1
    n_holdout = ws.holdout_csv.read_text(encoding="utf-8").count("\n") - 1
    found = sum(f.severity != "pass" for f in report.findings)
    dropped = f", {len(both.dropped)} holdout images dropped" if both.dropped else ""
    console.print(
        f"\nlinked: {escape(str(ws.root))}\n  raw_files/  a copy of what you gave\n"
        f"  train/      {n_train} images\n  holdout/    {n_holdout} images, sealed\n"
        f"  train.csv, holdout.csv, link.json\n"
        f"  monitor.json  {found} finding(s){dropped}"
    )
    return ws


def _install_saved_packages(*, install: bool | None, compute: str | None) -> None:
    from iterate.adapters.compute import deps

    cfg = userconfig.load_user_config()
    venue = (compute or cfg.get("compute") or "local").lower()
    if venue != "local":
        return
    consent = install if install is not None else bool(cfg.get("install", False))
    installer = deps.Installer(pending=deps.pending_path())
    waiting = installer.pending()
    if not waiting:
        return
    if not consent:
        names = ", ".join(sorted(e["package"] for e in waiting))
        console.print(
            f"[dim]installs: {escape(names)} saved by an earlier run, waiting for --install[/dim]"
        )
        return
    for line in installer.install_pending():
        console.print(f"[dim]installs: {escape(line)}[/dim]")


@app.command()
def run(
    data: Path | None = typer.Option(
        None,
        "--data",
        help="Your CSV, or a folder of images with their labels inside laid out however "
        "they came. Split here, 80/20, stratified on a classification target, and the "
        "holdout is sealed. To bring your own split, pass --train and --holdout instead.",
        exists=True,
    ),
    train: Path | None = typer.Option(
        None,
        "--train",
        help="Your training CSV or folder, when you split the data yourself. Needs "
        "--holdout; cannot be combined with --data.",
        exists=True,
    ),
    holdout: Path | None = typer.Option(
        None,
        "--holdout",
        help="Your holdout CSV or folder, sealed exactly as given: its labels never enter "
        "the kernel and nothing is reshuffled. Needs --train.",
        exists=True,
    ),
    target: str | None = typer.Option(
        None,
        "--target",
        help="Name of the target column. Not needed for a folder of images, where the "
        "label is found for you; with --labels it names that table's label column.",
    ),
    labels: Path | None = typer.Option(
        None,
        "--labels",
        help="Folders only: the table that holds the labels, when the rules should not pick one.",
        exists=True,
        dir_okay=False,
    ),
    key: str | None = typer.Option(
        None, "--key", help="Folders only: the column in --labels that names each image."
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        help="Folders only: accept a link the rules could not fully prove, or one the "
        "Linker proposed, without asking. Every such plan was measured first.",
    ),
    metric: str | None = typer.Option(
        None,
        "--metric",
        help="Primary metric: f1 | accuracy | roc_auc | average_precision | rmse | r2 | … "
        "Omit it and the agent picks one from your data profile and the literature, "
        "then states why. An explicit choice always wins.",
    ),
    task: str | None = typer.Option(
        None,
        "--task",
        help='One line describing the job, e.g. "say whether the comment is toxic". '
        "Supplying it switches the run to PROMPT iteration: the agent writes and "
        "improves a prompt instead of a model, scored on the same sealed holdout.",
    ),
    prompt_file: Path | None = typer.Option(
        None,
        "--prompt-file",
        help="Your current production prompt (txt/md, or yaml with system + user_template). "
        "Becomes the baseline the agent has to beat. Omit it and a minimal prompt is "
        "built from --task.",
        exists=True,
        dir_okay=False,
    ),
    target_model: str | None = typer.Option(
        None,
        "--target-model",
        help="The model whose prompt is being tuned. Separate from the model DRIVING "
        "the run (--model). Defaults to the same one.",
    ),
    target_backend: str | None = typer.Option(
        None, "--target-backend", help="Backend for the model under test. Defaults to --backend."
    ),
    allow_free_text: bool = typer.Option(
        False,
        "--allow-free-text",
        help="Prompt runs only: score a free-text answer column on EXACT string "
        "matches. Off by default because a correct answer worded differently scores "
        "as wrong, which produces a confident and meaningless zero.",
    ),
    loop_holdout: int = typer.Option(
        100,
        "--loop-holdout",
        min=0,
        help="Prompt runs only: how many holdout records each candidate is scored on "
        "DURING the search. The same records every time, so the comparison stays "
        "paired; the winner is then re-scored on the whole holdout at the end. 0 uses "
        "the full holdout throughout (slower, and no more reliable for ranking).",
    ),
    average: str | None = typer.Option(
        None,
        "--average",
        help="Averaging for f1/precision/recall: binary | micro | macro | weighted. "
        "Default adapts to the target (binary on two classes, macro otherwise).",
    ),
    research: bool = typer.Option(
        True,
        "--research/--no-research",
        help="Ground briefs in retrievable literature (OpenAlex + arXiv, no API key). "
        "Cached on disk; --no-research skips it entirely for offline or fast runs.",
    ),
    critique: bool = typer.Option(
        True,
        "--critique/--no-critique",
        help="Review each experiment for leakage and for gains that look like luck. "
        "A proven leak stops that experiment banking as the run's best.",
    ),
    baseline: float | None = typer.Option(
        None, "--baseline", help="Your reported baseline score (sanity check; requires --source)."
    ),
    source: Path | None = typer.Option(
        None,
        "--source",
        help="md / txt / .py / .ipynb describing the user's approach (read as TEXT, never executed).",
        exists=True,
        dir_okay=False,
    ),
    backend: str | None = typer.Option(
        None, "--backend", help="ollama | openai-compatible (or aliases: groq, together, …)"
    ),
    model: str | None = typer.Option(None, "--model", help="Override the backend's default model."),
    base_url: str | None = typer.Option(
        None, "--base-url", help="Override the backend's base URL."
    ),
    api_key: str | None = typer.Option(None, "--api-key", help="Override the backend's API key."),
    think: bool = typer.Option(
        False,
        "--think/--no-think",
        help="Enable thinking mode for the CODING agent (ollama only; slower per call). "
        "The supervisor always runs with thinking off — its reply must be a single "
        "tool call, and a long thinking trace crowds that call out.",
    ),
    code: bool = typer.Option(
        True,
        "--code/--spec",
        help="Agent WRITES model code (default), or picks an allow-listed estimator (--spec).",
    ),
    compute: str | None = typer.Option(
        None, "--compute", help="Where generated code runs: local (default) | e2b."
    ),
    install: bool | None = typer.Option(
        None,
        "--install/--no-install",
        help="Let iterate install packages a session imports (local; e2b always installs in "
        "its sandbox). Cells never install, and on local runs torch and torchvision never "
        "change mid-run.",
    ),
    max_iterations: int = typer.Option(
        10, "--max-iterations", min=1, help="Hard cap on experiments."
    ),
    patience: int = typer.Option(
        3, "--patience", min=1, help="Stop after N consecutive non-improvements."
    ),
    until: str | None = typer.Option(
        None, "--until", help='Wall-clock deadline, e.g. "30m" or "2h".'
    ),
    fresh: bool = typer.Option(
        False,
        "--fresh",
        help="Archive the existing memory db and start a new chapter with the factory default baseline.",
    ),
    memory_path: Path | None = typer.Option(None, "--memory", help="Override the memory db path."),
    output: Path | None = typer.Option(
        None,
        "--output",
        help="Where to save the best model. Default: .iterate/runs/<run_id>/best_model.joblib",
    ),
    notebooks: str = typer.Option(
        "best",
        "--notebooks",
        help=(
            "Runnable .ipynb deliverable: best (winner only) | all (one per experiment) | none. "
            "Saved under .iterate/runs/<run_id>/ (best.ipynb; all → notebooks/)."
        ),
    ),
    plain: bool = typer.Option(
        False,
        "--plain",
        help=(
            "Plain terminal output instead of the interactive UI (log pane + input box). "
            "Chat still works on a terminal: type a line and press Enter. Non-tty runs "
            "are always plain and non-interactive."
        ),
    ),
) -> None:
    """Run the agent on a tabular dataset."""
    # An install saved by the last run may move numpy, pandas or scikit-learn, which
    # this process must not have loaded yet.
    _install_saved_packages(install=install, compute=compute)
    # Heavy imports live here, not at module top, so `iterate version`/`--help`
    # don't pay the pandas + scikit-learn import cost.
    from iterate.adapters.compute.kernel import E2BKernel, LocalKernel, StatefulKernel
    from iterate.adapters.compute.local import LocalExecutor
    from iterate.adapters.data.tabular import load_csv, load_split
    from iterate.core import codegen
    from iterate.core.agent_loop import run_supervised
    from iterate.core.coder import CodingAgent
    from iterate.core.critic import Critic
    from iterate.core.memory import SqliteMemory
    from iterate.core.orchestrator import Orchestrator
    from iterate.core.proposer import Proposer, summarize_dataset
    from iterate.core.reconstructor import Reconstructor
    from iterate.core.researcher import Researcher
    from iterate.core.scoring import AVERAGES, CLASSIFICATION_METRICS, REGRESSION_METRICS
    from iterate.core.scoring import direction as metric_direction
    from iterate.core.summarizer import Summarizer
    from iterate.core.supervisor import Supervisor
    from iterate.core.terminator import default_terminator
    from iterate.llm.factory import api_key_for, build_client
    from iterate.targets.model import ModelTarget

    # ─── One input split here, or two inputs the user split ────────────────
    if data is None and (train is None or holdout is None):
        raise typer.BadParameter(
            "pass --data for a dataset split here, or both --train and --holdout for a "
            "split you made yourself"
        )
    if data is not None and (train is not None or holdout is not None):
        raise typer.BadParameter(
            "--data is split here; --train and --holdout are your own split. Pass one "
            "form, not both"
        )
    if data is None and not code:
        raise typer.BadParameter(
            "--train and --holdout are not supported on the --spec fast lane; pass --data"
        )
    if train is not None and holdout is not None and train.resolve() == holdout.resolve():
        raise typer.BadParameter("--train and --holdout are the same file")

    given = [p for p in (data, train, holdout) if p is not None]
    folders = [p for p in given if p.is_dir()]
    if folders and len(folders) != len(given):
        raise typer.BadParameter("give folders for every input, or files for every input")
    if not folders:
        if target is None:
            raise typer.BadParameter("--target is required for a CSV")
        if labels is not None or key is not None or yes:
            raise typer.BadParameter("--labels, --key and --yes describe a folder of images")

    # ─── First run with no saved config? Offer the setup wizard. ───────────
    if not userconfig.exists() and sys.stdin.isatty():
        console.print("[dim]No saved config found — let's set your defaults once.[/dim]\n")
        setup()
        console.print()
    cfg = userconfig.load_user_config()

    # ─── Resolve: explicit flag > saved config > built-in default ──────────
    backend = backend or cfg.get("backend") or "ollama"
    model = model or cfg.get("model")
    compute = (compute or cfg.get("compute") or "local").lower()
    install = install if install is not None else bool(cfg.get("install", False))
    if compute not in ("local", "e2b"):
        raise typer.BadParameter(f"--compute must be 'local' or 'e2b', got {compute!r}")
    notebooks = notebooks.lower()
    if notebooks not in ("best", "all", "none"):
        raise typer.BadParameter(f"--notebooks must be best | all | none, got {notebooks!r}")

    # ─── Cloud backend? API key required wherever a client is built. ───────
    settings = get_settings()
    if backend != "ollama":
        api_key = api_key or cfg.get("api_key") or _resolved_api_key_from_env(settings, backend)

    def _need_key() -> None:
        if backend != "ollama" and not api_key:
            raise typer.BadParameter(
                f"backend {backend!r} requires --api-key or a corresponding env var "
                f"(ITERATE_BACKEND_API_KEY / OPENAI_API_KEY / GROQ_API_KEY / …)"
            )

    # ─── A folder of images is linked first and shown before anything runs ──
    # Rules alone where they can prove it; the Linker only where they could not,
    # so a rules-only folder never builds a client and needs no key.
    if folders:
        from iterate.core.linker import Linker

        def _make_linker() -> Linker:
            _need_key()
            return Linker(build_client(backend, model=model, base_url=base_url, api_key=api_key))

        ws = _link_folder(
            data=data,
            train=train,
            holdout=holdout,
            labels=labels,
            key=key,
            target=target,
            yes=yes,
            make_linker=_make_linker,
        )
        console.print(
            "[dim]the folder is ready; the vision target that runs on it lands later in "
            f"v0.6, so this run stops here. Its CSVs: {ws.train_csv} and {ws.holdout_csv}[/dim]"
        )
        raise typer.Exit(code=0)
    assert target is not None  # a CSV run was checked above
    _need_key()

    # ─── Validate ──────────────────────────────────────────────────────────
    if baseline is not None and source is None:
        raise typer.BadParameter("--baseline requires --source")
    if metric is not None:
        metric = metric.lower()
        if metric not in CLASSIFICATION_METRICS and metric not in REGRESSION_METRICS:
            raise typer.BadParameter(
                f"unknown metric {metric!r}; expected one of "
                f"{sorted(CLASSIFICATION_METRICS | REGRESSION_METRICS)}"
            )
    if average is not None:
        average = average.lower()
        if average not in AVERAGES:
            raise typer.BadParameter(
                f"unknown average {average!r}; expected one of {list(AVERAGES)}"
            )

    resolved_memory_path = memory_path or Path(settings.iterate_memory_db)

    # ─── New chapter? Archive the existing db. ─────────────────────────────
    # Any of --fresh, --source, --baseline+--source means "new chapter."
    starting_new_chapter = fresh or source is not None
    if starting_new_chapter:
        archived = _archive_memory_db(resolved_memory_path)
        if archived is not None:
            console.print(
                f"[dim]memory: archived [/dim]{resolved_memory_path}[dim] → "
                f"[/dim]{archived.name}[dim]; starting fresh[/dim]"
            )

    # ─── e2b compute? API key required. ────────────────────────────────────
    e2b_api_key = cfg.get("e2b_api_key") or settings.e2b_api_key
    if compute == "e2b" and not e2b_api_key:
        raise typer.BadParameter(
            "--compute e2b needs an E2B API key — run 'iterate setup' or set E2B_API_KEY "
            "(get a free key at e2b.dev)."
        )

    # The TUI installs its own log handler; a stdout handler would vanish into the
    # alternate screen. Interactivity needs FOREGROUND tty ownership: a backgrounded
    # job reading its tty gets SIGTTIN and the whole process is suspended.
    interactive_tty = _stdin_owns_tty()
    use_tui = code and not plain and interactive_tty and sys.stdout.isatty()
    if not use_tui:
        _configure_logging()

    # ─── Load data ─────────────────────────────────────────────────────────
    # An explicit metric names the task; only without one does the loader guess.
    from iterate.adapters.data.tabular import describe_target
    from iterate.core.scoring import task_for_metric as _task_for_metric

    named_task = _task_for_metric(metric) if metric is not None else None
    if data is not None:
        dataset = load_csv(data, target=target, task=named_task)
    else:
        assert train is not None  # validated above
        assert holdout is not None
        dataset = load_split(train, holdout, target=target, task=named_task)
        console.print(
            f"[dim]your split: {dataset.n_train} train rows, {dataset.n_test} holdout rows, "
            "sealed as given[/dim]"
        )
    if named_task is None:
        console.print(
            f"[dim]target {target!r} read as {dataset.task} "
            f"({describe_target(dataset.train_target)}); pass --metric to override[/dim]"
        )
    from iterate.adapters.data.images import OutsideDataError, image_column

    try:
        found = image_column(dataset)
    except OutsideDataError as exc:
        raise _refused(exc) from exc
    if found is not None:
        console.print(
            "[dim]this CSV holds image paths; the vision target that runs on it lands "
            "later in v0.6, so this run stops here[/dim]"
        )
        raise typer.Exit(code=0)
    data_summary = summarize_dataset(dataset)

    # ─── LLM clients + memory ──────────────────────────────────────────────
    # Two clients on purpose: thinking applies to the CODER only. The supervisor
    # (and every other tool-only role) must answer with a single structured tool
    # call, and a thinking trace eats the reply budget before the call is emitted —
    # observed live: every supervisor turn failed with "no plan" under thinking.
    client = build_client(backend, model=model, base_url=base_url, api_key=api_key)
    coder_client = (
        build_client(backend, model=model, base_url=base_url, api_key=api_key, think=True)
        if think
        else client
    )
    # The metric is resolved BEFORE the target is built, since every role is
    # constructed around it, and then fixed for the run.
    run_setup = _resolve_setup(
        dataset=dataset,
        explicit=metric,
        data_summary=data_summary,
        client=client,
        enabled=research and metric is None,
    )
    metric = run_setup.metric
    direction = metric_direction(metric)
    is_prompt_run = task is not None
    # ABSOLUTE: the kernel reads this from meta.json in its own temp working
    # directory, where a relative path would resolve to a throwaway cache.
    answer_cache = (Path(settings.iterate_runs_dir).parent / "prompt-answers.db").resolve()
    if is_prompt_run:
        from iterate.adapters.data.tabular import with_smaller_holdout

        # The search runs on a cheap slice of the holdout; the winner is re-scored on
        # all of it once the loop ends. One model call per record makes the full
        # holdout too expensive to spend on every candidate — and since every
        # candidate sees the SAME slice, ranking stays paired and reliable while the
        # number finally quoted is measured on more data than any intermediate one.
        full_dataset = dataset
        dataset = with_smaller_holdout(dataset, loop_holdout) if loop_holdout else dataset
        if dataset.n_test < full_dataset.n_test:
            console.print(
                f"[dim]scoring candidates on {dataset.n_test} of "
                f"{full_dataset.n_test} holdout records; the winner is re-scored on "
                f"all {full_dataset.n_test} at the end[/dim]"
            )
        model_target = _build_prompt_target(
            dataset,
            metric=metric,
            average=average,
            task=str(task),
            prompt_file=prompt_file,
            backend=target_backend or backend,
            model=target_model or model,
            base_url=base_url,
            cache_path=answer_cache,
            allow_free_text=allow_free_text,
        )
    else:
        model_target = ModelTarget(dataset, metric=metric, average=average)
    if line := run_setup.render():
        console.print(f"[dim]{line}[/dim]")
        if run_setup.starting_model:
            console.print(f"[dim]starting model: {run_setup.starting_model}[/dim]")

    # NOTE: the sqlite Memory is deliberately NOT constructed here. A sqlite
    # connection can only be used on the thread that created it, and in TUI mode
    # the supervised loop runs on a worker thread — each branch below constructs
    # its Memory on the thread that will actually use it.

    # ─── Terminator ────────────────────────────────────────────────────────
    deadline_seconds = _parse_duration(until) if until is not None else None
    terminator = default_terminator(
        max_iterations=max_iterations, patience=patience, deadline_seconds=deadline_seconds
    )

    mode = "cell-by-cell" if code else "spec"
    console.print(
        f"\n[dim]Running on {model_target.name}; target={target!r}, metric={metric}, "
        f"mode={mode}, compute={compute}[/dim]\n"
    )

    if code:
        # --until bounds the WHOLE run via the terminator; a session's budget is
        # kernel-execution seconds. Tool-only roles use the no-think client even
        # under --think, because thinking crowds out the call.
        if not is_prompt_run:
            supervisor_family = "tabular"
        else:
            from iterate.core.scoring import task_for_metric

            supervisor_family = (
                "prompt_scoring" if task_for_metric(metric) == "regression" else "prompt"
            )
        n_classes = int(dataset.train_target.nunique()) if not is_prompt_run else 0
        supervisor = Supervisor(
            client,
            metric=metric,
            family=supervisor_family,
            multiclass=n_classes > 2,
        )
        summarizer = Summarizer(client, metric=metric)
        # Same no-think client as the other strict roles: the Researcher must emit
        # a single structured tool call, and a thinking trace crowds that out.
        # Cached beside the runs so a re-run on the same data pays nothing.
        critic_agent = (
            Critic(
                client,
                metric=metric,
                direction=direction,
                family="prompt" if is_prompt_run else "tabular",
            )
            if critique
            else None
        )
        researcher = (
            Researcher(
                client,
                metric=metric,
                direction=direction,
                family="prompt" if is_prompt_run else "tabular",
                cache_dir=Path(settings.iterate_runs_dir).parent / "research",
            )
            if research
            else None
        )
        # Local models live in a small context window (num_ctx); cap the coder's
        # prompt well under it so the system prompt is never what truncates.
        context_budget = 48_000 if backend == "ollama" else 400_000

        # Interactive input: the TUI on a terminal, a stdin thread under --plain or
        # piped stdout, nothing when not a tty. Control words act immediately;
        # other messages wait for the next safe boundary.
        controller: RunController | None = None
        if interactive_tty:
            from iterate.core.interactive import RunController as _RunController

            controller = _RunController()
            if not use_tui:
                import threading

                # Replies carry user/LLM text with brackets (code, lists) — print
                # them literally, never through rich markup (which would eat
                # "df[cols]" or raise on a stray closing tag).
                controller.bind_reply(
                    lambda text: console.print(f">> {text}", style="cyan", markup=False)
                )
                # If the run is later backgrounded (Ctrl-Z + bg), the listener's
                # tty read must fail with EIO (ending the thread quietly) instead
                # of SIGTTIN-stopping the whole process mid-run.
                with contextlib.suppress(ValueError, OSError):
                    signal.signal(signal.SIGTTIN, signal.SIG_IGN)

                def _listen(ctrl: RunController = controller) -> None:
                    try:
                        for line in sys.stdin:
                            ctrl.submit_line(line)
                    except Exception:  # a dying listener must never touch the run
                        pass

                threading.Thread(target=_listen, daemon=True, name="iterate-chat").start()
                console.print(
                    "[dim]interactive: type anything, anytime — pause / resume work as plain "
                    "words, stop (or /stop) quits now; messages reach the agent at the next "
                    "safe point[/dim]"
                )

                def _stop_now(ctrl: RunController = controller) -> None:
                    # A typed stop quits NOW but still shows the scoreboard:
                    # render whatever the loop finished, then leave. os._exit,
                    # deliberately — the main thread is blocked inside the loop
                    # and cannot be joined; notebooks + memory are already on
                    # disk, and orphaned kernels self-reap.
                    snap = ctrl.snapshot
                    result = snap() if callable(snap) else None
                    if result is not None:
                        with contextlib.suppress(Exception):
                            _render_summary(cast("RunResult", result), metric)
                    console.print(
                        ">> stopped — everything already saved is under .iterate/",
                        style="cyan",
                        markup=False,
                    )
                    os._exit(0)

                controller.on_stop_now = _stop_now

                _sigint_presses: list[int] = []

                def _sigint(signum: int, frame: object) -> None:
                    # First Ctrl-C: the existing graceful interrupt (floor banked,
                    # memory finalized). Second Ctrl-C: quit immediately.
                    if _sigint_presses:
                        _stop_now()
                    _sigint_presses.append(1)
                    raise KeyboardInterrupt

                with contextlib.suppress(ValueError, OSError):
                    signal.signal(signal.SIGINT, _sigint)

        from iterate.adapters.compute import confine, deps

        confinement = confine.Confinement(
            weights=confine.weights_dir(),
            files=(answer_cache,) if is_prompt_run else (),
            protected=(
                Path(settings.iterate_runs_dir).parent.resolve(),
                resolved_memory_path.resolve(),
                *(p.resolve() for p in given),
                *((labels.resolve(),) if labels is not None else ()),
            ),
        )
        target_key = (
            os.environ.get("ITERATE_TARGET_API_KEY") or api_key_for(target_backend or backend)
            if is_prompt_run
            else None
        )
        if compute == "local":
            if confine.sandbox_available():
                console.print(
                    "[dim]cells are confined: they open their own folder, "
                    + ("the answer cache, " if is_prompt_run else "")
                    + f"and model weights under {escape(_home_tilde(confinement.weights))}, "
                    "nothing else[/dim]"
                )
            else:
                console.print(
                    "[dim]cells are NOT confined on this machine (no sandbox here yet): a "
                    "cell can read any file you can. --compute e2b runs them isolated[/dim]"
                )

        installer = (
            deps.Installer(pending=deps.pending_path()) if compute == "local" and install else None
        )

        def make_coder() -> CodingAgent:
            kernel: StatefulKernel = (
                E2BKernel(api_key=e2b_api_key)
                if compute == "e2b"
                else LocalKernel(confinement=confinement, target_key=target_key)
            )
            family: dict[str, Any] = {}
            if is_prompt_run:
                # The four things that differ: what the session opens with, what
                # else is in its working directory, which floor catches it, and
                # which instructions it gets. The agent's job — write cells until
                # you can submit — is unchanged.
                family = {
                    "preamble": model_target.session_preamble(),
                    "extra_inputs": {codegen.META_JSON: model_target.meta_json()},
                    "floor_cell": codegen.prompt_fallback_baseline(),
                    "family": "prompt",
                    # A tabular cell is a fit: seconds. A prompt cell is one model
                    # call per record: minutes. The first live run spent both its
                    # sessions hitting the 120s cell timeout and never submitted,
                    # so these are not generosity, they are the unit of work being
                    # three orders of magnitude slower.
                    "cell_timeout": 600.0,
                    "deadline_seconds": 1800.0,
                    "wall_ceiling_seconds": 5400.0,
                }
            return CodingAgent(
                coder_client,
                kernel,
                metric=metric,
                average=average,
                install=(install or compute == "e2b"),
                installer=installer,
                context_budget_chars=context_budget,
                controller=controller,
                **family,
            )

        def on_experiment(
            *, experiment: Experiment, baseline: ExperimentResult, is_best: bool, run_id: str
        ) -> None:
            # Incremental deliverable: each finished iteration's notebook is saved
            # NOW (and best.ipynb tracks the best-so-far), so a crash or Ctrl-C
            # mid-run still leaves everything finished on disk.
            if notebooks == "none":
                return
            _write_experiment_notebook(
                experiment,
                baseline=baseline,
                is_best=is_best,
                run_dir=Path(settings.iterate_runs_dir) / run_id,
                mode=notebooks,
                data_path=str(data or train),
                holdout_path=str(holdout) if holdout is not None else None,
                target=target,
                metric=metric,
            )

        def _run_loop() -> RunResult:
            # The Memory is born HERE so its sqlite connection lives on the
            # thread that runs the loop (the TUI's worker, or the main thread
            # in plain mode) — sqlite objects are single-thread by design.
            loop_memory: Memory = SqliteMemory(resolved_memory_path)
            return run_supervised(
                target=model_target,
                dataset=dataset,
                supervisor=supervisor,
                make_coder=make_coder,
                terminator=terminator,
                memory=loop_memory,
                data_summary=data_summary,
                summarizer=summarizer,
                researcher=researcher,
                critic=critic_agent,
                on_experiment=on_experiment,
                controller=controller,
            )

        if use_tui and controller is not None:
            from iterate.ui.tui import run_in_tui

            result = run_in_tui(
                _run_loop,
                controller,
                title=(
                    f"iterate · {model_target.name} · target={target} · {metric} · "
                    f"{mode} · {compute} — type below; / for commands"
                ),
                configure_logging=_configure_logging,
            )
            if result is None:
                # Hard stop before the loop recorded anything: nothing to show.
                console.print("stopped — the run ended before anything finished.")
                raise typer.Exit(0)
        else:
            result = _run_loop()
    else:
        # ─── Spec path (allow-listed estimators) + the baseline precedence ─────
        memory: Memory = SqliteMemory(resolved_memory_path)  # main thread creates + uses
        baseline_candidate: Candidate | None
        baseline_model: str
        if source is not None:
            baseline_candidate = Reconstructor(client).reconstruct(
                data_summary=data_summary,
                source_text=_read_source(source),
                metric=metric,
                direction=direction,
            )
            baseline_model = _candidate_model(baseline_candidate)
            console.print(
                f"[bold]baseline from source[/bold] ({source.name}): "
                f"{baseline_candidate.description}"
            )
        elif not fresh and (prior := _prior_best(memory, model_target.name, direction)) is not None:
            assert prior.result is not None
            assert prior.result.metrics is not None
            baseline_candidate = prior.candidate
            baseline_model = _candidate_model(prior.candidate)
            console.print(
                f"[bold]baseline from memory[/bold]: {prior.candidate.description} "
                f"({metric}={prior.result.metrics.primary_value:.4f}); re-measuring"
            )
        else:
            baseline_candidate = None
            baseline_model = run_setup.starting_model

        orchestrator = Orchestrator(
            model_target,
            Proposer(client),
            LocalExecutor(),
            terminator,
            memory,
            data_summary=data_summary,
            baseline_model=baseline_model,
            baseline_candidate=baseline_candidate,
        )
        result = orchestrator.run()

    # ─── Sanity check on user-reported baseline ────────────────────────────
    if baseline is not None and result.baseline.metrics is not None:
        _check_baseline_divergence(
            reported=baseline, measured=result.baseline.metrics.primary_value
        )

    # ─── The deliverable the user actually leaves with ─────────────────────
    run_dir = Path(settings.iterate_runs_dir) / (result.run_id or "run")
    if is_prompt_run:
        # For a prompt run the artifact is the prompt, and a notebook cannot say
        # which of its cells held the winner. Written by the harness, never by the
        # agent, so `best` is decided by recorded scores and honours the Critic.
        from iterate.deliver import prompt_record

        final_score = _rescore_winner_on_full_holdout(
            result, full_dataset, model_target, console=console
        )
        record = prompt_record.write(
            run_dir,
            task=str(task),
            metric=metric,
            direction=direction,
            model_under_test=model_target.model_under_test,
            baseline_prompt=model_target.baseline_prompt,
            final_score=final_score,
            baseline_score=(
                result.baseline.metrics.primary_value
                if result.baseline.metrics is not None
                else None
            ),
            history=result.history,
        )
        console.print(f"[dim]prompts written to {record}[/dim]")
    elif result.best is not None and result.best.result is not None:
        out_path = output or (run_dir / "best_model.joblib")
        _save_best_model(model_target, result, metric, out_path)

    # ─── Notebook deliverable (full record is already in Memory) ───────────
    if notebooks != "none":
        _write_notebooks(
            result,
            mode=notebooks,
            run_dir=run_dir,
            data_path=str(data or train),
            holdout_path=str(holdout) if holdout is not None else None,
            target=target,
            metric=metric,
        )

    # ─── Summary ───────────────────────────────────────────────────────────
    _render_summary(result, metric)


# ── helpers ──────────────────────────────────────────────────────────────


def _rescore_winner_on_full_holdout(
    result: Any, full_dataset: Any, loop_target: Any, *, console: Any
) -> dict[str, Any] | None:
    """Measure the winning prompt once on the WHOLE holdout.

    The search scored every candidate on a cheap slice so the comparison could be
    paired and affordable. That slice carries real sampling error — at 100 records
    around ±0.03 — so it is fine for choosing between prompts and not fine as the
    number a user quotes. This pays for one full pass, on the winner only, and that
    is the figure `prompts.yaml` reports.

    Never raises: a re-score that fails leaves the loop's number in place with a
    note, rather than losing a finished run to a last-minute model call.
    """
    from iterate.core import codegen
    from iterate.core.prompting import Prompt
    from iterate.targets.prompt import PromptTarget

    best = result.best
    if best is None or best.result is None or full_dataset.n_test <= loop_target._dataset.n_test:
        return None
    raw = best.result.artifacts.get(codegen.PROMPT_JSON)
    if not raw:
        return None
    try:
        winner = Prompt.from_dict(json.loads(raw))
    except (ValueError, TypeError):
        return None

    console.print(
        f"[dim]re-scoring the winning prompt on all {full_dataset.n_test} holdout "
        f"records for the final number[/dim]"
    )
    full_target = PromptTarget(
        full_dataset,
        metric=loop_target._metric,
        average=loop_target._average,
        task=loop_target._task,
        target_backend=loop_target._target_backend,
        target_model=loop_target._target_model,
        target_base_url=loop_target._target_base_url,
        cache_path=loop_target._cache_path,
        starting_prompt=winner,
    )
    try:
        final = full_target.baseline()
    except Exception as exc:
        console.print(
            f"[dim]final re-score failed ({type(exc).__name__}); keeping the loop score[/dim]"
        )
        return None
    if final.metrics is None:
        return None
    return {"score": final.metrics.primary_value, "n": full_dataset.n_test}


def _read_starting_prompt(path: Path) -> Any:
    """The user's production prompt, from yaml (system + user_template) or plain text.

    Plain text becomes the system message with a bare `{input}` template, which is
    what someone pasting the prompt they already run actually means. A yaml file with
    no `system` key is treated as text too, rather than silently producing an empty
    prompt that would score like a broken run.
    """
    import yaml

    from iterate.core.prompting import Prompt

    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            payload = yaml.safe_load(raw)
        except yaml.YAMLError as exc:
            raise typer.BadParameter(f"{path} is not valid yaml: {exc}") from exc
        if isinstance(payload, dict) and payload.get("system"):
            return Prompt(
                system=str(payload["system"]),
                user_template=str(payload.get("user_template") or "{input}"),
            )
    if not raw.strip():
        raise typer.BadParameter(f"{path} is empty")
    return Prompt(system=raw.strip(), user_template="{input}")


def _build_prompt_target(
    dataset: Any,
    *,
    metric: str,
    average: str | None,
    task: str,
    prompt_file: Path | None,
    backend: str,
    model: str | None,
    base_url: str | None,
    cache_path: Path,
    allow_free_text: bool = False,
) -> Any:
    from iterate.core.scoring import requires_proba
    from iterate.targets.prompt import PromptTarget, target_kind

    if target_kind(dataset) == "free_text" and not allow_free_text:
        # Refused BEFORE the run rather than scored afterwards. Exact-string
        # matching rates three genuinely correct summaries at 0.0000, so without
        # this a summarisation dataset gets a confident, meaningless zero — worse
        # than an error, because it looks like an answer.
        distinct = int(dataset.train_target.dropna().nunique())
        raise typer.BadParameter(
            f"the answer column has {distinct} distinct values and almost every row "
            "differs, which reads as free text. A prompt run scores one answer "
            "against a known set or a number; iterate has no metric for free-form "
            "text yet, and exact-string matching would score correct answers as "
            "wrong. Pass --allow-free-text to score on exact matches anyway."
        )

    if requires_proba(metric):
        # Checked before anything runs, in the same spirit as validating a metric
        # against the target column: a text model returns a label, not a calibrated
        # probability, so this run could never score and should not start.
        raise typer.BadParameter(
            f"{metric!r} needs probabilities, which a prompt run cannot produce. "
            "Pick a label metric such as f1, accuracy or f1_macro."
        )

    return PromptTarget(
        dataset,
        metric=metric,
        average=average,
        task=task,
        target_backend=backend,
        target_model=model or get_settings().iterate_model,
        target_base_url=base_url,
        cache_path=cache_path,
        starting_prompt=_read_starting_prompt(prompt_file) if prompt_file else None,
    )


def _home_tilde(path: Path | None) -> str:
    home = str(Path.home())
    text = str(path)
    return "~" + text[len(home) :] if text == home or text.startswith(home + os.sep) else text


def _mask(secret: str) -> str:
    if len(secret) <= 4:
        return "****"
    return f"{secret[:2]}…{secret[-2:]}"


def _stdin_owns_tty() -> bool:
    """True only when stdin is a tty AND this process owns it (foreground job).

    `isatty()` alone is not enough: a backgrounded job still has a tty stdin,
    and reading it raises SIGTTIN, which by default STOPS the entire process —
    the run would silently freeze until foregrounded."""
    try:
        return sys.stdin.isatty() and os.tcgetpgrp(sys.stdin.fileno()) == os.getpgrp()
    except (OSError, ValueError, AttributeError):
        return False


def _configure_logging(handler: logging.Handler | None = None) -> None:
    """Stream orchestrator INFO logs to the console via rich, once — or into
    ``handler`` (the TUI's log pane) when one is given."""
    root = logging.getLogger()
    if handler is not None:
        root.addHandler(handler)
        if root.level == logging.NOTSET or root.level > logging.INFO:
            root.setLevel(logging.INFO)
    elif root.handlers:
        return  # something already configured (likely a test)
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(console=console, show_path=False, markup=False)],
        )
    # Third-party HTTP chatter (one line per LLM call, plus e2b keepalive/execute
    # pairs around every cell) drowns the run's own progress; keep it debug-only.
    for name in ("httpx", "httpcore", "e2b"):
        logging.getLogger(name).setLevel(logging.WARNING)


def _read_source(path: Path) -> str:
    """Read a source document as text; for `.ipynb`, walk cells."""
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() != ".ipynb":
        return raw
    try:
        nb = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"could not parse {path.name} as a notebook: {exc}") from exc
    parts: list[str] = []
    for cell in nb.get("cells", []):
        src = cell.get("source", "")
        text = "".join(src) if isinstance(src, list) else str(src)
        if not text.strip():
            continue
        if cell.get("cell_type") == "code":
            parts.append(f"```python\n{text}\n```")
        else:
            parts.append(text)
    return "\n\n".join(parts)


def _parse_duration(text: str) -> float:
    """Parse '30s' / '15m' / '2h' / '1h30m' to seconds."""
    text = text.strip().lower()
    total = 0.0
    number = ""
    for ch in text:
        if ch.isdigit() or ch == ".":
            number += ch
            continue
        if not number:
            raise typer.BadParameter(f"invalid duration {text!r}")
        value = float(number)
        if ch == "h":
            total += value * 3600
        elif ch == "m":
            total += value * 60
        elif ch == "s":
            total += value
        else:
            raise typer.BadParameter(f"invalid duration unit {ch!r} in {text!r}")
        number = ""
    if number:  # trailing bare number — assume seconds
        total += float(number)
    if total <= 0:
        raise typer.BadParameter(f"duration must be > 0 ({text!r} parsed as 0)")
    return total


def _archive_memory_db(path: Path) -> Path | None:
    """Rename an existing memory db to a timestamped `.bak`. Returns the new path."""
    if not path.exists():
        return None
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    archived = path.with_name(f"{path.stem}.{stamp}.bak{path.suffix}")
    path.rename(archived)
    return archived


def _resolved_api_key_from_env(settings: object, backend: str) -> str | None:
    """Pull an api key from settings env-overrides for a cloud backend.

    Delegates to the llm factory, which owns the backend -> key-field table. The
    model under test on a prompt run resolves its key through the same function, so
    a `--target-backend groq` picks up `GROQ_API_KEY` exactly as the driving model
    does — one table, not two that agree until one changes.
    """
    from iterate.llm.factory import api_key_for

    return api_key_for(backend, settings)


def _candidate_model(candidate: Candidate) -> str:
    """A display string for a candidate's approach: the estimator name (spec path)
    or the description (code path, where there is no single estimator name)."""
    model = candidate.changes.get("model")
    if isinstance(model, str) and model.strip():
        return model.strip()
    return candidate.description or "generated code"


def _resolve_setup(
    *,
    dataset: Any,
    explicit: str | None,
    data_summary: str,
    client: Any,
    enabled: bool,
) -> Any:
    """Decide the run's metric + starting model. Never raises: every failure path
    lands on the deterministic default, so the dial can only upgrade a working run."""
    from iterate.core import setup as run_setup_mod
    from iterate.core.scoring import CLASSIFICATION_METRICS, REGRESSION_METRICS

    if explicit or not enabled:
        return run_setup_mod.resolve(dataset, explicit=explicit)

    from iterate.core.researcher import Researcher

    task = run_setup_mod.target_task(dataset)
    allowed = CLASSIFICATION_METRICS if task == "classification" else REGRESSION_METRICS
    try:
        findings = Researcher(client).research(
            profile=data_summary, choose_setup=True, allowed_metrics=sorted(allowed)
        )
        proposed = findings.setup
    except Exception:  # a specialist must never stop a run from starting
        proposed = None
    resolved = run_setup_mod.resolve(dataset, proposed=proposed)
    if proposed is not None and not resolved.chosen_by_agent and resolved.why:
        logging.getLogger(__name__).info("setup: %s", resolved.why)
    return resolved


def _default_baseline_model(metric: str) -> str:
    """Factory default per task. Mirrors the private mapping in adapters.models.registry."""
    from iterate.core.scoring import task_for_metric

    if task_for_metric(metric) == "classification":
        return "sklearn.ensemble.HistGradientBoostingClassifier"
    return "sklearn.ensemble.HistGradientBoostingRegressor"


def _prior_best(memory: Memory, target_name: str, direction: str) -> Experiment | None:
    """Return the best succeeded experiment for this target across memory, or None."""
    history = memory.history(target_name)
    succeeded = [e for e in history if e.result and e.result.succeeded and e.result.metrics]
    if not succeeded:
        return None

    def _score(experiment: Experiment) -> float:
        assert experiment.result is not None
        assert experiment.result.metrics is not None
        return experiment.result.metrics.primary_value

    return min(succeeded, key=_score) if direction == "minimize" else max(succeeded, key=_score)


def _check_baseline_divergence(
    *, reported: float, measured: float, threshold: float = 0.10
) -> None:
    if reported == 0:
        return
    divergence = abs(measured - reported) / abs(reported)
    if divergence > threshold:
        console.print(
            f"\n[yellow]warning:[/yellow] your reported baseline {reported:.4f} "
            f"differs from our re-measurement {measured:.4f} ({divergence:.1%} divergence). "
            f"Likely a feature/eval mismatch."
        )


def _save_best_model(target: ModelTarget, result: RunResult, metric: str, path: Path) -> None:
    """Persist the winning approach + a sidecar best.json.

    Spec winner → refit and pickle the fitted pipeline (joblib). Code winner →
    save the `train_and_predict` source (a code-gen winner returns predictions, not
    a pickled model — by design; see LIMITATIONS.md). The notebook deliverable
    (Day 6) turns that source into a runnable artifact.
    """
    best = result.best
    assert best is not None
    best_result = best.result
    assert best_result is not None
    spec = best.candidate.changes
    score = best_result.metrics.primary_value if best_result.metrics else None
    path.parent.mkdir(parents=True, exist_ok=True)  # the run dir (the code path skips save_model)

    if spec.get("code"):
        # Code winner: the runnable artifact is best.ipynb (written by _write_notebooks);
        # a code-gen winner returns predictions, not a pickle — by design.
        artifact = None
        load_hint = "the winning approach is in best.ipynb (runnable)"
    else:
        target.save_model(spec, path)
        artifact = path
        load_hint = f"load it: joblib.load({str(path)!r}).predict(X)"

    sidecar = path.with_name("best.json")
    sidecar.write_text(
        json.dumps(
            {
                "run_id": result.run_id,
                "model": spec.get("model"),
                "params": spec.get("params", {}),
                "code": spec.get("code"),
                "description": best.candidate.description,
                "rationale": best.candidate.rationale,
                "metric": metric,
                "score": score,
                "artifact_path": str(artifact) if artifact else None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if artifact is not None:
        console.print(f"\n[bold]saved best model[/bold] → {artifact}\n[dim]{load_hint}[/dim]")
    else:
        console.print(f"\n[dim]{load_hint}[/dim]")


def _render_experiment(
    exp: Experiment,
    *,
    is_best: bool,
    baseline_score: float | None,
    metric: str,
    data_path: str,
    holdout_path: str | None = None,
    target: str,
    leaderboard: list[Experiment] | None = None,
) -> Any:
    """Render ONE experiment to a notebook node — shared by the incremental
    per-iteration save and the end-of-run write, so both produce identical files.
    Cell-by-cell experiments carry their session ("cells"); render the real
    session. Spec / one-shot experiments render through the contract."""
    from iterate.deliver.notebook import build_notebook, build_session_notebook

    cells = exp.candidate.changes.get("cells")
    if isinstance(cells, list):
        score = exp.result.metrics.primary_value if exp.result and exp.result.metrics else None
        title = ("best: " if is_best else "") + exp.candidate.description
        if exp.candidate.changes.get("duplicate_submission"):
            note = (
                "this submission is byte-identical to an earlier experiment's — "
                "the score adds no new information beyond it"
            )
        elif exp.candidate.changes.get("lever_unmeasured"):
            note = (
                "the briefed change never executed successfully — "
                "the score belongs to the carried pipeline, not the briefed idea"
            )
        else:
            note = None
        return build_session_notebook(
            cells,
            title=title,
            metric=metric,
            score=score,
            baseline_score=baseline_score,
            hypothesis=exp.hypothesis,
            findings=exp.digest,
            honesty_note=note,
        )
    return build_notebook(
        exp,
        data_path=data_path,
        holdout_path=holdout_path,
        target=target,
        metric=metric,
        baseline_score=baseline_score,
        is_best=is_best,
        leaderboard=leaderboard,
    )


def _write_experiment_notebook(
    exp: Experiment,
    *,
    baseline: ExperimentResult,
    is_best: bool,
    run_dir: Path,
    mode: str,
    data_path: str,
    holdout_path: str | None = None,
    target: str,
    metric: str,
) -> None:
    """Save one finished iteration's notebook the moment it completes (and keep
    best.ipynb pointing at the best-so-far), so a crash or Ctrl-C mid-run still
    leaves every finished deliverable on disk. The end-of-run `_write_notebooks`
    rewrite is idempotent on top of these."""
    from iterate.deliver.notebook import save_notebook, slug

    baseline_score = baseline.metrics.primary_value if baseline.metrics is not None else None
    if mode == "all":
        name = f"iter_{exp.iteration:02d}_{slug(exp.candidate.description)}.ipynb"
        save_notebook(
            _render_experiment(
                exp,
                is_best=False,
                baseline_score=baseline_score,
                metric=metric,
                data_path=data_path,
                holdout_path=holdout_path,
                target=target,
            ),
            run_dir / "notebooks" / name,
        )
    if is_best and exp.result is not None and exp.result.succeeded:
        save_notebook(
            _render_experiment(
                exp,
                is_best=True,
                baseline_score=baseline_score,
                metric=metric,
                data_path=data_path,
                holdout_path=holdout_path,
                target=target,
            ),
            run_dir / "best.ipynb",
        )


def _write_notebooks(
    result: RunResult,
    *,
    mode: str,
    run_dir: Path,
    data_path: str,
    holdout_path: str | None = None,
    target: str,
    metric: str,
) -> None:
    """Render the run as runnable notebooks: the winner (`best`) or one per
    experiment (`all`). The full record is already in Memory; this just renders it."""
    from iterate.deliver.notebook import save_notebook, slug

    baseline_score = (
        result.baseline.metrics.primary_value if result.baseline.metrics is not None else None
    )

    written: list[Path] = []
    if mode == "all":
        for exp in result.history:
            is_best = result.best is not None and exp.id == result.best.id
            name = f"iter_{exp.iteration:02d}_{slug(exp.candidate.description)}.ipynb"
            written.append(
                save_notebook(
                    _render_experiment(
                        exp,
                        is_best=is_best,
                        baseline_score=baseline_score,
                        metric=metric,
                        data_path=data_path,
                        holdout_path=holdout_path,
                        target=target,
                    ),
                    run_dir / "notebooks" / name,
                )
            )

    if result.best is not None:
        written.append(
            save_notebook(
                _render_experiment(
                    result.best,
                    is_best=True,
                    baseline_score=baseline_score,
                    metric=metric,
                    data_path=data_path,
                    holdout_path=holdout_path,
                    target=target,
                    leaderboard=result.history,
                ),
                run_dir / "best.ipynb",
            )
        )

    if not written:
        return
    if mode == "all":
        n_journey = len([p for p in written if p.parent.name == "notebooks"])
        console.print(f"\n[bold]notebooks[/bold] → {run_dir / 'notebooks'}/ ({n_journey} files)")
    if result.best is not None:
        console.print(f"[bold]best notebook[/bold] → {run_dir / 'best.ipynb'}")


def _render_summary(result: RunResult, metric: str) -> None:
    baseline_score = (
        result.baseline.metrics.primary_value if result.baseline.metrics is not None else None
    )
    best_id = result.best.id if result.best is not None else None

    table = Table(title="Run summary", show_lines=False)
    table.add_column("iter", justify="right")
    table.add_column("model")
    table.add_column(metric, justify="right")
    table.add_column("Δ vs baseline", justify="right")

    if baseline_score is not None:
        table.add_row("base", "baseline", f"{baseline_score:.4f}", "—")

    for exp in result.history:
        model_name = _candidate_model(exp.candidate)
        if exp.id == best_id:
            model_name += "  [bold green]← best[/bold green]"
        if exp.result is None or exp.result.metrics is None:
            err = exp.result.error if exp.result else "no result"
            table.add_row(str(exp.iteration), model_name, "[red]FAILED[/red]", str(err)[:40])
            continue
        score = exp.result.metrics.primary_value
        delta = (score - baseline_score) if baseline_score is not None else 0.0
        arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "—")
        table.add_row(str(exp.iteration), model_name, f"{score:.4f}", f"{arrow} {delta:+.4f}")

    console.print()
    console.print(table)
    console.print(f"\n[bold]stopped:[/bold] {result.stopped_because}")
    if result.best is not None and result.best.result and result.best.result.metrics:
        improvement = (
            result.best.result.metrics.primary_value - baseline_score
            if baseline_score is not None
            else 0.0
        )
        console.print(
            f"[bold]best:[/bold] {result.best.candidate.description} "
            f"({metric}={result.best.result.metrics.primary_value:.4f}, "
            f"{improvement:+.4f} vs baseline)"
        )
    else:
        console.print("[dim]no candidate beat the baseline.[/dim]")


if __name__ == "__main__":
    app()
