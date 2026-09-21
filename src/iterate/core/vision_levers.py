"""The image family's lever classes, read from what a session's helpers printed.

A tabular lever is a class name in the code. An image fit is one call that names every
lever it sets, so the code cannot say which lever moved. The helpers print what ran:
the runner's epoch lines and ``FIT {json}`` from ``fit()``, ``MODEL {json}`` from
``evaluate(..., model=...)`` on the agent's own model, and ``SUBMITTED {json}`` from the
submit helpers. Everything here reads those lines against the recipe each try started
from, which ``fit()`` prints under "from".
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from importlib import resources
from typing import TYPE_CHECKING, Any

from iterate.targets import layers as arch

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from iterate.schemas.experiment import Experiment

LEVERS: tuple[str, ...] = (
    "backbone",
    "own-model",
    "image-size",
    "epochs",
    "augmentation",
    "regularisation",
    "fine-tune-depth",
    "layer-stack",
    "custom-head",
)
FIT_FIELDS: dict[str, tuple[str, ...]] = {
    "backbone": ("backbone",),
    "image-size": ("image_size",),
    "epochs": ("epochs",),
    "augmentation": ("augment",),
    "regularisation": ("label_smoothing",),
    "fine-tune-depth": ("unfreeze", "lr", "head_init", "optimizer", "schedule"),
    "layer-stack": ("layers",),
    "custom-head": ("head", "drop_stages"),
}
LAYER_LEVERS = frozenset({"layer-stack", "custom-head"})
MODEL_FIELDS: dict[str, tuple[str, ...]] = {
    "image-size": ("image_size",),
    "epochs": ("epochs",),
    "augmentation": ("augment",),
    "regularisation": (
        "label_smoothing",
        "dropout",
        "weight_decay",
        "mixup",
        "cutmix",
        "drop_path",
    ),
    "fine-tune-depth": ("trainable", "unfreeze", "lr", "optimizer", "schedule"),
}
# The networks fit() itself trains: naming one of these is a backbone move, never an
# own-model one, however the brief words it.
FIT_BACKBONES = frozenset({"simple_cnn", "resnet18", "resnet50", "convnext_tiny"})
# The kinds fit() trains from zero. Mirrors targets.net.SCRATCH, which cannot be imported
# here because it pulls in torch. A layers net stays out of STRENGTH and COST: it has no
# pretrained rank and no measured seconds, so ready() branches on it by name.
FROM_ZERO = frozenset({"simple_cnn", "layers_net"})
# ImageNet-1k top-1 of the pinned weights, the order a stronger backbone is read in.
STRENGTH: dict[str, int] = {"simple_cnn": 0, "resnet18": 1, "resnet50": 2, "convnext_tiny": 3}
# Seconds per epoch against resnet18 at one size, from the Day 4 sweep's rows.
COST: dict[str, float] = {"simple_cnn": 0.5, "resnet18": 1.0, "resnet50": 3.5, "convnext_tiny": 3.5}
BENCH_EPOCHS = 3
MAX_EPOCHS = 12
BUDGET = 540.0
# Below this the last epoch's gain does not pay for doubling: measured on EuroSAT
# (+0.0405 then +0.0007 for five epochs) and Flowers102 (+0.0921 then +0.0130).
RISING = 0.05
GAP = {"train_acc": 0.03, "train_r2": 0.05}

_ALIASES: dict[str, str] = {
    "backbone-swap": "backbone",
    "pretrained-model": "backbone",
    "pretrained-backbone": "backbone",
    "model-swap": "backbone",
    "own-code": "own-model",
    "own-code-model": "own-model",
    "custom-model": "own-model",
    "resolution": "image-size",
    "input-size": "image-size",
    "image-resolution": "image-size",
    "epoch": "epochs",
    "augment": "augmentation",
    "data-augmentation": "augmentation",
    "regularization": "regularisation",
    "label-smoothing": "regularisation",
    "fine-tuning-depth": "fine-tune-depth",
    "finetune-depth": "fine-tune-depth",
    "unfreeze": "fine-tune-depth",
    "learning-rate": "fine-tune-depth",
    # Multi-word only: bare "head", "layers" and "architecture" are words the 12B
    # already writes into fine-tune and backbone tags.
    "from-scratch": "layer-stack",
    "new-head": "custom-head",
}
# Longest first so an alias wins over the class word inside it; the name breaks the tie,
# because set order is hash-randomised and the first two names are printed to the model.
_NAMES = sorted({*LEVERS, *_ALIASES}, key=lambda name: (-len(name), name))
_NEXT = re.compile(r"\bnext\s*:", re.IGNORECASE)
_TAG_CHARS = 40
_BECAUSE = re.compile(r"\(\s*because\b", re.IGNORECASE)
_SO = re.compile(r",?\s+so\s+", re.IGNORECASE)
_CALLS = {
    "FIT": re.compile(r"\bfit\s*\("),
    "MODEL": re.compile(r"\bevaluate\s*\("),
    "SUBMITTED": re.compile(r"\bsubmit(?:_probabilities|_numbers)?\s*\("),
}
_EPOCH = re.compile(r"^epoch (\d+)/(\d+) loss=\S+ (train_acc|train_r2)=(-?[\d.]+|nan) (\d+)s$")
_CUT = re.compile(r"^stopped in epoch (\d+)/(\d+): the fit budget ran out")
_OOM = re.compile(r"out of memory|allocate memory|invalid buffer size", re.IGNORECASE)
# An org/name id, kept apart from a DOI or a path by the lookbehind, and required to
# carry a digit or a hyphen somewhere so "training/validation" is not a model.
_HF_ID = re.compile(r"(?<![\w/.:])[a-z][\w-]*/(?=[\w.-]*[\d-])[a-z][\w.-]*[a-z0-9]", re.IGNORECASE)
_TOKEN = re.compile(r"(?<![\w/.:])(?:timm[_/])?([a-z][a-z0-9_]*[a-z0-9])", re.IGNORECASE)
# The recipe keys whose value is a layer spec: a list cannot go in the set of tried
# values, so everything that compares one compares its canonical text instead.
_SPEC_KEYS = frozenset({"layers", "head"})
# A word of refusal anywhere in an ask shuts every lever the ask could open. "never build
# a network from scratch" is a ban, and reading it as a request opened the lever it bans.
_NO = re.compile(r"\b(?:never|no|not|don'?t|do not|stop|skip|avoid|without)\b", re.IGNORECASE)
_ASK_ZERO = re.compile(
    r"from[\s-]+(?:scratch|zero)|layer[\s-]+stack|(?:custom|own)[\s-]+(?:cnn|network|architecture)",
    re.IGNORECASE,
)
_ASK_HEAD = re.compile(
    r"(?:own|custom|new|bigger|deeper)[\s-]+(?:classifier[\s-]+)?head|head with", re.IGNORECASE
)
_ASK_BLOCK = re.compile(r"last[\s_-]+(?:block|stage)", re.IGNORECASE)
# A stack written the way `fit_call` prints one, inside a sentence: a repair line hands
# the model a whole call, and what it copies back has to read as the value it names.
_IN_CALL = {
    key: re.compile(rf"\b{key}\s*=\s*(\[[^][]*\])", re.IGNORECASE) for key in ("layers", "head")
}


def _timm_names() -> frozenset[str]:
    text = resources.files("iterate.core").joinpath("timm_models.txt").read_text(encoding="utf-8")
    return frozenset(text.split())


_TIMM = _timm_names()


def classes_named(brief: str) -> list[str]:
    """The lever classes in the move's tag, the text between "next:" and the colon after
    it, so a class word in the reason ("keep 3 epochs") names nothing."""
    match = _NEXT.search(brief)
    if match is None:
        return []
    rest = brief[match.end() :]
    colon = rest.find(":")
    if colon < 0 or colon > _TAG_CHARS:
        return []
    head = re.sub(r"[\s_]+", "-", rest[:colon].strip().lower())
    found: list[str] = []
    for name in _NAMES:
        pattern = rf"(?<![a-z]){re.escape(name)}(?![a-z])"
        if re.search(pattern, head):
            canonical = _ALIASES.get(name, name)
            if canonical not in found:
                found.append(canonical)
            head = re.sub(pattern, " ", head)
    return found


def lever_class(brief: str) -> str | None:
    named = classes_named(brief)
    return named[0] if len(named) == 1 else None


def change_clause(brief: str) -> str:
    """The change a move proposes, with its reason cut away: after the class tag, before
    "(because", after the last "so". A value quoted as the reason ("resnet18 took 12s an
    epoch, so swap to convnext_tiny") is not a value the brief proposes."""
    match = _NEXT.search(brief)
    rest = brief[match.end() :] if match else brief
    tag, colon, after = rest.partition(":")
    rest = after if colon and len(tag) <= _TAG_CHARS else rest
    return _SO.split(_BECAUSE.split(rest, maxsplit=1)[0])[-1].strip()


def models_named(text: str) -> list[str]:
    """Names a library loads: a timm pretrained architecture or a Hugging Face id, never
    a plain English word and never a network fit() already trains."""
    found = {m.group(1).lower() for m in _TOKEN.finditer(text) if m.group(1).lower() in _TIMM}
    found |= {m.group(0) for m in _HF_ID.finditer(text)}
    return sorted(found - FIT_BACKBONES)


def _parts(cell: Any) -> tuple[str, str, str]:
    if isinstance(cell, dict):
        return (
            str(cell.get("code") or ""),
            str(cell.get("stdout") or ""),
            str(cell.get("error") or ""),
        )
    return (
        str(getattr(cell, "code", "") or ""),
        str(getattr(cell, "stdout", "") or ""),
        str(getattr(cell, "error", "") or ""),
    )


def _agent(cell: Any) -> bool:
    source = cell.get("source") if isinstance(cell, dict) else getattr(cell, "source", "agent")
    return source in ("agent", None)


def _payload(line: str, tag: str) -> dict[str, Any] | None:
    try:
        value = json.loads(line[len(tag) + 1 :])
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


@dataclass
class Try:
    """One fit or one scored own-code model, with the evidence its cell printed."""

    kind: str
    recipe: dict[str, Any]
    start: dict[str, Any] = field(default_factory=dict)
    train_key: str = ""
    train: list[float] = field(default_factory=list)
    planned: int | None = None
    ran: int | None = None
    epoch_seconds: float | None = None
    stopped: bool = False

    @property
    def name(self) -> str:
        return model_name(self.recipe)

    @property
    def val(self) -> float | None:
        v = self.recipe.get("val")
        return float(v) if isinstance(v, int | float) else None

    @property
    def val_like_train(self) -> float | None:
        key = {"train_acc": "val_accuracy", "train_r2": "val_r2"}.get(self.train_key)
        v = self.recipe.get(key) if key else None
        return float(v) if isinstance(v, int | float) else None

    @property
    def asked(self) -> int | None:
        v = self.recipe.get("epochs")
        return int(v) if isinstance(v, int) else None

    @property
    def cut(self) -> bool:
        """The fit trained fewer epochs than the recipe asked for, whether the budget
        shrank the plan before the first epoch or stopped it part way through."""
        if self.kind != "fit":
            return False
        if self.stopped:
            return True
        if self.planned is not None and (self.ran or 0) < self.planned:
            return True
        return self.asked is not None and self.planned is not None and self.planned < self.asked

    @property
    def rising(self) -> bool:
        return len(self.train) >= 2 and self.train[-1] - self.train[-2] > RISING

    @property
    def gap(self) -> float | None:
        like = self.val_like_train
        if like is None or not self.train:
            return None
        return self.train[-1] - like

    @property
    def seconds_per_epoch(self) -> float | None:
        """From the epoch lines for a fit; for own code, from the seconds and the epochs
        the cell reported, the only timing a MODEL line carries."""
        if self.epoch_seconds is not None:
            return self.epoch_seconds
        seconds, epochs = self.recipe.get("seconds"), self.asked
        if isinstance(seconds, int | float) and epochs:
            return float(seconds) / epochs
        return None


def model_name(recipe: dict[str, Any] | None) -> str:
    if not recipe:
        return ""
    raw = str(recipe.get("model") or recipe.get("backbone") or "").strip().lower()
    for prefix in ("timm/", "timm_", "hf-hub:", "hf_hub:", "torchvision.models."):
        raw = raw.removeprefix(prefix)
    return raw.replace("-", "_")


def tries(cells: Iterable[Any]) -> list[Try]:
    """Every fit and own-code model the session ran, from cells that call the helper, so
    a cell printing a tagged line of its own counts for nothing.

    A cell that raised keeps the lines it printed first: the helpers print FIT, MODEL and
    SUBMITTED only after their work is done and their files are written, so a fit that
    finished before a later statement failed really did run."""
    out: list[Try] = []
    for cell in cells:
        if not _agent(cell):
            continue
        code, stdout, error = _parts(cell)
        calls_fit = bool(_CALLS["FIT"].search(code))
        pending: list[tuple[str, float, int, int, float]] = []
        stopped = False
        for raw in stdout.splitlines():
            line = raw.strip()
            if calls_fit and (m := _EPOCH.match(line)):
                value = float(m.group(4)) if m.group(4) != "nan" else float("nan")
                pending.append(
                    (m.group(3), value, int(m.group(1)), int(m.group(2)), float(m.group(5)))
                )
            elif calls_fit and _CUT.match(line):
                stopped = True
            elif calls_fit and line.startswith("FIT ") and (p := _payload(line, "FIT")):
                start = p.get("from")
                t = Try(
                    kind="fit",
                    recipe=p,
                    start=dict(start) if isinstance(start, dict) else {},
                    stopped=stopped,
                )
                if pending:
                    t.train_key = pending[0][0]
                    t.train = [v for _, v, _, _, _ in pending]
                    t.epoch_seconds = sum(s for *_, s in pending) / len(pending)
                planned, ran = p.get("epochs_planned"), p.get("epochs_run")
                t.planned = (
                    int(planned)
                    if isinstance(planned, int)
                    else (pending[-1][3] if pending else None)
                )
                t.ran = int(ran) if isinstance(ran, int) else len(pending)
                if t.planned is None:
                    t.planned = t.ran
                out.append(t)
                pending, stopped = [], False
            elif (
                _CALLS["MODEL"].search(code)
                and line.startswith("MODEL ")
                and (p := _payload(line, "MODEL"))
            ):
                t = Try(kind="model", recipe=p)
                if "val_accuracy" in p:
                    t.train_key = "train_acc"
                elif "val_r2" in p:
                    t.train_key = "train_r2"
                out.append(t)
        # Last, so the ledger stays in the order the cell ran: whatever finished comes
        # before the fit the memory refused.
        if error and calls_fit and _OOM.search(error):
            out.append(Try(kind="oom", recipe={}))
    return out


def submitted(cells: Iterable[Any]) -> dict[str, Any] | None:
    """The payload of the last submission the session wrote.

    A cell that raised after its submit still counts: the helper prints SUBMITTED only
    once predictions.csv, probabilities.csv and recipe.json are on disk, and the host
    scores that file whatever the cell did afterwards, so dropping the payload would
    carry a recipe the run did not score."""
    found: dict[str, Any] | None = None
    for cell in cells:
        if not _agent(cell):
            continue
        code, stdout, _ = _parts(cell)
        if not _CALLS["SUBMITTED"].search(code):
            continue
        for raw in stdout.splitlines():
            line = raw.strip()
            if line.startswith("SUBMITTED ") and (p := _payload(line, "SUBMITTED")):
                found = p
    return found


def submitted_try(cells: Sequence[Any]) -> Try | None:
    """The try whose numbers the submitted payload carries: a submit helper writes the
    fit's or the model's own line, so the payload IS one of them."""
    sub = submitted(cells)
    if sub is None:
        return None
    ran = [t for t in tries(cells) if t.kind != "oom"]
    for t in reversed(ran):
        if t.recipe == sub:
            return t
    for t in reversed(ran):
        if t.name and t.name == model_name(sub) and t.recipe.get("val") == sub.get("val"):
            return t
    return None


def _differs(value: Any, reference: Any) -> bool:
    return bool(value != reference)


def _value_of(recipe: dict[str, Any], key: str) -> Any:
    """One key's value as something a set can hold and two runs can compare: a layer spec
    as its canonical text, everything else untouched. A spec that does not read is no
    value, which leaves it untried rather than crashing the run that holds it."""
    value = recipe.get(key)
    if key not in _SPEC_KEYS or value is None:
        return value
    try:
        return arch.text(value, key) or None
    except arch.RecipeError:
        return None


def moved(lever: str, cells: Sequence[Any], carried: dict[str, Any]) -> bool:
    """Whether a helper line the session printed changed ``lever``.

    The backbone is read against the recipe the RUN carried in, and every other setting
    against the recipe the fit says it started from. Leaving the plain CNN starts a fit
    from the fine-tune reference, whose epochs and depth are not the agent's doing; the
    network it named is.

    A stack names no network, so a from-zero fit is a layer-stack move and nothing else:
    crediting the backbone too would let two losing stacks close the backbone as well."""
    current = model_name(carried)
    for t in tries(cells):
        if t.kind == "oom":
            continue
        if t.kind == "fit":
            if lever == "backbone" and t.name != current and not t.recipe.get("layers"):
                return True
            if lever in LAYER_LEVERS:
                # A line leaves a layer setting out when it is off, so an absent key is
                # the value None and not "nothing to compare": the first head on a
                # pretrained best has no key to match and would never be credited.
                if any(
                    _differs(_value_of(t.recipe, k), _value_of(t.start, k))
                    for k in FIT_FIELDS[lever]
                ):
                    return True
            elif (
                lever in FIT_FIELDS
                and lever != "backbone"
                and any(
                    k in t.recipe and k in t.start and _differs(t.recipe[k], t.start[k])
                    for k in FIT_FIELDS[lever]
                )
            ):
                return True
        else:
            if not t.name:
                continue
            if lever == "own-model" and t.name != current:
                return True
            if lever in MODEL_FIELDS and any(
                k in t.recipe and k in carried and _differs(t.recipe[k], carried[k])
                for k in MODEL_FIELDS[lever]
            ):
                return True
    return False


def moved_levers(cells: Sequence[Any], carried: dict[str, Any]) -> list[str]:
    return [lever for lever in LEVERS if moved(lever, cells, carried)]


def describe(recipe: dict[str, Any] | None) -> str:
    if not recipe:
        return ""
    if recipe.get("model"):
        bits = [f"own code: {model_name(recipe)}"]
        bits += [f"{recipe[k]}px" for k in ("image_size",) if recipe.get(k)]
        bits += [f"{recipe[k]} epochs" for k in ("epochs",) if recipe.get(k)]
        bits += [
            f"{k} {recipe[k]}"
            for k in ("trainable", "augment", "lr", "dropout", "weight_decay")
            if k in recipe
        ]
        return ", ".join(bits)
    depth = {
        "none": "linear probe",
        "head": "head only",
        "last_block": "last stage and head",
        "all": "all layers",
    }.get(str(recipe.get("unfreeze")), "")
    bits = [f"{recipe.get('backbone')} {recipe.get('image_size')}px {depth}".strip()]
    if recipe.get("layers"):
        bits.append(f"layers {arch.text(recipe['layers'])}")
    if recipe.get("drop_stages"):
        bits.append(f"{recipe['drop_stages']} stages dropped")
    if recipe.get("head"):
        bits.append(f"head {arch.text(recipe['head'])}")
    if recipe.get("unfreeze") != "none":
        bits.append(f"{recipe.get('epochs')} epochs")
    bits.append(f"augment {recipe.get('augment')}")
    bits.append(f"lr {recipe.get('lr')}")
    if recipe.get("label_smoothing"):
        bits.append(f"label_smoothing {recipe['label_smoothing']}")
    if recipe.get("head_init") == "probe":
        bits.append("head from the probe")
    return ", ".join(bits)


def fit_call(recipe: dict[str, Any], **changes: Any) -> str:
    """A repair names the recipe it repairs, not "the recipe": the coder starts from the
    carried best, which is not the try that failed."""
    if recipe.get("model"):
        base: dict[str, Any] = {"model": model_name(recipe)}
    elif recipe.get("layers"):
        # A from-zero net names no backbone, and merge() gives it the from-zero
        # reference's epochs; naming the pretrained best's would retrain it for three.
        base = {k: recipe[k] for k in ("layers", "image_size") if recipe.get(k)}
    else:
        base = {
            k: recipe[k]
            for k in ("backbone", "image_size", "epochs", "head", "drop_stages")
            if recipe.get(k)
        }
    merged = {**base, **changes}
    inside = ", ".join(
        f"{k}={arch.code(v, k)}"
        if k in _SPEC_KEYS
        else f"{k}={v!r}"
        if isinstance(v, str)
        else f"{k}={v}"
        for k, v in merged.items()
        if v is not None
    )
    if merged.get("model"):
        return f"your own code for {merged['model']}, " + ", ".join(
            f"{k} {v}" for k, v in merged.items() if k != "model" and v is not None
        )
    return f"fit({inside})"


def evidence(cells: Sequence[Any], metric: str) -> str:
    """The submitted try's numbers as one line, the words the ladder's triggers name."""
    all_tries = tries(cells)
    ran = [t for t in all_tries if t.kind != "oom"]
    chosen = submitted_try(cells) or (ran[-1] if ran else None)
    parts: list[str] = []
    if chosen is not None:
        parts.append(describe(chosen.recipe))
        if chosen.kind == "fit" and chosen.planned:
            at = f" at {chosen.epoch_seconds:.0f}s each" if chosen.epoch_seconds else ""
            parts.append(f"ran {chosen.ran} of {chosen.planned} epochs{at}")
            if chosen.asked and chosen.planned < chosen.asked:
                parts.append(f"planned {chosen.planned} of the {chosen.asked} asked")
            if chosen.cut:
                parts.append("STOPPED BY THE FIT BUDGET")
        if chosen.train:
            trend = " (still rising)" if chosen.rising else ""
            parts.append(
                f"{chosen.train_key} {chosen.train[0]:.3f} -> {chosen.train[-1]:.3f}{trend}"
            )
        if chosen.val is not None:
            parts.append(f"val {metric} {chosen.val:.4f}")
        gap = chosen.gap
        if gap is not None and gap > GAP.get(chosen.train_key, 0.03):
            parts.append(f"train above val by {gap:.3f}")
    if any(t.kind == "oom" for t in all_tries):
        parts.append("a fit ran OUT OF MEMORY")
    if len(ran) > 1:
        parts.append(f"{len(ran)} tries this session")
    return "; ".join(p for p in parts if p)


def recipe_of(exp: Experiment | None) -> dict[str, Any]:
    if exp is None:
        return {}
    value = exp.candidate.changes.get("recipe")
    return dict(value) if isinstance(value, dict) else {}


def fit_recipe_of(exp: Experiment | None) -> dict[str, Any]:
    """The recipe the next session's `fit()` may start from.

    A submission from the agent's own code is not one: its `epochs` counts loops the
    agent wrote and it names no backbone, so `fit()` would depart from a recipe nothing
    ran. That best travels forward as code instead, and the last fit of its session, or
    nothing, is what a fit can start from."""
    recipe = recipe_of(exp)
    if not recipe.get("model"):
        return recipe
    for attempt in reversed(tries(_cells(exp)) if exp is not None else []):
        if attempt.kind == "fit":
            return dict(attempt.recipe)
    return {}


def _cells(exp: Experiment) -> list[Any]:
    cells = exp.candidate.changes.get("cells")
    return cells if isinstance(cells, list) else []


def best_try(exp: Experiment | None) -> Try | None:
    return submitted_try(_cells(exp)) if exp is not None else None


def attempted(exp: Experiment) -> tuple[str, Any] | None:
    """What a brief set out to change, and to what. An experiment that failed printed no
    helper line, so without this a value that keeps failing is never counted as tried."""
    lever = lever_class(exp.hypothesis or "")
    if lever is None:
        return None
    clause = change_clause(exp.hypothesis or "")
    value = (
        next(iter(models_named(clause)), None)
        if lever == "own-model"
        else proposed_value(lever, clause)
    )
    return lever, value


def tried(history: Sequence[Experiment]) -> set[str]:
    out: set[str] = set()
    for exp in history:
        levers = exp.candidate.changes.get("levers_moved")
        if isinstance(levers, list):
            out.update(str(lv) for lv in levers)
    return out


_LEVER_KEY = {
    "backbone": "backbone",
    "own-model": "name",
    "image-size": "image_size",
    "epochs": "epochs",
    "augmentation": "augment",
    "layer-stack": "layers",
    "custom-head": "head",
}


def _tried_values(history: Sequence[Experiment], key: str) -> set[Any]:
    """Values this run has spent an experiment on, whether its session printed a helper
    line or died first: a failed try is still a value that did not pay."""
    values: set[Any] = set()
    for exp in history:
        for t in tries(_cells(exp)):
            if key == "name" and t.name:
                values.add(t.name)
            elif key in t.recipe:
                value = _value_of(t.recipe, key)
                if key not in _SPEC_KEYS or value is not None:
                    values.add(value)
        pair = attempted(exp)
        if pair is not None and pair[1] is not None and _LEVER_KEY.get(pair[0]) == key:
            values.add(model_name({"model": pair[1]}) if key == "name" else pair[1])
    return values


def _score(exp: Experiment) -> float | None:
    r = exp.result
    return (
        r.metrics.primary_value if r is not None and r.succeeded and r.metrics is not None else None
    )


def _classes_spent(exp: Experiment) -> set[str]:
    spent = {str(lv) for lv in (exp.candidate.changes.get("levers_moved") or [])}
    pair = attempted(exp)
    if pair is not None:
        spent.add(pair[0])
    return spent


def _pivot_closed(
    history: Sequence[Experiment], best_score: float | None, direction: str
) -> set[str]:
    """Classes the last two experiments both spent an experiment on without either
    beating the best. A failed try counts: it spent the experiment either way."""
    if len(history) < 2 or best_score is None:
        return set()
    last_two = history[-2:]
    shared = _classes_spent(last_two[0]) & _classes_spent(last_two[1])

    def beat(e: Experiment) -> bool:
        s = _score(e)
        return s is not None and (s < best_score if direction == "minimize" else s > best_score)

    return set() if any(beat(e) for e in last_two) else shared


@dataclass(frozen=True)
class Ready:
    lever: str
    reason: str
    move: str
    # The canonical layer stack this move names, for the guard that refuses a brief whose
    # stack is not an entry's. Empty on every class that holds no stack.
    stack: str = ""
    # True when the harness chose the value because the ask named none. Such an entry may
    # steer the model, but it never becomes the harness's own brief.
    invented: bool = False


def ready(
    history: Sequence[Experiment],
    carried: Experiment | None,
    *,
    task: str,
    direction: str,
    findings: str = "",
    asks: str = "",
    median_width: int | None = None,
    default_size: int | None = None,
) -> list[Ready]:
    """The lever classes this run's evidence opens, each with the fact that opened it.
    A failure closes everything but its repair.

    ``asks`` is what the human typed into THIS iteration, a recorded fact like any other:
    it adds an entry, and every guard the supervisor runs still runs on the brief."""
    last = history[-1] if history else None
    best = best_try(carried)
    recipe = recipe_of(carried)
    size = int(recipe.get("image_size") or default_size or 64)
    asked = _asked(asks, history, recipe)
    # An ask that wrote its own stack leads; one the harness filled in follows the ladder,
    # so a vague ask can never become the fallback brief.
    led = [r for r in asked if not r.invented]
    trailing = [r for r in asked if r.invented]
    if last is not None and (repair := _repair(history, recipe)) is not None:
        return [repair, *led, *trailing]
    if last is not None:
        cut = submitted_try(_cells(last))
        if cut is not None and cut.kind == "fit" and cut.cut:
            ran = max(1, cut.ran or 1)
            return [
                Ready(
                    "epochs",
                    f"the last fit ran {cut.ran} of the {cut.asked or cut.planned} epochs it "
                    "asked for before the budget cut it",
                    f"retry that try as {fit_call(cut.recipe, epochs=ran)}, the epochs that fit",
                ),
                *led,
                *trailing,
            ]
    best_score = _score(carried) if carried is not None else None
    closed = _pivot_closed(history, best_score, direction)
    out: list[Ready] = []
    spe = best.seconds_per_epoch if best is not None else None
    epochs = int(recipe.get("epochs") or BENCH_EPOCHS)
    name = model_name(recipe)
    own_code = bool(recipe.get("model"))
    if not recipe or name in FROM_ZERO:
        out.append(
            Ready(
                "backbone",
                "no pretrained model has scored this run",
                "fine-tune resnet18 through fit(), all layers, 3 epochs, at the session image size",
            )
        )
    else:
        if not own_code:
            stronger = [
                b
                for b, rank in STRENGTH.items()
                if rank > STRENGTH.get(name, len(STRENGTH))
                and b not in _tried_values(history, "backbone")
            ]
            affordable = [
                b
                for b in stronger
                if spe is not None and spe * epochs * COST[b] / COST.get(name, 1.0) <= BUDGET * 0.9
            ]
            if affordable and spe is not None:
                pick = affordable[-1]
                cost = spe * epochs * COST[pick] / COST.get(name, 1.0)
                out.append(
                    Ready(
                        "backbone",
                        f"{name} took {spe:.0f}s an epoch, so {pick} fits the budget at about "
                        f"{cost:.0f}s",
                        f"keep the recipe and swap the backbone to {pick}",
                    )
                )
        bigger = min(224, size * 2)
        if spe is not None and size < 224 and bigger not in _tried_values(history, "image_size"):
            cost = spe * epochs * (bigger / size) ** 2
            if cost <= BUDGET * 0.9:
                why = (
                    f"the images are {median_width} px wide and the best trains at {size}"
                    if median_width and median_width >= 2 * size
                    else f"a pretrained network sees more detail above {size} px"
                )
                out.append(
                    Ready(
                        "image-size",
                        f"{why}, and {bigger} px fits the budget at about {cost:.0f}s",
                        f"keep the best and train at {bigger} px",
                    )
                )
        more = min(MAX_EPOCHS, epochs * 2)
        room = spe is not None and spe * more <= BUDGET * 0.9 and epochs < MAX_EPOCHS
        if best is not None and room and not own_code and best.rising and not best.cut:
            out.append(
                Ready(
                    "epochs",
                    f"training was still rising at the last epoch ({best.train[-2]:.3f} -> "
                    f"{best.train[-1]:.3f})",
                    f"keep the recipe and train {more} epochs",
                )
            )
        elif best is not None and room and own_code:
            # Own code prints no epoch trail, so "still rising" cannot be read; what the
            # MODEL line does carry is how few epochs the model actually trained.
            out.append(
                Ready(
                    "epochs",
                    f"your own {name} trained only {epochs} epochs in {best.recipe.get('seconds')}s",
                    f"keep that code and train {more} epochs",
                )
            )
        gap = best.gap if best is not None else None
        if gap is not None and best is not None and gap > GAP.get(best.train_key, 0.03):
            if recipe.get("augment") != "flip_crop":
                out.append(
                    Ready(
                        "augmentation",
                        f"training sits {gap:.3f} above validation",
                        "keep the best and set augment to flip_crop",
                    )
                )
            if task == "classification" and not recipe.get("label_smoothing"):
                out.append(
                    Ready(
                        "regularisation",
                        f"training sits {gap:.3f} above validation",
                        "keep the best and set label_smoothing to 0.1",
                    )
                )
        depth = {
            "none": "no layers",
            "head": "the head only",
            "last_block": "the last stage and the head",
        }.get(str(recipe.get("unfreeze")))
        if not own_code and depth is not None:
            out.append(
                Ready(
                    "fine-tune-depth",
                    f"the best trains {depth}",
                    "keep the recipe and fine-tune all layers",
                )
            )
    untried = [n for n in models_named(findings) if n not in _tried_values(history, "name")]
    if untried:
        out.append(
            Ready(
                "own-model",
                f"a literature finding names {' and '.join(untried[:2])}",
                f"write torch code for {untried[0]} with pretrained weights, all layers, at the "
                "input size its pretrained_cfg names",
            )
        )
    opened = {r.lever for r in asked}
    return [*led, *(r for r in out if r.lever not in closed and r.lever not in opened), *trailing]


def _stack_kind(spec: arch.Spec) -> tuple[str | None, str]:
    """Which recipe key a written stack belongs to, or why it is neither. A stack that
    opens with a conv is a whole network; anything else can only be a head."""
    looks = "layers" if spec and spec[0][0] == "conv" else "head"
    try:
        arch.check(spec, looks)
    except arch.RecipeError as exc:
        return None, str(exc)
    return looks, ""


def _backbone_named(text: str) -> str:
    """The one pretrained network the words name, for the head entry that has to carry
    it: a brief names one class, so the backbone the user asked for travels in the move."""
    named = {b for b in STRENGTH if b not in FROM_ZERO and b in text.lower().replace("-", "_")}
    return named.pop() if len(named) == 1 else ""


def _stack_move(stack: str) -> str:
    return f"train a network from zero through fit(), with layers {stack}"


def _head_move(stack: str, recipe: dict[str, Any], ask: str) -> str:
    """The whole ask as ONE move. A brief may name one class, so the backbone the user
    named and the depth they asked for ride along inside the head entry."""
    name = _backbone_named(ask) or model_name(recipe)
    if not name or name in FROM_ZERO or recipe.get("model"):
        name = "resnet18"
    if _ASK_BLOCK.search(ask):
        depth = "the last stage and the head (unfreeze last_block)"
    else:
        depth = "all layers (unfreeze all)"
    return f"fine-tune {name} through fit(), {depth}, {BENCH_EPOCHS} epochs, with head {stack}"


def _spent_asks(history: Sequence[Experiment], ask: str) -> set[str]:
    """The classes an experiment these same words already steered spent. One ask buys one
    experiment per class; after that the run's own closed rule decides, so a steer cannot
    pin the rest of the run on a stack the model keeps mis-copying."""
    out: set[str] = set()
    for exp in history:
        stamped = exp.candidate.changes.get("user_guidance")
        if isinstance(stamped, str) and stamped.strip() and stamped.strip() in ask:
            out |= _classes_spent(exp)
    return out


def _asked(ask: str, history: Sequence[Experiment], recipe: dict[str, Any]) -> list[Ready]:
    """What a one-shot user ask opens, with the ask itself as the recorded fact. A word of
    refusal opens nothing, a value already tried stays shut, a spent ask stays spent, and
    every brief guard still runs on whatever the model writes from it."""
    if not ask.strip() or _NO.search(ask):
        return []
    spec = arch.found_strict(ask)
    key, _ = _stack_kind(spec) if spec is not None else (None, "")
    invented = False
    if key is None:
        if _ASK_ZERO.search(ask):
            key, invented = "layers", True
        elif _ASK_HEAD.search(ask):
            key, invented = "head", True
        else:
            return []
        spec = arch.found_strict(arch.EXAMPLE if key == "layers" else arch.HEAD_EXAMPLE)
    if spec is None:  # pragma: no cover - the two examples parse, this keeps mypy honest
        return []
    lever = "layer-stack" if key == "layers" else "custom-head"
    stack = arch.text(spec, key)
    if lever in _spent_asks(history, ask) or stack in _tried_values(history, key):
        return []
    what = "a network from zero" if key == "layers" else "its own head"
    reason = (
        f"the user asked for {what} and named no layers, so this is the example stack"
        if invented
        else f"the user asked for {'this layer stack' if key == 'layers' else 'this head'}"
    )
    move = _stack_move(stack) if key == "layers" else _head_move(stack, recipe, ask)
    return [Ready(lever, reason, move, stack=stack, invented=invented)]


def ask_note(text: str) -> tuple[str, str]:
    """What the harness reads in a typed ask, BEFORE the note is clipped: the canonical
    stack to store in front of it, because a clip mid-stack still parses as a shorter
    network, and one line for the user. Two empty strings mean the ask named no layer
    lever, and the run says nothing about it."""
    spec = arch.found_strict(text)
    key, why = _stack_kind(spec) if spec is not None else (None, "")
    vague = _ASK_ZERO.search(text) or _ASK_HEAD.search(text)
    if key is None and not (vague or why):
        return "", ""
    if _NO.search(text):
        return "", "read as a limit on the layers, so it opens no lever"
    if key is None and why:
        return "", f"read a layer stack and refused it: {why}"
    if key is None:
        what = "a network from zero" if _ASK_ZERO.search(text) else "your own head"
        return "", f"read as {what} with no layers named; the example stack is what opens"
    stack = arch.text(spec, key)
    return f"{key} {stack}", f"read {key} {stack}; it opens the next experiment"


_WHERE = re.compile(r"batch_size=(\d+), image_size=(\d+)")


def _repair(history: Sequence[Experiment], carried: dict[str, Any]) -> Ready | None:
    """The one move a failure opens, once: a second failure of the same class is left to
    the ladder, which by then counts that value as tried."""
    last = history[-1]
    lever = lever_class(last.hypothesis or "") or ""
    if len(history) >= 2:
        before = history[-2]
        if (
            before.result is not None
            and not before.result.succeeded
            and lever_class(before.hypothesis or "") == lever
        ):
            return None
    cells = _cells(last)
    errors = " ".join(_parts(c)[2] for c in cells if _agent(c))
    failed = _failed_recipe(last, carried)
    if any(t.kind == "oom" for t in tries(cells)) and _score(last) is None:
        # An out-of-memory fit the session recovered from and submitted is not a
        # failure to repair: the run has its number, and the ladder has its evidence.
        where = _WHERE.search(errors)
        batch, size = (int(where.group(1)), int(where.group(2))) if where else (64, None)
        change = (
            {"batch_size": batch // 2}
            if batch > 16 or size is None
            else {"image_size": max(32, size // 2)}
        )
        if size is not None:
            failed = {**failed, "image_size": size}
        at = f" at batch_size {batch}, image_size {size}" if where else ""
        # A from-zero net holds every activation of a map nothing has shrunk yet, so the
        # pool is the fix that costs nothing; a smaller batch is the fallback.
        pool = ", or add a pool after the first conv" if failed.get("layers") else ""
        return Ready(
            lever or "image-size",
            f"a fit in the last experiment ran out of memory{at}",
            f"retry that try as {fit_call(failed, **change)}{pool}",
            stack=_repair_stack(lever, failed),
        )
    if last.result is not None and not last.result.succeeded:
        if any(t.kind == "model" for t in tries(cells)) or lever == "own-model":
            return Ready(
                "backbone",
                "the last experiment's own-code model failed",
                "return to fit() with the best recipe and swap only the backbone to resnet50",
            )
        first = (last.result.error or "").strip().splitlines()[:1]
        why = (
            f"the last experiment failed ({first[0][:80]})"
            if first
            else "the last experiment failed"
        )
        return Ready(
            lever or "backbone",
            why,
            f"repair that try, {fit_call(failed)}: fix the error it names and run it again",
            stack=_repair_stack(lever, failed),
        )
    return None


def _repair_stack(lever: str, failed: dict[str, Any]) -> str:
    """The stack a repair of a layer try names, so the guard that compares the brief's
    stack to the line's has something to compare against."""
    key = _LEVER_KEY.get(lever, "")
    return _value_of(failed, key) or "" if lever in LAYER_LEVERS else ""


def _failed_recipe(exp: Experiment, carried: dict[str, Any]) -> dict[str, Any]:
    """What the failed experiment was training: the carried best with the brief's own
    change laid over it, since no helper line survived to say."""
    recipe = {k: carried[k] for k in ("backbone", "image_size", "epochs") if carried.get(k)}
    if carried.get("model"):
        recipe = {"model": model_name(carried)}
    pair = attempted(exp)
    if pair is None or pair[1] is None:
        return recipe
    lever, value = pair
    if lever == "own-model":
        return {"model": str(value)}
    key = _LEVER_KEY.get(lever)
    if key == "layers":
        # Only the stack: merge() starts a from-zero fit from the from-zero reference, so
        # carrying the pretrained best's three epochs would retrain the network at three.
        return {"layers": value}
    return {**recipe, key: value} if key else recipe


def ready_line(items: Sequence[Ready]) -> str:
    if not items:
        return "Levers ready now: none; no lever's evidence fires on this run's numbers."
    return (
        "Levers ready now: "
        + "; ".join(f"{r.lever}: {r.move} (because {r.reason})" for r in items)
        + "."
    )


def ledger_line(history: Sequence[Experiment]) -> str:
    """A layer class is listed only once a try has spent an experiment on it, so a run
    that never opens one sends the bytes it sent before the classes existed."""
    done = tried(history)
    shown = [lv for lv in LEVERS if lv not in LAYER_LEVERS or lv in done]
    yes = ", ".join(lv for lv in shown if lv in done) or "none"
    no = ", ".join(lv for lv in shown if lv not in done) or "none"
    return f"Levers tried: {yes} | Levers NOT yet tried: {no}"


def fallback_move(items: Sequence[Ready]) -> tuple[str, str] | None:
    """The harness's own brief when the model will not write one. An entry whose value the
    harness filled in for a vague ask is not one: the run would then be briefing itself on
    a network nobody chose."""
    first = next((r for r in items if not r.invented), None)
    if first is None:
        return None
    return (
        f"evidence: {first.lever}",
        f"next: {first.lever}: {first.move} (because {first.reason}).",
    )


def tried_components(history: Sequence[Experiment]) -> list[str]:
    seen: dict[str, None] = {}
    for exp in history:
        text = describe(recipe_of(exp))
        if text:
            seen.setdefault(text, None)
    return list(seen)


_VALUE_WORDS = {
    "image-size": ("px", "pixels?", "image_size"),
    "epochs": ("epochs?",),
}


def _ints_for(text: str, words: Sequence[str], low: int, high: int) -> set[int]:
    """Integers this text offers FOR one lever: the word sits against the number, either
    just before it ("image_size=128", "epochs 6") or just after it ("128 px")."""
    found: set[int] = set()
    for m in re.finditer(r"(?<![\d.])(\d{1,3})(?![\d.])", text):
        value = int(m.group(1))
        if not low <= value <= high:
            continue
        before = text[max(0, m.start() - 20) : m.start()].lower()
        after = text[m.end() : m.end() + 12].lower()
        if any(
            re.search(rf"\b(?:{word})[\s=:]*$", before) or re.match(rf"\s*(?:{word})\b", after)
            for word in words
        ):
            found.add(value)
    return found


def _spec_text(clause: str, key: str) -> str | None:
    """The one stack a change clause states, in the canonical text. Loose here on purpose:
    a brief is a model's own words, and the ask and the findings are the strict readers."""
    written = _IN_CALL[key].search(clause) if key in _IN_CALL else None
    try:
        spec = arch.parse(written.group(1) if written else clause, key)
        if spec is None:
            return None
        arch.check(spec, key)
    except arch.RecipeError:
        return None
    return arch.text(spec, key)


def proposed_value(lever: str, clause: str) -> Any:
    """The one value the change clause proposes for its lever, or None unless it names
    exactly one."""
    move = clause.lower()
    if lever in LAYER_LEVERS:
        return _spec_text(clause, _LEVER_KEY[lever])
    if lever == "backbone":
        named = {b for b in STRENGTH if b in move.replace("-", "_")}
        return named.pop() if len(named) == 1 else None
    if lever in _VALUE_WORDS:
        low, high = (32, 384) if lever == "image-size" else (1, 30)
        values = _ints_for(move, _VALUE_WORDS[lever], low, high)
        return values.pop() if len(values) == 1 else None
    if lever == "augmentation":
        return "flip_crop" if ("flip_crop" in move or "crop" in move) else None
    return None


def banked(brief: str, carried: Experiment | None) -> str | None:
    lever = lever_class(brief)
    if carried is None or lever not in _LEVER_KEY or lever == "own-model":
        return None
    value = proposed_value(lever, change_clause(brief))
    key = _LEVER_KEY[lever]
    if value is not None and _value_of(recipe_of(carried), key) == value:
        return f"the carried best already trains with {key}={value}"
    return None


def measured_lost(
    brief: str, history: Sequence[Experiment], carried: Experiment | None, direction: str
) -> str | None:
    lever = lever_class(brief)
    best = _score(carried) if carried is not None else None
    if best is None or lever not in _LEVER_KEY or lever == "own-model":
        return None
    value = proposed_value(lever, change_clause(brief))
    if value is None:
        return None
    key = _LEVER_KEY[lever]
    for exp in reversed(history):
        score = _score(exp)
        if exp is carried or score is None or _value_of(recipe_of(exp), key) != value:
            continue
        if (score > best) if direction == "minimize" else (score < best):
            return f"{key}={value} was already submitted this run (holdout {score:.4f}, did not beat {best:.4f})"
    return None


def missing_value(brief: str) -> str | None:
    """Why a move names its class but not the one value it changes, or None."""
    lever = lever_class(brief)
    clause = change_clause(brief)
    if lever == "own-model":
        if len(models_named(clause)) != 1:
            return "an own-model move names exactly ONE model by the name its library loads"
        return None
    if lever in LAYER_LEVERS:
        if proposed_value(lever, clause) is None:
            shape = arch.HEAD_EXAMPLE if _LEVER_KEY[lever] == "head" else arch.EXAMPLE
            return f"a {lever} move gives ONE layer stack, written like {shape}"
        return None
    if lever in _LEVER_KEY and lever != "augmentation" and proposed_value(lever, clause) is None:
        return f"the {lever} move states exactly ONE new value"
    return None


def brief_call(brief: str) -> str | None:
    """The brief's layer change as the `fit()` call that makes it, or None for any other
    class. The head call carries the backbone and the depth the move named, so the whole
    ask runs as one experiment."""
    lever = lever_class(brief)
    if lever not in LAYER_LEVERS:
        return None
    clause = change_clause(brief)
    value = proposed_value(lever, clause)
    if value is None:
        return None
    key = _LEVER_KEY[lever]
    call: dict[str, Any] = {}
    if key == "head" and (name := _backbone_named(clause)):
        call["backbone"] = name
    call[key] = value
    if key == "head" and _ASK_BLOCK.search(clause):
        call["unfreeze"] = "last_block"
    inside = ", ".join(
        f"{k}={arch.code(v, k)}" if k in _SPEC_KEYS else f"{k}={v!r}" for k, v in call.items()
    )
    return f"fit({inside})"
