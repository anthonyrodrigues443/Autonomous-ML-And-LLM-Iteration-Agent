"""The image session inside the kernel: the twin of `prompt_runtime`.

`start()` is what the preamble calls. It reads the three files the host wrote, decodes
the images once at the session size, holds back a fifth of each class to score a try on,
and returns the names the agent's cells work with. torch loads before anything else
touches the device, with the pool cap and the fallback flag already set.

Every helper prints one tagged line the host reads afterwards: ``FIT`` when `fit()`
returns, ``MODEL`` when `evaluate()` scores the agent's own model, and ``SUBMITTED``
when a submit helper writes predictions. Those lines, not the text of a cell, are how
the run knows which lever moved.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from iterate.core import codegen

if TYPE_CHECKING:
    import pandas as pd

    from iterate.targets.dl import Recipe, Runner

VAL_SHARE = 0.2
DECODE_THREADS = 8
DECODE_CHUNK = 256
SESSION_JSON = ".vision-session.json"
BENCH_EPOCHS = 3
SIZE_STEP = 32
SIZE_FLOOR = 32


def set_device_env() -> None:
    # Read at torch's static init: they must be set before the first import of torch.
    from iterate.targets.dl import MPS_HIGH_RATIO, MPS_LOW_RATIO

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    if "PYTORCH_MPS_HIGH_WATERMARK_RATIO" not in os.environ:
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = MPS_HIGH_RATIO
        os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = MPS_LOW_RATIO


def merge(
    best: Recipe, changes: dict[str, Any], task: str, *, baseline: Recipe
) -> tuple[Recipe, dict[str, Any]]:
    """What `fit(**changes)` trains, and the recipe it started from.

    Every fit starts from the carried best. A switch between the plain CNN and a
    pretrained network starts from that kind's reference instead, since the plain CNN's
    20 epochs are not a legal fine-tune.
    """
    from iterate.targets.dl import SCRATCH, Recipe

    start = best
    new = changes.get("backbone", best.backbone)
    if (new in SCRATCH) != (best.backbone in SCRATCH):
        start = (
            replace(baseline, seed=best.seed)
            if new in SCRATCH
            else Recipe(unfreeze="all", epochs=BENCH_EPOCHS, seed=best.seed)
        )
    base = asdict(start)
    if "unfreeze" in changes and "epochs" not in changes:
        if changes["unfreeze"] == "none":
            base["epochs"] = 0
        elif base["epochs"] == 0:
            base["epochs"] = BENCH_EPOCHS
    elif "epochs" in changes and "unfreeze" not in changes and new not in SCRATCH:
        if changes["epochs"] == 0:
            base["unfreeze"] = "none"
        elif base["unfreeze"] == "none":
            base["unfreeze"] = "all"
    return Recipe.from_changes({**base, **changes}, task=task), asdict(start)


@dataclass(frozen=True)
class Fit:
    """One fit on the fold. `line` is what was printed; the holdout predictions stay
    inside, and leave only through `submit`."""

    line: dict[str, Any]
    _holdout_out: np.ndarray = field(repr=False)

    @property
    def val(self) -> float:
        return float(self.line["val"])

    @property
    def recipe(self) -> dict[str, Any]:
        return dict(self.line)

    @property
    def epochs_run(self) -> int:
        return int(self.line["epochs_run"])

    @property
    def seconds(self) -> float:
        return float(self.line["seconds"])


class Session:
    def __init__(
        self,
        meta: dict[str, Any],
        train: pd.DataFrame,
        holdout: pd.DataFrame,
        *,
        workdir: Path,
        runner: Runner | None = None,
    ) -> None:
        from iterate.targets.dl import BASELINE, FIT_BUDGET_SECONDS, Recipe, TorchRunner

        self.workdir = workdir
        self.task = str(meta["task"])
        self.metric = str(meta["metric"])
        self.average = meta.get("average")
        self.size = int(meta["image_size"])
        self.budget = float(meta.get("budget_seconds") or FIT_BUDGET_SECONDS)
        column, target = meta["image_column"], meta["target"]
        self.train_paths = [str(p) for p in train[column]]
        self.holdout_paths = [str(p) for p in holdout[column]]
        self.runner: Runner = runner if runner is not None else TorchRunner()
        self.classes: list[Any] = list(meta["classes"] or [])
        raw = train[target].tolist()
        if self.task == "classification":
            index = {str(c): i for i, c in enumerate(self.classes)}
            self.labels = np.array([index[str(v)] for v in raw], dtype=np.int64)
        else:
            self.labels = np.asarray(raw, dtype=np.float64)
        self.labels.flags.writeable = False
        self.val_idx, self.fit_idx = self._fold(int(meta.get("seed") or 42))
        self.baseline = Recipe.from_changes(
            {**asdict(BASELINE), **(meta.get("baseline") or {})}, task=self.task
        )
        self.best = self._start_from()
        self._cell_start: float | None = None
        self._fits_in_cell = 0
        self._models: dict[str, dict[str, Any]] = {}
        self._pixels: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._features: dict[tuple[str, int], np.ndarray] = {}

    # ─── the fold, the carried recipe, the session's own state ───

    def _fold(self, seed: int) -> tuple[np.ndarray, np.ndarray]:
        """A fifth of each class, as tables hold back a fifth of the rows. The floor of
        one row per class keeps a column for every class when a probability metric
        scores the fold."""
        rng = np.random.default_rng(seed)
        n = len(self.labels)
        if self.task == "classification":
            chosen: list[int] = []
            for k in range(len(self.classes)):
                rows = np.flatnonzero(self.labels == k)
                if len(rows):
                    chosen.extend(rng.permutation(rows)[: max(1, round(len(rows) * VAL_SHARE))])
            val = np.sort(np.array(chosen, dtype=np.int64))
        else:
            val = np.sort(rng.permutation(n)[: max(1, round(n * VAL_SHARE))])
        fit = np.setdiff1d(np.arange(n), val)
        val.flags.writeable = False
        fit.flags.writeable = False
        return val, fit

    def _recipe_keys(self, saved: dict[str, Any]) -> dict[str, Any]:
        from iterate.targets.dl import Recipe

        return {k: v for k, v in saved.items() if k in Recipe.__dataclass_fields__}

    def _saved(self) -> dict[str, Any] | None:
        """What this session last fitted, so a restart rebuilds the same state."""
        try:
            saved = json.loads((self.workdir / SESSION_JSON).read_text())["best"]
        except (OSError, ValueError, KeyError, TypeError):
            return None
        return self._recipe_keys(saved) if isinstance(saved, dict) else None

    def _carried(self) -> dict[str, Any] | None:
        """The best recipe the run carried in, written by the host before the session.

        A payload naming a `model` came from the agent's own code, where `epochs` counts
        loops it wrote itself; it is not a fit recipe and the own-code best travels back
        as code, not as a recipe."""
        try:
            saved = json.loads((self.workdir / codegen.INCUMBENT_JSON).read_text())
        except (OSError, ValueError):
            return None
        if not isinstance(saved, dict) or saved.get("model"):
            return None
        return self._recipe_keys(saved)

    def _start_from(self) -> Recipe:
        """The recipe the first fit of this session departs from. Carried keys fill in
        over this run's baseline, never over `Recipe`'s own defaults, and a carry the
        runner refuses falls back to the baseline: no file the host wrote may raise out
        of the preamble, which would leave the cell with no helper bound."""
        from iterate.targets.dl import Recipe, RecipeError

        base = asdict(self.baseline)
        carried = self._saved() or self._carried()
        if carried:
            try:
                return Recipe.from_changes({**base, **carried}, task=self.task)
            except RecipeError as exc:
                print(f"the carried recipe was dropped ({exc}); starting from the baseline")
        return Recipe.from_changes(base, task=self.task)

    def _remember(self, recipe: Recipe) -> None:
        (self.workdir / SESSION_JSON).write_text(json.dumps({"best": asdict(recipe)}))

    # ─── the cell clock ───

    def begin_cell(self) -> None:
        from iterate.targets.dl import release

        forget_last_cell()
        release(self.device)
        self._fits_in_cell = 0
        self._cell_start = time.perf_counter()

    @property
    def device(self) -> str:
        return self.runner.device

    def _tick(self) -> float:
        return self._cell_start if self._cell_start is not None else time.perf_counter()

    def seconds_left(self) -> float:
        return max(0.0, self._tick() + self.budget - time.perf_counter())

    def seconds_used(self) -> float:
        return time.perf_counter() - self._tick()

    # ─── pixels ───

    def pixels(self, size: int) -> tuple[np.ndarray, np.ndarray]:
        if size not in self._pixels:
            # The session size is kept; any other size makes room for this one.
            for other in [s for s in self._pixels if s != self.size]:
                del self._pixels[other]
            self._pixels[size] = (
                self._decoded(self.train_paths, size),
                self._decoded(self.holdout_paths, size),
            )
        return self._pixels[size]

    def _fits_in_ram(self, size: int) -> bool:
        from iterate.targets.dl import PIXEL_RAM_SHARE, _ram_bytes

        n = len(self.train_paths) + len(self.holdout_paths)
        return n * 3 * size * size <= PIXEL_RAM_SHARE * _ram_bytes()

    def _decoded(self, paths: list[str], size: int) -> np.ndarray:
        from iterate.targets.dl import decode

        if not self._fits_in_ram(size):
            n = len(self.train_paths) + len(self.holdout_paths)
            raise ValueError(
                f"{n} images at {size} px take {n * 3 * size * size / 2**30:.1f} GiB decoded, "
                "over a quarter of this machine's RAM; lower image_size"
            )
        chunks = [paths[i : i + DECODE_CHUNK] for i in range(0, len(paths), DECODE_CHUNK)]
        with ThreadPoolExecutor(DECODE_THREADS) as pool:
            parts = list(pool.map(lambda chunk: decode(chunk, size), chunks))
        array = np.concatenate(parts) if parts else np.zeros((0, 3, size, size), np.uint8)
        array.flags.writeable = False
        return array

    def largest_size_that_fits(self, wanted: int) -> int:
        """The session size, lowered in steps of 32 until the decoded pixels fit. Without
        it a set too big to decode leaves the preamble with no helpers bound, so the
        advice to lower image_size could not be taken."""
        size = wanted
        while size > SIZE_FLOOR and not self._fits_in_ram(size):
            size = max(SIZE_FLOOR, (size - 1) // SIZE_STEP * SIZE_STEP)
        return size

    # ─── fit ───

    def fit(self, **changes: Any) -> Fit:
        from iterate.targets.dl import FitJob, RecipeError

        tick = self._tick()
        recipe, start = merge(self.best, changes, self.task, baseline=self.baseline)
        size = recipe.image_size or self.size
        recipe = replace(recipe, image_size=size)
        # A start with no size of its own is the session size, not the size this fit
        # arrived at: the lever gate compares the two dicts key by key, and taking the
        # fit's own size would read an explicit `image_size=` as no move.
        start["image_size"] = start.get("image_size") or self.size
        train_px, holdout_px = self.pixels(size)
        stack = np.concatenate([train_px[self.val_idx], holdout_px])
        n_val = len(self.val_idx)
        fit_labels, centre, spread = self._fit_labels()
        outputs = 1 if self.task == "regression" else len(self.classes)
        if recipe.unfreeze == "none":
            out = self._probe(recipe.backbone, size, train_px, stack, fit_labels)
            planned, ran = 0, 0
        else:
            head = None
            if recipe.head_init == "probe":
                head = self._probe_head(recipe.backbone, size, train_px, fit_labels)
            job = FitJob(
                np.asarray(train_px[self.fit_idx]),
                stack,
                fit_labels,
                recipe,
                outputs,
                head,
                tick + self.budget,
                print,
                task=self.task,
            )
            try:
                report = self.runner.fit(job)
            except RecipeError as exc:
                if self._fits_in_cell:
                    raise RecipeError(
                        f"{exc}. This cell already ran a fit, which spent the budget: run this "
                        "fit at the top of a new cell"
                    ) from None
                raise
            out, planned, ran = report.outputs, report.epochs_planned, report.epochs_run
        if self.task == "regression":
            out = np.asarray(out, dtype=np.float64) * spread + centre
        val_out = out[:n_val]
        line = {
            **asdict(recipe),
            "epochs_planned": planned,
            "epochs_run": ran,
            "seconds": round(time.perf_counter() - tick),
            "val": self._score(val_out),
            **self._plain(val_out),
            "from": start,
        }
        self.best = recipe
        self._fits_in_cell += 1
        self._remember(recipe)
        print("FIT " + json.dumps(line, default=str))
        print(
            f"val {self.metric} = {line['val']:.4f} ({recipe.backbone} {size}px, "
            f"{ran}/{recipe.epochs} epochs, {line['seconds']}s)"
        )
        return Fit(line, out[n_val:])

    def _fit_labels(self) -> tuple[np.ndarray, float, float]:
        labels = self.labels[self.fit_idx]
        if self.task != "regression":
            return labels, 0.0, 1.0
        centre, spread = float(labels.mean()), float(labels.std()) or 1.0
        return ((labels - centre) / spread).astype(np.float32), centre, spread

    def _embed(self, backbone: str, size: int, pixels: np.ndarray, part: str) -> np.ndarray:
        key = (f"{backbone}:{part}", size)
        if key not in self._features:
            self._features[key] = self.runner.embed(np.asarray(pixels), backbone=backbone)
        return self._features[key]

    def _probe(
        self, backbone: str, size: int, train_px: np.ndarray, stack: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.preprocessing import StandardScaler

        features = self._embed(backbone, size, train_px[self.fit_idx], "fit")
        scaler = StandardScaler().fit(features)
        held = scaler.transform(self._embed(backbone, size, stack, "stack"))
        if self.task == "regression":
            return np.asarray(Ridge(alpha=1.0).fit(scaler.transform(features), y).predict(held))
        model = LogisticRegression(max_iter=3000).fit(scaler.transform(features), y)
        return _widen(model.predict_proba(held), model.classes_, len(self.classes))

    def _probe_head(
        self, backbone: str, size: int, train_px: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.preprocessing import StandardScaler

        features = self._embed(backbone, size, train_px[self.fit_idx], "fit")
        scaler = StandardScaler().fit(features)
        if self.task == "regression":
            model: Any = Ridge(alpha=1.0).fit(scaler.transform(features), y)
        else:
            model = LogisticRegression(max_iter=3000).fit(scaler.transform(features), y)
        weight = np.atleast_2d(model.coef_) / scaler.scale_
        bias = np.atleast_1d(model.intercept_) - weight @ scaler.mean_
        if self.task == "classification":
            full_w = np.zeros((len(self.classes), weight.shape[1]))
            full_b = np.full(len(self.classes), -1e4)
            seen = np.asarray(model.classes_, dtype=np.int64)
            if len(weight) == 1:
                weight, bias = np.vstack([np.zeros_like(weight), weight]), np.array([0.0, bias[0]])
            full_w[seen], full_b[seen] = weight, bias
            weight, bias = full_w, full_b
        return weight.astype(np.float32), bias.astype(np.float32)

    # ─── scoring ───

    def _score(self, out: np.ndarray) -> float:
        from iterate.core.scoring import requires_proba, score

        truth = self.labels[self.val_idx]
        if self.task == "regression":
            values = score("regression", truth, out, include=(self.metric,))
        else:
            names = [self.classes[i] for i in truth]
            guess = [self.classes[i] for i in np.asarray(out).argmax(1)]
            values = score(
                "classification",
                names,
                guess,
                y_proba=out if requires_proba(self.metric) else None,
                average=self.average,
                include=(self.metric,),
            )
        return float(values[self.metric])

    def _plain(self, out: np.ndarray) -> dict[str, float]:
        """A second number in words everyone knows, beside a metric that may not be.
        The supervisor reads it to say what a try actually did."""
        truth = self.labels[self.val_idx]
        if self.task == "regression":
            spread = float(((truth - truth.mean()) ** 2).sum())
            if spread == 0.0:
                return {}
            error = float(((truth - np.asarray(out, dtype=np.float64)) ** 2).sum())
            return {"val_r2": round(1.0 - error / spread, 4)}
        guess = np.asarray(out).argmax(1)
        return {"val_accuracy": round(float((guess == truth).mean()), 4)}

    # ─── the agent's own code ───

    def batches(self, rows: Any, size: int = 64) -> list[np.ndarray]:
        order = np.random.default_rng().permutation(np.asarray(rows))
        return [order[i : i + size] for i in range(0, len(order), size)]

    def as_input(self, pixels: np.ndarray) -> Any:
        from iterate.targets.dl import _to_device, _torch

        torch = _torch()
        return _to_device(torch, np.asarray(pixels), torch.device(self.device))

    def as_labels(self, rows: Any) -> Any:
        from iterate.targets.dl import _torch

        torch = _torch()
        values = torch.from_numpy(np.asarray(self.labels[np.asarray(rows)]))
        kind = torch.long if self.task == "classification" else torch.float32
        return values.to(device=self.device, dtype=kind)

    def predict(self, model: Any, pixels: np.ndarray, size: int = 256) -> np.ndarray:
        from iterate.targets.dl import _torch

        torch = _torch()
        model.eval()
        parts = []
        with torch.no_grad():
            for i in range(0, len(pixels), size):
                out = model(self.as_input(pixels[i : i + size]))
                # A Hugging Face model answers with an output object, not a tensor.
                out = getattr(out, "logits", out)
                out = (out[0] if isinstance(out, tuple | list) else out).float()
                if self.task == "classification":
                    parts.append(torch.softmax(out, dim=1).cpu())
                else:
                    parts.append(out.reshape(len(out), -1)[:, 0].cpu())
        return np.asarray(torch.cat(parts).numpy(), dtype=np.float64)

    def evaluate(self, outputs: Any, *, model: str, **recipe: Any) -> float:
        """Score the agent's own model on the fold. `model` is required: without a name
        the run cannot record what was tried."""
        out = self._checked(outputs, len(self.val_idx), "VAL_IDX")
        line = {
            "model": model,
            **recipe,
            "seconds": round(self.seconds_used()),
            "val": self._score(out),
            **self._plain(out),
        }
        self._models[model] = line
        print("MODEL " + json.dumps(line, default=str))
        print(f"val {self.metric} = {line['val']:.4f} ({model})")
        return float(line["val"])

    # ─── submitting ───

    def submit(self, fit: Fit) -> None:
        if not isinstance(fit, Fit):
            helper = (
                "submit_numbers(values, model=NAME)"
                if self.task == "regression"
                else "submit_probabilities(probs, model=NAME)"
            )
            raise TypeError(f"submit() takes what fit() returned; for your own model call {helper}")
        self._write(fit._holdout_out, dict(fit.line))

    def submit_probabilities(self, probs: Any, *, model: str) -> None:
        if self.task == "regression":
            raise ValueError("this run predicts a number: call submit_numbers(values, model=NAME)")
        out = _probabilities(_numpy(probs), len(self.holdout_paths), len(self.classes), "X_holdout")
        self._write(out, self._models.get(model, {"model": model}))

    def submit_numbers(self, values: Any, *, model: str) -> None:
        if self.task != "regression":
            raise ValueError(
                "this run predicts classes: call submit_probabilities(probs, model=NAME)"
            )
        out = _numbers(_numpy(values), len(self.holdout_paths), "X_holdout")
        self._write(out, self._models.get(model, {"model": model}))

    def _checked(self, outputs: Any, rows: int, order: str) -> np.ndarray:
        out = _numpy(outputs)
        if self.task == "regression":
            return _numbers(out, rows, order)
        return _probabilities(out, rows, len(self.classes), order)

    def _write(self, out: np.ndarray, line: dict[str, Any]) -> None:
        import pandas as pd

        if self.task == "regression":
            predictions = pd.Series(np.asarray(out, dtype=np.float64))
            (self.workdir / codegen.PROBABILITIES_CSV).unlink(missing_ok=True)
        else:
            # A float32 softmax leaves rows about 2.5e-7 off 1, inside the tolerance
            # these were accepted under and outside the one scikit-learn scores with.
            totals = out.sum(axis=1, keepdims=True)
            out = np.divide(out, totals, out=np.zeros_like(out), where=totals > 0)
            predictions = pd.Series([self.classes[i] for i in out.argmax(1)])
            pd.DataFrame(out).to_csv(
                self.workdir / codegen.PROBABILITIES_CSV, index=False, header=False
            )
        predictions.to_csv(self.workdir / codegen.PREDICTIONS_CSV, index=False, header=False)
        digest = hashlib.sha256((self.workdir / codegen.PREDICTIONS_CSV).read_bytes()).hexdigest()
        (self.workdir / codegen.RECIPE_JSON).write_text(
            json.dumps({**line, "predictions_sha256": digest}, default=str)
        )
        print("SUBMITTED " + json.dumps(line, default=str))
        print(f"submitted {len(out)} predictions for the holdout")


# ─── what IPython keeps of the last cell ───


def forget_last_cell() -> None:
    """Drop what IPython keeps of the last cell: its result, its traceback, and the
    `_` names, which it pushes into the hidden namespace as well as the visible one.
    Each one of the three pins a whole fit's device memory until the cell after next."""
    ip = _ipython()
    if ip is None:
        return
    ip.last_execution_result = None
    for formatter in (getattr(ip, "InteractiveTB", None), getattr(ip, "SyntaxTB", None)):
        if formatter is not None and hasattr(formatter, "tb"):
            formatter.tb = None
    hook = ip.displayhook
    for key in ("_", "__", "___"):
        cached = getattr(hook, key, None)
        for ns in (ip.user_ns, ip.user_ns_hidden):
            if key in ns and ns[key] is cached and cached is not None:
                ns[key] = ""
        setattr(hook, key, "")


def stop_output_cache() -> None:
    ip = _ipython()
    if ip is not None:
        ip.displayhook.do_full_cache = 0
        ip.displayhook.cache_size = 0


def _ipython() -> Any:
    try:
        from IPython.core.getipython import get_ipython
    except ImportError:
        return None
    return get_ipython()  # type: ignore[no-untyped-call]


# ─── shapes the helpers accept ───


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _probabilities(out: np.ndarray, rows: int, classes: int, order: str) -> np.ndarray:
    if out.shape != (rows, classes):
        raise ValueError(
            f"expected probabilities of shape ({rows}, {classes}): one row per image in "
            f"{order} order, one column per class in CLASSES order; got {out.shape}"
        )
    if not np.isfinite(out).all():
        raise ValueError("probabilities must be finite")
    if (out < 0).any() or not np.allclose(out.sum(axis=1), 1.0, atol=1e-3):
        raise ValueError(
            "each row must be probabilities that sum to 1; for logits, call "
            "torch.softmax(logits, dim=1) first"
        )
    return out


def _numbers(out: np.ndarray, rows: int, order: str) -> np.ndarray:
    out = out.reshape(-1) if out.ndim == 2 and out.shape[1] == 1 else out
    if out.shape != (rows,):
        raise ValueError(
            f"expected {rows} numbers, one per image in {order} order, in the label's own "
            f"units; got shape {out.shape}"
        )
    if not np.isfinite(out).all():
        raise ValueError(f"{int((~np.isfinite(out)).sum())} of {rows} predictions are not finite")
    return out


def _widen(proba: np.ndarray, seen: np.ndarray, classes: int) -> np.ndarray:
    full = np.zeros((len(proba), classes))
    full[:, np.asarray(seen, dtype=np.int64)] = proba
    return full


# ─── the session this kernel is running ───

_CURRENT: tuple[Session, dict[str, Any]] | None = None


def begin_cell(namespace: dict[str, Any]) -> None:
    """The first statement of every image cell: frees the last cell's device memory,
    starts this cell's fit clock, and puts back any session name a cell rebound."""
    if _CURRENT is None:
        return
    session, names = _CURRENT
    session.begin_cell()
    namespace.update(names)


def current() -> Session | None:
    return _CURRENT[0] if _CURRENT is not None else None


def start(workdir: str = ".") -> dict[str, Any]:
    """The preamble's one call: every name the agent's cells see."""
    import pandas as pd

    set_device_env()
    import torch

    stop_output_cache()

    folder = Path(os.path.realpath(workdir))
    with (folder / codegen.META_JSON).open() as handle:
        meta = json.load(handle)
    train = pd.read_csv(folder / codegen.TRAIN_CSV)
    holdout = pd.read_csv(folder / codegen.HOLDOUT_CSV)
    torch.manual_seed(int(meta.get("seed") or 42))
    session = Session(meta, train, holdout, workdir=folder)
    wanted = session.size
    session.size = session.largest_size_that_fits(wanted)
    tick = time.perf_counter()
    train_px, holdout_px = session.pixels(session.size)
    decoded = time.perf_counter() - tick
    names: dict[str, Any] = {
        "SESSION": session,
        "TASK": session.task,
        "CLASSES": session.classes or None,
        "DEVICE": session.device,
        "IMAGE_SIZE": session.size,
        "train_px": train_px,
        "holdout_px": holdout_px,
        "labels": session.labels,
        "FIT_IDX": session.fit_idx,
        "VAL_IDX": session.val_idx,
        "fit": session.fit,
        "submit": session.submit,
        "evaluate": session.evaluate,
        "submit_probabilities": session.submit_probabilities,
        "submit_numbers": session.submit_numbers,
        "seconds_left": session.seconds_left,
        "pixels": session.pixels,
        "batches": session.batches,
        "as_input": session.as_input,
        "as_labels": session.as_labels,
        "predict": session.predict,
    }
    global _CURRENT
    _CURRENT = (session, names)
    kind = (
        f"{len(session.classes)} classes"
        if session.task == "classification"
        else "a number (regression)"
    )
    if session.size != wanted:
        print(
            f"image_size lowered from {wanted} to {session.size}px: {len(train) + len(holdout)} "
            "images do not fit in a quarter of this machine's RAM at the larger size"
        )
    print(
        f"loaded: {len(train)} train / {len(holdout)} holdout images at {session.size}px "
        f"on {session.device} in {decoded:.1f}s; predicting {kind}"
    )
    print(
        f"fold: {len(session.fit_idx)} images to fit, {len(session.val_idx)} to validate "
        "(FIT_IDX, VAL_IDX), the same fold every session"
    )
    print("recipe now: " + json.dumps(asdict(session.best)))
    print("\n".join(codegen.vision_worked_example(session.task)))
    gc.collect()
    return names


__all__ = ["Fit", "Session", "begin_cell", "merge", "set_device_env", "start"]
