"""Running one cell: one version, one dataset, one repeat, fully isolated.

The governing idea is that a version is measured AT ITS OWN DEFAULTS. Only the
things that would otherwise make two cells incomparable get pinned — the data, the
metric, the model, the iteration budget. Everything a release changed about its own
behaviour is left alone, because that is the thing being measured. Forcing v0.4 to
run with its Researcher disabled so it "matches" v0.3 would measure a version that
was never shipped.

Isolation is per cell and deliberate about what it does NOT isolate:

*Memory is isolated.* Every cell gets its own `memory.db`. Sharing one would let a
previous cell's best score carry over as the next cell's baseline, so v0.5 would
inherit v0.4's work and the grid would measure the order the cells ran in.

*Saved user config is isolated.* `XDG_CONFIG_HOME` points at a throwaway directory
so whatever backend the developer once saved with `iterate setup` cannot leak in
and quietly re-point one cell at a different model.

*The research cache is SHARED, on purpose.* Every version that retrieves literature
then sees the same papers, so a difference between two cells is the agent rather
than what OpenAlex happened to return that minute. It also stops a long sweep
hammering a free API.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import time
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from evals.config import DEV_VERSION, WORK_DIR
from evals.readout import UnreadableRunError, read
from evals.store import (
    STATUS_OK,
    STATUS_RUN_FAILED,
    STATUS_TIMEOUT,
    STATUS_UNREADABLE,
    CellRecord,
)

if TYPE_CHECKING:
    from evals.config import Conditions
    from evals.corpus import Dataset

# Which CLI flags a released version understands.
#
# VERIFIED for `dev` and 0.4.0. PROVISIONAL for older releases: the flags are read
# off this repo's history rather than from the installed wheels, so the first
# backfill sweep is what confirms them. Every cell records the argv it ran, so a
# version that rejects a flag fails loudly with the command in the store rather
# than being silently mis-measured.
_MIN_VERSION: dict[str, tuple[int, int, int]] = {
    "--plain": (0, 3, 0),
    "--backend": (0, 2, 0),
}

_FUTURE = (999, 0, 0)


def parse_version(version: str) -> tuple[int, int, int]:
    """`dev` sorts above every release: it is the working tree, which is ahead of
    everything published."""
    if version == DEV_VERSION:
        return _FUTURE
    parts = version.split(".")
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        raise ValueError(f"version must be x.y.z or {DEV_VERSION!r}, got {version!r}")
    major, minor, patch = (int(p) for p in parts)
    return (major, minor, patch)


def supports(version: str, flag: str) -> bool:
    return parse_version(version) >= _MIN_VERSION.get(flag, (0, 0, 0))


def cell_dir(version: str, dataset_name: str, repeat: int, work_dir: Path | None = None) -> Path:
    return (work_dir or WORK_DIR) / version / dataset_name / f"repeat-{repeat}"


def command_for(version: str, dataset: Dataset, conditions: Conditions) -> list[str]:
    """The exact argv for one cell.

    Pure and separately tested, because it is the part most likely to be wrong for
    a version nobody has run in months, and the part hardest to debug from inside a
    six-hour sweep.
    """
    launcher = (
        ["uv", "run"]
        if version == DEV_VERSION
        # An ephemeral environment per released version. --no-project keeps it from
        # resolving this repo's own package and shadowing the version under test.
        else ["uv", "run", "--no-project", "--with", f"iterate=={version}"]
    )

    argv = [
        *launcher,
        "iterate",
        "run",
        "--data",
        str(dataset.path.resolve()),
        "--target",
        dataset.target,
        # Always explicit, even from v0.4 on where it became optional. Letting the
        # agent choose its own metric would mean two versions scoring on different
        # rulers, and the column would be meaningless.
        "--metric",
        dataset.metric,
        "--max-iterations",
        str(conditions.max_iterations),
        "--patience",
        str(conditions.patience),
        "--model",
        conditions.model,
    ]
    if supports(version, "--backend"):
        argv += ["--backend", conditions.backend]
    if supports(version, "--plain"):
        argv.append("--plain")
    return argv


def environment_for(cell_path: Path, conditions: Conditions, work_dir: Path) -> dict[str, str]:
    """Env for one cell. Explicit variables beat any `.env` in the working
    directory, so a stray local file cannot re-point a sweep."""
    research_cache = work_dir / "research-cache"
    research_cache.mkdir(parents=True, exist_ok=True)

    # The package derives its research cache from the runs directory's parent, so a
    # symlink at that spot is what makes the cache shared without patching iterate.
    link = cell_path / "research"
    if not link.exists():
        link.symlink_to(research_cache, target_is_directory=True)

    return {
        **os.environ,
        "ITERATE_MEMORY_DB": str(cell_path / "memory.db"),
        "ITERATE_RUNS_DIR": str(cell_path / "runs"),
        "ITERATE_MODEL": conditions.model,
        "XDG_CONFIG_HOME": str(work_dir / "xdg-config"),
    }


@dataclass(frozen=True)
class CellSpec:
    version: str
    dataset: Dataset
    repeat: int


def run_cell(
    spec: CellSpec,
    conditions: Conditions,
    *,
    work_dir: Path | None = None,
    repo_root: Path | None = None,
) -> CellRecord:
    """Run one cell to completion and read its result. Never raises: a cell that
    crashes, times out or writes an unreadable database is a recorded outcome, so a
    long sweep survives one bad dataset."""
    root = work_dir or WORK_DIR
    path = cell_dir(spec.version, spec.dataset.name, spec.repeat, root)
    path.mkdir(parents=True, exist_ok=True)

    argv = command_for(spec.version, spec.dataset, conditions)
    env = environment_for(path, conditions, root)
    log_path = path / "run.log"
    started_at = datetime.now(UTC).isoformat()
    started = time.monotonic()

    base = CellRecord(
        version=spec.version,
        dataset=spec.dataset.name,
        dataset_hash=spec.dataset.content_hash(),
        repeat=spec.repeat,
        conditions_fingerprint=conditions.fingerprint(),
        status=STATUS_RUN_FAILED,
        started_at=started_at,
        memory_db=str(path / "memory.db"),
        log_path=str(log_path),
        argv=shlex.join(argv),
    )

    try:
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                argv,
                cwd=str(repo_root or Path.cwd()),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=conditions.timeout_minutes * 60,
                check=False,
            )
        exit_code = completed.returncode
        timed_out = False
    except subprocess.TimeoutExpired:
        exit_code, timed_out = -1, True
    except OSError as exc:
        return _finish(
            base,
            status=STATUS_RUN_FAILED,
            seconds=time.monotonic() - started,
            error=str(exc),
        )

    seconds = time.monotonic() - started

    # The database is read even after a non-zero exit. A run killed by the deadline
    # still banked every experiment it finished, and throwing those away would turn
    # a partial result into no result.
    try:
        runs = read(path / "memory.db")
    except (UnreadableRunError, OSError) as exc:
        return _finish(
            base,
            status=STATUS_TIMEOUT if timed_out else STATUS_UNREADABLE,
            seconds=seconds,
            error=f"exit {exit_code}: {exc}",
        )

    if not runs:
        return _finish(
            base,
            status=STATUS_TIMEOUT if timed_out else STATUS_RUN_FAILED,
            seconds=seconds,
            error=f"exit {exit_code}: no run recorded (see {log_path.name})",
        )

    latest = runs[-1]
    status = STATUS_TIMEOUT if timed_out else (STATUS_OK if exit_code == 0 else STATUS_RUN_FAILED)
    return replace(
        base,
        status=status,
        finished_at=datetime.now(UTC).isoformat(),
        duration_seconds=seconds,
        metric=latest.metric,
        direction=latest.direction,
        baseline=latest.baseline,
        best=latest.best,
        n_experiments=latest.n_experiments,
        n_failed=latest.n_failed,
        n_rejected=latest.n_rejected,
        stopped_because=latest.stopped_because,
        error="" if status == STATUS_OK else f"exit {exit_code}",
    )


def _finish(base: CellRecord, *, status: str, seconds: float, error: str) -> CellRecord:
    return replace(
        base,
        status=status,
        finished_at=datetime.now(UTC).isoformat(),
        duration_seconds=seconds,
        error=error,
    )


__all__ = [
    "CellSpec",
    "cell_dir",
    "command_for",
    "environment_for",
    "parse_version",
    "run_cell",
    "supports",
]
