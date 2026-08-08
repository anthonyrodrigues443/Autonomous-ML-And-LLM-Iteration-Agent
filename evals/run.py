"""Entrypoint: `make eval`, `make eval-ceilings`, `make eval-report`.

Deliberately argparse and not typer. This never ships, so it should not add a
dependency to the package, and a dev tool with four subcommands does not need one.

The default behaviour of `sweep` is RESUME: it runs only the cells the store is
missing. That is what makes adding a version affordable, and it is why `--force`
exists as an explicit opt-in rather than a habit.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from evals import ceilings as ceilings_mod
from evals import config as config_mod
from evals import corpus, report
from evals.runner import CellSpec, command_for, run_cell
from evals.store import STATUS_OK, Store


def _harness_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _split(value: str | None) -> list[str] | None:
    return [item.strip() for item in value.split(",") if item.strip()] if value else None


def _available(names: list[str] | None) -> list[corpus.Dataset]:
    selected = corpus.select(names)
    missing = [d for d in selected if not d.available]
    for dataset in missing:
        print(f"  skip {dataset.name}: no data at {dataset.path}")
    return [d for d in selected if d.available]


def cmd_sweep(args: argparse.Namespace) -> int:
    config = config_mod.load()
    conditions = config.conditions
    fingerprint = conditions.fingerprint()
    versions = _split(args.versions) or config.versions
    datasets = _available(_split(args.datasets))
    if not datasets:
        print("no datasets available; nothing to sweep")
        return 1

    work_dir = Path(args.work_dir)
    with Store(args.store) as store:
        done = set() if args.force else store.completed(fingerprint)

        todo = [
            CellSpec(version=version, dataset=dataset, repeat=repeat)
            for version in versions
            for dataset in datasets
            for repeat in range(1, conditions.repeats + 1)
            if (version, dataset.name, dataset.content_hash(), repeat) not in done
        ]

        print(
            f"{len(todo)} cells to run "
            f"({len(versions)} versions x {len(datasets)} datasets x {conditions.repeats} repeats, "
            f"{len(done)} already recorded)"
        )
        if args.dry_run:
            for spec in todo:
                print(f"  {spec.version} {spec.dataset.name} #{spec.repeat}")
                print(f"    {' '.join(command_for(spec.version, spec.dataset, conditions))}")
            return 0
        if not todo:
            return 0

        sweep_id = store.start_sweep(conditions, harness_sha=_harness_sha(), note=args.note)
        for index, spec in enumerate(todo, start=1):
            print(
                f"[{index}/{len(todo)}] {spec.version} {spec.dataset.name} #{spec.repeat} ... ",
                end="",
                flush=True,
            )
            cell = run_cell(spec, conditions, work_dir=work_dir, repo_root=Path.cwd())
            store.record_cell(sweep_id, cell)
            minutes = (cell.duration_seconds or 0) / 60
            detail = f"best={cell.best:.4f}" if cell.best is not None else "no improvement"
            print(f"{cell.status} ({detail}, {minutes:.0f}m)")
        store.finish_sweep(sweep_id)

    return cmd_report(args)


def cmd_ceilings(args: argparse.Namespace) -> int:
    config = config_mod.load()
    datasets = _available(_split(args.datasets))
    if not datasets:
        print("no datasets available; nothing to measure")
        return 1

    with Store(args.store) as store:
        for dataset in datasets:
            data_hash = dataset.content_hash()
            existing = store.get_ceiling(dataset.name, data_hash, dataset.metric)
            if existing and not args.force:
                print(
                    f"{dataset.name}: have {existing.ceiling:.4f} from {existing.measured_at[:10]}"
                )
                continue

            print(f"{dataset.name} ({dataset.metric}) sweeping:")

            def show(result: ceilings_mod.SpecResult) -> None:
                value = (
                    f"{result.score:.4f}"
                    if result.score is not None
                    else f"skip ({result.error[:60]})"
                )
                print(f"    {result.label:<60} {value}  {result.seconds:.0f}s")

            try:
                ceiling, _ = ceilings_mod.sweep(
                    dataset, threads=config.conditions.sweep_threads, on_progress=show
                )
            except Exception as exc:
                print(f"  FAILED: {type(exc).__name__}: {exc}")
                continue

            store.put_ceiling(ceiling)
            gap = f", baseline {ceiling.baseline:.4f}" if ceiling.baseline is not None else ""
            print(f"  ceiling {ceiling.ceiling:.4f}{gap}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    config = config_mod.load()
    with Store(args.store) as store:
        markdown = report.build(store, config)
    out = Path(args.out)
    out.write_text(markdown, encoding="utf-8")
    print(f"wrote {out}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    config = config_mod.load()
    print(f"conditions: {config.conditions.fingerprint()}  model={config.conditions.model}")
    print(f"versions:   {', '.join(config.versions)}")
    print("datasets:")
    with Store(args.store) as store:
        cells = store.cells(config.conditions.fingerprint())
        for dataset in corpus.load():
            if not dataset.available:
                print(f"  {dataset.name:<24} MISSING  {dataset.path}")
                continue
            data_hash = dataset.content_hash()
            ceiling = store.get_ceiling(dataset.name, data_hash, dataset.metric)
            done = sum(1 for c in cells if c.dataset == dataset.name and c.status == STATUS_OK)
            bar = f"{ceiling.ceiling:.4f}" if ceiling else "unmeasured"
            print(
                f"  {dataset.name:<24} {dataset.metric:<20} hash={data_hash} "
                f"ceiling={bar:<12} cells={done}"
            )
    return 0


def build_parser() -> argparse.ArgumentParser:
    # A parent parser rather than top-level arguments, so the shared flags work
    # AFTER the subcommand. `make eval ARGS="--datasets churn"` appends them, and
    # top-level argparse arguments only parse before the subcommand name.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--store", default=str(config_mod.STORE_PATH))
    common.add_argument("--out", default=str(config_mod.REPORT_PATH))
    common.add_argument("--datasets", help="comma-separated dataset names (default: all)")
    common.add_argument("--work-dir", default=str(config_mod.WORK_DIR))

    parser = argparse.ArgumentParser(prog="evals", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sweep = sub.add_parser("sweep", parents=[common], help="fill the cells the store is missing")
    sweep.add_argument("--versions", help="comma-separated (default: config.toml)")
    sweep.add_argument("--force", action="store_true", help="re-run cells already recorded")
    sweep.add_argument("--dry-run", action="store_true", help="print the commands, run nothing")
    sweep.add_argument("--note", default="", help="free text stored with the sweep")
    sweep.set_defaults(func=cmd_sweep)

    ceilings = sub.add_parser(
        "ceilings", parents=[common], help="measure the brute-force ceiling per dataset"
    )
    ceilings.add_argument("--force", action="store_true", help="re-measure datasets that have one")
    ceilings.set_defaults(func=cmd_ceilings)

    report_cmd = sub.add_parser(
        "report", parents=[common], help="rebuild the markdown table from the store"
    )
    report_cmd.set_defaults(func=cmd_report)

    list_cmd = sub.add_parser(
        "list", parents=[common], help="show the corpus and what the store holds"
    )
    list_cmd.set_defaults(func=cmd_list)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    sys.exit(main())
