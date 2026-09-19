"""Installs during a run: the route a missing import takes, and the one install ladder.

A missing import is planned with dry runs before anything is installed. Every install
carries a constraints file, and torch and torchvision never change during a run. Cells
never install: the harness does, on the host.
"""

from __future__ import annotations

import contextlib
import enum
import hashlib
import importlib.metadata
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from iterate import userconfig

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

    from packaging.requirements import Requirement

    Runner = Callable[..., subprocess.CompletedProcess[str]]

FROZEN = frozenset({"torch", "torchvision"})
OWN = "iterate-ai"
DRY_RUN_TIMEOUT = 120.0
INSTALL_TIMEOUT = 600.0
_UV_CHANGE = re.compile(r"^ ([+-]) (\S+)==(\S+)$", re.MULTILINE)
_EXPLAINED = re.compile(r"(?:╰─▶|Because)\s*(.*?)(?:, we can conclude|\.\s|$)")


def canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


class Route(enum.StrEnum):
    INSTALL = "install"
    RESTART = "restart"
    NEXT_RUN = "next_run"
    REFUSE = "refuse"


@dataclass(frozen=True)
class Plan:
    package: str
    route: Route
    version: str | None = None
    moves: dict[str, tuple[str, str]] = field(default_factory=dict)
    pins: tuple[str, ...] = ()
    reason: str = ""
    detail: str = ""
    seconds: float = 0.0

    def moved(self) -> str:
        return ", ".join(f"{n} {a} -> {b}" for n, (a, b) in sorted(self.moves.items()))


@dataclass(frozen=True)
class _DryRun:
    ok: bool
    changes: dict[str, tuple[str | None, str | None]]
    output: str


def pending_path(prefix: str = sys.prefix) -> Path:
    """Where installs saved for the next run live: one file per environment, found from
    any folder a run starts in."""
    venv = hashlib.sha256(os.path.realpath(prefix).encode()).hexdigest()[:16]
    return userconfig.cache_dir() / "installs" / f"{venv}.json"


def find_uv(python: str = sys.executable) -> str | None:
    """uv on PATH, else where uv installs itself: a PATH from launchd, cron or an IDE
    often leaves those folders out."""
    if found := shutil.which("uv"):
        return found
    home = Path.home()
    for candidate in (
        Path(python).parent / "uv",
        home / ".local" / "bin" / "uv",
        home / ".cargo" / "bin" / "uv",
    ):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def installed() -> dict[str, str]:
    # A leftover dist-info folder can carry no name.
    return {
        canonical(name): d.version
        for d in importlib.metadata.distributions()
        if (name := d.metadata["Name"])
    }


def distributions_of(
    modules: Iterable[str], mapping: Mapping[str, list[str]] | None = None
) -> frozenset[str]:
    mapping = mapping if mapping is not None else importlib.metadata.packages_distributions()
    tops = {m.partition(".")[0] for m in modules}
    return frozenset(canonical(d) for top in tops for d in mapping.get(top, ()))


def _requirements(extra: str) -> list[Requirement]:
    from packaging.requirements import InvalidRequirement, Requirement

    try:
        lines = importlib.metadata.requires(OWN) or []
    except importlib.metadata.PackageNotFoundError:
        return []
    kept = []
    for line in lines:
        try:
            req = Requirement(line)
        except InvalidRequirement:
            continue
        base = req.marker is None or req.marker.evaluate({"extra": ""})
        if (not extra and base) or (
            extra and req.marker is not None and not base and req.marker.evaluate({"extra": extra})
        ):
            kept.append(req)
    return kept


def own_requirements() -> tuple[str, ...]:
    # pip refuses a constraint that carries extras.
    return tuple(f"{req.name}{req.specifier}" for req in _requirements(""))


def vision_requirements() -> tuple[str, ...]:
    return tuple(f"{req.name}{req.specifier}" for req in _requirements("vision")) or (
        "torch",
        "torchvision",
    )


def pin_lines(names: Iterable[str], versions: Mapping[str, str]) -> tuple[str, ...]:
    return tuple(sorted({f"{n}=={versions[n]}" for n in map(canonical, names) if n in versions}))


def pin_everything(directory: Path) -> Path:
    versions = installed()
    path = directory / "constraints-installed.txt"
    path.write_text("\n".join(pin_lines(versions, versions)) + "\n")
    return path


def broken_dependents(
    changes: Mapping[str, tuple[str | None, str | None]],
) -> dict[str, list[str]]:
    """Installed distributions whose own requirements the planned changes would break."""
    from packaging.requirements import InvalidRequirement, Requirement

    after = installed()
    for name, (_, new) in changes.items():
        if new is None:
            after.pop(name, None)
        else:
            after[name] = new
    broken: dict[str, list[str]] = {}
    for dist in importlib.metadata.distributions():
        name = canonical(dist.metadata["Name"] or "")
        if name in changes or name not in after:
            continue
        for line in dist.requires or []:
            try:
                req = Requirement(line)
            except InvalidRequirement:
                continue
            if req.marker is not None and not req.marker.evaluate({"extra": ""}):
                continue
            dep = canonical(req.name)
            if (
                dep in changes
                and dep in after
                and not req.specifier.contains(after[dep], prereleases=True)
            ):
                broken.setdefault(name, []).append(str(req))
    return broken


def closure(roots: Iterable[str]) -> frozenset[str]:
    from packaging.requirements import InvalidRequirement, Requirement

    seen: set[str] = set()
    todo = [canonical(r) for r in roots]
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        seen.add(name)
        try:
            lines = importlib.metadata.requires(name) or []
        except importlib.metadata.PackageNotFoundError:
            continue
        for line in lines:
            try:
                req = Requirement(line)
            except InvalidRequirement:
                continue
            if req.marker is None or req.marker.evaluate({"extra": ""}):
                todo.append(canonical(req.name))
    return frozenset(seen)


def _why(output: str, own: Sequence[str]) -> tuple[str, str]:
    flat = " ".join(output.replace("╰─▶", " ").split())
    found = _EXPLAINED.search(flat)
    detail = (found.group(1) if found else flat)[:240]
    if "No module named pip" in flat:
        return "no_installer", detail
    if re.search(
        r"was not found in the package registry|No matching distribution found"
        r"|there are no versions of",
        flat,
    ):
        return "not_found", detail
    if re.search(
        r"network was disabled|Failed to fetch|error sending request|Could not connect", flat
    ):
        return "network", detail
    # uv writes a marker between the name and the version: torch{sys_platform == 'darwin'}>=2.10
    if re.search(r"\btorch(?:vision)?(?:\{[^}]*\})?\s*(?:==|>=|<=|~=|>|<)", flat):
        return "frozen_dep", detail
    if any(line.replace(" ", "") in flat.replace(" ", "") for line in own):
        return "iterate", detail
    if re.search(r"no wheels with a matching|has no usable wheels", flat):
        return "no_wheel", detail
    return "unresolvable", detail


def _shipper(dists: Iterable[str], module: str) -> str | None:
    """The installed distribution that ships ``module``, or owns its top-level name as a
    regular package. A namespace folder other distributions share is owned by none."""
    parts = module.split(".")
    top, inner = parts[0], "/".join(parts)
    for dist in dists:
        try:
            files = importlib.metadata.files(dist)
        except importlib.metadata.PackageNotFoundError:
            continue
        if files is None or any(
            f.as_posix() == f"{top}/__init__.py"
            or f.as_posix().startswith((f"{top}.", f"{inner}.", f"{inner}/"))
            for f in files
        ):
            return canonical(dist)
    return None


def _floor(name: str, version: str | None) -> str:
    from packaging.version import InvalidVersion, Version

    try:
        release = Version(version or "").release
    except InvalidVersion:
        return name
    return f"{name}>={'.'.join(map(str, release[:2]))}"


def _moved(run: _DryRun) -> set[str]:
    return {n for n, (old, _) in run.changes.items() if old is not None}


class Installer:
    """Plans and runs the harness's installs for one interpreter. Host-side only."""

    def __init__(
        self,
        python: str = sys.executable,
        *,
        pending: Path | None = None,
        host_modules: Callable[[], Iterable[str]] = lambda: list(sys.modules),
        run: Runner | None = None,
    ) -> None:
        self._python = python
        self._pending = pending
        self._host_modules = host_modules
        self._run = run
        self._tmp: tempfile.TemporaryDirectory[str] | None = None
        self._files = itertools.count()
        self._ensured = False

    def plan(
        self, package: str, *, kernel_modules: Iterable[str] | None, module: str | None = None
    ) -> Plan:
        t0 = time.monotonic()

        def done(
            route: Route,
            version: str | None = None,
            run: _DryRun | None = None,
            pins: tuple[str, ...] = (),
            reason: str = "",
            detail: str = "",
        ) -> Plan:
            moves = {n: (a, b) for n, (a, b) in (run.changes if run else {}).items() if a and b}
            return Plan(package, route, version, moves, pins, reason, detail, time.monotonic() - t0)

        name = canonical(package)
        if name in FROZEN:
            return done(Route.REFUSE, reason="frozen")
        versions = installed()
        if name in versions:
            return done(Route.REFUSE, versions[name], reason="installed")
        mapping = importlib.metadata.packages_distributions()
        top, _, inner = (module or "").partition(".")
        if module and (providers := mapping.get(top)):
            if (owner := _shipper(providers, module)) is not None:
                return done(Route.REFUSE, versions.get(owner), reason="installed", detail=owner)
            # The longest name first: google.cloud.storage is google-cloud-storage, not
            # google-cloud, and the bare top name is PyPI's unrelated `google`.
            parts = [p for p in inner.split(".") if p]
            under: Plan | None = None
            for depth in range(len(parts), 0, -1):
                dotted = canonical("-".join([top, *parts[:depth]]))
                if dotted == name:
                    continue
                under = self.plan(dotted, kernel_modules=kernel_modules)
                if under.reason != "not_found":
                    return under
            if under is not None:
                return under
        own = own_requirements()
        latest = self._dry_run(package, ())
        if not latest.ok:
            reason, detail = _why(latest.output, ())
            return done(Route.REFUSE, reason=reason, detail=detail)
        version = (latest.changes.get(name) or (None, None))[1]
        if absent := sorted((set(latest.changes) & FROZEN) - set(versions)):
            return done(Route.REFUSE, version, reason="needs_torch", detail=", ".join(absent))
        frozen = pin_lines(FROZEN, versions)
        # Pins make uv backtrack to years-old releases, so the floor keeps the version research means.
        floor = _floor(name, version)
        protected = closure((OWN, *FROZEN))
        host = distributions_of(self._host_modules(), mapping)
        kernel = (
            distributions_of(kernel_modules, mapping)
            if kernel_modules is not None
            else frozenset(versions)
        )
        base = latest
        if _moved(latest) & (FROZEN | protected):
            base = self._dry_run(floor, (*frozen, *own))
            if not base.ok:
                if torch := sorted(_moved(latest) & FROZEN):
                    moves = (f"{n} {latest.changes[n][0]} -> {latest.changes[n][1]}" for n in torch)
                    return done(Route.REFUSE, version, reason="frozen_dep", detail=", ".join(moves))
                reason, detail = _why(base.output, own)
                return done(
                    Route.REFUSE,
                    version,
                    reason="iterate" if reason == "unresolvable" else reason,
                    detail=detail,
                )
        # uv's dry run replaces a distribution under an installed dependent without a word.
        broken = broken_dependents(base.changes) if _moved(base) else {}
        if clash := sorted(set(broken) & protected):
            return done(
                Route.REFUSE,
                version,
                base,
                reason="iterate",
                detail="; ".join(f"{d} needs {', '.join(broken[d])}" for d in clash),
            )
        if not (_moved(base) | set(broken)) & (host | kernel):
            pins = (*pin_lines(host | kernel, versions), *frozen, *own)
            return done(Route.INSTALL, version, base, pins)
        refused: tuple[str, str] | None = None
        for route, names in ((Route.INSTALL, host | kernel), (Route.RESTART, host)):
            if route is Route.INSTALL and kernel_modules is None:
                continue
            pins = (*pin_lines(names, versions), *frozen, *own)
            run = self._dry_run(floor, pins)
            if not run.ok:
                continue
            # The pins can change the resolution, so the checks on the first runs hold again.
            broken = broken_dependents(run.changes) if _moved(run) else {}
            if absent := sorted((set(run.changes) & FROZEN) - set(versions)):
                refused = refused or ("needs_torch", ", ".join(absent))
            elif clash := sorted(set(broken) & protected):
                refused = refused or (
                    "iterate",
                    "; ".join(f"{d} needs {', '.join(broken[d])}" for d in clash),
                )
            elif not set(broken) & names:
                return done(route, version, run, pins)
        if refused is not None:
            return done(Route.REFUSE, version, reason=refused[0], detail=refused[1])
        return done(Route.NEXT_RUN, version, base, (*frozen, *own))

    def install(self, plan: Plan, *, timeout: float | None = INSTALL_TIMEOUT) -> str:
        spec = _floor(canonical(plan.package), plan.version) if plan.version else plan.package
        return install(
            self._python, [spec], self._constraints(plan.pins), timeout=timeout, run=self._run
        )

    def save_for_next_run(self, plan: Plan, *, module: str) -> None:
        if self._pending is None:
            return
        entries = [e for e in self.pending() if canonical(e["package"]) != canonical(plan.package)]
        entries.append(
            {
                "python": self._python,
                "package": plan.package,
                "module": module,
                "version": plan.version,
                "moves": {n: list(v) for n, v in plan.moves.items()},
                "saved": datetime.now().isoformat(timespec="seconds"),
            }
        )
        self._write(entries)

    def pending(self) -> list[dict[str, Any]]:
        """The saved entries; a file of any other shape reads as nothing waiting."""
        if self._pending is None or not self._pending.is_file():
            return []
        try:
            saved = json.loads(self._pending.read_text())
        except (OSError, ValueError):
            return []
        entries = saved.get("entries") if isinstance(saved, dict) else None
        if not isinstance(entries, list):
            return []
        return [
            e
            for e in entries
            if isinstance(e, dict)
            and isinstance(e.get("package"), str)
            and e["package"]
            and isinstance(e.get("module"), str | None)
        ]

    def install_pending(self) -> list[str]:
        """At the start of a run, before anything heavy is imported. One line per entry."""
        lines: list[str] = []
        kept: list[dict[str, Any]] = []
        waiting = self.pending()
        for entry in waiting:
            plan = self.plan(entry["package"], kernel_modules=(), module=entry.get("module"))
            if plan.route in (Route.INSTALL, Route.RESTART):
                if error := self.install(plan):
                    lines.append(f"could not install {plan.package}: {error.strip()[-200:]}")
                    kept.append(entry)
                    continue
                moved = f" ({plan.moved()})" if plan.moves else ""
                lines.append(
                    f"installed {plan.package} {plan.version} saved by an earlier run{moved}"
                )
            elif plan.reason == "installed":
                lines.append(f"{entry['package']} is already installed")
            elif plan.reason == "network":
                lines.append(f"{plan.package} waits: the package index is unreachable")
                kept.append(entry)
            elif plan.reason == "no_installer":
                lines.append(
                    f"{plan.package} waits: this environment has no pip, and uv was not found"
                )
                kept.append(entry)
            elif plan.route is Route.NEXT_RUN:
                lines.append(
                    f"dropped {plan.package}: it needs {plan.moved()}, which iterate loads "
                    "before any run starts"
                )
            else:
                lines.append(
                    f"dropped {plan.package}: it cannot be installed beside iterate "
                    f"({plan.detail or plan.reason})"
                )
        if waiting:
            self._write(kept)
        return lines

    def _write(self, entries: list[dict[str, Any]]) -> None:
        assert self._pending is not None
        self._pending.parent.mkdir(parents=True, exist_ok=True)
        self._pending.write_text(json.dumps({"entries": entries}, indent=1) + "\n")

    def _constraints(self, lines: Sequence[str]) -> Path:
        if self._tmp is None:
            self._tmp = tempfile.TemporaryDirectory(prefix="iterate-deps-")
        path = Path(self._tmp.name) / f"constraints-{next(self._files)}.txt"
        path.write_text("\n".join(lines) + "\n")
        return path

    def _dry_run(self, spec: str, pins: Sequence[str]) -> _DryRun:
        constraints = self._constraints(pins)
        run = self._run or subprocess.run
        if uv := find_uv(self._python):
            cmd = [
                uv,
                "pip",
                "install",
                "--dry-run",
                "--python",
                self._python,
                "-c",
                str(constraints),
                spec,
            ]
            try:
                proc = run(
                    cmd, capture_output=True, text=True, timeout=DRY_RUN_TIMEOUT, check=False
                )
            except subprocess.TimeoutExpired:
                return _DryRun(False, {}, "Failed to fetch: dry run timed out")
            except OSError as exc:
                return _DryRun(False, {}, str(exc))
            output = proc.stdout + proc.stderr
            changes: dict[str, list[str | None]] = {}
            for sign, name, version in _UV_CHANGE.findall(output):
                changes.setdefault(canonical(name), [None, None])[0 if sign == "-" else 1] = version
            return _DryRun(
                proc.returncode == 0, {k: (v[0], v[1]) for k, v in changes.items()}, output
            )
        report = constraints.with_suffix(".report.json")
        cmd = [
            self._python,
            "-m",
            "pip",
            "install",
            "--dry-run",
            "--quiet",
            "--report",
            str(report),
            "-c",
            str(constraints),
            spec,
        ]
        try:
            proc = run(cmd, capture_output=True, text=True, timeout=DRY_RUN_TIMEOUT, check=False)
        except subprocess.TimeoutExpired:
            return _DryRun(False, {}, "Failed to fetch: dry run timed out")
        except OSError as exc:
            return _DryRun(False, {}, str(exc))
        if proc.returncode != 0:
            output = proc.stdout + proc.stderr
            # uv venvs ship without pip; the run already has install consent, as the ladder does.
            if "No module named pip" in output and not self._ensured:
                self._ensured = True
                with contextlib.suppress(subprocess.TimeoutExpired, OSError):
                    run(
                        [self._python, "-m", "ensurepip", "--upgrade"],
                        capture_output=True,
                        text=True,
                        timeout=DRY_RUN_TIMEOUT,
                        check=False,
                    )
                return self._dry_run(spec, pins)
            return _DryRun(False, {}, output)
        versions = installed()
        planned: dict[str, tuple[str | None, str | None]] = {}
        try:
            for item in json.loads(report.read_text()).get("install", []):
                dist = canonical(item["metadata"]["name"])
                planned[dist] = (versions.get(dist), item["metadata"]["version"])
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            return _DryRun(False, {}, f"unreadable pip report: {exc}")
        return _DryRun(True, planned, proc.stdout + proc.stderr)


def install(
    python: str,
    specs: Sequence[str],
    constraints: Path,
    *,
    timeout: float | None = INSTALL_TIMEOUT,
    run: Runner | None = None,
) -> str:
    """The one install ladder: "" on success, else the error log. Never raises."""
    if not specs:
        return ""
    from packaging.requirements import InvalidRequirement, Requirement

    for spec in specs:
        try:
            name = canonical(Requirement(spec).name)
        except InvalidRequirement:
            return f"not a package requirement: {spec!r}"
        if name in FROZEN:
            return f"{name} never changes during a run"
    return _ladder(python, list(specs), constraints, timeout=timeout, run=run)


def _ladder(
    python: str,
    specs: list[str],
    constraints: Path,
    *,
    timeout: float | None,
    run: Runner | None,
) -> str:
    def attempt(cmd: list[str]) -> str:
        try:
            proc = (run or subprocess.run)(
                cmd, capture_output=True, text=True, timeout=timeout, check=False
            )
        except subprocess.TimeoutExpired:
            return f"install timed out: {' '.join(cmd)}"
        except FileNotFoundError:
            return f"command not found: {cmd[0]}"
        return (
            "" if proc.returncode == 0 else (proc.stderr.strip() or f"exit code {proc.returncode}")
        )

    pip = [python, "-m", "pip", "install", "--quiet", "-c", str(constraints), *specs]
    log = attempt(pip)
    if not log or "No module named pip" not in log:
        return log
    # uv venvs ship without pip.
    if uv := find_uv(python):
        return attempt(
            [uv, "pip", "install", "--quiet", "--python", python, "-c", str(constraints), *specs]
        )
    attempt([python, "-m", "ensurepip", "--upgrade"])
    return attempt(pip)


def ensure_torch(
    *, consent: bool, python: str = sys.executable, host_modules: Iterable[str] | None = None
) -> str:
    """At the start of an image run only: "" when torch and torchvision are installed,
    else why not. No time cap: the wheels are about a gigabyte."""
    versions = installed()
    if set(versions) >= FROZEN:
        return ""
    if not consent:
        return (
            "an image run needs torch and torchvision: pip install 'iterate-ai[vision]' "
            "(or pass --install)"
        )
    loaded = distributions_of(host_modules if host_modules is not None else list(sys.modules))
    with tempfile.TemporaryDirectory(prefix="iterate-pins-") as tmp:
        pins = Path(tmp) / "constraints-start.txt"
        pins.write_text("\n".join((*pin_lines(loaded, versions), *own_requirements())) + "\n")
        return _ladder(python, list(vision_requirements()), pins, timeout=None, run=None)


__all__ = [
    "DRY_RUN_TIMEOUT",
    "FROZEN",
    "INSTALL_TIMEOUT",
    "Installer",
    "Plan",
    "Route",
    "canonical",
    "ensure_torch",
    "find_uv",
    "install",
    "pending_path",
    "pin_everything",
]
