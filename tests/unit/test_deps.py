"""The install routes, planned from scripted dry runs. No network, nothing installed."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any

import pytest

from iterate.adapters.compute import deps
from iterate.adapters.compute.deps import Installer, Plan, Route

VERSIONS = {
    "numpy": "2.4.6",
    "pandas": "3.0.3",
    "narwhals": "1.14.0",
    "lightgbm": "4.6.0",
    "torch": "2.14.0",
    "torchvision": "0.29.0",
    "scikit-learn": "1.8.0",
}
MAPPING = {
    "numpy": ["numpy"],
    "pandas": ["pandas"],
    "narwhals": ["narwhals"],
    "lightgbm": ["lightgbm"],
    "torch": ["torch"],
    "sklearn": ["scikit-learn"],
    "google": ["protobuf"],
}
FILES = {
    "scikit-learn": ["sklearn/__init__.py", "sklearn/externals/__init__.py"],
    "protobuf": ["google/protobuf/__init__.py", "google/protobuf/message.py"],
}


SKOPT = """  \u00d7 No solution found when resolving dependencies:
  ╰─▶ Because there are no versions of skopt and you require skopt, we can
      conclude that your requirements are unsatisfiable."""
AUTOGLUON = """  \u00d7 No solution found when resolving dependencies:
  ╰─▶ Because autogluon-tabular[all]>=1.6.0 depends on torch{sys_platform ==
      'darwin'}>=2.10,<2.11 and torch{sys_platform == 'darwin'}==2.14.0, we
      can conclude that autogluon-tabular[all]>=1.6.0 cannot be used."""


def _uv(changes: str = "", *, ok: bool = True, error: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], 0 if ok else 1, "", changes if ok else error)


class _Scripted:
    """Answers each dry run in order and records the constraints it was given."""

    def __init__(self, answers: list[subprocess.CompletedProcess[str]]) -> None:
        self.answers = list(answers)
        self.calls: list[tuple[list[str], list[str]]] = []

    def __call__(self, cmd: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        pins = Path(cmd[cmd.index("-c") + 1]).read_text().split()
        self.calls.append((cmd, pins))
        return self.answers.pop(0)


@pytest.fixture(autouse=True)
def _environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(deps, "installed", lambda: dict(VERSIONS))
    monkeypatch.setattr(deps, "own_requirements", lambda: ("pandas>=2.2.0",))
    monkeypatch.setattr(
        deps, "closure", lambda roots: frozenset({"iterate-ai", "pandas", "numpy", "torch"})
    )
    monkeypatch.setattr(deps, "broken_dependents", lambda changes: {})
    monkeypatch.setattr(deps.importlib.metadata, "packages_distributions", lambda: MAPPING)
    monkeypatch.setattr(
        deps.importlib.metadata,
        "files",
        lambda name: [PurePosixPath(f) for f in FILES.get(name, [])],
    )
    monkeypatch.setattr(deps.shutil, "which", lambda name: "/bin/uv")


def _installer(run: _Scripted, host: tuple[str, ...] = ("numpy", "pandas"), **kw: Any) -> Installer:
    return Installer("/venv/bin/python", host_modules=lambda: host, run=run, **kw)


def test_a_package_that_moves_nothing_loaded_installs_after_one_dry_run() -> None:
    run = _Scripted([_uv(" + tabulate==0.10.0\n")])
    plan = _installer(run).plan("tabulate", kernel_modules=["numpy", "lightgbm"])
    assert (plan.route, plan.version) == (Route.INSTALL, "0.10.0")
    assert len(run.calls) == 1
    assert run.calls[0][1] == []
    expected = {
        "numpy==2.4.6",
        "lightgbm==4.6.0",
        "torch==2.14.0",
        "torchvision==0.29.0",
        "pandas>=2.2.0",
    }
    assert expected <= set(plan.pins)


def test_a_clash_with_the_kernel_only_restarts_pinned_to_the_host() -> None:
    moved = " - narwhals==1.14.0\n + narwhals==2.26.0\n + plotly==7.1.0\n"
    unsolvable = _uv(
        ok=False, error="Because plotly>=7.1 depends on narwhals>=1.15.1 and narwhals==1.14.0"
    )
    run = _Scripted([_uv(moved), unsolvable, _uv(moved)])
    plan = _installer(run).plan("plotly", kernel_modules=["numpy", "narwhals"])
    assert plan.route == Route.RESTART
    assert plan.moves == {"narwhals": ("1.14.0", "2.26.0")}
    assert run.calls[1][0][-1] == "plotly>=7.1"
    assert "narwhals==1.14.0" in run.calls[1][1]
    assert "narwhals==1.14.0" not in run.calls[2][1]
    assert {"torch==2.14.0", "numpy==2.4.6"} <= set(run.calls[2][1])


def test_a_clash_with_what_iterate_runs_on_waits_for_the_next_run(tmp_path: Path) -> None:
    moved = " - pandas==3.0.3\n + pandas==2.3.3\n + sktime==1.1.0\n"
    unsolvable = _uv(
        ok=False, error="Because sktime==1.1.0 depends on pandas<2.4 and pandas==3.0.3"
    )
    run = _Scripted([_uv(moved), _uv(moved), unsolvable, unsolvable])
    pending = tmp_path / "pending.json"
    installer = _installer(run, pending=pending)
    plan = installer.plan("sktime", kernel_modules=["pandas"])
    assert plan.route == Route.NEXT_RUN
    installer.save_for_next_run(plan, module="sktime")
    [entry] = json.loads(pending.read_text())["entries"]
    assert (entry["package"], entry["version"], entry["moves"]) == (
        "sktime",
        "1.1.0",
        {"pandas": ["3.0.3", "2.3.3"]},
    )


@pytest.mark.parametrize(
    ("error", "reason"),
    [
        ("Because nope was not found in the package registry and you require nope", "not_found"),
        ("hint: Packages were unavailable because the network was disabled", "network"),
        ("Caused by: Failed to fetch: `https://pypi.org/simple/x/`", "network"),
        ("Because x has no wheels with a matching Python ABI tag (e.g., `cp312`)", "no_wheel"),
        (SKOPT, "not_found"),
    ],
)
def test_a_first_dry_run_that_fails_is_refused_with_its_reason(error: str, reason: str) -> None:
    plan = _installer(_Scripted([_uv(ok=False, error=error)])).plan("x", kernel_modules=[])
    assert (plan.route, plan.reason) == (Route.REFUSE, reason)


def test_a_torch_clash_uv_writes_with_a_platform_marker_is_a_torch_refusal() -> None:
    assert deps._why(AUTOGLUON, ("pandas>=2.2.0",))[0] == "frozen_dep"
    moved = " - torch==2.14.0\n + torch==2.10.0\n + autogluon==1.6.2\n"
    unrelated = "Because autogluon-core==1.6.2 depends on scipy<1.17 and scipy==1.17.1"
    run = _Scripted([_uv(moved), _uv(ok=False, error=unrelated)])
    plan = _installer(run).plan("autogluon", kernel_modules=[])
    assert (plan.route, plan.reason, plan.detail) == (
        Route.REFUSE,
        "frozen_dep",
        "torch 2.14.0 -> 2.10.0",
    )


def test_torch_and_torchvision_are_refused_by_name_before_any_dry_run() -> None:
    run = _Scripted([])
    for name in ("torch", "torchvision", "Torch"):
        assert _installer(run).plan(name, kernel_modules=[]).reason == "frozen"
    assert run.calls == []


def test_an_installed_distribution_is_never_reinstalled_for_a_missing_submodule() -> None:
    plan = _installer(_Scripted([])).plan("scikit-learn", kernel_modules=[])
    assert (plan.route, plan.reason, plan.version) == (Route.REFUSE, "installed", "1.8.0")


def test_an_import_inside_an_installed_top_level_name_installs_nothing() -> None:
    run = _Scripted([])
    plan = _installer(run).plan("sklearn", kernel_modules=[], module="sklearn.externals.joblib")
    assert (plan.route, plan.reason, plan.detail) == (Route.REFUSE, "installed", "scikit-learn")
    assert run.calls == []


def test_a_module_under_a_namespace_another_package_shares_plans_its_own_distribution() -> None:
    run = _Scripted([_uv(" + google-generativeai==0.8.5\n")])
    plan = _installer(run).plan("google", kernel_modules=[], module="google.generativeai")
    assert (plan.package, plan.route, plan.version) == (
        "google-generativeai",
        Route.INSTALL,
        "0.8.5",
    )
    assert run.calls[0][0][-1] == "google-generativeai"

    unknown = SKOPT.replace("skopt", "google-genai")
    run = _Scripted([_uv(ok=False, error=unknown), _uv(" + google==3.0.0\n")])
    plan = _installer(run).plan("google", kernel_modules=[], module="google.genai")
    assert (plan.package, plan.route) == ("google", Route.INSTALL)
    assert [c[0][-1] for c in run.calls] == ["google-genai", "google"]


def test_a_package_that_needs_another_torch_is_refused() -> None:
    moved = " - torch==2.14.0\n + torch==2.9.0\n + oldthing==1.0\n"
    run = _Scripted(
        [
            _uv(moved),
            _uv(ok=False, error="Because oldthing depends on torch==2.9.0 and torch==2.14.0"),
        ]
    )
    plan = _installer(run).plan("oldthing", kernel_modules=[])
    assert (plan.route, plan.reason) == (Route.REFUSE, "frozen_dep")


def test_a_package_that_needs_torch_where_there_is_none_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        deps, "installed", lambda: {k: v for k, v in VERSIONS.items() if "torch" not in k}
    )
    run = _Scripted([_uv(" + torch==2.14.0\n + tabnet==4.1.0\n")])
    plan = _installer(run).plan("tabnet", kernel_modules=[])
    assert (plan.route, plan.reason, plan.detail) == (Route.REFUSE, "needs_torch", "torch")


def test_a_package_that_breaks_what_iterate_requires_is_refused_not_saved(tmp_path: Path) -> None:
    moved = " - pandas==3.0.3\n + pandas==2.1.4\n + pycaret==3.3.2\n"
    error = "Because pycaret>=3.3.0 depends on pandas<2.2.0 and pandas>=2.2.0"
    pending = tmp_path / "pending.json"
    installer = _installer(_Scripted([_uv(moved), _uv(ok=False, error=error)]), pending=pending)
    plan = installer.plan("pycaret", kernel_modules=[])
    assert (plan.route, plan.reason) == (Route.REFUSE, "iterate")
    assert "pandas<2.2.0" in plan.detail
    assert not pending.exists()


def test_a_resolution_that_breaks_an_installed_part_of_iterate_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        deps, "broken_dependents", lambda changes: {"pandas": ["numpy>=2.3"]} if changes else {}
    )
    moved = " - numpy==2.4.6\n + numpy==2.2.0\n + oldnumpy==1.0\n"
    run = _Scripted([_uv(moved), _uv(moved)])
    plan = _installer(run).plan("oldnumpy", kernel_modules=[])
    assert (plan.route, plan.reason, plan.detail) == (
        Route.REFUSE,
        "iterate",
        "pandas needs numpy>=2.3",
    )


def test_a_pinned_resolution_that_adds_torch_is_refused_not_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        deps, "installed", lambda: {k: v for k, v in VERSIONS.items() if "torch" not in k}
    )
    latest = _uv(" - narwhals==1.14.0\n + narwhals==2.26.0\n + fakepkg==2.0.0\n")
    pinned = _uv(" + torch==2.8.0\n + fakepkg==2.0.0\n")
    run = _Scripted([latest, pinned, pinned])
    plan = _installer(run).plan("fakepkg", kernel_modules=["narwhals"])
    assert (plan.route, plan.reason, plan.detail) == (Route.REFUSE, "needs_torch", "torch")
    assert len(run.calls) == 3


def test_a_pinned_resolution_that_breaks_an_unloaded_part_of_iterate_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        deps,
        "closure",
        lambda roots: frozenset({"iterate-ai", "pandas", "numpy", "torch", "e2b", "wcmatch"}),
    )
    monkeypatch.setattr(
        deps,
        "broken_dependents",
        lambda changes: {"e2b": ["wcmatch>=10.1"]} if "wcmatch" in changes else {},
    )
    latest = _uv(" - narwhals==1.14.0\n + narwhals==2.26.0\n + fakepkg==2.0.0\n")
    pinned = _uv(" - wcmatch==10.1\n + wcmatch==9.0\n + fakepkg==2.0.0\n")
    run = _Scripted([latest, pinned, pinned])
    plan = _installer(run).plan("fakepkg", kernel_modules=["narwhals"])
    assert (plan.route, plan.reason, plan.detail) == (
        Route.REFUSE,
        "iterate",
        "e2b needs wcmatch>=10.1",
    )


def test_an_unknown_kernel_never_installs_a_move_without_a_restart() -> None:
    moved = " - narwhals==1.14.0\n + narwhals==2.26.0\n + plotly==7.1.0\n"
    run = _Scripted([_uv(moved), _uv(moved)])
    assert _installer(run).plan("plotly", kernel_modules=None).route == Route.RESTART


def test_without_uv_a_dry_run_reads_pips_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(deps.shutil, "which", lambda name: None)
    monkeypatch.setenv("HOME", str(tmp_path))
    commands: list[list[str]] = []

    def pip(cmd: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        commands.append(cmd)
        report = {"install": [{"metadata": {"name": "tabulate", "version": "0.10.0"}}]}
        Path(cmd[cmd.index("--report") + 1]).write_text(json.dumps(report))
        return subprocess.CompletedProcess(cmd, 0, "", "")

    plan = Installer("/venv/bin/python", host_modules=lambda: (), run=pip).plan(
        "tabulate", kernel_modules=[]
    )
    assert (plan.route, plan.version) == (Route.INSTALL, "0.10.0")
    assert commands[0][:6] == ["/venv/bin/python", "-m", "pip", "install", "--dry-run", "--quiet"]
    assert "-c" in commands[0]


def test_uv_off_path_is_found_where_it_installs_itself(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(deps.shutil, "which", lambda name: None)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert deps.find_uv("/venv/bin/python") is None
    uv = tmp_path / ".local" / "bin" / "uv"
    uv.parent.mkdir(parents=True)
    uv.write_text("#!/bin/sh\n")
    uv.chmod(0o755)
    run = _Scripted([_uv(" + tabulate==0.10.0\n")])
    plan = _installer(run).plan("tabulate", kernel_modules=[])
    assert plan.route == Route.INSTALL
    assert run.calls[0][0][:3] == [str(uv), "pip", "install"]


def _pip_less(*, ensurepip_works: bool) -> tuple[list[list[str]], Any]:
    calls: list[list[str]] = []
    state = {"pip": False}

    def fake(cmd: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        if cmd[1:3] == ["-m", "ensurepip"]:
            state["pip"] = ensurepip_works
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if not state["pip"]:
            return subprocess.CompletedProcess(cmd, 1, "", "/venv/bin/python: No module named pip")
        report = {"install": [{"metadata": {"name": "tabulate", "version": "0.10.0"}}]}
        Path(cmd[cmd.index("--report") + 1]).write_text(json.dumps(report))
        return subprocess.CompletedProcess(cmd, 0, "", "")

    return calls, fake


def test_without_uv_or_pip_the_planner_runs_ensurepip_once_and_plans(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(deps.shutil, "which", lambda name: None)
    monkeypatch.setenv("HOME", str(tmp_path))
    calls, fake = _pip_less(ensurepip_works=True)
    plan = Installer("/venv/bin/python", host_modules=lambda: (), run=fake).plan(
        "tabulate", kernel_modules=[]
    )
    assert (plan.route, plan.version) == (Route.INSTALL, "0.10.0")
    assert sum(c[1:3] == ["-m", "ensurepip"] for c in calls) == 1


def test_an_environment_with_no_installer_keeps_its_saved_installs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(deps.shutil, "which", lambda name: None)
    monkeypatch.setenv("HOME", str(tmp_path))
    calls, fake = _pip_less(ensurepip_works=False)
    pending = tmp_path / "pending.json"
    installer = Installer("/venv/bin/python", host_modules=lambda: (), run=fake, pending=pending)
    plan = installer.plan("tabulate", kernel_modules=[])
    assert (plan.route, plan.reason) == (Route.REFUSE, "no_installer")
    installer.save_for_next_run(Plan("sktime", Route.NEXT_RUN, "1.1.0"), module="sktime")
    [line] = installer.install_pending()
    assert line.startswith("sktime waits: this environment has no pip")
    assert [e["package"] for e in installer.pending()] == ["sktime"]
    assert sum(c[1:3] == ["-m", "ensurepip"] for c in calls) == 1


def test_every_install_carries_the_constraints_file_and_refuses_frozen_names(
    tmp_path: Path,
) -> None:
    calls: list[list[str]] = []

    def fake(cmd: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        if cmd[1:3] == ["-m", "pip"]:
            return subprocess.CompletedProcess(cmd, 1, "", "No module named pip")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    pins = tmp_path / "pins.txt"
    pins.write_text("numpy==2.4.6\n")
    assert deps.install("/venv/bin/python", ["catboost"], pins, run=fake) == ""
    assert [c[:3] for c in calls] == [
        ["/venv/bin/python", "-m", "pip"],
        ["/bin/uv", "pip", "install"],
    ]
    assert all(c[c.index("-c") : c.index("-c") + 2] == ["-c", str(pins)] for c in calls)
    refused = deps.install("/venv/bin/python", ["torchvision==0.24.0"], pins, run=fake)
    assert refused == "torchvision never changes during a run"
    assert len(calls) == 2


def test_the_ensurepip_rung_retries_pip_with_the_constraints_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(deps.shutil, "which", lambda name: None)
    monkeypatch.setenv("HOME", str(tmp_path))
    calls: list[list[str]] = []

    def fake(cmd: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        missing = cmd[1:3] == ["-m", "pip"] and len(calls) == 1
        return subprocess.CompletedProcess(cmd, 1 if missing else 0, "", "No module named pip")

    pins = tmp_path / "pins.txt"
    pins.write_text("")
    assert deps.install("/venv/bin/python", ["catboost"], pins, run=fake) == ""
    installs = [c for c in calls if "install" in c]
    assert len(installs) == 2
    assert all("-c" in c for c in installs)
    assert any("ensurepip" in c for c in calls)


def test_a_route_install_asks_for_the_floor_of_the_planned_version(tmp_path: Path) -> None:
    seen: list[list[str]] = []

    def fake(cmd: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        seen.append([*cmd, *Path(cmd[cmd.index("-c") + 1]).read_text().split()])
        return subprocess.CompletedProcess(cmd, 0, "", "")

    plan = Plan("plotly", Route.RESTART, "7.1.0", pins=("numpy==2.4.6",))
    assert Installer("/venv/bin/python", run=fake).install(plan) == ""
    assert "plotly>=7.1" in seen[0]
    assert "numpy==2.4.6" in seen[0]


def test_install_pending_installs_what_resolves_and_drops_what_never_will(tmp_path: Path) -> None:
    pending = tmp_path / "pending.json"
    sktime = _uv(" - pandas==3.0.3\n + pandas==2.3.3\n + sktime==1.1.0\n")
    run = _Scripted(
        [
            sktime,
            sktime,
            _uv(" - pandas==3.0.3\n + pandas==2.1.4\n + pycaret==3.3.2\n"),
            _uv(ok=False, error="Because pycaret>=3.3.0 depends on pandas<2.2.0 and pandas>=2.2.0"),
            _uv(ok=False, error="Caused by: Failed to fetch: `https://pypi.org/simple/lazy/`"),
        ]
    )
    installer = _installer(run, host=("typer",), pending=pending)
    for package in ("sktime", "pycaret", "lazy"):
        installer.save_for_next_run(Plan(package, Route.NEXT_RUN, "1.0"), module=package)
    installs: list[str] = []

    def record(plan: Plan, **_: Any) -> str:
        installs.append(plan.package)
        return ""

    installer.install = record  # type: ignore[method-assign]
    lines = installer.install_pending()
    assert installs == ["sktime"]
    assert lines[0].startswith("installed sktime 1.1.0 saved by an earlier run (pandas 3.0.3")
    assert lines[1].startswith("dropped pycaret: it cannot be installed beside iterate")
    assert lines[2] == "lazy waits: the package index is unreachable"
    assert [e["package"] for e in json.loads(pending.read_text())["entries"]] == ["lazy"]


def test_a_saved_install_is_found_from_any_folder_and_either_interpreter_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    venv = tmp_path / "venv"
    venv.mkdir()
    (tmp_path / "link").symlink_to(venv)
    for folder in ("projA", "projB"):
        (tmp_path / folder).mkdir()
    monkeypatch.chdir(tmp_path / "projA")
    saving = Installer(str(venv / "bin" / "python3"), pending=deps.pending_path(str(venv)))
    saving.save_for_next_run(Plan("sktime", Route.NEXT_RUN, "1.1.0"), module="sktime")
    monkeypatch.chdir(tmp_path / "projB")
    through_a_link = deps.pending_path(str(tmp_path / "link"))
    found = Installer(str(venv / "bin" / "python"), pending=through_a_link).pending()
    assert [e["package"] for e in found] == ["sktime"]
    assert through_a_link.parent == tmp_path / "cache" / "iterate" / "installs"
    other = tmp_path / "other-venv"
    other.mkdir()
    assert Installer(pending=deps.pending_path(str(other))).pending() == []


def test_a_broken_pending_file_is_read_as_nothing_waiting(tmp_path: Path) -> None:
    pending = tmp_path / "pending.json"
    pending.write_text("{not json")
    assert Installer(pending=pending).pending() == []
    pending.write_text(json.dumps({"entries": [{"package": 3}, "sktime", {"package": "ok"}]}))
    assert Installer(pending=pending).pending() == [{"package": "ok"}]
    for shape in ({"entries": None}, {"entries": 5}, {"entries": {"package": "ok"}}, [1], "x"):
        pending.write_text(json.dumps(shape))
        assert Installer(pending=pending).pending() == []
    entries = [
        {"package": "tabulate", "module": 5},
        {"package": ""},
        {"package": "ok", "module": None},
        {"package": "fine", "module": "fine.sub"},
    ]
    pending.write_text(json.dumps({"entries": entries}))
    assert Installer(pending=pending).pending() == entries[2:]


def test_saved_installs_are_not_read_by_a_run_off_the_local_venue(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from iterate import cli

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))

    def unread(self: Installer) -> list[dict[str, Any]]:
        raise AssertionError("the saved installs were read")

    monkeypatch.setattr(Installer, "pending", unread)
    cli._install_saved_packages(install=True, compute="e2b")
    cli._install_saved_packages(install=False, compute="e2b")


def test_a_constraint_never_carries_extras(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.undo()
    lines = [
        "pandas>=2.2.0",
        "uvicorn[standard]>=0.30",
        'torch>=2.9; extra == "vision"',
        'pytest>=8.3; extra == "dev"',
    ]
    monkeypatch.setattr(deps.importlib.metadata, "requires", lambda name: lines)
    assert deps.own_requirements() == ("pandas>=2.2.0", "uvicorn>=0.30")
    assert deps.vision_requirements() == ("torch>=2.9",)


def test_ensure_torch_installs_the_vision_extra_only_with_consent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], list[str], float | None]] = []

    def fake(cmd: list[str], **kw: Any) -> subprocess.CompletedProcess[str]:
        calls.append((cmd, Path(cmd[cmd.index("-c") + 1]).read_text().split(), kw["timeout"]))
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(deps.subprocess, "run", fake)
    monkeypatch.setattr(deps, "vision_requirements", lambda: ("torch>=2.9", "torchvision>=0.24"))
    assert deps.ensure_torch(consent=True) == ""
    assert calls == []
    monkeypatch.setattr(
        deps, "installed", lambda: {k: v for k, v in VERSIONS.items() if "torch" not in k}
    )
    assert "pip install 'iterate-ai[vision]'" in deps.ensure_torch(consent=False)
    assert calls == []
    assert deps.ensure_torch(consent=True, host_modules=["numpy"]) == ""
    [(cmd, pins, timeout)] = calls
    assert cmd[-2:] == ["torch>=2.9", "torchvision>=0.24"]
    assert {"numpy==2.4.6", "pandas>=2.2.0"} <= set(pins)
    assert timeout is None


def test_the_saved_install_runs_before_the_run_imports_pandas(tmp_path: Path) -> None:
    script = (
        "import sys, json\n"
        "from typer.testing import CliRunner\n"
        "import iterate.cli as cli\n"
        "from iterate.adapters.compute import deps\n"
        "seen = {}\n"
        "def fake(self):\n"
        "    seen['pandas'] = 'pandas' in sys.modules\n"
        "    seen['numpy'] = 'numpy' in sys.modules\n"
        "    return ['installed sktime 1.1.0 saved by an earlier run']\n"
        "def no_llm(*a, **kw):\n"
        "    raise SystemExit('no LLM in this test')\n"
        "deps.Installer.install_pending = fake\n"
        "deps.Installer.pending = lambda self: [{'package': 'sktime'}]\n"
        "import iterate.llm.factory as factory\n"
        "factory.build_client = no_llm\n"
        "open('d.csv', 'w').write('num,churn\\n' + ''.join(f'{i},{i % 2}\\n' for i in range(40)))\n"
        "args = ['run', '--data', 'd.csv', '--target', 'churn', '--metric', 'f1', '--install',\n"
        "        '--compute', 'local', '--no-research', '--plain']\n"
        "out = CliRunner().invoke(cli.app, args, env={'COLUMNS': '200'}).output\n"
        "print(json.dumps({**seen, 'output': out}))\n"
    )
    env = {
        **os.environ,
        "XDG_CACHE_HOME": str(tmp_path / "cache"),
        "XDG_CONFIG_HOME": str(tmp_path / "config"),
    }
    out = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=180,
        env=env,
        cwd=tmp_path,
        check=True,
    )
    seen = json.loads(out.stdout.strip().splitlines()[-1])
    assert (seen["pandas"], seen["numpy"]) == (False, False)
    assert seen["output"].startswith("installs: installed sktime 1.1.0 saved by an earlier run")
