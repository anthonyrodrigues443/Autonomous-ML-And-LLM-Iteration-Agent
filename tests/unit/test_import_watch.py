"""The watch that records a missing import a cell caught itself.

A live cell wrapped `import catboost` in try/except, so nothing was raised and nothing
was installed. The watch is a finder appended LAST on sys.meta_path, asked only after
every real finder has failed, and it records a name only when the import came straight
from a cell. Everything here runs in a real IPython shell, which is what gives a cell
its line-cache entry.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

import pytest

from iterate.core import codegen

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

pytestmark = pytest.mark.unit


@pytest.fixture
def shell(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    """A shell with the watch installed, and sys.meta_path put back afterwards."""
    from IPython.core.interactiveshell import InteractiveShell

    monkeypatch.chdir(tmp_path)
    saved = list(sys.meta_path)
    started = InteractiveShell.instance()
    started.run_cell(codegen.IMPORT_WATCH)
    try:
        yield started
    finally:
        sys.meta_path[:] = saved
        InteractiveShell.clear_instance()


def recorded(tmp_path: Path) -> list[str]:
    path = tmp_path / codegen.MISSING_IMPORTS
    return path.read_text().split() if path.exists() else []


# ─── what it records ─────────────────────────────────────────────────────────


def test_an_import_a_cell_caught_is_recorded(shell: Any, tmp_path: Path) -> None:
    shell.run_cell("try:\n    import notreal_top\nexcept ImportError:\n    pass")
    assert recorded(tmp_path) == ["notreal_top"]


def test_a_dynamic_import_and_a_spec_lookup_are_recorded(shell: Any, tmp_path: Path) -> None:
    shell.run_cell(
        "import importlib, importlib.util\n"
        "try:\n"
        "    importlib.import_module('notreal_dynamic')\n"
        "except ImportError:\n"
        "    pass\n"
        "importlib.util.find_spec('notreal_spec')\n"
    )
    assert recorded(tmp_path) == ["notreal_dynamic", "notreal_spec"]


def test_an_import_inside_a_function_from_an_earlier_cell_is_recorded(
    shell: Any, tmp_path: Path
) -> None:
    shell.run_cell("def load():\n    import notreal_inside\n    return notreal_inside")
    shell.run_cell("try:\n    load()\nexcept ImportError:\n    pass")
    assert recorded(tmp_path) == ["notreal_inside"]


def test_a_child_of_a_namespace_package_is_recorded(
    shell: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`google.colab` is the live shape: the parent exists as a namespace with no file
    of its own, so nothing rules the child out."""
    (tmp_path / "ns_probe").mkdir()
    monkeypatch.syspath_prepend(str(tmp_path))
    shell.run_cell("try:\n    import ns_probe.missing\nexcept ImportError:\n    pass")
    assert recorded(tmp_path) == ["ns_probe.missing"]


def test_a_dotted_name_whose_top_level_is_absent_records_the_top_level(
    shell: Any, tmp_path: Path
) -> None:
    """`google.colab` in a venv that has no `google` folder at all: the parent import
    fails first, so the top name is what the watch is asked for and what it records."""
    shell.run_cell("try:\n    import notreal_top.colab\nexcept ImportError:\n    pass")
    assert recorded(tmp_path) == ["notreal_top"]


def test_a_name_is_recorded_once_however_often_it_is_tried(shell: Any, tmp_path: Path) -> None:
    for _ in range(3):
        shell.run_cell("try:\n    import notreal_twice\nexcept ImportError:\n    pass")
    assert recorded(tmp_path) == ["notreal_twice"]


# ─── what it leaves alone ────────────────────────────────────────────────────


def test_a_library_probing_its_own_optional_dependency_is_not_recorded(
    shell: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reason the frame is checked at all: pandas and friends import optional
    libraries inside try/except on every call, and none of those is the agent's."""
    (tmp_path / "probing_lib.py").write_text(
        "def read():\n    try:\n        import notreal_optional\n    except ImportError:\n"
        "        return 'fallback'\n    return 'real'\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    shell.run_cell("import probing_lib\nprobing_lib.read()")
    assert recorded(tmp_path) == []


def test_a_missing_submodule_of_an_installed_package_is_not_recorded(
    shell: Any, tmp_path: Path
) -> None:
    shell.run_cell("try:\n    import pandas.notreal\nexcept ImportError:\n    pass")
    assert recorded(tmp_path) == []


def test_a_stdlib_name_is_not_recorded(shell: Any, tmp_path: Path) -> None:
    """msvcrt is stdlib and missing off Windows: a name the index cannot help with."""
    shell.run_cell("try:\n    import msvcrt\nexcept ImportError:\n    pass")
    assert recorded(tmp_path) == []


def test_a_name_a_later_finder_provides_is_not_recorded(shell: Any, tmp_path: Path) -> None:
    shell.run_cell(
        "import importlib.machinery, importlib.util, sys\n"
        "class Provider:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name != 'provided_later':\n"
        "            return None\n"
        "        return importlib.util.spec_from_loader(name, loader=None)\n"
        "sys.meta_path.append(Provider())\n"
        "importlib.util.find_spec('provided_later')\n"
    )
    assert recorded(tmp_path) == []


def test_an_import_that_succeeds_is_not_recorded(shell: Any, tmp_path: Path) -> None:
    shell.run_cell("import json, pandas")
    assert recorded(tmp_path) == []


# ─── where it sits ───────────────────────────────────────────────────────────


def test_the_watch_is_asked_last(shell: Any) -> None:
    assert hasattr(sys.meta_path[-1], "iterate_log")


def test_running_the_preamble_again_leaves_one_watch(shell: Any, tmp_path: Path) -> None:
    """A restart re-runs the preamble in the same process on some paths, and two watches
    would record every name twice."""
    other = tmp_path / "second"
    other.mkdir()
    shell.run_cell(f"import os; os.chdir({str(other)!r})")
    shell.run_cell(codegen.IMPORT_WATCH)
    assert sum(1 for f in sys.meta_path if hasattr(f, "iterate_log")) == 1
    shell.run_cell("try:\n    import notreal_after\nexcept ImportError:\n    pass")
    assert recorded(other) == ["notreal_after"]  # the log follows the new folder
    assert recorded(tmp_path) == []


def test_a_failed_write_does_not_break_the_cell(shell: Any, tmp_path: Path) -> None:
    """The log is written by the cell's own process, into a folder the sandbox may have
    made read-only. A missing import must not become an OSError in the agent's cell."""
    shell.run_cell("_iterate_watch = None\nimport sys\nsys.meta_path[-1].iterate_log = '/'")
    shell.run_cell("try:\n    import notreal_unwritable\nexcept ImportError:\n    pass\nok = True")
    assert shell.user_ns["ok"] is True


# ─── the preambles carry it ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    "preamble",
    [
        codegen.session_preamble(),
        codegen.prompt_session_preamble(),
        codegen.vision_session_preamble(),
    ],
)
def test_every_preamble_ends_with_the_watch(preamble: str) -> None:
    assert preamble.endswith(codegen.IMPORT_WATCH)
    compile(preamble, "<preamble>", "exec")
