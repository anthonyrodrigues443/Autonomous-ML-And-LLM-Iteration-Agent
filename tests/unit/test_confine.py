"""A local kernel opens its own folder and what its run gave it, nothing else.

The profile, the kernel env and the error reading run everywhere; the sandboxed
kernels run only where sandbox-exec can apply a profile, which CI's ubuntu runner
never can.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from iterate.adapters.compute import confine
from iterate.adapters.compute.kernel import (
    KERNEL_SECRETS,
    KEY_HIDDEN,
    Blocked,
    CellResult,
    E2BKernel,
    LocalKernel,
    is_secret,
)
from iterate.config import Settings

pytestmark = pytest.mark.unit

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

sandboxed = pytest.mark.skipif(
    not confine.sandbox_available(), reason="needs macOS sandbox-exec outside another sandbox"
)


@pytest.fixture(autouse=True)
def _fresh_probe() -> Iterator[None]:
    confine.sandbox_available.cache_clear()
    confine._import_path.cache_clear()
    yield
    confine.sandbox_available.cache_clear()


# ─── the profile ───────────────────────────────────────────────────────────


def test_profile_quotes_awkward_paths_and_lists_every_folder(tmp_path: Path) -> None:
    odd = tmp_path / 'we"ird\\name'
    conf = confine.Confinement(
        reads=(tmp_path / "v1",), weights=tmp_path / "weights", files=(tmp_path / "a.db",)
    )
    text = confine.profile(odd, tmp_path / "scratch", conf)
    real = os.path.realpath(tmp_path)
    assert f'(subpath "{real}/we\\"ird\\\\name")' in text
    read_rule = next(line for line in text.splitlines() if line.startswith("(allow file-read-data"))
    write_rule = next(line for line in text.splitlines() if line.startswith("(allow file-write*"))
    for folder in ("scratch", "weights"):
        assert f'(subpath "{real}/{folder}")' in read_rule
        assert f'(subpath "{real}/{folder}")' in write_rule
    assert f'(subpath "{real}/v1")' in read_rule
    assert f"{real}/v1" not in write_rule
    for side in ("", "-journal", "-wal", "-shm"):
        assert f'(literal "{real}/a.db{side}")' in read_rule
        assert f'(literal "{real}/a.db{side}")' in write_rule
    assert "network" not in text


def test_an_import_path_that_holds_home_or_a_protected_path_is_not_opened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, site_packages = tmp_path / "proj", tmp_path / "venv" / "site-packages"
    monkeypatch.setattr(
        confine, "_import_path", lambda _: (str(project), str(site_packages), str(Path.home()), "/")
    )
    reads = confine.interpreter_reads((project / ".iterate", project / "data.csv"))
    assert os.path.realpath(site_packages) in reads
    assert os.path.realpath(project) not in reads
    assert os.path.realpath(Path.home()) not in reads
    assert "/" not in reads


def test_the_import_path_is_asked_of_the_interpreter_not_copied_from_the_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.syspath_prepend(str(tmp_path / "host-only"))
    assert os.path.realpath(tmp_path / "host-only") not in confine.interpreter_reads()


def _fake_installers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "bin").mkdir()
    for name in ("uv", "pip"):
        (tmp_path / "bin" / name).write_text("#!/bin/sh\n")
        (tmp_path / "bin" / name).chmod(0o755)
    monkeypatch.setenv("PATH", os.pathsep.join([str(tmp_path / "bin"), os.environ["PATH"]]))
    return tmp_path / "bin"


def test_installers_and_anything_a_cell_writes_cannot_be_executed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_installers(tmp_path, monkeypatch)
    weights = tmp_path / "weights"
    text = confine.profile(
        tmp_path / "work", tmp_path / "scratch", confine.Confinement(weights=weights)
    )
    rule = next(line for line in text.splitlines() if line.startswith("(deny process-exec"))
    real = os.path.realpath(tmp_path)
    for path in ("bin/uv", "bin/pip"):
        assert f'(literal "{real}/{path}")' in rule
    for folder in ("work", "scratch", "weights"):
        assert f'(subpath "{real}/{folder}")' in rule


def test_every_installer_spelling_on_path_is_listed_shadowed_ones_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    names = {first: ("pip3.14", "uvx", "pipeline", "uv.txt"), second: ("pip", "pip3", "pdm")}
    for folder, found in names.items():
        folder.mkdir()
        for name in found:
            (folder / name).write_text("#!/bin/sh\n")
            (folder / name).chmod(0o755)
    (first / "pip3").symlink_to(sys.executable)
    monkeypatch.setenv("PATH", os.pathsep.join([str(first), str(second), "/no/such/folder"]))
    listed = confine.installer_paths()
    real = os.path.realpath(tmp_path)
    for path in ("first/pip3.14", "first/uvx", "second/pip", "second/pip3", "second/pdm"):
        assert f"{real}/{path}" in listed
    assert f"{real}/first/pipeline" not in listed
    assert f"{real}/first/uv.txt" not in listed
    assert os.path.realpath(sys.executable) not in listed


def test_the_weights_folder_follows_the_cache_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert confine.weights_dir() == tmp_path / "xdg" / "iterate" / "weights"


# ─── the probe ─────────────────────────────────────────────────────────────


def test_no_sandbox_off_macos(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    confine.sandbox_available.cache_clear()
    assert confine.sandbox_available() is False


def test_no_sandbox_when_the_probe_carrying_a_deny_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[list[str]] = []

    def refused(cmd: list[str], **_: Any) -> subprocess.CompletedProcess[bytes]:
        seen.append(cmd)
        return subprocess.CompletedProcess(cmd, 71, b"", b"sandbox_apply: Operation not permitted")

    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(Path, "exists", lambda self: True)
    monkeypatch.setattr(subprocess, "run", refused)
    confine.sandbox_available.cache_clear()
    assert confine.sandbox_available() is False
    assert "(deny " in seen[0][2]


# ─── reading a refusal ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("error", "named"),
    [
        (
            "PermissionError: [Errno 1] Operation not permitted: '/Users/x/Desktop'",
            Blocked("/Users/x/Desktop", program=False),
        ),
        (
            "PermissionError: [Errno 1] Operation not permitted: '../other/train.csv'",
            Blocked("../other/train.csv", program=False),
        ),
        ("PermissionError: [Errno 1] Operation not permitted: 'predictions.csv'", None),
        ("RuntimeError: Operation not permitted", None),
        ("URLError: <urlopen error [Errno 1] Operation not permitted>", None),
        ("sqlite3.OperationalError: unable to open database file", None),
        ("PermissionError: [Errno 13] Permission denied: '/etc/sudoers'", None),
        (None, None),
    ],
)
def test_only_a_refused_path_outside_the_folder_is_named(
    tmp_path: Path, error: str | None, named: Blocked | None
) -> None:
    kernel = LocalKernel()
    kernel.confined, kernel._workdir = True, tmp_path / "work"
    assert kernel.blocked(error) == named


def test_a_refused_installer_is_named_as_a_program(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bin_dir = _fake_installers(tmp_path, monkeypatch)
    kernel = LocalKernel()
    kernel.confined, kernel._workdir = True, tmp_path / "work"
    kernel._programs = confine.installer_paths()
    by_path = f"PermissionError: [Errno 1] Operation not permitted: '{bin_dir / 'uv'}'"
    by_name = "PermissionError: [Errno 1] Operation not permitted: 'pip'"
    assert kernel.blocked(by_path) == Blocked(str(bin_dir / "uv"), program=True)
    assert kernel.blocked(by_name) == Blocked("pip", program=True)


def test_an_unconfined_or_remote_kernel_names_nothing(tmp_path: Path) -> None:
    error = "PermissionError: [Errno 1] Operation not permitted: '/Users/x/Desktop'"
    kernel = LocalKernel()
    kernel.confined, kernel._workdir = False, tmp_path
    assert kernel.blocked(error) is None
    assert E2BKernel().blocked(error) is None


# ─── the environment every local kernel starts with ────────────────────────


def _env_of(kernel: LocalKernel, names: tuple[str, ...]) -> dict[str, str | None]:
    kernel.start({})
    try:
        result = kernel.run_cell(
            f"import json, os\nprint(json.dumps({{k: os.environ.get(k) for k in {names!r}}}))",
            timeout=60,
        )
    finally:
        kernel.close()
    assert result.ok, result.error
    return dict(json.loads(result.stdout.strip().splitlines()[-1]))


def test_the_kernel_env_points_at_its_own_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OLDPWD", str(tmp_path / "project"))
    kernel = LocalKernel()
    kernel.start({})
    try:
        out = kernel.run_cell(
            "import os\nprint(os.environ['PWD'] == os.getcwd(), 'OLDPWD' in os.environ, "
            "len(os.environ['TMPDIR']), "
            "os.environ['IPYTHONDIR'].startswith(os.environ['TMPDIR'].rstrip('/') + '/'))",
            timeout=60,
        ).stdout.split()
    finally:
        kernel.close()
    assert out[:2] == ["True", "False"]
    assert int(out[2]) < 40
    assert out[3] == "True"


def test_without_a_sandbox_the_weights_caches_stay_inherited(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inherited = {
        "TORCH_HOME": str(tmp_path / "my-torch"),
        "HF_HOME": str(tmp_path / "my-hf"),
        "HF_HUB_CACHE": str(tmp_path / "big-disk"),
        "MPLCONFIGDIR": str(tmp_path / "my-matplotlib"),
        "XDG_CACHE_HOME": str(tmp_path / "my-cache"),
    }
    for name, value in inherited.items():
        monkeypatch.setenv(name, value)
    kernel = LocalKernel(confinement=confine.Confinement(weights=tmp_path / "weights"))
    kernel.confined = False
    assert _env_of(kernel, tuple(inherited)) == inherited
    assert not (tmp_path / "weights").exists()


def test_a_tabular_kernel_sees_no_provider_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    planted = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "ITERATE_TARGET_API_KEY", "groq_api_key")
    for name in planted:
        monkeypatch.setenv(name, f"secret-{name}")
    monkeypatch.setenv("HF_TOKEN", "hf-token")
    seen = _env_of(LocalKernel(), (*planted, "HF_TOKEN"))
    assert seen == {**dict.fromkeys(planted), "HF_TOKEN": "hf-token"}


def test_a_prompt_kernel_sees_only_its_target_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Read back reversed: printed as it is, the key would come out hidden."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk-driver")
    monkeypatch.setenv("ITERATE_TARGET_API_KEY", "inherited")
    kernel = LocalKernel(target_key="gsk-target")
    kernel.start({})
    try:
        result = kernel.run_cell(
            "import os\n"
            "print(os.environ.get('GROQ_API_KEY'), os.environ['ITERATE_TARGET_API_KEY'][::-1])",
            timeout=60,
        )
    finally:
        kernel.close()
    assert result.ok, result.error
    assert result.stdout.split() == ["None", "gsk-target"[::-1]]


def test_every_settings_key_is_kept_from_the_kernel() -> None:
    keys = {name.upper() for name in Settings.model_fields if name.endswith("_key")}
    assert keys
    assert keys <= KERNEL_SECRETS


def test_every_providers_own_key_is_kept_from_the_kernel() -> None:
    """A provider added to the factory brings a key variable with it."""
    from iterate.llm import factory

    own = {factory.own_key_env(name) for name in factory.known_providers()} - {None}
    assert {"OPENAI_API_KEY", "GROQ_API_KEY"} <= own
    assert all(is_secret(str(name)) for name in own)


@pytest.mark.parametrize(
    "name",
    [
        *["MISTRAL_API_KEY", "openrouter_api_key", "GROQ_API_KEY_2", "API_KEY", "OPENAI_KEY"],
        *["ANTHROPIC_AUTH_TOKEN", "GITHUB_TOKEN", "API_TOKEN_GITHUB", "AWS_BEARER_TOKEN_BEDROCK"],
        *["AWS_SECRET_ACCESS_KEY", "AZURE_CLIENT_SECRET", "GITHUB_PAT", "MYSQL_PWD", "PASSWD"],
        *["DISCORD_WEBHOOK_URL", "GOOGLE_APPLICATION_CREDENTIALS", "ITERATE_BACKEND_URL"],
        *["KAGGLE_KEY", "STRIPE_KEY_LIVE"],
    ],
)
def test_a_name_that_reads_as_a_secret_is_kept_from_the_kernel(name: str) -> None:
    assert is_secret(name)


@pytest.mark.parametrize(
    "name",
    [
        *["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_HUB_DISABLE_IMPLICIT_TOKEN"],
        *["TOKENIZERS_PARALLELISM", "KEYRING_BACKEND", "MONKEY", "PATH", "HOME", "TMPDIR"],
        *["PYTHONPATH", "OLLAMA_HOST", "OLLAMA_NUM_PARALLEL", "TORCH_HOME", "HF_HOME"],
        *["XDG_CACHE_HOME", "CUDA_VISIBLE_DEVICES", "PYTORCH_ENABLE_MPS_FALLBACK"],
        *["OMP_NUM_THREADS", "KMP_DUPLICATE_LIB_OK", "SSH_AUTH_SOCK", "ITERATE_MODEL"],
        *["SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "HTTPS_PROXY", "NO_PROXY"],
        # Read by the OpenAI client a cell's `ask` builds, which sends them where the
        # host's client does.
        *["OPENAI_ORG_ID", "OPENAI_PROJECT_ID", "OPENAI_CUSTOM_HEADERS"],
    ],
)
def test_what_a_cell_needs_is_not_taken_for_a_secret(name: str) -> None:
    assert not is_secret(name)


def test_a_key_nobody_listed_does_not_reach_a_cell(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MISTRAL_API_KEY", "secret")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "secret")
    names = ("MISTRAL_API_KEY", "ANTHROPIC_AUTH_TOKEN")
    assert _env_of(LocalKernel(), names) == dict.fromkeys(names)


def test_a_cell_that_prints_its_key_shows_it_hidden() -> None:
    kernel = LocalKernel(target_key="gsk-the-target-key")
    result = kernel._hidden(
        CellResult(
            stdout="key is gsk-the-target-key\n",
            stderr="gsk-the-target-key",
            error="AuthError: gsk-the-target-key was refused",
            outputs=[{"type": "stream", "name": "stdout", "text": "key is gsk-the-target-key\n"}],
            restarted=True,
        )
    )
    assert "gsk-the-target-key" not in repr(result)
    assert result.stdout == f"key is {KEY_HIDDEN}\n"
    assert result.outputs[0]["text"] == f"key is {KEY_HIDDEN}\n"
    assert result.restarted is True


@pytest.mark.parametrize("key", ['sk-with"quote', "sk-with\\backslash", "sk-ünïcode-key"])
def test_a_key_with_an_odd_character_is_hidden_too(key: str) -> None:
    outputs = [
        {"type": "display_data", "data": {"text/plain": f"<{key}>", "n": 3}, "metadata": {}},
        {"type": "error", "traceback": [f"Bearer {key}", "line 2"], "evalue": key},
    ]
    result = LocalKernel(target_key=key)._hidden(CellResult(stdout=key, stderr="", outputs=outputs))
    assert key not in repr(result)
    assert result.outputs[0]["data"] == {"text/plain": f"<{KEY_HIDDEN}>", "n": 3}
    assert result.outputs[1]["traceback"] == [f"Bearer {KEY_HIDDEN}", "line 2"]


def test_a_key_printed_in_two_writes_is_hidden_as_the_notebook_shows_it() -> None:
    from iterate.deliver.notebook import _settled

    key = "gsk-the-target-key-0123456789"
    outputs = [
        {"type": "stream", "name": "stdout", "text": key[:12]},
        {"type": "stream", "name": "stdout", "text": key[12:] + "\n"},
        {"type": "stream", "name": "stderr", "text": "a warning\n"},
    ]
    result = LocalKernel(target_key=key)._hidden(
        CellResult(stdout=key + "\n", stderr="a warning\n", outputs=outputs)
    )
    assert key not in repr(_settled(result.outputs))
    assert [o["text"] for o in result.outputs] == [f"{KEY_HIDDEN}\n", "a warning\n"]
    assert outputs[0]["text"] == key[:12]  # the captured record is not changed


def test_a_file_a_cell_wrote_comes_back_with_the_key_hidden(tmp_path: Path) -> None:
    key = "gsk-the-target-key-0123456789"
    kernel = LocalKernel(target_key=key)
    kernel._workdir = tmp_path
    (tmp_path / "prompt.json").write_text(f'{{"system": "use {key}"}}', encoding="utf-8")
    (tmp_path / "predictions.csv").write_bytes(b"prediction\ntoxic\n")
    assert kernel.read_output("prompt.json") == f'{{"system": "use {KEY_HIDDEN}"}}'.encode()
    assert kernel.read_output("predictions.csv") == b"prediction\ntoxic\n"
    assert kernel.read_output("missing.csv") is None


def test_a_key_cut_short_is_hidden_too() -> None:
    """The harness's own summary of a cell's names cuts every value at 40 characters."""
    key = "gsk_" + "a1B2c3D4e5" * 5
    shown = f"k = '{key[:39]}\n"
    result = LocalKernel(target_key=key)._hidden(CellResult(stdout=shown, stderr=""))
    assert result.stdout == f"k = '{KEY_HIDDEN}\n"
    untouched = "gsk_ is how a groq key starts\n"
    assert LocalKernel(target_key=key)._hidden(CellResult(untouched, "")).stdout == untouched


def test_a_key_in_pieces_under_a_name_or_under_colour_is_hidden_too() -> None:
    key = "gsk-the-target-key-0123456789"
    red = f"\x1b[31m{key[:10]}\x1b[0m{key[10:]}"
    outputs = [
        {"type": "stream", "name": "stdout", "text": key[:8]},
        {"type": "stream", "name": "stderr", "text": "a warning\n"},
        {"type": "stream", "name": "stdout", "text": key[8:20]},
        {"type": "stream", "name": "stdout", "text": key[20:] + "\n"},
        {"type": "display_data", "data": {"application/json": {key: 1}}, "metadata": {}},
        {"type": "error", "ename": "E", "evalue": "x", "traceback": [red, "line 2"]},
    ]
    result = LocalKernel(target_key=key)._hidden(CellResult("", "", outputs=outputs))
    assert key not in repr(result.outputs)
    assert key[:10] not in repr(result.outputs)
    assert [o.get("text") for o in result.outputs if o["type"] == "stream"] == [
        f"{KEY_HIDDEN}\n",
        "a warning\n",
    ]
    assert result.outputs[2]["data"] == {"application/json": {KEY_HIDDEN: 1}}
    assert result.outputs[3]["traceback"] == [KEY_HIDDEN, "line 2"]


def test_a_file_that_is_not_text_comes_back_byte_for_byte(tmp_path: Path) -> None:
    key = "gsk-the-target-key-0123456789"
    kernel = LocalKernel(target_key=key)
    kernel._workdir = tmp_path
    weights = b"\x80\x04\x95 weights " + key.encode() + b" \xff\xfe more"
    (tmp_path / "best_model.pt").write_bytes(weights)
    assert kernel.read_output("best_model.pt") == weights


def test_a_cell_of_a_run_with_no_key_is_left_as_it_is() -> None:
    result = CellResult(stdout="nothing to hide", stderr="", outputs=[{"text": "x"}])
    assert LocalKernel()._hidden(result) is result
    assert LocalKernel(target_key="short")._hidden(result) is result


def test_the_key_a_live_cell_prints_is_hidden() -> None:
    kernel = LocalKernel(target_key="gsk-the-target-key")
    kernel.start({})
    try:
        result = kernel.run_cell(
            "import os\nprint(os.environ['ITERATE_TARGET_API_KEY'])", timeout=60
        )
    finally:
        kernel.close()
    assert result.ok, result.error
    assert result.stdout.strip() == KEY_HIDDEN


# ─── sandboxed kernels (macOS) ─────────────────────────────────────────────


@pytest.fixture
def boot() -> Iterator[Callable[..., LocalKernel]]:
    started: list[LocalKernel] = []

    def _boot(**kwargs: Any) -> LocalKernel:
        kernel = LocalKernel(confinement=confine.Confinement(**kwargs))
        kernel.start({"train.csv": b"a,b\n1,2\n"})
        started.append(kernel)
        return kernel

    yield _boot
    for kernel in started:
        kernel.close()


def _cell(kernel: LocalKernel, code: str) -> str:
    result = kernel.run_cell(code, timeout=60)
    return result.stdout.strip() if result.ok else (result.error or "").strip().splitlines()[-1]


@sandboxed
def test_a_cell_opens_its_folder_and_nothing_the_project_holds(
    boot: Callable[..., LocalKernel], tmp_path: Path
) -> None:
    project = tmp_path / "proj"
    (project / ".iterate").mkdir(parents=True)
    (project / "data.csv").write_text("x,y\n1,2\n")
    sqlite3.connect(project / ".iterate" / "memory.db").execute("create table t(x)")
    kernel = boot()
    assert kernel.confined
    opened = "print(open('train.csv').read().split()[0]); open('predictions.csv', 'w').write('1')"
    assert _cell(kernel, opened) == "a,b"
    assert kernel.read_output("predictions.csv") == b"1"
    refused = _cell(kernel, f"open({str(project / 'data.csv')!r}).read()")
    assert refused.startswith("PermissionError: [Errno 1]")
    assert kernel.blocked(refused) == Blocked(str(project / "data.csv"), program=False)
    memory = project / ".iterate" / "memory.db"
    assert "unable to open database file" in _cell(
        kernel, f"import sqlite3; sqlite3.connect({str(memory)!r}).execute('select 1')"
    )
    for code in (
        "import os; os.listdir(os.path.expanduser('~'))",
        "import os; os.listdir('..')",
        f"open({str(tmp_path / 'elsewhere.txt')!r}, 'w')",
    ):
        assert _cell(kernel, code).startswith("PermissionError: [Errno 1]"), code


@sandboxed
def test_a_read_folder_opens_but_its_siblings_do_not(
    boot: Callable[..., LocalKernel], tmp_path: Path
) -> None:
    for version in ("v1", "v2"):
        (tmp_path / "images" / version).mkdir(parents=True)
        (tmp_path / "images" / version / "img").write_bytes(b"png")
    kernel = boot(reads=(tmp_path / "images" / "v1",))
    read = f"print(open({str(tmp_path / 'images/v1/img')!r}, 'rb').read())"
    assert _cell(kernel, read) == "b'png'"
    write = f"open({str(tmp_path / 'images/v1/new')!r}, 'w')"
    assert _cell(kernel, write).startswith("PermissionError")
    for code in (
        f"open({str(tmp_path / 'images/v2/img')!r}, 'rb')",
        f"import os; os.listdir({str(tmp_path / 'images')!r})",
    ):
        assert _cell(kernel, code).startswith("PermissionError"), code


@sandboxed
def test_the_answer_cache_is_the_one_file_outside_the_folder_a_cell_writes(
    boot: Callable[..., LocalKernel], tmp_path: Path
) -> None:
    db, other = tmp_path / "prompt-answers.db", tmp_path / "memory.db"
    kernel = boot(files=(db,))
    code = (
        "import sqlite3\nc = sqlite3.connect({!r}); c.execute('create table if not exists t(x)')\n"
        "c.execute('insert into t values (1)'); c.commit(); print('ok')"
    )
    assert _cell(kernel, code.format(str(db))) == "ok"
    assert "unable to open database file" in _cell(kernel, code.format(str(other)))
    assert sqlite3.connect(db).execute("select count(*) from t").fetchone() == (1,)


@sandboxed
def test_a_sandboxed_kernel_downloads_weights_into_the_weights_folder_only(
    boot: Callable[..., LocalKernel], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "big-disk"))
    monkeypatch.setenv("TORCH_HOME", str(tmp_path / "torch-elsewhere"))
    weights = tmp_path / "weights"
    kernel = boot(weights=weights)
    seen = "import os; print(os.environ['TORCH_HOME'], os.environ['HF_HOME'], 'HF_HUB_CACHE' in os.environ)"
    assert _cell(kernel, seen).split() == [
        str(weights / "torch"),
        str(weights / "huggingface"),
        "False",
    ]
    write = (
        "import os\np = os.path.join({}, 'hub', 'checkpoints')\nos.makedirs(p, exist_ok=True)\n"
        "open(os.path.join(p, 'w.pth'), 'wb').write(b'w')\nprint('ok')"
    )
    assert _cell(kernel, write.format("os.environ['TORCH_HOME']")) == "ok"
    assert (weights / "torch" / "hub" / "checkpoints" / "w.pth").read_bytes() == b"w"
    assert _cell(kernel, write.format(repr(str(tmp_path / "torch-elsewhere")))).startswith(
        "PermissionError"
    )


@sandboxed
def test_a_sandboxed_kernel_keeps_font_caches_in_the_weights_folder(
    boot: Callable[..., LocalKernel], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "home-matplotlib"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "home-cache"))
    weights = tmp_path / "weights"
    kernel = boot(weights=weights)
    write = (
        "import os\n"
        "for name, sub in (('MPLCONFIGDIR', ''), ('XDG_CACHE_HOME', 'fontconfig')):\n"
        "    folder = os.path.join(os.environ[name], sub)\n"
        "    os.makedirs(folder, exist_ok=True)\n"
        "    open(os.path.join(folder, 'fonts.json'), 'w').write('{}')\n"
        "print('ok')"
    )
    assert _cell(kernel, write) == "ok"
    assert (weights / "matplotlib" / "fonts.json").is_file()
    assert (weights / "cache" / "fontconfig" / "fonts.json").is_file()
    assert not (tmp_path / "home-matplotlib").exists()
    assert not (tmp_path / "home-cache").exists()


@sandboxed
def test_a_cell_cannot_run_an_installer_or_a_binary_it_wrote(
    boot: Callable[..., LocalKernel],
) -> None:
    kernel = boot()
    code = "import subprocess\nsubprocess.run([{!r}, '--version'], capture_output=True)"
    uv = shutil.which("uv")
    if uv is not None:
        refused = _cell(kernel, code.format("uv"))
        assert refused.startswith("PermissionError")
        assert kernel.blocked(refused) == Blocked("uv", program=True)
    _cell(kernel, "import shutil, os\nshutil.copy('/bin/echo', 'echo2'); os.chmod('echo2', 0o755)")
    assert _cell(kernel, code.format("./echo2")).startswith("PermissionError")
    assert _cell(kernel, "import platform; print(platform.uname().system)") == "Darwin"


@sandboxed
def test_a_restarted_kernel_is_still_confined(boot: Callable[..., LocalKernel]) -> None:
    kernel = boot()
    before = _cell(kernel, "import os; print(os.getpid())")
    kernel.restart()
    assert _cell(kernel, "import os; print(os.getpid())") != before
    assert _cell(kernel, "import os; os.listdir(os.path.expanduser('~'))").startswith(
        "PermissionError"
    )
    assert _cell(kernel, "print(open('train.csv').read().split()[0])") == "a,b"


@sandboxed
def test_a_module_on_pythonpath_outside_the_venv_still_imports(
    boot: Callable[..., LocalKernel], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "extra").mkdir()
    (tmp_path / "extra" / "extra_mod.py").write_text("VALUE = 42\n")
    monkeypatch.setenv(
        "PYTHONPATH",
        os.pathsep.join(filter(None, [str(tmp_path / "extra"), os.environ.get("PYTHONPATH")])),
    )
    assert _cell(boot(), "import extra_mod; print(extra_mod.VALUE)") == "42"


@sandboxed
def test_inside_another_sandbox_the_kernel_starts_unconfined() -> None:
    code = (
        "from iterate.adapters.compute.kernel import LocalKernel\n"
        "k = LocalKernel(); k.start({'a.txt': b'x'})\n"
        "print(k.confined, k.run_cell(\"print(open('a.txt').read())\", timeout=30).stdout.strip())\n"
        "k.close()\n"
    )
    outer = "(version 1)(allow default)"
    done = subprocess.run(
        [confine.SANDBOX_EXEC, "-p", outer, sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert done.stdout.split() == ["False", "x"], done.stderr[-500:]
