"""Tests for the CLI scaffold — guards against the single-command collapse bug."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner

from iterate import __version__, userconfig
from iterate.cli import app
from iterate.userconfig import SavedProvider

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit

runner = CliRunner()


def test_help_lists_both_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "version" in result.output
    assert "config" in result.output
    assert "setup" in result.output


def test_run_help_shows_code_and_compute_flags() -> None:
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    assert "--code" in result.output
    assert "--spec" in result.output
    assert "--compute" in result.output
    assert "--install" in result.output


def test_setup_saves_local_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    # backend=ollama, model=blank, compute=local, install=no, prompt providers=any
    result = runner.invoke(app, ["setup"], input="ollama\n\nlocal\nn\n\n")
    assert result.exit_code == 0, result.output
    cfg = userconfig.load_user_config()
    assert cfg == {"backend": "ollama", "compute": "local", "install": False}
    assert userconfig.load_prompt_settings() == userconfig.PromptSettings()


def test_setup_saves_e2b_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    # backend=openai, model=gpt-4o, api_key=sk-x, compute=e2b, e2b key=e2b-x, providers=any
    result = runner.invoke(app, ["setup"], input="openai\ngpt-4o\nsk-x\ne2b\ne2b-x\n\n")
    assert result.exit_code == 0, result.output
    cfg = userconfig.load_user_config()
    assert cfg["backend"] == "openai"
    assert cfg["model"] == "gpt-4o"
    assert cfg["api_key"] == "sk-x"
    assert cfg["compute"] == "e2b"
    assert cfg["e2b_api_key"] == "e2b-x"


def test_version_runs_as_a_subcommand() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_config_shows_what_a_run_will_use_and_masks_every_key() -> None:
    userconfig.save_user_config(
        {"backend": "groq", "model": "llama-3.3-70b", "api_key": "gsk-harness-key"},
        allowed=["openai", "vllm"],
        providers={
            "openai": SavedProvider("openai", api_key="sk-prompt-key"),
            "vllm": SavedProvider("vllm"),
            "together": SavedProvider("together", api_key="tg-held-back"),
        },
    )
    result = runner.invoke(app, ["config"])
    assert result.exit_code == 0, result.output
    out = result.output
    for label in ("backend:     groq", "model:       llama-3.3-70b", "timeout:     "):
        assert label in out
    assert "base_url:    https://api.groq.com/openai/v1" in out
    assert "api_key:     gs…ey from the saved config" in out
    assert "prompt providers allowed: openai, vllm" in out
    assert "openai: https://api.openai.com/v1 key sk…ey from the saved config" in out
    assert "vllm: NOT READY, no base URL saved" in out
    assert "saved but not allowed, never called: together" in out
    assert f"saved in {userconfig.config_path()}" in out.replace("\n", "")
    for secret in ("gsk-harness-key", "sk-prompt-key", "tg-held-back"):
        assert secret not in out


def test_config_shows_no_more_of_an_address_than_may_be_shown() -> None:
    userconfig.save_user_config(
        {"backend": "vllm", "base_url": "https://me:harness-pw@gw.test/v1?token=harness-t"},
        allowed=["vllm", "groq"],
        providers={
            "vllm": SavedProvider("vllm", base_url="http://me:saved-pw@gpu:8000/v1"),
            "groq": SavedProvider("groq", api_key="short"),
        },
    )
    out = runner.invoke(app, ["config"]).output
    assert "base_url:    https://gw.test/v1" in out
    assert "vllm: http://gpu:8000/v1 NOT READY, its saved address cannot be used" in out
    assert "groq: https://api.groq.com/openai/v1 key **** from the saved config" in out
    for secret in ("harness-pw", "harness-t", "saved-pw", "short"):
        assert secret not in out


def test_config_shows_a_harness_the_environment_points_at(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from iterate.config import get_settings

    monkeypatch.chdir(tmp_path)  # away from a project's .env, which holds keys of its own
    userconfig.save_user_config({"backend": "openai-compatible"}, allowed=["groq"])
    monkeypatch.setenv("ITERATE_BACKEND_URL", "http://gpu-box:8000/v1")
    monkeypatch.setenv("ITERATE_BACKEND_API_KEY", "box-key-0123456789")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    get_settings.cache_clear()
    try:
        out = runner.invoke(app, ["config"]).output
    finally:
        get_settings.cache_clear()
    assert "base_url:    http://gpu-box:8000/v1" in out
    assert "api_key:     bo…89 from ITERATE_BACKEND_API_KEY" in out
    assert "groq: https://api.groq.com/openai/v1 NOT READY, no key, set GROQ_API_KEY" in out


def test_config_says_when_a_provider_would_run_on_the_harnesss_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from iterate.config import get_settings

    monkeypatch.chdir(tmp_path)  # away from a project's .env, which holds keys of its own
    userconfig.save_user_config({"backend": "groq", "api_key": "gsk-harness-key"}, allowed=["groq"])
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    get_settings.cache_clear()
    try:
        out = runner.invoke(app, ["config"]).output
    finally:
        get_settings.cache_clear()
    assert "no key of its own; the harness's key when it is the harness model" in out
    assert "NOT READY" not in out


def test_config_with_nothing_saved_says_each_run_names_its_provider() -> None:
    result = runner.invoke(app, ["config"])
    assert result.exit_code == 0, result.output
    assert "backend:     ollama" in result.output
    assert "no list saved" in result.output
    assert "nothing saved yet: run `iterate setup`" in result.output
    assert "saved in" not in result.output


def test_setup_saves_the_providers_a_prompt_run_may_call() -> None:
    # harness: ollama, blank model, local, no install; then the list and a key for each
    answers = "ollama\n\nlocal\nn\nopenai, groq\nsk-prompt\n\n"
    result = runner.invoke(app, ["setup"], input=answers)
    assert result.exit_code == 0, result.output
    saved = userconfig.load_prompt_settings()
    assert saved.allowed == ("openai", "groq")
    assert saved.providers["openai"].api_key == "sk-prompt"
    assert saved.providers["groq"].api_key is None
    assert "read GROQ_API_KEY at run time" in result.output
    assert "readable by you only" in result.output


def test_setup_asks_again_for_a_name_it_does_not_know() -> None:
    answers = "grok\ngroq\nllama-3.3-70b\ngsk-h\nlocal\nn\nopenia\nopenai\nsk-p\n"
    result = runner.invoke(app, ["setup"], input=answers)
    assert result.exit_code == 0, result.output
    assert "grok is not one of" in result.output
    assert "openia: not a provider iterate knows. Choose from" in result.output
    assert userconfig.load_user_config()["backend"] == "groq"
    assert userconfig.load_prompt_settings().allowed == ("openai",)


def test_setup_run_again_keeps_the_keys_it_was_not_given() -> None:
    userconfig.save_user_config(
        {"backend": "groq", "api_key": "gsk-harness", "compute": "e2b", "e2b_api_key": "e2b-k"},
        allowed=["openai"],
        providers={
            "openai": SavedProvider("openai", api_key="sk-prompt"),
            "together": SavedProvider("together", api_key="tg-k"),
        },
    )
    # every answer left blank: the saved backend, keys and list stand
    result = runner.invoke(app, ["setup"], input="\n\n\nlocal\nn\n\n\n")
    assert result.exit_code == 0, result.output
    cfg = userconfig.load_user_config()
    assert (cfg["backend"], cfg["api_key"], cfg["e2b_api_key"]) == ("groq", "gsk-harness", "e2b-k")
    assert userconfig.load_prompt_settings().allowed == ("openai",)
    assert userconfig.load_saved_providers()["together"].api_key == "tg-k"
    assert userconfig.load_saved_providers()["openai"].api_key == "sk-prompt"


def test_setup_can_drop_the_list() -> None:
    userconfig.save_user_config(
        {"backend": "ollama"},
        allowed=["openai"],
        providers={"openai": SavedProvider("openai", api_key="sk-prompt")},
    )
    result = runner.invoke(app, ["setup"], input="ollama\n\nlocal\nn\nany\n")
    assert result.exit_code == 0, result.output
    assert userconfig.load_prompt_settings().allowed is None
    assert userconfig.load_saved_providers()["openai"].api_key == "sk-prompt"


def test_no_args_shows_help() -> None:
    result = runner.invoke(app, [])
    assert "Usage" in result.output


def _broken_file() -> Path:
    path = userconfig.config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('backend = "groq\n', encoding="utf-8")
    return path


def test_config_names_a_saved_file_it_cannot_read() -> None:
    path = _broken_file()
    result = runner.invoke(app, ["config"])
    assert result.exit_code == 1
    assert "the saved config cannot be read" in result.output
    assert str(path) in result.output.replace("\n", "")
    assert "Traceback" not in result.output


def test_setup_puts_right_a_saved_file_it_cannot_read() -> None:
    path = _broken_file()
    broken = path.read_bytes()
    result = runner.invoke(app, ["setup"], input="ollama\n\nlocal\nn\n\n")
    assert result.exit_code == 0, result.output
    assert "Starting from nothing" in result.output
    assert userconfig.load_user_config() == {
        "backend": "ollama",
        "compute": "local",
        "install": False,
    }
    assert path.with_name("config.toml.broken").read_bytes() == broken


def test_setup_left_half_way_leaves_a_file_it_cannot_read_where_it_was() -> None:
    path = _broken_file()
    broken = path.read_bytes()
    result = runner.invoke(app, ["setup"], input="ollama\n")
    assert result.exit_code != 0
    assert path.read_bytes() == broken
    assert [p.name for p in path.parent.iterdir()] == ["config.toml"]


def test_setup_asks_again_for_an_empty_list_and_for_a_server_with_no_address() -> None:
    answers = "ollama\n\nlocal\nn\n,\nvllm\n\ngpu:8000/v1\nhttp://gpu:8000/v1\n\n"
    result = runner.invoke(app, ["setup"], input=answers)
    assert result.exit_code == 0, result.output
    assert "Name at least one provider, or answer any" in result.output
    assert "vllm: no base URL saved. Give it as http:// or https://" in result.output
    assert "vllm: its saved address cannot be used" in result.output
    assert "--target-base-url" not in result.output
    saved = userconfig.load_prompt_settings()
    assert saved.allowed == ("vllm",)
    assert saved.providers["vllm"].base_url == "http://gpu:8000/v1"


def test_answers_piped_in_from_before_v07_still_save_and_keep_the_providers() -> None:
    """They end where the questions used to end. A terminal that is closed at the new
    question saves nothing; a script is not a terminal."""
    userconfig.save_user_config(
        {"backend": "ollama"},
        allowed=["openai"],
        providers={"openai": SavedProvider("openai", api_key="sk-prompt")},
    )
    result = runner.invoke(app, ["setup"], input="openai\ngpt-4o\nsk-x\ne2b\ne2b-x\n")
    assert result.exit_code == 0, result.output
    cfg = userconfig.load_user_config()
    assert (cfg["backend"], cfg["api_key"], cfg["e2b_api_key"]) == ("openai", "sk-x", "e2b-x")
    assert userconfig.load_prompt_settings().allowed == ("openai",)
    assert userconfig.load_saved_providers()["openai"].api_key == "sk-prompt"


def test_config_says_what_a_saved_provider_is_when_no_list_is_saved() -> None:
    userconfig.save_user_config(
        {"backend": "ollama"},
        allowed=None,
        providers={"groq": SavedProvider("groq", api_key="gsk-saved-key")},
    )
    out = runner.invoke(app, ["config"]).output
    assert "saved, used when a run names it: groq" in out
    assert "never called" not in out


def test_config_says_when_a_target_key_is_exported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ITERATE_TARGET_API_KEY", "gsk-exported-0123456789")
    out = runner.invoke(app, ["config"]).output
    assert "ITERATE_TARGET_API_KEY is exported (gs…89)" in out
    assert "gsk-exported-0123456789" not in out
    monkeypatch.setenv("ITERATE_TARGET_API_KEY", "  ")
    assert "is exported" not in runner.invoke(app, ["config"]).output


def test_setup_asks_again_for_a_harness_server_with_no_address() -> None:
    answers = "vllm\nmy-llama\n\n\nhttp://gpu:8000/v1\nlocal\nn\n\n"
    result = runner.invoke(app, ["setup"], input=answers)
    assert result.exit_code == 0, result.output
    assert "vllm: no base URL saved" in result.output
    cfg = userconfig.load_user_config()
    assert (cfg["backend"], cfg["base_url"]) == ("vllm", "http://gpu:8000/v1")
