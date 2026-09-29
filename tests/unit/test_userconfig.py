"""Tests for the persisted user config (~/.config/iterate/config.toml)."""

from __future__ import annotations

import os
import stat
from typing import TYPE_CHECKING

import pytest

from iterate import userconfig
from iterate.userconfig import SavedProvider

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit


def test_config_path_honors_xdg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert userconfig.config_path() == tmp_path / "iterate" / "config.toml"


def test_cache_dir_honors_xdg_and_ignores_a_relative_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert userconfig.cache_dir() == tmp_path / "xdg" / "iterate"
    monkeypatch.setenv("XDG_CACHE_HOME", "relative/cache")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert userconfig.cache_dir() == tmp_path / "home" / ".cache" / "iterate"


def test_save_then_load_round_trips(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert not userconfig.exists()
    userconfig.save_user_config(
        {"backend": "groq", "model": "llama-3.3-70b", "compute": "e2b", "install": True}
    )
    assert userconfig.exists()
    loaded = userconfig.load_user_config()
    assert loaded == {
        "backend": "groq",
        "model": "llama-3.3-70b",
        "compute": "e2b",
        "install": True,
    }


def test_save_drops_unknown_and_empty(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    userconfig.save_user_config(
        {"backend": "ollama", "nonsense": "x", "model": "", "install": False}
    )
    loaded = userconfig.load_user_config()
    assert "nonsense" not in loaded
    assert "model" not in loaded  # empty string not written
    assert loaded == {"backend": "ollama", "install": False}


def test_load_missing_is_empty(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert userconfig.load_user_config() == {}


# ─── two sets of model settings (v0.7 Day 5) ───────────────────────────────

_PROVIDERS = {
    "openai": SavedProvider("openai", api_key="sk-saved"),
    "groq": SavedProvider("groq", api_key="gsk-saved"),
    "vllm": SavedProvider("vllm", base_url="http://gpu:8000/v1"),
}


def test_a_file_from_before_is_read_as_the_harness() -> None:
    path = userconfig.config_path()
    path.parent.mkdir(parents=True)
    path.write_text(
        'backend = "groq"\nmodel = "llama-3.3-70b"\napi_key = "gsk-old"\n'
        'compute = "local"\ninstall = false\n',
        encoding="utf-8",
    )
    assert userconfig.load_user_config() == {
        "backend": "groq",
        "model": "llama-3.3-70b",
        "api_key": "gsk-old",
        "compute": "local",
        "install": False,
    }
    assert userconfig.load_prompt_settings() == userconfig.PromptSettings()


def test_the_harness_the_list_and_the_providers_round_trip() -> None:
    values = {"backend": "ollama", "model": "gemma4:12b", "compute": "local", "install": False}
    path = userconfig.save_user_config(values, allowed=["openai", "groq"], providers=_PROVIDERS)
    text = path.read_text(encoding="utf-8")
    assert text.index('backend = "ollama"') < text.index("[prompt]") < text.index("[providers.")
    assert userconfig.load_user_config() == values
    saved = userconfig.load_prompt_settings()
    assert saved.allowed == ("openai", "groq")
    assert saved.providers == {"openai": _PROVIDERS["openai"], "groq": _PROVIDERS["groq"]}


def test_a_provider_outside_the_list_keeps_its_key_in_the_file_and_is_not_loaded() -> None:
    path = userconfig.save_user_config(
        {"backend": "ollama"}, allowed=["openai"], providers=_PROVIDERS
    )
    saved = userconfig.load_prompt_settings()
    assert "gsk-saved" in path.read_text(encoding="utf-8")
    assert set(saved.providers) == {"openai"}
    assert saved.held_back == ("groq", "vllm")
    assert "gsk-saved" not in repr(saved)


def test_a_runs_own_list_replaces_the_saved_one() -> None:
    userconfig.save_user_config({"backend": "ollama"}, allowed=["openai"], providers=_PROVIDERS)
    saved = userconfig.load_prompt_settings(["groq"])
    assert saved.allowed == ("groq",)
    assert set(saved.providers) == {"groq"}


def test_with_no_list_saved_no_provider_is_loaded() -> None:
    userconfig.save_user_config({"backend": "ollama"}, allowed=None, providers=_PROVIDERS)
    saved = userconfig.load_prompt_settings()
    assert saved.allowed is None
    assert saved.providers == {}
    assert saved.held_back == ("groq", "openai", "vllm")


def test_saving_the_harness_again_keeps_the_list_and_the_keys() -> None:
    userconfig.save_user_config({"backend": "ollama"}, allowed=["groq"], providers=_PROVIDERS)
    userconfig.save_user_config({"backend": "openai", "api_key": "sk-harness"})
    assert userconfig.load_user_config() == {"backend": "openai", "api_key": "sk-harness"}
    assert userconfig.load_prompt_settings().allowed == ("groq",)
    assert userconfig.load_saved_providers() == _PROVIDERS


def test_a_key_never_shows_in_a_repr() -> None:
    assert "sk-saved" not in repr(_PROVIDERS["openai"])


@pytest.mark.skipif(os.name == "nt", reason="Windows has no owner-only mode bits")
def test_the_file_is_readable_by_its_owner_only() -> None:
    path = userconfig.save_user_config({"backend": "groq", "api_key": "gsk"})
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    path.chmod(0o644)
    userconfig.save_user_config({"backend": "groq", "api_key": "gsk-2"})
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert [p.name for p in path.parent.iterdir()] == ["config.toml"]


@pytest.mark.skipif(os.name == "nt", reason="Windows has no owner-only mode bits")
def test_a_file_anyone_could_read_is_narrowed_when_it_is_read() -> None:
    path = userconfig.config_path()
    path.parent.mkdir(parents=True)
    path.write_text('backend = "groq"\napi_key = "gsk-old"\n', encoding="utf-8")
    path.chmod(0o644)
    assert userconfig.load_user_config()["api_key"] == "gsk-old"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


@pytest.mark.skipif(os.name == "nt", reason="links need a privilege on Windows")
def test_a_config_that_is_a_link_stays_a_link(tmp_path: Path) -> None:
    real = tmp_path / "dotfiles" / "iterate.toml"
    real.parent.mkdir()
    real.write_text('backend = "ollama"\n', encoding="utf-8")
    path = userconfig.config_path()
    path.parent.mkdir(parents=True)
    path.symlink_to(real)
    userconfig.save_user_config({"backend": "groq", "api_key": "gsk"})
    assert path.is_symlink()
    assert 'backend = "groq"' in real.read_text(encoding="utf-8")


@pytest.mark.parametrize("value", ['a"b', "a\\b", "line\nbreak", "tab\there", "del\x7f", "é"])
def test_a_value_with_a_quote_or_a_line_break_survives(value: str) -> None:
    userconfig.save_user_config(
        {"backend": "groq", "api_key": value},
        allowed=["groq"],
        providers={"groq": SavedProvider("groq", api_key=value)},
    )
    assert userconfig.load_user_config()["api_key"] == value
    assert userconfig.load_prompt_settings().providers["groq"].api_key == value


@pytest.mark.parametrize("name", ["open ai", "a.b", "x]\n[harness", ""])
def test_a_name_that_is_not_a_bare_key_is_refused(name: str) -> None:
    with pytest.raises(ValueError, match="provider name"):
        userconfig.save_user_config(
            {"backend": "ollama"}, allowed=None, providers={name: SavedProvider(name)}
        )


def test_a_version_before_v07_still_reads_the_harness_from_the_new_file() -> None:
    """Its loader keeps the plain keys at the top and drops every table."""
    import tomllib

    path = userconfig.save_user_config(
        {"backend": "groq", "model": "llama", "api_key": "gsk", "compute": "local"},
        allowed=["openai"],
        providers=_PROVIDERS,
    )
    old_keys = {"backend", "model", "api_key", "e2b_api_key", "compute", "install"}
    seen = {k: v for k, v in tomllib.loads(path.read_text("utf-8")).items() if k in old_keys}
    assert seen == {"backend": "groq", "model": "llama", "api_key": "gsk", "compute": "local"}


def _written(text: str) -> None:
    path = userconfig.config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_a_harness_table_written_by_hand_wins_key_by_key() -> None:
    _written(
        'backend = "groq"\napi_key = "gsk-top"\ncompute = "local"\n\n'
        '[harness]\nmodel = "llama"\n\n[harness.extra]\nx = 1\n'
    )
    assert userconfig.load_user_config() == {
        "backend": "groq",
        "api_key": "gsk-top",
        "model": "llama",
        "compute": "local",
    }
    _written('backend = "ollama"\n\n[harness]\nbackend = "groq"\napi_key = "k"\n')
    assert userconfig.load_user_config() == {"backend": "groq", "api_key": "k"}


def test_a_harness_table_for_another_backend_leaves_the_key_at_the_top_behind() -> None:
    _written(
        'backend = "openai"\napi_key = "sk-top"\nbase_url = "https://gw.test/v1"\n'
        'compute = "local"\n\n[harness]\nbackend = "groq"\n'
    )
    assert userconfig.load_user_config() == {"backend": "groq", "compute": "local"}


def test_a_list_written_as_one_name_is_a_list_of_one() -> None:
    _written('[prompt]\nallowed = "OpenAI"\n\n[providers.openai]\napi_key = "sk"\n')
    saved = userconfig.load_prompt_settings()
    assert saved.allowed == ("openai",)
    assert saved.providers["openai"].api_key == "sk"


@pytest.mark.parametrize("listed", ["5", "true", '["openai", 5]', '{ name = "openai" }'])
def test_a_list_that_cannot_be_read_is_refused_not_taken_for_no_list(listed: str) -> None:
    _written(f"[prompt]\nallowed = {listed}\n")
    with pytest.raises(userconfig.ConfigError, match="has to be a list of names"):
        userconfig.load_prompt_settings()


def test_a_provider_table_is_found_whatever_the_case_of_its_name() -> None:
    _written('[prompt]\nallowed = ["groq"]\n\n[providers.GROQ]\napi_key = "gsk"\n')
    saved = userconfig.load_prompt_settings()
    assert saved.providers["groq"].api_key == "gsk"
    assert saved.held_back == ()


def test_a_provider_that_is_not_a_table_is_refused_with_how_to_write_it() -> None:
    _written('[providers]\ngroq = "gsk-as-a-string"\n')
    with pytest.raises(userconfig.ConfigError, match=r"\[providers.groq\]") as refused:
        userconfig.load_saved_providers()
    assert "gsk-as-a-string" not in str(refused.value)


def test_a_file_that_is_not_toml_is_refused_by_its_path() -> None:
    _written('backend = "groq\n')
    with pytest.raises(userconfig.ConfigError, match=r"config.toml is not valid TOML"):
        userconfig.load_user_config()


def test_a_draft_left_by_a_save_that_was_killed_is_cleared_by_the_next() -> None:
    path = userconfig.save_user_config({"backend": "groq", "api_key": "gsk"})
    stale = path.with_name(".config.toml.4242.tmp")
    stale.write_text('api_key = "a-key-since-removed"\n', encoding="utf-8")
    fresh = path.with_name(".config.toml.4243.tmp")
    fresh.write_text("another save, still running\n", encoding="utf-8")
    old = stale.stat().st_mtime - 600
    os.utime(stale, (old, old))
    userconfig.save_user_config({"backend": "groq", "api_key": "gsk"})
    assert sorted(p.name for p in path.parent.iterdir()) == [".config.toml.4243.tmp", "config.toml"]


@pytest.mark.skipif(os.name == "nt", reason="Windows has no owner-only mode bits")
def test_the_file_is_swapped_in_whole_and_was_never_wider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    swaps: list[tuple[int, str]] = []
    real = os.replace

    def spy(source: Path, target: Path) -> None:
        swaps.append((stat.S_IMODE(source.stat().st_mode), source.read_text(encoding="utf-8")))
        real(source, target)

    monkeypatch.setattr(os, "replace", spy)
    path = userconfig.save_user_config({"backend": "groq", "api_key": "gsk"})
    ((mode, text),) = swaps
    assert mode == 0o600
    assert text == path.read_text(encoding="utf-8")
    assert 'api_key = "gsk"' in text


def test_only_a_draft_of_a_save_is_cleared() -> None:
    path = userconfig.save_user_config({"backend": "ollama"})
    theirs = path.with_name(".config.toml.notes.tmp")
    theirs.write_text("not ours", encoding="utf-8")
    old = theirs.stat().st_mtime - 600
    os.utime(theirs, (old, old))
    userconfig.save_user_config({"backend": "ollama"})
    assert theirs.exists()


def test_a_file_that_cannot_be_opened_as_text_is_refused_by_its_path() -> None:
    path = userconfig.config_path()
    path.parent.mkdir(parents=True)
    path.write_bytes(b'backend = "\xff\xfe"\n')
    with pytest.raises(userconfig.ConfigError, match=r"config\.toml is not valid TOML"):
        userconfig.load_user_config()


@pytest.mark.skipif(os.name == "nt", reason="Windows has no owner-only mode bits")
def test_narrowing_takes_from_others_and_gives_the_owner_nothing() -> None:
    path = userconfig.config_path()
    path.parent.mkdir(parents=True)
    path.write_text('backend = "groq"\n', encoding="utf-8")
    path.chmod(0o444)
    userconfig.load_user_config()
    assert stat.S_IMODE(path.stat().st_mode) == 0o400
    path.chmod(0o600)


@pytest.mark.skipif(os.name == "nt", reason="Windows has no owner-only mode bits")
def test_a_config_path_that_is_a_folder_is_refused_and_left_as_it_is() -> None:
    path = userconfig.config_path()
    path.mkdir(parents=True)
    before = stat.S_IMODE(path.stat().st_mode)
    with pytest.raises(userconfig.ConfigError, match="cannot be opened"):
        userconfig.load_user_config()
    assert stat.S_IMODE(path.stat().st_mode) == before


@pytest.mark.skipif(os.name == "nt", reason="Windows has no owner-only mode bits")
def test_a_copy_kept_beside_the_file_is_the_owners_alone() -> None:
    path = userconfig.save_user_config({"backend": "groq", "api_key": "gsk"})
    copy = userconfig.keep_beside(path, "broken")
    assert copy is not None
    assert copy.name == "config.toml.broken"
    assert stat.S_IMODE(copy.stat().st_mode) == 0o600
    assert copy.read_bytes() == path.read_bytes()
    assert userconfig.keep_beside(path.with_name("nothing.toml"), "broken") is None
