"""Tests for the LLM backend factory."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from iterate.llm import factory
from iterate.llm.factory import UnknownBackendError, build_client, resolve_base_url
from iterate.llm.ollama_client import OllamaClient
from iterate.llm.openai_compatible import OpenAICompatibleClient
from iterate.userconfig import SavedProvider

pytestmark = pytest.mark.unit


def test_default_backend_is_ollama() -> None:
    assert isinstance(build_client(), OllamaClient)


def test_explicit_ollama_backend() -> None:
    assert isinstance(build_client("ollama"), OllamaClient)


def test_openai_compatible_backend() -> None:
    client = build_client("openai-compatible", api_key="sk-fake")
    assert isinstance(client, OpenAICompatibleClient)


def test_cloud_aliases_route_to_openai_compatible() -> None:
    for alias in ("groq", "together", "deepseek", "openai", "vllm"):
        assert isinstance(build_client(alias, api_key="sk-fake"), OpenAICompatibleClient)


def test_cloud_alias_supplies_its_base_url() -> None:
    # A known cloud alias resolves its own URL (so the user needs only backend+key+model).
    assert resolve_base_url("groq", None) == "https://api.groq.com/openai/v1"
    assert resolve_base_url("openai", None) == "https://api.openai.com/v1"
    assert resolve_base_url("deepseek", None) == "https://api.deepseek.com/v1"


def test_explicit_base_url_overrides_alias() -> None:
    assert resolve_base_url("groq", "http://localhost:8000/v1") == "http://localhost:8000/v1"


def test_generic_aliases_have_no_default_url() -> None:
    # openai-compatible / vllm have no canonical URL; the user must supply one.
    assert resolve_base_url("openai-compatible", None) is None
    assert resolve_base_url("vllm", None) is None


def test_unknown_backend_raises() -> None:
    with pytest.raises(UnknownBackendError, match="unknown backend"):
        build_client("not-a-real-backend")


def test_model_override_threads_through_to_ollama() -> None:
    client = build_client("ollama", model="qwen3:8b")
    assert client.model == "qwen3:8b"


def test_think_defaults_off_and_threads_through_to_ollama() -> None:
    assert build_client("ollama")._think is False
    assert build_client("ollama", think=True)._think is True


def test_think_warns_and_is_ignored_for_openai_compatible(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("WARNING", logger="iterate.llm.factory"):
        client = build_client("groq", api_key="sk-fake", think=True)
    assert isinstance(client, OpenAICompatibleClient)
    assert "only applies to the ollama backend" in caplog.text


# ─── the providers a prompt run may call (v0.7 Day 5) ──────────────────────


def _settings(**keys: str | None) -> Any:
    """Settings with the harness's generic slot full, so a test sees it leak."""
    base = {
        "iterate_backend_api_key": "harness-generic",
        "openai_api_key": None,
        "groq_api_key": None,
        "together_api_key": None,
        "deepseek_api_key": None,
        "ollama_host": "http://localhost:11434",
    }
    return SimpleNamespace(**{**base, **keys})


def _provider(backend: str, **kwargs: Any) -> factory.Provider:
    kwargs.setdefault("settings", _settings())
    kwargs.setdefault("environ", {})
    return factory.prompt_provider(backend, **kwargs)


@pytest.mark.parametrize("name", ["openai", "groq", "together", "deepseek"])
def test_a_provider_with_no_key_of_its_own_never_takes_the_generic_slot(name: str) -> None:
    found = _provider(name)
    assert found.api_key is None
    assert f"set {name.upper()}_API_KEY" in str(factory.not_callable(found))


def test_a_provider_takes_its_own_key_from_the_environment() -> None:
    found = _provider("groq", settings=_settings(groq_api_key="gsk-env"))
    assert (found.api_key, found.key_from) == ("gsk-env", "GROQ_API_KEY")
    assert found.base_url == "https://api.groq.com/openai/v1"
    assert factory.not_callable(found) is None


def test_another_providers_key_is_never_taken() -> None:
    found = _provider("groq", settings=_settings(openai_api_key="sk-openai"))
    assert found.api_key is None


def test_the_saved_key_beats_the_environment_and_the_target_key_beats_both() -> None:
    saved = {"groq": SavedProvider("groq", api_key="gsk-saved")}
    settings = _settings(groq_api_key="gsk-env")
    found = _provider("groq", saved=saved, settings=settings)
    assert (found.api_key, found.key_from) == ("gsk-saved", "the saved config")
    found = _provider(
        "groq", saved=saved, settings=settings, environ={"ITERATE_TARGET_API_KEY": "gsk-run"}
    )
    assert (found.api_key, found.key_from) == ("gsk-run", "ITERATE_TARGET_API_KEY")


def test_an_exported_key_of_spaces_is_no_key() -> None:
    found = _provider(
        "groq",
        settings=_settings(groq_api_key="gsk-env"),
        environ={"ITERATE_TARGET_API_KEY": "   "},
    )
    assert (found.api_key, found.key_from) == ("gsk-env", "GROQ_API_KEY")


@pytest.mark.parametrize("name", ["openai-compatible", "vllm"])
def test_a_server_you_run_never_takes_openais_key(name: str) -> None:
    found = _provider(
        name, base_url="http://gpu:8000/v1", settings=_settings(openai_api_key="sk-openai")
    )
    assert found.name == name
    assert found.api_key is None
    assert factory.not_callable(found) is None


def test_a_server_you_run_needs_its_address() -> None:
    assert "has no address of its own" in str(factory.not_callable(_provider("vllm")))
    saved = {"vllm": SavedProvider("vllm", api_key="k", base_url="http://gpu:8000/v1")}
    found = _provider("vllm", saved=saved)
    assert (found.base_url, found.api_key) == ("http://gpu:8000/v1", "k")


def test_openai_compatible_aimed_at_openai_is_openai() -> None:
    found = _provider(
        "openai-compatible",
        base_url="https://API.openai.com/v1",
        settings=_settings(openai_api_key="sk-openai"),
    )
    assert (found.name, found.api_key) == ("openai", "sk-openai")


def test_the_address_given_beats_the_saved_one_and_the_saved_one_beats_the_public_one() -> None:
    saved = {"groq": SavedProvider("groq", api_key="k", base_url="https://gateway.corp/groq")}
    assert _provider("groq", saved=saved).base_url == "https://gateway.corp/groq"
    assert _provider("groq", saved=saved, base_url="http://x/v1").base_url == "http://x/v1"


def test_ollama_takes_no_key_and_its_host_is_settled() -> None:
    found = _provider("ollama", environ={"ITERATE_TARGET_API_KEY": "stray"})
    assert found.api_key is None
    assert found.base_url == "http://localhost:11434"
    assert factory.not_callable(found) is None
    assert _provider("ollama", base_url="http://box:11434").base_url == "http://box:11434"


def test_a_name_iterate_does_not_know_is_refused_with_the_names_it_does() -> None:
    with pytest.raises(factory.ProviderError, match=r"mistral: not a provider.*Choose from"):
        _provider("mistral")
    assert factory.unknown("grok", flag="--target-backend").startswith("--target-backend grok:")
    with pytest.raises(factory.ProviderError, match="grok, openia: not a provider"):
        factory.allowed_names(["groq", "grok", "openia"])
    assert factory.allowed_names([" Groq ", "openai", "groq", ""]) == ("groq", "openai")


def test_a_refused_key_is_a_reason_and_a_provider_out_of_reach_is_not() -> None:
    from openai import APIConnectionError, AuthenticationError

    found = _provider("groq", settings=_settings(groq_api_key="gsk-wrong"))
    response = SimpleNamespace(request=None, status_code=401, headers={})

    def refuses(provider: factory.Provider, timeout: float, model: str | None) -> None:
        raise AuthenticationError("bad key", response=response, body=None)  # type: ignore[arg-type]

    def unreachable(provider: factory.Provider, timeout: float, model: str | None) -> None:
        raise APIConnectionError(request=None)  # type: ignore[arg-type]

    why = factory.refused_key(found, lister=refuses)
    assert why == (
        "api.groq.com refused the groq key from GROQ_API_KEY. Check the key, or export "
        "ITERATE_TARGET_API_KEY for this run"
    )
    exported = _provider(
        "groq", environ={"ITERATE_TARGET_API_KEY": "gsk-wrong"}, settings=_settings()
    )
    assert factory.refused_key(exported, lister=refuses) == (
        "api.groq.com refused the groq key from ITERATE_TARGET_API_KEY. Check the key, or "
        "unset ITERATE_TARGET_API_KEY to use the key saved for groq"
    )
    assert factory.refused_key(found, lister=unreachable) is None
    assert factory.refused_key(found, lister=lambda provider, timeout, model: None) is None


def test_a_company_that_does_not_serve_the_model_named_stops_the_run() -> None:
    """A typo in --target-model is found by the call that checks the key, which already
    holds the names the key is served, and not by every record of the baseline."""
    found = _provider("groq", settings=_settings(groq_api_key="gsk-env"))
    served = ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b"]

    def lister(provider: factory.Provider, timeout: float, model: str | None) -> list[str]:
        return served

    assert factory.refused_key(found, model="openai/gpt-oss-20b", lister=lister) is None
    assert factory.refused_key(found, model="llama-3.3-70b-versatile", lister=lister) == (
        "groq does not serve llama-3.3-70b-versatile to this key. It serves: "
        "openai/gpt-oss-120b, openai/gpt-oss-20b, qwen/qwen3.8-27b"
    )
    assert factory.refused_key(found, lister=lister) is None


def test_a_provider_that_is_sent_no_key_is_not_asked() -> None:
    def never(provider: factory.Provider, timeout: float, model: str | None) -> None:
        raise AssertionError("asked")

    assert factory.refused_key(_provider("ollama"), lister=never) is None
    assert factory.refused_key(_provider("groq"), lister=never) is None
    assert factory.refused_key(_provider("vllm", base_url="http://gpu/v1"), lister=never) is None


def test_a_key_sent_to_a_server_you_run_is_checked_too() -> None:
    class RefusedError(Exception):
        status_code = 401

    def refuses(provider: factory.Provider, timeout: float, model: str | None) -> None:
        raise RefusedError

    saved = {"vllm": SavedProvider("vllm", api_key="key-a", base_url="http://gpu.test/v1")}
    why = str(factory.refused_key(_provider("vllm", saved=saved), lister=refuses))
    assert why.startswith("gpu.test refused the vllm key from the saved config")


def test_the_target_client_is_always_given_its_address_and_its_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[dict[str, Any]] = []
    monkeypatch.setattr(
        factory, "build_client", lambda name, **kw: seen.append({"name": name, **kw})
    )
    factory.build_target_client("groq", model="m", base_url=None, api_key="gsk")
    factory.build_target_client("vllm", model="m", base_url="http://gpu:8000/v1", api_key=None)
    factory.build_target_client("ollama", model="m", base_url="http://box:11434", api_key=None)
    assert seen == [
        {
            "name": "groq",
            "model": "m",
            "base_url": "https://api.groq.com/openai/v1",
            "api_key": "gsk",
        },
        {"name": "vllm", "model": "m", "base_url": "http://gpu:8000/v1", "api_key": "not-needed"},
        {"name": "ollama", "model": "m", "base_url": "http://box:11434", "api_key": None},
    ]
    with pytest.raises(factory.ProviderError, match="groq has no key here: set GROQ_API_KEY"):
        factory.build_target_client("groq", model="m", base_url=None, api_key=None)


def test_a_notebook_from_before_v07_finds_its_server_where_that_run_did(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Its meta.json holds no address for a server the user runs: that run found it in
    the environment, and the key in the generic slot is not sent."""
    from iterate.config import get_settings

    seen: list[dict[str, Any]] = []
    monkeypatch.setattr(factory, "build_client", lambda name, **kw: seen.append(kw))
    monkeypatch.setenv("ITERATE_BACKEND_URL", "http://gpu-box:8000/v1")
    monkeypatch.setenv("ITERATE_BACKEND_API_KEY", "the-harness-key")
    get_settings.cache_clear()
    try:
        factory.build_target_client("vllm", model="m", base_url=None, api_key=None)
    finally:
        get_settings.cache_clear()
    assert seen == [{"model": "m", "base_url": "http://gpu-box:8000/v1", "api_key": "not-needed"}]


@pytest.mark.parametrize(
    ("backend", "base_url", "scope"),
    [
        ("groq", None, "groq|api.groq.com/openai/v1"),
        ("openai-compatible", "https://API.openai.com/v1/", "openai|api.openai.com/v1"),
        ("vllm", "http://gpu:8000/v1", "vllm|gpu:8000/v1"),
        ("ollama", "http://box:11434/", "ollama|box:11434"),
        ("ollama", None, "ollama|"),
    ],
)
def test_the_cache_scope_names_the_provider_and_where_it_was_called(
    backend: str, base_url: str | None, scope: str
) -> None:
    assert factory.cache_scope(backend, base_url) == scope


def test_a_key_never_shows_in_a_provider_repr() -> None:
    assert "gsk-env" not in repr(_provider("groq", settings=_settings(groq_api_key="gsk-env")))


@pytest.mark.parametrize(
    ("address", "said"),
    [
        ("http://me:secret@gpu:8000/v1", "carries a user name, a password or a token."),
        ("https://gpu.test/v1?api_key=secret", "carries a user name, a password or a token."),
        ("https://gpu.test/v1#secret", "carries a user name, a password or a token."),
        ("https://gpu.test:abc/v1", "cannot be read: check its port"),
        ("ftp://gpu.test/v1", "has to start with http:// or https://"),
        ("gpu.test/v1", "has to start with http:// or https://"),
        ("https://gpu test/v1", "holds a space"),
    ],
)
def test_an_address_that_cannot_be_used_is_refused_without_showing_it(
    address: str, said: str
) -> None:
    found = _provider("vllm", base_url=address)
    why = str(factory.not_callable(found))
    assert said in why
    assert why.startswith("the base URL saved for vllm")
    assert "secret" not in why
    assert str(factory.not_callable(found, given_as="--target-base-url")).startswith(
        "--target-base-url "
    )


def test_an_address_is_shown_without_what_rides_on_it() -> None:
    assert factory.shown("https://me:pw@gw.test:8443/v1/?token=t#f") == "https://gw.test:8443/v1"
    assert factory.shown("https://api.groq.com/openai/v1") == "https://api.groq.com/openai/v1"
    assert factory.shown("https://gw.test:abc/v1") == "an address that cannot be read"


@pytest.mark.parametrize(
    "address",
    [
        *[
            "https://api.openai.com:443/v1",
            "https://API.OPENAI.COM/v1",
            "https://api.openai.com./v1",
        ],
        "http://api.openai.com/v1",
    ],
)
def test_a_companys_host_is_that_company_however_it_is_written(address: str) -> None:
    assert factory.alias_for_base_url(address) == "openai"
    assert factory.provider_name("groq", address) == "openai"


@pytest.mark.parametrize(
    "address",
    [
        *["https://api.openai.com.evil.test/v1", "https://evil.test/api.openai.com"],
        *["https://api.openai.com@evil.test/v1", "https://evil.test/v1?x=@api.openai.com"],
        "https://gw.test:abc/v1",
    ],
)
def test_a_host_that_only_looks_like_a_company_is_not_one(address: str) -> None:
    assert factory.alias_for_base_url(address) is None


def test_a_company_reached_in_the_clear_is_not_sent_its_key() -> None:
    found = _provider(
        "openai", base_url="http://api.openai.com/v1", settings=_settings(openai_api_key="sk")
    )
    why = str(factory.not_callable(found))
    assert "takes its key over https only" in why
    assert why.endswith("Pass the https:// address")
    own_server = _provider(
        "vllm", base_url="http://gpu:8000/v1", environ={"ITERATE_TARGET_API_KEY": "k"}
    )
    assert factory.not_callable(own_server) is None


def test_a_key_saved_for_a_server_you_run_goes_with_its_address() -> None:
    saved = {"vllm": SavedProvider("vllm", api_key="key-a", base_url="http://a.test:8000/v1")}
    assert _provider("vllm", saved=saved).api_key == "key-a"
    assert _provider("vllm", saved=saved, base_url="http://a.test:8000/other").api_key == "key-a"
    assert _provider("vllm", saved=saved, base_url="http://a.test:9000/v1").api_key is None
    assert _provider("vllm", saved=saved, base_url="http://b.test:8000/v1").api_key is None
    company = {"groq": SavedProvider("groq", api_key="gsk")}
    assert _provider("groq", saved=company, base_url="https://gateway.test/v1").api_key == "gsk"


@pytest.mark.parametrize(
    "address",
    [
        *["https://api.openai.com/v1", "https://api.groq.com/openai/v1"],
        "https://eu.api.openai.com/v1",
    ],
)
def test_aimed_at_a_company_the_key_of_a_server_you_run_stays_with_that_server(
    address: str,
) -> None:
    """The address makes the provider the company. The key was saved under the server's
    name, beside the server's address, and is still the server's."""
    saved = {"vllm": SavedProvider("vllm", api_key="key-a", base_url="http://a.test:8000/v1")}
    found = _provider(
        "vllm", saved=saved, base_url=address, settings=_settings(openai_api_key="sk-env")
    )
    assert found.name in ("openai", "groq")
    assert found.api_key in (None, "sk-env")
    assert found.api_key != "key-a"


@pytest.mark.parametrize(
    ("one", "other", "same"),
    [
        ("https://gpu.test/v1", "https://gpu.test/other", True),
        ("https://gpu.test/v1", "https://GPU.test.:443/v1", True),
        ("http://gpu.test/v1", "http://gpu.test:80/v1", True),
        ("https://gpu.test/v1", "http://gpu.test/v1", False),
        ("http://gpu.test:8000/v1", "http://gpu.test:9000/v1", False),
        ("http://a.test/v1", "http://b.test/v1", False),
        ("http://a.test/v1", None, False),
        (None, None, False),
        ("https://gpu.test:abc/v1", "https://gpu.test:abc/v1", False),
    ],
)
def test_one_place_is_one_host_and_one_port(one: str | None, other: str | None, same: bool) -> None:
    assert factory.same_place(one, other) is same


def test_a_key_saved_under_the_name_given_is_found_when_the_address_is_a_companys() -> None:
    saved = {
        "openai-compatible": SavedProvider(
            "openai-compatible", api_key="sk-saved", base_url="https://api.openai.com/v1"
        )
    }
    found = _provider("openai-compatible", saved=saved)
    assert (found.name, found.api_key) == ("openai", "sk-saved")


def test_the_words_for_a_missing_key_fit_where_they_are_read() -> None:
    found = _provider("groq")
    assert "save one with `iterate setup`" in str(factory.not_callable(found))
    in_a_cell = str(factory.not_callable(found, in_a_cell=True))
    assert in_a_cell == "groq has no key here: set GROQ_API_KEY or ITERATE_TARGET_API_KEY"
    assert factory.not_ready(found) == "no key, set GROQ_API_KEY"
    assert factory.not_ready(_provider("vllm")) == "no base URL saved"
    assert factory.not_ready(_provider("ollama")) is None


def test_every_table_is_read_off_the_one_row_a_provider_has(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A provider added later is one row: its key, its address and the harness question
    are read off it, and a wire with no lister is never checked with another wire's."""
    row = factory._Known("mistral", "https://api.mistral.ai/v1", "anthropic_api_key")
    monkeypatch.setitem(factory._PROVIDERS, "mistral", row)
    assert "mistral" in factory.known_providers()
    assert "mistral" not in factory.harness_backends()
    assert factory.own_key_env("mistral") == "ANTHROPIC_API_KEY"
    found = _provider("mistral", settings=_settings(anthropic_api_key="key"))
    assert (found.api_key, found.needs_key) == ("key", True)

    # Recorded, not raised: the check swallows what a lister raises.
    asked: list[Any] = []
    monkeypatch.setattr(factory, "_list_models", lambda *args: asked.append(args))
    monkeypatch.setattr(factory, "_claude_serves", lambda *args: asked.append(args))
    assert factory.refused_key(found) is None
    assert asked == []


def test_the_claude_library_is_asked_for_only_where_its_client_is_built(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "anthropic", None)
    at_claude = factory.Provider(
        name="anthropic", base_url="https://api.anthropic.com/v1", api_key="k"
    )
    assert "pip install 'iterate-ai[anthropic]'" in str(factory.not_ready(at_claude))
    assert factory.not_ready(at_claude, wire="openai") is None
    # A notebook from a run that called Claude through the OpenAI-compatible layer.
    client = factory.build_target_client(
        "openai-compatible",
        model="claude-haiku-4-5",
        base_url="https://api.anthropic.com/v1",
        api_key="k",
    )
    assert type(client).__name__ == "OpenAICompatibleClient"


def test_ollama_is_asked_for_its_models_before_anything_is_written() -> None:
    """Free and local: a server that is down, one that is not Ollama, or a model not
    pulled is found before the run, not on every record of its baseline."""
    found = factory.Provider(name="ollama", base_url="http://localhost:11434")

    def has(*names: str) -> Any:
        return lambda host, timeout: list(names)

    def down(host: str, timeout: float) -> list[str]:
        raise ConnectionError("refused")

    assert factory.ollama_refusal(found, "gemma4:12b", lister=has("gemma4:12b")) is None
    assert factory.ollama_refusal(found, "qwen3", lister=has("qwen3:latest")) is None
    assert "has no model gpt-4o-mini: `ollama pull gpt-4o-mini`" in str(
        factory.ollama_refusal(found, "gpt-4o-mini", lister=has("gemma4:12b"))
    )
    assert "no Ollama server answers at http://localhost:11434 (ConnectionError)" in str(
        factory.ollama_refusal(found, "gemma4:12b", lister=down)
    )


def test_an_ollama_address_ending_in_v1_is_another_servers() -> None:
    found = factory.Provider(name="ollama", base_url="http://gpu-box:8000/v1")
    assert "ends in /v1, the OpenAI-compatible door of a server" in str(
        factory.not_callable(found, given_as="--target-base-url x")
    )


def test_claude_is_a_provider_for_the_model_under_test_and_not_a_harness() -> None:
    assert "anthropic" in factory.known_providers()
    assert "anthropic" not in factory.harness_backends()
    assert factory.own_key_env("anthropic") == "ANTHROPIC_API_KEY"
    found = _provider("anthropic", settings=_settings(anthropic_api_key="sk-ant-key"))
    assert (found.name, found.base_url, found.api_key) == (
        "anthropic",
        "https://api.anthropic.com",
        "sk-ant-key",
    )
    assert factory.cache_scope("anthropic", None) == "anthropic|api.anthropic.com"
    with pytest.raises(factory.UnknownBackendError, match=r"comes in v1\.0"):
        factory.build_client("anthropic", model="claude-haiku-4-5", api_key="k")


def test_the_harnesss_key_fields_are_read_off_the_same_rows() -> None:
    assert factory._KEY_FIELDS["groq"] == ("groq_api_key", "iterate_backend_api_key")
    assert factory._KEY_FIELDS["openai-compatible"] == ("iterate_backend_api_key",)
    assert "ollama" not in factory._KEY_FIELDS
    assert set(factory.known_providers()) - set(factory.harness_backends()) == {"anthropic"}


def test_a_refusal_is_read_off_the_status_whatever_the_library() -> None:
    class RefusedError(Exception):
        status_code = 401

    class MissingError(Exception):
        status_code = 403

    found = _provider("groq", settings=_settings(groq_api_key="gsk-wrong"))

    def raises(error: Exception) -> Any:
        def lister(provider: factory.Provider, timeout: float, model: str | None) -> None:
            raise error

        return lister

    assert "refused the groq key" in str(factory.refused_key(found, lister=raises(RefusedError())))
    # A key that may chat and may not list models, a list that cannot be read, a server
    # out of reach: none is a no to the key.
    for not_a_refusal in (MissingError(), ValueError("bad url"), AttributeError("no .data")):
        assert factory.refused_key(found, lister=raises(not_a_refusal)) is None


_the_real_key_check = factory._list_models


@pytest.fixture
def sent_to(monkeypatch: pytest.MonkeyPatch) -> Any:
    """What a client built for an address sends there: the headers of one request, by
    name in lower case. The transport is handed to the client as it is built, since a
    copy of a client reads the environment again."""
    import httpx
    import openai

    from iterate.llm import openai_compatible

    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.clear()
        seen.update({k.lower(): v for k, v in request.headers.items()})
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"object": "list", "data": []})

    real = openai.OpenAI

    def built(**kwargs: Any) -> Any:
        return real(http_client=httpx.Client(transport=httpx.MockTransport(handler)), **kwargs)

    monkeypatch.setattr(openai, "OpenAI", built)
    monkeypatch.setattr(openai_compatible, "OpenAI", built)

    def send(base_url: str) -> dict[str, str]:
        OpenAICompatibleClient(base_url=base_url, model="m", api_key="k")._client.models.list()
        return dict(seen)

    send.seen = seen  # type: ignore[attr-defined]
    return send


_OPENAIS = ("openai-organization", "openai-project", "x-gateway-auth", "x-team")


def _openais_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_ORG_ID", "org-mine")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "proj-mine")
    monkeypatch.setenv("OPENAI_CUSTOM_HEADERS", "X-Gateway-Auth: secret-token\nX-Team: a")


def test_the_key_check_sends_another_company_nothing_of_openais(
    monkeypatch: pytest.MonkeyPatch, sent_to: Any
) -> None:
    _openais_environment(monkeypatch)
    found = _provider("groq", settings=_settings(groq_api_key="gsk-env"))

    _the_real_key_check(found, 5.0, None)

    seen = sent_to.seen
    assert seen["url"] == "https://api.groq.com/openai/v1/models"
    assert seen["authorization"] == "Bearer gsk-env"
    assert not set(_OPENAIS) & set(seen)


@pytest.mark.parametrize(
    ("body", "said"),
    [
        (
            {"object": "list", "data": [{"id": "openai/gpt-oss-20b", "object": "model"}]},
            "groq does not serve llama-3.3-70b-versatile to this key. It serves: "
            "openai/gpt-oss-20b",
        ),
        # Together's shape: a bare list, which the OpenAI library cannot read.
        ([{"id": "openai/gpt-oss-20b"}], None),
    ],
)
def test_the_names_a_company_serves_are_read_off_its_own_list(
    monkeypatch: pytest.MonkeyPatch, body: Any, said: str | None
) -> None:
    import httpx
    import openai

    real = openai.OpenAI

    def built(**kwargs: Any) -> Any:
        answer = httpx.MockTransport(lambda request: httpx.Response(200, json=body))
        return real(http_client=httpx.Client(transport=answer), **kwargs)

    monkeypatch.setattr(openai, "OpenAI", built)
    found = _provider("groq", settings=_settings(groq_api_key="gsk-env"))
    check = _the_real_key_check

    assert factory.refused_key(found, model="llama-3.3-70b-versatile", lister=check) == said
    assert factory.refused_key(found, model="openai/gpt-oss-20b", lister=check) is None


def test_what_the_environment_holds_for_openai_is_sent_to_openai_only(
    monkeypatch: pytest.MonkeyPatch, sent_to: Any
) -> None:
    """The OpenAI library reads these for every client it builds, wherever it points."""
    _openais_environment(monkeypatch)
    # Every company in the table but OpenAI, so one added later is covered.
    others = {k: v for k, v in factory._ALIAS_BASE_URLS.items() if k != "openai"}
    assert set(others) >= {"groq", "together", "deepseek"}
    for address in others.values():
        sent = sent_to(address)
        assert sent["authorization"] == "Bearer k"
        assert not set(_OPENAIS) & set(sent)
    for openai_itself in ("https://api.openai.com/v1", "https://API.openai.com:443/v1"):
        sent = sent_to(openai_itself)
        assert [sent[name] for name in _OPENAIS] == ["org-mine", "proj-mine", "secret-token", "a"]


def test_a_gateway_the_user_named_keeps_the_headers_it_was_given(
    monkeypatch: pytest.MonkeyPatch, sent_to: Any
) -> None:
    """An address the user typed or saved is theirs: the headers they set for it go
    with it. OpenAI's own two ids still go to OpenAI alone."""
    _openais_environment(monkeypatch)
    for theirs in ("https://my-gateway.test/groq/v1", "http://localhost:11434/v1"):
        sent = sent_to(theirs)
        assert (sent["x-gateway-auth"], sent["x-team"]) == ("secret-token", "a")
        assert not {"openai-organization", "openai-project"} & set(sent)


@pytest.mark.parametrize("spelling", ["Authorization", "authorization", "AUTHORIZATION"])
def test_a_key_in_the_custom_headers_never_replaces_the_key_given(
    monkeypatch: pytest.MonkeyPatch, sent_to: Any, spelling: str
) -> None:
    monkeypatch.setenv("OPENAI_CUSTOM_HEADERS", f"{spelling}: Bearer from-the-environment")
    assert sent_to("https://api.groq.com/openai/v1")["authorization"] == "Bearer k"


@pytest.mark.parametrize("spelling", ["OpenAI-Organization", "openai-organization"])
def test_a_line_the_user_wrote_for_a_gateway_goes_through(
    monkeypatch: pytest.MonkeyPatch, sent_to: Any, spelling: str
) -> None:
    """The id from OPENAI_ORG_ID stays home. The same header written by hand for the
    address the user typed is theirs to send."""
    monkeypatch.setenv("OPENAI_ORG_ID", "org-from-the-variable")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "proj-from-the-variable")
    monkeypatch.setenv("OPENAI_CUSTOM_HEADERS", f"{spelling}: org-for-the-gateway")
    sent = sent_to("https://my-gateway.test/v1")
    assert sent["openai-organization"] == "org-for-the-gateway"
    assert "openai-project" not in sent


def test_a_host_under_a_companys_is_that_company(
    monkeypatch: pytest.MonkeyPatch, sent_to: Any
) -> None:
    monkeypatch.setenv("OPENAI_ORG_ID", "org-mine")
    assert factory.alias_for_base_url("https://eu.api.openai.com/v1") == "openai"
    assert sent_to("https://eu.api.openai.com/v1")["openai-organization"] == "org-mine"
    assert factory.alias_for_base_url("https://notapi.openai.com.test/v1") is None


def test_a_client_that_was_cleaned_holds_nothing_of_the_environments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Read off the client, with no request made: what the library read in its
    constructor is gone from it, whatever the library's version merges or drops."""
    _openais_environment(monkeypatch)
    monkeypatch.setenv(
        "OPENAI_CUSTOM_HEADERS", "X-Gateway-Auth: secret-token\nauthorization: Bearer env"
    )
    groq = OpenAICompatibleClient(
        base_url="https://api.groq.com/openai/v1", model="m", api_key="k"
    )._client
    assert (groq.organization, groq.project, dict(groq._custom_headers)) == (None, None, {})
    assert groq.api_key == "k"
    gateway = OpenAICompatibleClient(
        base_url="https://my-gateway.test/v1", model="m", api_key="k"
    )._client
    assert (gateway.organization, gateway.project) == (None, None)
    assert {k.lower() for k in gateway._custom_headers} == {"x-gateway-auth", "authorization"}
    openai_itself = OpenAICompatibleClient(
        base_url="https://api.openai.com/v1", model="m", api_key="k"
    )._client
    assert (openai_itself.organization, openai_itself.project) == ("org-mine", "proj-mine")


def _merged_as_openai_3_does(client: Any) -> dict[str, str]:
    """The headers a client's defaults become under the OpenAI library 3.x, which CI
    installs: names merged without regard to case, a later mark of left out removing
    an earlier value. Replayed here so a machine on 2.x sees what CI sees."""
    from openai import Omit

    merged: dict[str, tuple[str, str]] = {}
    for name, value in client.default_headers.items():
        if isinstance(value, Omit):
            merged.pop(name.lower(), None)
        else:
            merged[name.lower()] = (name, str(value))
    return {name.lower(): value for name, value in merged.values()}


@pytest.mark.parametrize(
    "spelling", ["OpenAI-Organization", "openai-organization", "OPENAI-ORGANIZATION"]
)
def test_a_gateways_own_id_line_survives_the_merge_of_either_library(
    monkeypatch: pytest.MonkeyPatch, spelling: str
) -> None:
    monkeypatch.setenv("OPENAI_ORG_ID", "org-from-the-variable")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "proj-from-the-variable")
    monkeypatch.setenv("OPENAI_CUSTOM_HEADERS", f"{spelling}: org-for-the-gateway\nX-Team: a")
    client = OpenAICompatibleClient(
        base_url="https://my-gateway.test/v1", model="m", api_key="k"
    )._client
    assert (client.organization, client.project) == ("org-for-the-gateway", None)
    sent = _merged_as_openai_3_does(client)
    assert sent["openai-organization"] == "org-for-the-gateway"
    assert "openai-project" not in sent
    assert sent["x-team"] == "a"
    groq = OpenAICompatibleClient(
        base_url="https://api.groq.com/openai/v1", model="m", api_key="k"
    )._client
    assert not {"openai-organization", "openai-project", "x-team"} & set(
        _merged_as_openai_3_does(groq)
    )
