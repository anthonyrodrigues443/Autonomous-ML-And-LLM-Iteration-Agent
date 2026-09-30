"""LLM client factory — pick the right adapter for a named backend.

The CLI's ``--backend`` flag dispatches through here. Both `OllamaClient` and
`OpenAICompatibleClient` implement the `LLMClient` protocol, so nothing
downstream notices which one came back.

Two kinds of caller. The harness, the model that runs the loop, comes through
`build_client` and `api_key_for`. The model under test of a prompt run comes through
`prompt_provider` and `build_target_client`: its address and its key are its
provider's own, looked up by the provider's name, and never the harness's.
"""

from __future__ import annotations

import importlib.util
import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from iterate.llm.ollama_client import OllamaClient
from iterate.llm.openai_compatible import OpenAICompatibleClient, keep_openais_own_at_home

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

    from iterate.llm.base import LLMClient
    from iterate.userconfig import SavedProvider

logger = logging.getLogger(__name__)

# The generic slot: whatever key the harness backend needs. Never a prompt provider's.
_HARNESS_SLOT = "iterate_backend_api_key"


@dataclass(frozen=True)
class _Known:
    """What iterate knows of a provider before anyone sets anything."""

    # Which client speaks to it. A name is a company or a server, never a wire format.
    wire: str
    # Its public address. None for a server the user runs, who gives the address.
    address: str | None = None
    # The Settings field that holds its own key. None when no company's key is its own.
    key_field: str | None = None


# One row per provider. Every other table here, and the hosts the OpenAI client keeps
# OpenAI's headers from, are read off it.
_PROVIDERS: dict[str, _Known] = {
    "ollama": _Known("ollama"),
    "openai": _Known("openai", "https://api.openai.com/v1", "openai_api_key"),
    "groq": _Known("openai", "https://api.groq.com/openai/v1", "groq_api_key"),
    "together": _Known("openai", "https://api.together.xyz/v1", "together_api_key"),
    "deepseek": _Known("openai", "https://api.deepseek.com/v1", "deepseek_api_key"),
    "openai-compatible": _Known("openai"),
    "vllm": _Known("openai"),
    "anthropic": _Known("anthropic", "https://api.anthropic.com", "anthropic_api_key"),
}

# Cloud aliases → their OpenAI-compatible base URL, so `--backend groq` (or a saved
# config) needs only a model + key, never a hand-typed --base-url. An explicit
# --base-url still overrides. "openai-compatible"/"vllm" have no canonical URL (the
# user supplies one), so they're not mapped.
_ALIAS_BASE_URLS = {name: known.address for name, known in _PROVIDERS.items() if known.address}

# Names that route to the OpenAI-compatible client.
_OPENAI_COMPATIBLE_ALIASES = frozenset(
    name for name, known in _PROVIDERS.items() if known.wire == "openai"
)

_KNOWN = ("ollama", "openai-compatible")

# One key for the one model under test a run's flags name. Read from the exported
# environment only, never from a project's .env.
TARGET_KEY_ENV = "ITERATE_TARGET_API_KEY"
# What a server that asks for no key is sent, so the client never reaches for the
# harness's.
NO_KEY = "not-needed"


class UnknownBackendError(ValueError):
    """Raised when ``--backend`` names a backend we don't recognize."""


class ProviderError(ValueError):
    """A prompt provider that cannot be called as it stands. The message says why and
    what to do."""


@dataclass(frozen=True)
class Provider:
    """One provider a prompt run may call, resolved: where it is and the key it takes."""

    name: str
    base_url: str | None = None
    api_key: str | None = field(default=None, repr=False)
    # Where the key was found, in words a person can act on. Never the key.
    key_from: str = ""

    @property
    def needs_key(self) -> bool:
        return takes_a_key(self.name) and not is_self_hosted(self.name)


@dataclass(frozen=True)
class UnderTest:
    """What a prompt run settled before it wrote anything: the model its prompt is
    tuned for, and every provider it may call, each with its key in hand."""

    provider: Provider
    model: str
    # The name the run's flags gave, which picks the client: `openai-compatible` aimed
    # at OpenAI is the provider openai, called through the client of the name given.
    backend: str
    allowed: tuple[Provider, ...] = ()
    # None when no list was saved or given: the run named its one provider itself.
    listed: tuple[str, ...] | None = None


def resolve_base_url(name: str, base_url: str | None) -> str | None:
    """Explicit ``base_url`` wins; otherwise a known cloud alias supplies its own."""
    return base_url if base_url is not None else _ALIAS_BASE_URLS.get(name)


def _host(url: str | None) -> str:
    """The machine an address names, in one spelling: lower case, no closing dot, and
    no port, so OpenAI's host with `:443` on it is still OpenAI's."""
    try:
        return (urlparse(url or "").hostname or "").rstrip(".").lower()
    except ValueError:
        return ""


def alias_for_base_url(base_url: str | None) -> str | None:
    """The cloud alias whose endpoint an explicit base URL points at, or None."""
    host = _host(base_url)
    if not host:
        return None
    for alias, url in _ALIAS_BASE_URLS.items():
        # The company's host, or one under it: eu.api.openai.com is OpenAI's.
        if host == _host(url) or host.endswith(f".{_host(url)}"):
            return alias
    return None


def build_client(
    name: str = "ollama",
    *,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    think: bool = False,
) -> LLMClient:
    """Build an `LLMClient` for a named backend.

    - ``"ollama"`` (default) → `OllamaClient` (native ``/api/chat``; ``think`` is
      forwarded, default ``False``).
    - ``"openai-compatible"`` (or any of the cloud aliases) → `OpenAICompatibleClient`
      (``think`` does not apply — the `/v1` layer can't control it).

    Any of `model`/`base_url`/`api_key` left ``None`` falls through to the client's
    own defaults (which read the central `Settings` / `.env`). API-key validation
    is the CLI layer's job — this factory just dispatches.
    """
    if name == "ollama":
        return OllamaClient(host=base_url, model=model, think=think)
    if name in _OPENAI_COMPATIBLE_ALIASES:
        if think:
            logger.warning(
                "think=True only applies to the ollama backend (native /api/chat); ignored for %r",
                name,
            )
        return OpenAICompatibleClient(
            base_url=resolve_base_url(name, base_url), model=model, api_key=api_key
        )
    if name in _PROVIDERS:
        raise UnknownBackendError(not_a_harness(name))
    raise UnknownBackendError(f"unknown backend {name!r}; choose one of {_KNOWN}")


def not_a_harness(name: str) -> str:
    """Why a provider cannot run the loop."""
    return (
        f"{name} runs as the model under test of a prompt run (--target-backend {name}); "
        f"as the harness that runs the loop it comes in v1.0. Harness backends: "
        f"{', '.join(harness_backends())}"
    )


def _sdk_missing(wire: str | None) -> str | None:
    """Why a client of this wire cannot be built here, or None. Claude's SDK is an
    optional extra."""
    if wire == "anthropic" and not importlib.util.find_spec("anthropic"):
        return "the anthropic package is not installed: pip install 'iterate-ai[anthropic]'"
    return None


# Which settings field holds the harness's key for each cloud backend.
_KEY_FIELDS: dict[str, tuple[str, ...]] = {
    name: tuple(f for f in (known.key_field, _HARNESS_SLOT) if f)
    for name, known in _PROVIDERS.items()
    if known.wire != "ollama"
}


def api_key_for(backend: str, settings: object | None = None) -> str | None:
    """The harness's api key for a backend, or None.

    Reads the vendor-specific env var (`GROQ_API_KEY`, `OPENAI_API_KEY`, …) and
    falls back to the generic `ITERATE_BACKEND_API_KEY`. Ollama has no key, and its
    placeholder default is treated as absent. A server the user runs has the generic
    slot alone: aimed at a company's address it is that company, so pass
    ``provider_name(backend, base_url)``.
    """
    if settings is None:
        from iterate.config import get_settings

        settings = get_settings()
    for attr in _KEY_FIELDS.get(backend, ()):
        value = getattr(settings, attr, None)
        if value and value != "ollama":  # the placeholder default, not a real key
            return str(value)
    return None


def known_providers() -> tuple[str, ...]:
    """Every name a prompt provider can have, the local one first."""
    return ("ollama", *sorted(n for n in _PROVIDERS if n != "ollama"))


def harness_backends() -> tuple[str, ...]:
    """Every name `--backend` takes: the providers a client of the harness's speaks to."""
    return tuple(n for n in known_providers() if _PROVIDERS[n].wire in ("ollama", "openai"))


def wire_of(name: str) -> str | None:
    """Which client speaks to this provider, or None for a name iterate does not know."""
    known = _PROVIDERS.get(name)
    return known.wire if known is not None else None


def is_self_hosted(name: str) -> bool:
    """A server the user runs. It has no public address, so the user gives one, and its
    name says nothing of who is called: a key saved for it goes with its address."""
    known = _PROVIDERS.get(name)
    return known is not None and known.wire != "ollama" and known.address is None


def takes_a_key(name: str) -> bool:
    known = _PROVIDERS.get(name)
    return known is not None and known.wire != "ollama"


def provider_name(backend: str, base_url: str | None) -> str:
    """The provider a backend and a base URL add up to: `openai-compatible` aimed at
    api.openai.com is openai, with openai's key and openai's prices."""
    return alias_for_base_url(base_url) or backend


def own_key_env(name: str) -> str | None:
    """The environment variable that holds this provider's own key, or None."""
    known = _PROVIDERS.get(name)
    return known.key_field.upper() if known is not None and known.key_field else None


def own_key_for(name: str, settings: object | None = None) -> str | None:
    """This provider's own key from the environment or the project's .env, or None.
    Never the generic slot and never another provider's."""
    variable = own_key_env(name)
    if variable is None:
        return None
    if settings is None:
        from iterate.config import get_settings

        settings = get_settings()
    value = getattr(settings, variable.lower(), None)
    return str(value) if value else None


_DEFAULT_PORTS = {"http": 80, "https": 443}


def same_place(one: str | None, other: str | None) -> bool:
    """Whether two addresses name one server: the same host and the same port, where a
    port left out is the scheme's own, so https and http to one host are two places."""

    def place(url: str | None) -> tuple[str, int | None]:
        parsed = urlparse(url or "")
        return _host(url), parsed.port or _DEFAULT_PORTS.get(parsed.scheme)

    try:
        return bool(_host(one)) and place(one) == place(other)
    except ValueError:
        return False


def shown(url: str | None) -> str:
    """An address as it may be printed or delivered: no user name, no password, and
    nothing after a `?` or a `#`, where a token can ride."""
    try:
        parsed = urlparse(url or "")
        port = f":{parsed.port}" if parsed.port else ""
    except ValueError:
        return "an address that cannot be read"
    if not parsed.hostname:
        return url or ""
    return f"{parsed.scheme}://{parsed.hostname}{port}{parsed.path.rstrip('/')}"


def prompt_provider(
    backend: str,
    *,
    base_url: str | None = None,
    saved: Mapping[str, SavedProvider] | None = None,
    settings: object | None = None,
    environ: Mapping[str, str] | None = None,
    harness_key: str | None = None,
    harness_key_from: str = "the harness key",
) -> Provider:
    """The provider a prompt run calls, with its endpoint and its key.

    The endpoint is the one given, else the one saved for this provider, else the
    provider's own public address: never the harness's. Ollama alone has no public
    address and is found at `OLLAMA_HOST`. The key is the exported target key, else the
    ``harness_key`` a caller hands on, else the one saved for this provider, else the
    provider's own variable. A key saved for a server the user runs goes only to the
    address saved beside it.
    """
    if backend not in _PROVIDERS:
        raise ProviderError(unknown(backend))
    env = os.environ if environ is None else environ
    entries = saved or {}
    named = entries.get(backend)
    url = base_url or (named.base_url if named is not None else None)
    name = provider_name(backend, url)
    # An Ollama host is settled here, so the cell calls the host the baseline called.
    url = (url or _ollama_host(settings)) if backend == "ollama" else resolve_base_url(name, url)
    entry = entries.get(name) or named
    saved_key = entry.api_key if entry is not None else None
    # By the name it was saved under: aimed at a company, the key of a server the user
    # runs is still that server's.
    if entry is not None and is_self_hosted(entry.name) and not same_place(url, entry.base_url):
        saved_key = None
    for key, source in (
        ((env.get(TARGET_KEY_ENV) or "").strip(), TARGET_KEY_ENV),
        (harness_key, harness_key_from),
        (saved_key, "the saved config"),
        (own_key_for(name, settings), own_key_env(name) or ""),
    ):
        if key and takes_a_key(name):
            return Provider(name=name, base_url=url, api_key=key, key_from=source)
    return Provider(name=name, base_url=url)


def unknown(*names: str, flag: str = "") -> str:
    """The one wording for a name that is no provider's, led by the flag it came from."""
    return (
        f"{flag + ' ' if flag else ''}{', '.join(names)}: not a provider iterate knows. "
        f"Choose from {', '.join(known_providers())}"
    )


def _ollama_host(settings: object | None) -> str | None:
    if settings is None:
        from iterate.config import get_settings

        settings = get_settings()
    host = getattr(settings, "ollama_host", None)
    return str(host) if host else None


def _bad_address(provider: Provider, given_as: str, wire: str | None = None) -> str | None:
    """Why an address cannot be used, or None. It is written into meta.json, which is
    delivered with the run, and a key is sent to it. ``given_as`` names where it came
    from, the flag or the saved file, so the refusal says what to change."""
    url = provider.base_url or ""
    what = given_as or f"the base URL saved for {provider.name}"
    try:
        parsed = urlparse(url)
        parsed.port  # noqa: B018  a port that is not a number raises here
    except ValueError:
        return f"{what} cannot be read: check its port"
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return f"{what} has to start with http:// or https:// and name a host"
    if any(ch.isspace() or ord(ch) < 32 for ch in url):
        return f"{what} holds a space or a character that cannot be sent"
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        return (
            f"{what} carries a user name, a password or a token. Give the address "
            f"without it, and the key in {TARGET_KEY_ENV} or `iterate setup`"
        )
    if (wire or wire_of(provider.name)) == "anthropic" and parsed.path.rstrip("/").endswith("/v1"):
        return f"{what} ends in /v1, which Anthropic's library adds itself. Drop the /v1"
    if (wire or wire_of(provider.name)) == "ollama" and parsed.path.rstrip("/").endswith("/v1"):
        return (
            f"{what} ends in /v1, the OpenAI-compatible door of a server, and Ollama is "
            "called at its own address. Drop the /v1, or pass --target-backend "
            "openai-compatible for a server that speaks OpenAI's API"
        )
    if provider.api_key and parsed.scheme == "http" and alias_for_base_url(url) is not None:
        return (
            f"{provider.name} takes its key over https only: {shown(url)} would send it "
            f"in the clear. Pass the https:// address"
        )
    return None


def not_callable(
    provider: Provider,
    *,
    in_a_cell: bool = False,
    given_as: str = "",
    wire: str | None = None,
    library: bool = True,
) -> str | None:
    """Why the model under test cannot be called, or None when it can. ``in_a_cell``
    words the remedy for a kernel or a notebook run by hand, which reads the
    environment alone. ``given_as`` is the flag its address came from, if one did.
    ``wire`` is the client it is called with, when that is not its provider's own.
    ``library`` False leaves out whether the client's library is installed, so every
    other reason is known before one is installed."""
    if library and (missing := _sdk_missing(wire or wire_of(provider.name))) is not None:
        return missing
    if is_self_hosted(provider.name) and not provider.base_url:
        return (
            f"{provider.name} is a server you run, so it has no address of its own: "
            f"save its base URL with `iterate setup` or pass --target-base-url"
        )
    if provider.base_url and (why := _bad_address(provider, given_as, wire)) is not None:
        return why
    if provider.needs_key and not provider.api_key:
        variable = own_key_env(provider.name)
        if in_a_cell:
            return f"{provider.name} has no key here: set {variable} or {TARGET_KEY_ENV}"
        return f"{provider.name} has no key: save one with `iterate setup` or set {variable}"
    return None


def not_ready(provider: Provider, *, wire: str | None = None) -> str | None:
    """Why a provider the list allows could not be called, in a few words, or None. No
    flag of a run mends it: the run calls another provider. ``wire`` is the client it
    is saved to be called with, when that is not its provider's own."""
    if (missing := _sdk_missing(wire or wire_of(provider.name))) is not None:
        return missing
    if is_self_hosted(provider.name) and not provider.base_url:
        return "no base URL saved"
    if provider.base_url and _bad_address(provider, "", wire) is not None:
        return "its saved address cannot be used"
    if provider.needs_key and not provider.api_key:
        return f"no key, set {own_key_env(provider.name)}"
    return None


def refused_key(
    provider: Provider,
    *,
    model: str | None = None,
    timeout: float = 5.0,
    lister: Callable[[Provider, float, str | None], list[str] | None] | None = None,
) -> str | None:
    """Ask the provider which models it serves this key, which costs nothing, and return
    why the run cannot go on, or None. Two reasons count: a plain no to the key, and a
    company that does not serve the model named. A provider that cannot be reached, a
    key that may chat and may not list models, a list that cannot be read: the run goes
    on, and the first real call says so."""
    if not provider.api_key or provider.name not in _PROVIDERS:
        return None
    ask = lister or _LISTERS.get(_PROVIDERS[provider.name].wire)
    if ask is None:
        return None
    try:
        served = ask(provider, timeout, model)
    except Exception as exc:
        # Read off the status, not the class: each wire's library has its own classes.
        if getattr(exc, "status_code", None) == 401:
            remedy = (
                f"unset {TARGET_KEY_ENV} to use the key saved for {provider.name}"
                if provider.key_from == TARGET_KEY_ENV
                else f"export {TARGET_KEY_ENV} for this run"
            )
            return (
                f"{_host(provider.base_url) or provider.name} refused the {provider.name} "
                f"key from {provider.key_from}. Check the key, or {remedy}"
            )
        logger.info("could not check the %s key (%s)", provider.name, type(exc).__name__)
        return None
    # A server the user runs, or a gateway, may list its models by other names.
    company = alias_for_base_url(provider.base_url) == provider.name
    if model is None or served is None or not company or model in served:
        return None
    shown_names = sorted(served)
    listing = ", ".join(shown_names) if 0 < len(shown_names) <= 12 else ""
    return f"{provider.name} does not serve {model} to this key." + (
        f" It serves: {listing}" if listing else " Check the name in its model list"
    )


def ollama_refusal(
    provider: Provider,
    model: str,
    *,
    timeout: float = 5.0,
    lister: Callable[[str, float], list[str]] | None = None,
) -> str | None:
    """Why Ollama at the model under test's address cannot answer for ``model``, or
    None. Its model list is free and local: a server that does not answer, one that is
    not Ollama, or a model not pulled is found before anything is written, where the
    run would find it on every record of the baseline."""
    host = provider.base_url or ""
    try:
        names = (lister or _ollama_models)(host, timeout)
    except Exception as exc:
        return (
            f"no Ollama server answers at {shown(host)} ({type(exc).__name__}): start it "
            "with `ollama serve`, or pass --target-backend and --target-model for the "
            "provider your prompt is for"
        )
    if model in names or f"{model}:latest" in names:
        return None
    return (
        f"Ollama at {shown(host)} has no model {model}: `ollama pull {model}`, or pass "
        "--target-backend and --target-model for the provider your prompt is for"
    )


def _ollama_models(host: str, timeout: float) -> list[str]:
    import httpx

    response = httpx.get(f"{host.rstrip('/')}/api/tags", timeout=timeout)
    response.raise_for_status()
    return [
        str(name)
        for entry in response.json().get("models", [])
        for name in (entry.get("name"), entry.get("model"))
        if name
    ]


def _list_models(provider: Provider, timeout: float, model: str | None) -> list[str]:
    from openai import OpenAI

    client = OpenAI(
        base_url=provider.base_url, api_key=provider.api_key, timeout=timeout, max_retries=0
    )
    listing = keep_openais_own_at_home(client, provider.base_url or "").models.list()
    return [str(entry.id) for entry in listing]


def _claude_serves(provider: Provider, timeout: float, model: str | None) -> list[str] | None:
    from iterate.llm import anthropic_client

    return anthropic_client.served(
        model, base_url=provider.base_url or "", api_key=provider.api_key or "", timeout=timeout
    )


# How each wire says which models it serves. The call is looked up when it is made.
_LISTERS: dict[str, Callable[[Provider, float, str | None], list[str] | None]] = {
    "openai": lambda provider, timeout, model: _list_models(provider, timeout, model),
    "anthropic": lambda provider, timeout, model: _claude_serves(provider, timeout, model),
}


def build_target_client(
    backend: str, *, model: str, base_url: str | None, api_key: str | None
) -> LLMClient:
    """The client for the model under test. Its endpoint and its key are always given
    to it, so it never falls through to the harness's settings."""
    if backend == "ollama":
        return build_client(backend, model=model, base_url=base_url, api_key=None)
    if _PROVIDERS.get(backend, _Known("")).wire == "anthropic":
        found = Provider(
            name=backend,
            base_url=resolve_base_url(backend, base_url),
            api_key=None if api_key == NO_KEY else api_key,
        )
        if (why := not_callable(found, in_a_cell=True)) is not None:
            raise ProviderError(why)
        from iterate.llm.anthropic_client import AnthropicClient

        return AnthropicClient(
            model=model, api_key=found.api_key or "", base_url=found.base_url or ""
        )
    if base_url is None and is_self_hosted(backend):
        # A run folder from before v0.7 wrote no address for a server the user runs:
        # its notebook finds the server where that run did, in the environment.
        from iterate.config import get_settings

        base_url = get_settings().iterate_backend_url
    provider = Provider(
        name=provider_name(backend, base_url),
        base_url=resolve_base_url(backend, base_url),
        api_key=None if api_key == NO_KEY else api_key,
    )
    if (why := not_callable(provider, in_a_cell=True, wire=wire_of(backend))) is not None:
        raise ProviderError(why)
    return build_client(
        backend, model=model, base_url=provider.base_url, api_key=provider.api_key or NO_KEY
    )


def cache_scope(backend: str, base_url: str | None) -> str:
    """Who answered: the provider and the host it was called at. Two providers that
    serve a model under one name give different answers."""
    url = base_url if backend == "ollama" else resolve_base_url(backend, base_url)
    try:
        parsed = urlparse(url or "")
        port = f":{parsed.port}" if parsed.port else ""
    except ValueError:
        return f"{backend}|"
    where = f"{_host(url)}{port}{parsed.path.rstrip('/')}" if parsed.hostname else ""
    return f"{provider_name(backend, base_url)}|{where}"


def allowed_names(names: Iterable[str]) -> tuple[str, ...]:
    """A list of provider names checked against the ones iterate knows."""
    cleaned = tuple(dict.fromkeys(n.strip().lower() for n in names if n and n.strip()))
    if strangers := [n for n in cleaned if n not in _PROVIDERS]:
        raise ProviderError(unknown(*strangers))
    return cleaned


__all__ = [
    "NO_KEY",
    "TARGET_KEY_ENV",
    "Provider",
    "ProviderError",
    "UnderTest",
    "UnknownBackendError",
    "alias_for_base_url",
    "allowed_names",
    "api_key_for",
    "build_client",
    "build_target_client",
    "cache_scope",
    "harness_backends",
    "is_self_hosted",
    "known_providers",
    "not_a_harness",
    "not_callable",
    "not_ready",
    "ollama_refusal",
    "own_key_env",
    "own_key_for",
    "prompt_provider",
    "provider_name",
    "refused_key",
    "resolve_base_url",
    "same_place",
    "shown",
    "takes_a_key",
    "unknown",
    "wire_of",
]
