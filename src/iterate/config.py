"""Central configuration — the single source of every runtime default.

All settings live here with their defaults, loaded from environment variables
(and a local ``.env``). To override in deployment, set the env var / secret —
no other module hardcodes a default. Each field name is the lowercase of its
env var (matching ``.env.example``), so pydantic-settings binds them
automatically (case-insensitive) with no aliases needed.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Process-wide settings, sourced from env + .env (env wins)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ─── LLM backend (default: local Ollama, $0, no key) ───────────
    iterate_backend_url: str = "http://localhost:11434/v1"
    iterate_backend_api_key: str = "ollama"
    iterate_model: str = "qwen3:14b"  # emits structured tool calls (qwen2.5-coder does not)
    iterate_backend_timeout: float = 120.0

    # ─── Sandbox + dataset access ──────────────────────────────────
    e2b_api_key: str | None = None
    kaggle_username: str | None = None
    kaggle_key: str | None = None

    # ─── Optional cloud LLM keys ───────────────────────────────────
    groq_api_key: str | None = None
    together_api_key: str | None = None
    deepseek_api_key: str | None = None
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None

    # ─── Optional logging adapters ─────────────────────────────────
    notion_api_key: str | None = None
    notion_database_id: str | None = None
    slack_webhook_url: str | None = None

    # ─── Optional tracing ──────────────────────────────────────────
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str | None = None

    # ─── Local paths ───────────────────────────────────────────────
    iterate_memory_db: str = ".iterate/memory.db"
    iterate_runs_dir: str = ".iterate/runs"


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings singleton (cached after first load)."""
    return Settings()


__all__ = ["Settings", "get_settings"]
