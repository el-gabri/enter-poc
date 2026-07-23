"""Application configuration.

Single source of truth for all runtime settings, loaded from environment
variables (prefix ``LITIGATION_``) or a local ``.env`` file. Every component
receives a ``Settings`` instance via dependency injection instead of reading
``os.environ`` directly, which keeps configuration testable and explicit.
"""

from enum import Enum
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(str, Enum):
    """Supported LLM backends. Adding a provider = new enum value + new client."""

    OPENAI = "openai"
    MOCK = "mock"


class Settings(BaseSettings):
    """Runtime configuration for the whole application."""

    model_config = SettingsConfigDict(
        env_prefix="LITIGATION_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM ---
    llm_provider: LLMProvider = LLMProvider.OPENAI
    openai_api_key: str | None = Field(default=None, repr=False)
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    # --- Storage ---
    data_dir: Path = Path("./data")

    # --- Output ---
    report_language: str = "pt-BR"

    # --- Logging ---
    log_level: str = "INFO"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def reports_dir(self) -> Path:
        return self.data_dir / "reports"


@lru_cache
def get_settings() -> Settings:
    """Cached settings accessor for entry points (API, CLI, UI).

    Components should still receive ``Settings`` as a constructor argument;
    this accessor exists only at composition roots.
    """
    return Settings()
