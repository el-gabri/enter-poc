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


class VectorStoreBackend(str, Enum):
    """Supported vector store backends (see ADR 0003)."""

    CHROMA = "chroma"
    MEMORY = "memory"


class EmbeddingProvider(str, Enum):
    """Embedding backends independent from the conversational LLM provider."""

    AUTO = "auto"
    OPENAI = "openai"
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    MOCK = "mock"


class RetrievalMode(str, Enum):
    """Candidate-generation strategies supported by the RAG pipeline."""

    DENSE = "dense"
    HYBRID = "hybrid"


class RerankerProvider(str, Enum):
    """Optional second-stage retrieval rankers."""

    NONE = "none"
    SENTENCE_TRANSFORMERS = "sentence_transformers"


class PromptInjectionScanMode(str, Enum):
    """How the pre-analysis document security gate reviews suspicious text."""

    RULES = "rules"
    BALANCED = "balanced"
    STRICT = "strict"


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
    embedding_provider: EmbeddingProvider = EmbeddingProvider.AUTO
    embedding_model: str = "text-embedding-3-small"
    embedding_model_revision: str | None = None
    embedding_query_instruction: str | None = None
    embedding_device: str | None = None
    embedding_batch_size: int = Field(default=8, ge=1)

    # --- Storage ---
    data_dir: Path = Path("./data")
    vector_store: VectorStoreBackend = VectorStoreBackend.CHROMA
    max_document_pages: int = Field(default=250, ge=1)
    retain_uploads: bool = False

    # --- RAG ---
    chunk_target_chars: int = 1200
    chunk_overlap_chars: int = 150
    retrieval_k: int = Field(default=6, ge=1)
    retrieval_trace_include_previews: bool = False
    # Preserve the established Business ranking. Consumer retrieval opts into
    # hybrid explicitly at its call site.
    retrieval_mode: RetrievalMode = RetrievalMode.DENSE
    retrieval_candidate_multiplier: int = Field(default=4, ge=1)
    retrieval_rrf_constant: int = Field(default=60, ge=1)
    retrieval_dense_weight: float = Field(default=1.0, gt=0)
    retrieval_lexical_weight: float = Field(default=1.0, gt=0)
    rag_corpus_version: str = Field(default="documents-v1", min_length=1)
    reranker_provider: RerankerProvider = RerankerProvider.NONE
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_model_revision: str | None = None
    reranker_device: str | None = None
    reranker_batch_size: int = Field(default=8, ge=1)

    # --- Document security ---
    # Every mode applies deterministic bilingual rules to every page. Balanced
    # reviews suspicious candidates; strict semantically reviews all page text.
    prompt_injection_scan_mode: PromptInjectionScanMode = PromptInjectionScanMode.BALANCED
    prompt_injection_strict_max_chars: int = Field(default=500_000, ge=1)
    prompt_injection_strict_max_batches: int = Field(default=64, ge=1)

    # --- DataJud (CNJ public API) ---
    # The key is published openly by CNJ at
    # https://datajud-wiki.cnj.jus.br/api-publica/acesso/ but is still
    # injected via environment so rotation never requires a code change.
    datajud_api_key: str | None = Field(default=None, repr=False)
    datajud_base_url: str = "https://api-publica.datajud.cnj.jus.br"

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
