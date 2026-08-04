"""Embedding client port + adapters.

The mock adapter is not random: it hashes tokens into a fixed number of
buckets (bag-of-words projection), so texts sharing vocabulary really are
closer in cosine space. Retrieval tests exercise true ranking behavior
offline instead of asserting against arbitrary vectors.
"""

import asyncio
import hashlib
import math
import time
from typing import Any, Protocol, runtime_checkable

from openai import AsyncOpenAI

from app.core.logging import get_logger
from app.llm.base import TokenUsage
from app.llm.pricing import estimate_cost_usd

logger = get_logger(__name__)


@runtime_checkable
class EmbeddingClient(Protocol):
    """Purpose-aware embedding port used by new adapters."""

    @property
    def model_name(self) -> str: ...

    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...


@runtime_checkable
class BatchedQueryEmbeddingClient(Protocol):
    """Optional optimization for embedding several retrieval queries at once."""

    async def embed_queries(self, texts: list[str]) -> list[list[float]]: ...


@runtime_checkable
class LegacyEmbeddingClient(Protocol):
    """Original embedding contract retained for third-party/test adapters."""

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


EmbeddingBackend = EmbeddingClient | LegacyEmbeddingClient


async def embed_document_texts(client: EmbeddingBackend, texts: list[str]) -> list[list[float]]:
    """Use the document-specific method, falling back to the v1 contract."""
    if isinstance(client, EmbeddingClient):
        return await client.embed_documents(texts)
    return await client.embed(texts)


async def embed_query_texts(client: EmbeddingBackend, texts: list[str]) -> list[list[float]]:
    """Embed retrieval queries without silently applying document semantics."""
    if isinstance(client, BatchedQueryEmbeddingClient):
        return await client.embed_queries(texts)
    if isinstance(client, EmbeddingClient):
        return list(await asyncio.gather(*(client.embed_query(text) for text in texts)))
    return await client.embed(texts)


class OpenAIEmbeddingClient:
    """EmbeddingClient backed by the OpenAI embeddings API."""

    def __init__(
        self,
        api_key: str,
        model: str,
        batch_size: int = 128,
        query_instruction: str | None = None,
        model_revision: str | None = None,
    ) -> None:
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model
        self._batch_size = batch_size
        self._query_instruction = query_instruction
        self._model_revision = model_revision

    @property
    def model_name(self) -> str:
        return self._model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Compatibility alias: unqualified text is treated as document text."""
        return await self._embed_batch(texts, purpose="document")

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self.embed(texts)

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed_queries([text]))[0]

    async def embed_queries(self, texts: list[str]) -> list[list[float]]:
        prepared = [_with_instruction(text, self._query_instruction) for text in texts]
        return await self._embed_batch(prepared, purpose="query")

    async def _embed_batch(self, texts: list[str], *, purpose: str) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            t0 = time.perf_counter()
            response = await self._client.embeddings.create(model=self._model, input=batch)
            usage = TokenUsage(prompt_tokens=getattr(response.usage, "prompt_tokens", 0) or 0)
            logger.info(
                "embeddings_created",
                model=self._model,
                purpose=purpose,
                texts=len(batch),
                latency_ms=round((time.perf_counter() - t0) * 1000, 1),
                tokens=usage.prompt_tokens,
                cost_usd=estimate_cost_usd(self._model, usage),
            )
            vectors.extend(item.embedding for item in response.data)
        return vectors


class MockEmbeddingClient:
    """Deterministic bag-of-words embeddings for tests/offline mode."""

    def __init__(self, dimensions: int = 128, query_instruction: str | None = None) -> None:
        self._dimensions = dimensions
        self._query_instruction = query_instruction
        self._model_revision = "mock-hashed-bow-v1"

    @property
    def model_name(self) -> str:
        return f"mock-hashed-bow-v1:{self._dimensions}"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self.embed(texts)

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed_queries([text]))[0]

    async def embed_queries(self, texts: list[str]) -> list[list[float]]:
        prepared = [_with_instruction(text, self._query_instruction) for text in texts]
        return await self.embed(prepared)

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self._dimensions
        for token in text.lower().split():
            digest = hashlib.md5(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self._dimensions
            vector[bucket] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]


class SentenceTransformerEmbeddingClient:
    """Local multilingual embeddings through the optional ST dependency.

    The model name is entirely configuration-driven, so the same adapter works
    with ``ufca-llms/jua-4B-mixed``, ``BAAI/bge-m3`` and compatible models.
    Model loading is intentionally explicit and never exercised by offline
    tests unless this provider is selected.
    """

    def __init__(
        self,
        model: str,
        *,
        query_instruction: str | None = None,
        device: str | None = None,
        batch_size: int = 8,
        model_revision: str | None = None,
    ) -> None:
        try:
            import sentence_transformers
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise RuntimeError(
                "sentence-transformers is required for local embeddings; "
                "install the 'local-embeddings' optional dependency"
            ) from exc

        self._model_name = model
        self._query_instruction = query_instruction
        self._batch_size = batch_size
        self._model_revision = model_revision
        self._encode_lock = asyncio.Lock()
        self._model: Any = sentence_transformers.SentenceTransformer(
            model, device=device, revision=model_revision
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Compatibility alias: unqualified text is treated as document text."""
        return await self.embed_documents(texts)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        async with self._encode_lock:
            return await asyncio.to_thread(self._encode, texts)

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed_queries([text]))[0]

    async def embed_queries(self, texts: list[str]) -> list[list[float]]:
        prepared = [_with_instruction(text, self._query_instruction) for text in texts]
        async with self._encode_lock:
            return await asyncio.to_thread(self._encode, prepared)

    def _encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        encoded = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=self._batch_size,
        )
        return [[float(value) for value in row] for row in encoded]


def _with_instruction(text: str, instruction: str | None) -> str:
    if not instruction or not instruction.strip():
        return text
    prefix = instruction.strip()
    separator = "" if prefix.endswith((":", " ")) else " "
    return f"{prefix}{separator}{text}"
