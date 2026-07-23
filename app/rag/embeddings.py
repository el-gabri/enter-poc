"""Embedding client port + adapters.

The mock adapter is not random: it hashes tokens into a fixed number of
buckets (bag-of-words projection), so texts sharing vocabulary really are
closer in cosine space. Retrieval tests exercise true ranking behavior
offline instead of asserting against arbitrary vectors.
"""

import hashlib
import math
import time
from typing import Protocol, runtime_checkable

from openai import AsyncOpenAI

from app.core.logging import get_logger
from app.llm.base import TokenUsage
from app.llm.pricing import estimate_cost_usd

logger = get_logger(__name__)


@runtime_checkable
class EmbeddingClient(Protocol):
    """Turns texts into vectors."""

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class OpenAIEmbeddingClient:
    """EmbeddingClient backed by the OpenAI embeddings API."""

    def __init__(self, api_key: str, model: str, batch_size: int = 128) -> None:
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model
        self._batch_size = batch_size

    async def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            t0 = time.perf_counter()
            response = await self._client.embeddings.create(
                model=self._model, input=batch
            )
            usage = TokenUsage(
                prompt_tokens=getattr(response.usage, "prompt_tokens", 0) or 0
            )
            logger.info(
                "embeddings_created",
                model=self._model,
                texts=len(batch),
                latency_ms=round((time.perf_counter() - t0) * 1000, 1),
                tokens=usage.prompt_tokens,
                cost_usd=estimate_cost_usd(self._model, usage),
            )
            vectors.extend(item.embedding for item in response.data)
        return vectors


class MockEmbeddingClient:
    """Deterministic bag-of-words embeddings for tests/offline mode."""

    def __init__(self, dimensions: int = 128) -> None:
        self._dimensions = dimensions

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self._dimensions
        for token in text.lower().split():
            digest = hashlib.md5(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self._dimensions
            vector[bucket] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]
