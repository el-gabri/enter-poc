"""Vector store port + adapters (Chroma for runtime, in-memory for tests).

The in-memory adapter is not a toy: it proves the port is complete (two
independent implementations) and documents exactly what a Pinecone/Qdrant
adapter would need to provide.
"""

import asyncio
import math
from pathlib import Path
from typing import Protocol, runtime_checkable

from app.schemas.rag import Chunk, RetrievedChunk


@runtime_checkable
class VectorStore(Protocol):
    """Persistence + similarity search over chunks."""

    async def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None: ...

    async def query(
        self, vector: list[float], doc_id: str, k: int
    ) -> list[RetrievedChunk]: ...

    async def delete_document(self, doc_id: str) -> None: ...


class InMemoryVectorStore:
    """Reference implementation with exact cosine similarity."""

    def __init__(self) -> None:
        self._rows: dict[str, tuple[Chunk, list[float]]] = {}

    async def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        for chunk, vector in zip(chunks, vectors, strict=True):
            self._rows[chunk.chunk_id] = (chunk, vector)

    async def query(
        self, vector: list[float], doc_id: str, k: int
    ) -> list[RetrievedChunk]:
        candidates = [
            (chunk, _cosine(vector, stored))
            for chunk, stored in self._rows.values()
            if chunk.doc_id == doc_id
        ]
        candidates.sort(key=lambda pair: pair[1], reverse=True)
        return [
            RetrievedChunk(chunk=chunk, score=score)
            for chunk, score in candidates[:k]
        ]

    async def delete_document(self, doc_id: str) -> None:
        self._rows = {
            cid: row for cid, row in self._rows.items() if row[0].doc_id != doc_id
        }


class ChromaVectorStore:
    """VectorStore backed by a persistent ChromaDB collection.

    Chroma's client is synchronous; calls are wrapped in ``asyncio.to_thread``
    to keep the async contract honest.
    """

    COLLECTION = "lawsuits"

    def __init__(self, persist_dir: Path) -> None:
        import chromadb

        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    async def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        def _upsert() -> None:
            self._collection.upsert(
                ids=[c.chunk_id for c in chunks],
                embeddings=vectors,
                documents=[c.text for c in chunks],
                metadatas=[
                    {
                        "doc_id": c.doc_id,
                        "section": c.section or "",
                        "page_start": c.page_start,
                        "page_end": c.page_end,
                    }
                    for c in chunks
                ],
            )

        await asyncio.to_thread(_upsert)

    async def query(
        self, vector: list[float], doc_id: str, k: int
    ) -> list[RetrievedChunk]:
        def _query() -> list[RetrievedChunk]:
            result = self._collection.query(
                query_embeddings=[vector],
                n_results=k,
                where={"doc_id": doc_id},  # per-document isolation
                include=["documents", "metadatas", "distances"],
            )
            retrieved: list[RetrievedChunk] = []
            for chunk_id, text, meta, distance in zip(
                result["ids"][0],
                result["documents"][0],
                result["metadatas"][0],
                result["distances"][0],
                strict=True,
            ):
                chunk = Chunk(
                    chunk_id=chunk_id,
                    doc_id=str(meta["doc_id"]),
                    text=text,
                    section=str(meta["section"]) or None,
                    page_start=int(meta["page_start"]),
                    page_end=int(meta["page_end"]),
                )
                # cosine distance -> similarity
                retrieved.append(RetrievedChunk(chunk=chunk, score=1.0 - distance))
            return retrieved

        return await asyncio.to_thread(_query)

    async def delete_document(self, doc_id: str) -> None:
        await asyncio.to_thread(self._collection.delete, where={"doc_id": doc_id})


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a)) or 1.0
    norm_b = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (norm_a * norm_b)
