"""RAG facade used by the rest of the application.

Two operations: index a parsed document, retrieve context for a question.
Everything else (chunking policy, embedding provider, store backend) is
injected - agents never know which vector database is running.
"""

import asyncio

from app.core.logging import get_logger
from app.rag.chunking import SectionAwareChunker
from app.rag.embeddings import EmbeddingClient
from app.rag.vector_store import VectorStore
from app.schemas.document import ParsedDocument
from app.schemas.rag import Chunk, RetrievedChunk

logger = get_logger(__name__)


class RagPipeline:
    """Index and retrieve within a single document's boundary."""

    def __init__(
        self,
        embedder: EmbeddingClient,
        store: VectorStore,
        chunker: SectionAwareChunker | None = None,
        default_k: int = 6,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._chunker = chunker or SectionAwareChunker()
        self._default_k = default_k

    async def index_document(self, document: ParsedDocument) -> list[Chunk]:
        """Chunk, embed and store a document. Idempotent per doc_id."""
        chunks = self._chunker.chunk(document)
        if not chunks:
            logger.warning("index_empty_document", doc_id=document.doc_id)
            return []
        vectors = await self._embedder.embed([c.text for c in chunks])
        await self._store.upsert(chunks, vectors)
        logger.info(
            "document_indexed",
            doc_id=document.doc_id,
            chunks=len(chunks),
            sections=len({c.section for c in chunks}),
        )
        return chunks

    async def retrieve(
        self, query: str, doc_id: str, k: int | None = None
    ) -> list[RetrievedChunk]:
        """Return the k chunks of ``doc_id`` most relevant to ``query``."""
        [vector] = await self._embedder.embed([query])
        results = await self._store.query(vector, doc_id=doc_id, k=k or self._default_k)
        logger.info(
            "chunks_retrieved",
            doc_id=doc_id,
            query_preview=query[:60],
            results=len(results),
            top_score=round(results[0].score, 3) if results else None,
        )
        return results

    async def retrieve_many(
        self, queries: list[str], doc_id: str, k: int | None = None
    ) -> list[list[RetrievedChunk]]:
        """Retrieve context for several queries with one embedding request.

        Agent prompts commonly ask several focused questions of the same
        document. Batching their embeddings reduces provider round-trips and
        querying the store concurrently preserves the async latency benefit.
        """
        if not queries:
            return []
        vectors = await self._embedder.embed(queries)
        results = await asyncio.gather(
            *(
                self._store.query(vector, doc_id=doc_id, k=k or self._default_k)
                for vector in vectors
            )
        )
        logger.info(
            "chunks_retrieved_batch",
            doc_id=doc_id,
            queries=len(queries),
            results=sum(len(items) for items in results),
        )
        return list(results)
