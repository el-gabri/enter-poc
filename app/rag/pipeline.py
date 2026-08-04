"""RAG facade used by the rest of the application.

Two operations: index a parsed document, retrieve context for a question.
Everything else (chunking policy, embedding provider, store backend) is
injected - agents never know which vector database is running.
"""

import asyncio
import hashlib
import time
import uuid

from app.core.logging import get_logger
from app.rag.chunking import SectionAwareChunker
from app.rag.embeddings import EmbeddingClient
from app.rag.vector_store import VectorStore
from app.schemas.document import ParsedDocument
from app.schemas.rag import Chunk, RetrievedChunk
from app.schemas.security import PromptInjectionAssessment
from app.schemas.trace import RetrievalTrace, RetrievedItemTrace
from app.security.sanitization import SANITIZER_VERSION, sanitized_document

logger = get_logger(__name__)

AUDIT_PREVIEW_CHARS = 240


class RetrievalBatchError(RuntimeError):
    """A failed retrieval batch whose query-level audit remains available."""

    def __init__(self, cause: Exception, traces: list[RetrievalTrace]) -> None:
        self.cause = cause
        self.traces = traces
        failed = sum(trace.error is not None for trace in traces)
        super().__init__(
            f"{failed}/{len(traces)} retrieval queries failed; "
            f"first error: {_error_text(cause)}"
        )


class RagPipeline:
    """Index and retrieve within a single document's boundary."""

    def __init__(
        self,
        embedder: EmbeddingClient,
        store: VectorStore,
        chunker: SectionAwareChunker | None = None,
        default_k: int = 6,
        include_trace_previews: bool = False,
    ) -> None:
        if default_k < 1:
            raise ValueError("default_k must be positive")
        self._embedder = embedder
        self._store = store
        self._chunker = chunker or SectionAwareChunker()
        self._default_k = default_k
        self._embedding_model = str(
            getattr(embedder, "model_name", type(embedder).__name__)
        )
        self._vector_store_name = type(store).__name__
        self._index_version = f"{self._chunker.index_version}:{SANITIZER_VERSION}"
        self._include_trace_previews = include_trace_previews

    async def index_document(
        self,
        document: ParsedDocument,
        security_assessment: PromptInjectionAssessment | None = None,
    ) -> list[Chunk]:
        """Chunk, embed and store a document. Idempotent per doc_id."""
        safe_document = sanitized_document(document, security_assessment)
        chunks = self._chunker.chunk(safe_document, doc_id=document.doc_id)
        if not chunks:
            await self._store.delete_document(document.doc_id)
            logger.warning("index_empty_document", doc_id=document.doc_id)
            return []
        vectors = await self._embedder.embed([c.text for c in chunks])
        await self._store.delete_document(document.doc_id)
        await self._store.upsert(chunks, vectors)
        logger.info(
            "document_indexed",
            doc_id=document.doc_id,
            chunks=len(chunks),
            sections=len({c.section for c in chunks}),
        )
        return chunks

    async def delete_document(self, doc_id: str) -> None:
        """Remove every indexed chunk for ``doc_id``.

        Case-oriented features use this public operation to honor deletion and
        retention requests without reaching through the RAG facade into a
        concrete vector-store adapter.
        """
        await self._store.delete_document(doc_id)
        logger.info("document_index_deleted", doc_id=doc_id)

    async def retrieve(
        self, query: str, doc_id: str, k: int | None = None
    ) -> list[RetrievedChunk]:
        """Return the k chunks of ``doc_id`` most relevant to ``query``."""
        results, _ = await self.retrieve_with_trace(
            query, doc_id=doc_id, agent="direct", k=k
        )
        return results

    async def retrieve_with_trace(
        self,
        query: str,
        *,
        doc_id: str,
        agent: str,
        k: int | None = None,
    ) -> tuple[list[RetrievedChunk], RetrievalTrace]:
        """Return one ranked result set and its durable provenance trace."""
        result_sets, traces = await self.retrieve_many_with_traces(
            [query], doc_id=doc_id, agent=agent, k=k
        )
        return result_sets[0], traces[0]

    async def retrieve_many(
        self, queries: list[str], doc_id: str, k: int | None = None
    ) -> list[list[RetrievedChunk]]:
        """Retrieve context for several queries with one embedding request.

        Agent prompts commonly ask several focused questions of the same
        document. Batching their embeddings reduces provider round-trips and
        querying the store concurrently preserves the async latency benefit.
        """
        results, _ = await self.retrieve_many_with_traces(
            queries, doc_id=doc_id, agent="direct", k=k
        )
        return results

    async def retrieve_many_with_traces(
        self,
        queries: list[str],
        *,
        doc_id: str,
        agent: str,
        k: int | None = None,
    ) -> tuple[list[list[RetrievedChunk]], list[RetrievalTrace]]:
        """Retrieve a query batch and retain every ranked result for auditing."""
        if not queries:
            return [], []

        requested_k = self._default_k if k is None else k
        if requested_k < 1:
            raise ValueError("k must be positive")
        batch_id = uuid.uuid4().hex
        batch_started = time.perf_counter()
        embedding_started = time.perf_counter()
        try:
            vectors = await self._embedder.embed(queries)
        except Exception as exc:
            embedding_duration_ms = (time.perf_counter() - embedding_started) * 1000
            batch_duration_ms = (time.perf_counter() - batch_started) * 1000
            traces = [
                self._build_retrieval_trace(
                    agent=agent,
                    batch_id=batch_id,
                    doc_id=doc_id,
                    query=query,
                    query_index=query_index,
                    requested_k=requested_k,
                    results=[],
                    embedding_duration_ms=embedding_duration_ms,
                    search_duration_ms=0.0,
                    batch_duration_ms=batch_duration_ms,
                    error=_error_text(exc),
                )
                for query_index, query in enumerate(queries)
            ]
            logger.warning(
                "retrieval_embedding_failed",
                doc_id=doc_id,
                agent=agent,
                queries=len(queries),
                error_type=type(exc).__name__,
            )
            raise RetrievalBatchError(exc, traces) from exc
        embedding_duration_ms = (time.perf_counter() - embedding_started) * 1000

        if len(vectors) != len(queries):
            cardinality_error = ValueError(
                "embedding provider returned "
                f"{len(vectors)} vectors for {len(queries)} queries"
            )
            batch_duration_ms = (time.perf_counter() - batch_started) * 1000
            traces = [
                self._build_retrieval_trace(
                    agent=agent,
                    batch_id=batch_id,
                    doc_id=doc_id,
                    query=query,
                    query_index=query_index,
                    requested_k=requested_k,
                    results=[],
                    embedding_duration_ms=embedding_duration_ms,
                    search_duration_ms=0.0,
                    batch_duration_ms=batch_duration_ms,
                    error=_error_text(cardinality_error),
                )
                for query_index, query in enumerate(queries)
            ]
            raise RetrievalBatchError(cardinality_error, traces) from cardinality_error

        async def query_store(
            vector: list[float],
        ) -> tuple[list[RetrievedChunk], float, Exception | None]:
            started = time.perf_counter()
            try:
                items = await self._store.query(
                    vector, doc_id=doc_id, k=requested_k
                )
            except Exception as exc:
                return [], (time.perf_counter() - started) * 1000, exc
            return items, (time.perf_counter() - started) * 1000, None

        timed_results = await asyncio.gather(*(query_store(vector) for vector in vectors))
        batch_duration_ms = (time.perf_counter() - batch_started) * 1000
        results = [items for items, _, _ in timed_results]
        traces = [
            self._build_retrieval_trace(
                agent=agent,
                batch_id=batch_id,
                doc_id=doc_id,
                query=query,
                query_index=query_index,
                requested_k=requested_k,
                results=items,
                embedding_duration_ms=embedding_duration_ms,
                search_duration_ms=search_duration_ms,
                batch_duration_ms=batch_duration_ms,
                error=_error_text(error) if error is not None else None,
            )
            for query_index, (
                query,
                (items, search_duration_ms, error),
            ) in enumerate(
                zip(queries, timed_results, strict=True)
            )
        ]
        failures = [error for _, _, error in timed_results if error is not None]
        logger.info(
            "chunks_retrieved_batch",
            doc_id=doc_id,
            agent=agent,
            queries=len(queries),
            results=sum(len(items) for items in results),
            query_hashes=[trace.query_sha256[:12] for trace in traces],
            top_scores=[
                round(trace.results[0].score, 4) if trace.results else None
                for trace in traces
            ],
            embedding_duration_ms=round(embedding_duration_ms, 1),
            failed_queries=len(failures),
        )
        if failures:
            raise RetrievalBatchError(failures[0], traces)
        return results, traces

    def _build_retrieval_trace(
        self,
        *,
        agent: str,
        batch_id: str,
        doc_id: str,
        query: str,
        query_index: int,
        requested_k: int,
        results: list[RetrievedChunk],
        embedding_duration_ms: float,
        search_duration_ms: float,
        batch_duration_ms: float,
        error: str | None = None,
    ) -> RetrievalTrace:
        return RetrievalTrace(
            agent=agent,
            batch_id=batch_id,
            doc_id=doc_id,
            query_index=query_index,
            query=query,
            query_sha256=hashlib.sha256(query.encode("utf-8")).hexdigest(),
            requested_k=requested_k,
            returned_count=len(results),
            embedding_model=self._embedding_model,
            vector_store=self._vector_store_name,
            index_version=self._index_version,
            embedding_duration_ms=embedding_duration_ms,
            search_duration_ms=search_duration_ms,
            total_duration_ms=embedding_duration_ms + search_duration_ms,
            batch_duration_ms=batch_duration_ms,
            error=error,
            results=[
                RetrievedItemTrace(
                    rank=rank,
                    chunk_id=item.chunk.chunk_id,
                    doc_id=item.chunk.doc_id,
                    section=item.chunk.section,
                    page_start=item.chunk.page_start,
                    page_end=item.chunk.page_end,
                    score=item.score,
                    content_sha256=hashlib.sha256(
                        item.chunk.text.encode("utf-8")
                    ).hexdigest(),
                    text_preview=(
                        _audit_preview(item.chunk.text)
                        if self._include_trace_previews
                        else None
                    ),
                )
                for rank, item in enumerate(results, start=1)
            ],
        )


def _audit_preview(text: str) -> str:
    """Return a bounded preview suitable for persisted audit records."""
    normalized = " ".join(text.split())
    if len(normalized) <= AUDIT_PREVIEW_CHARS:
        return normalized
    return normalized[: AUDIT_PREVIEW_CHARS - 1].rstrip() + "…"


def _error_text(error: Exception) -> str:
    """Stable, compact error text suitable for a persisted trace."""
    return f"{type(error).__name__}: {error}"
