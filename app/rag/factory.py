"""Composition root for the configurable RAG stack."""

from typing import cast

from app.core.config import (
    EmbeddingProvider,
    LLMProvider,
    RerankerProvider,
    Settings,
    VectorStoreBackend,
)
from app.rag.chunking import SectionAwareChunker
from app.rag.embeddings import (
    EmbeddingBackend,
    MockEmbeddingClient,
    OpenAIEmbeddingClient,
    SentenceTransformerEmbeddingClient,
)
from app.rag.pipeline import RagPipeline
from app.rag.reranking import Reranker, SentenceTransformerReranker
from app.rag.vector_store import (
    ChromaVectorStore,
    InMemoryVectorStore,
    VectorStore,
    versioned_collection_name,
)

_CONFIGURED_RERANKER = object()


def create_embedding_client(settings: Settings) -> EmbeddingBackend:
    provider = settings.embedding_provider
    if provider is EmbeddingProvider.AUTO:
        provider = (
            EmbeddingProvider.MOCK
            if settings.llm_provider is LLMProvider.MOCK
            else EmbeddingProvider.OPENAI
        )
    if provider is EmbeddingProvider.MOCK:
        return MockEmbeddingClient(query_instruction=settings.embedding_query_instruction)
    if provider is EmbeddingProvider.SENTENCE_TRANSFORMERS:
        return SentenceTransformerEmbeddingClient(
            model=settings.embedding_model,
            query_instruction=settings.embedding_query_instruction,
            device=settings.embedding_device,
            batch_size=settings.embedding_batch_size,
            model_revision=settings.embedding_model_revision,
        )
    if not settings.openai_api_key:
        raise ValueError("LITIGATION_OPENAI_API_KEY required for OpenAI embeddings")
    return OpenAIEmbeddingClient(
        api_key=settings.openai_api_key,
        model=settings.embedding_model,
        query_instruction=settings.embedding_query_instruction,
        model_revision=settings.embedding_model_revision,
    )


def create_vector_store(
    settings: Settings,
    *,
    embedding_model: str | None = None,
    corpus_version: str | None = None,
) -> VectorStore:
    collection_name = versioned_collection_name(
        corpus_version or settings.rag_corpus_version,
        embedding_model or settings.embedding_model,
        prefix="give-exit",
    )
    if settings.vector_store is VectorStoreBackend.MEMORY:
        return InMemoryVectorStore(index_name=collection_name)
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    return ChromaVectorStore(persist_dir=settings.chroma_dir, collection_name=collection_name)


def create_reranker(settings: Settings) -> Reranker | None:
    if settings.reranker_provider is RerankerProvider.NONE:
        return None
    return SentenceTransformerReranker(
        settings.reranker_model,
        device=settings.reranker_device,
        batch_size=settings.reranker_batch_size,
        model_revision=settings.reranker_model_revision,
    )


def create_rag_pipeline(
    settings: Settings,
    *,
    corpus_version: str | None = None,
    embedder: EmbeddingBackend | None = None,
    reranker: Reranker | None | object = _CONFIGURED_RERANKER,
) -> RagPipeline:
    effective_embedder = embedder or create_embedding_client(settings)
    effective_reranker = (
        create_reranker(settings)
        if reranker is _CONFIGURED_RERANKER
        else cast(Reranker | None, reranker)
    )
    effective_corpus_version = corpus_version or settings.rag_corpus_version
    embedding_model = str(getattr(effective_embedder, "model_name", settings.embedding_model))
    embedding_revision = getattr(effective_embedder, "_model_revision", None)
    embedding_identity = (
        f"{embedding_model}@{embedding_revision}" if embedding_revision else embedding_model
    )
    return RagPipeline(
        embedder=effective_embedder,
        store=create_vector_store(
            settings,
            embedding_model=embedding_identity,
            corpus_version=effective_corpus_version,
        ),
        chunker=SectionAwareChunker(
            target_chars=settings.chunk_target_chars,
            overlap_chars=settings.chunk_overlap_chars,
        ),
        default_k=settings.retrieval_k,
        include_trace_previews=settings.retrieval_trace_include_previews,
        retrieval_mode=settings.retrieval_mode,
        candidate_multiplier=settings.retrieval_candidate_multiplier,
        rrf_constant=settings.retrieval_rrf_constant,
        dense_weight=settings.retrieval_dense_weight,
        lexical_weight=settings.retrieval_lexical_weight,
        reranker=effective_reranker,
        corpus_version=effective_corpus_version,
    )
