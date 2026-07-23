"""Composition root for the RAG stack."""

from app.core.config import LLMProvider, Settings, VectorStoreBackend
from app.rag.chunking import SectionAwareChunker
from app.rag.embeddings import EmbeddingClient, MockEmbeddingClient, OpenAIEmbeddingClient
from app.rag.pipeline import RagPipeline
from app.rag.vector_store import ChromaVectorStore, InMemoryVectorStore, VectorStore


def create_embedding_client(settings: Settings) -> EmbeddingClient:
    if settings.llm_provider is LLMProvider.MOCK:
        return MockEmbeddingClient()
    if not settings.openai_api_key:
        raise ValueError("LITIGATION_OPENAI_API_KEY required for OpenAI embeddings")
    return OpenAIEmbeddingClient(
        api_key=settings.openai_api_key, model=settings.embedding_model
    )


def create_vector_store(settings: Settings) -> VectorStore:
    if settings.vector_store is VectorStoreBackend.MEMORY:
        return InMemoryVectorStore()
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    return ChromaVectorStore(persist_dir=settings.chroma_dir)


def create_rag_pipeline(settings: Settings) -> RagPipeline:
    return RagPipeline(
        embedder=create_embedding_client(settings),
        store=create_vector_store(settings),
        chunker=SectionAwareChunker(
            target_chars=settings.chunk_target_chars,
            overlap_chars=settings.chunk_overlap_chars,
        ),
        default_k=settings.retrieval_k,
    )
