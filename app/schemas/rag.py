"""Schemas for the retrieval layer."""

from pydantic import BaseModel, Field

MetadataValue = str | int | float | bool | None


class Chunk(BaseModel):
    """A retrievable slice of a document, with provenance."""

    chunk_id: str = Field(description="Stable id: '{doc_id}:{index:04d}'")
    doc_id: str
    text: str
    section: str | None = Field(
        default=None, description="Heading of the section this chunk belongs to"
    )
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    metadata: dict[str, MetadataValue] = Field(
        default_factory=dict,
        description=(
            "Structured source provenance such as law/article, corpus release, "
            "official URL and content hash"
        ),
    )


class RetrievedChunk(BaseModel):
    """A chunk returned by similarity search."""

    chunk: Chunk
    score: float = Field(description="Similarity score (higher = more relevant)")
