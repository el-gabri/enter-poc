"""Shared explainability primitives.

Every important conclusion in the system is a ``ConfidentConclusion``:
statement + confidence + reasoning + citations. This is the mechanism that
makes the product auditable instead of a black box.

Note on validation strategy: OpenAI structured outputs accepts only a subset
of JSON Schema, and numeric ``minimum``/``maximum`` keywords may be rejected
server-side. We therefore keep the generated schema plain and enforce the
0..1 confidence range with a client-side Pydantic validator.
"""

from pydantic import BaseModel, Field, field_validator


class Citation(BaseModel):
    """Pointer to the source text that supports a conclusion."""

    quote: str = Field(description="Verbatim excerpt from the document")
    page: int | None = Field(default=None, description="1-based page number")
    chunk_id: str | None = Field(
        default=None, description="RAG chunk id the quote was retrieved from"
    )


class ConfidentConclusion(BaseModel):
    """A conclusion the system is willing to defend."""

    statement: str = Field(description="The conclusion itself")
    confidence: float = Field(
        description="Confidence between 0.0 and 1.0 (e.g. 0.87)"
    )
    reasoning: str = Field(
        description="WHY this conclusion follows from the evidence"
    )
    citations: list[Citation] = Field(
        default_factory=list,
        description="Document excerpts supporting the conclusion",
    )

    @field_validator("confidence")
    @classmethod
    def _clamp_confidence(cls, v: float) -> float:
        """Clamp instead of reject: an out-of-range score from the LLM is a
        formatting slip, not a reason to fail the whole analysis."""
        return max(0.0, min(1.0, v))

    @property
    def confidence_pct(self) -> int:
        return round(self.confidence * 100)
