"""Shared state of the analysis graph.

Lists that multiple nodes append to (traces, errors) use an additive
reducer, so nodes can run in parallel later (M3b) without write conflicts.
"""

import operator
from typing import Annotated

from pydantic import BaseModel, Field

from app.schemas.analysis import LawsuitClassification, LegalAnalysis
from app.schemas.document import ParsedDocument
from app.schemas.lawsuit import LawsuitExtraction
from app.schemas.rag import Chunk
from app.schemas.trace import AgentTrace


class AnalysisState(BaseModel):
    """Everything the pipeline knows about one lawsuit analysis run."""

    document: ParsedDocument

    # Filled by the graph as it advances
    chunks: list[Chunk] = Field(default_factory=list)
    classification: LawsuitClassification | None = None
    extraction: LawsuitExtraction | None = None
    legal_analysis: LegalAnalysis | None = None

    # Observability (append-only, parallel-safe)
    traces: Annotated[list[AgentTrace], operator.add] = Field(default_factory=list)
    errors: Annotated[list[str], operator.add] = Field(default_factory=list)
