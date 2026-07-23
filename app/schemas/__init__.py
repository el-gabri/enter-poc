"""Domain schemas - the typed contract between all agents and layers."""

from app.schemas.common import Citation, ConfidentConclusion
from app.schemas.document import DocumentPage, ExtractionMethod, ParsedDocument
from app.schemas.lawsuit import (
    Deadline,
    LawsuitExtraction,
    LawsuitType,
    MonetaryValue,
    Party,
    PartyRole,
)

__all__ = [
    "Citation",
    "ConfidentConclusion",
    "Deadline",
    "DocumentPage",
    "ExtractionMethod",
    "LawsuitExtraction",
    "LawsuitType",
    "MonetaryValue",
    "ParsedDocument",
    "Party",
    "PartyRole",
]
