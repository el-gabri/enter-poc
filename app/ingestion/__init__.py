"""Document ingestion: PDF -> ParsedDocument (text, language, OCR fallback)."""

from app.ingestion.service import DocumentIngestionService

__all__ = ["DocumentIngestionService"]
