"""Ingestion use case: PDF file -> ParsedDocument.

Flow:
    native text extraction
        -> looks like a scan? -> OCR (if engine available, else warn)
        -> language detection
        -> ParsedDocument

CPU/disk-bound work runs in a worker thread (``asyncio.to_thread``) so the
async API event loop is never blocked by a 200-page PDF.
"""

import asyncio
from pathlib import Path

from app.core.logging import get_logger
from app.ingestion import pdf_reader
from app.ingestion.language import detect_language
from app.ingestion.ocr import OcrEngine
from app.schemas.document import DocumentPage, ExtractionMethod, ParsedDocument

logger = get_logger(__name__)


class DocumentIngestionService:
    """Turns an uploaded PDF into a ParsedDocument ready for analysis."""

    def __init__(
        self, ocr_engine: OcrEngine | None = None, max_pages: int | None = None
    ) -> None:
        self._ocr_engine = ocr_engine
        self._max_pages = max_pages

    async def ingest(self, path: Path) -> ParsedDocument:
        return await asyncio.to_thread(self._ingest_sync, path)

    def _ingest_sync(self, path: Path) -> ParsedDocument:
        extraction = pdf_reader.extract_text(path, max_pages=self._max_pages)
        warnings: list[str] = []
        method = ExtractionMethod.NATIVE_TEXT
        ocr_applied = False
        page_texts = extraction.page_texts

        if extraction.needs_ocr:
            if self._ocr_engine is not None:
                ocr_indexes = extraction.ocr_page_indexes
                logger.info("ocr_started", file=path.name, pages=len(ocr_indexes))
                images = pdf_reader.render_page_images(path, page_indexes=ocr_indexes)
                successful_pages = 0
                for index, image in zip(ocr_indexes, images, strict=True):
                    try:
                        page_texts[index] = self._ocr_engine.ocr_image(image)
                        successful_pages += 1
                    except Exception as exc:
                        warnings.append(
                            f"OCR failed on page {index + 1}; extracted text may be incomplete."
                        )
                        logger.warning(
                            "ocr_page_failed", file=path.name, page=index + 1, error=str(exc)
                        )
                if successful_pages:
                    method = (
                        ExtractionMethod.OCR
                        if successful_pages == len(page_texts)
                        else ExtractionMethod.HYBRID
                    )
                    ocr_applied = True
            else:
                warnings.append(
                    f"{len(extraction.ocr_page_indexes)} page(s) appear to be scanned but "
                    "no OCR engine is available; extracted text is likely incomplete."
                )
                logger.warning("ocr_needed_but_unavailable", file=path.name)

        pages = [
            DocumentPage(number=i + 1, text=text)
            for i, text in enumerate(page_texts)
        ]
        full_text = "\n\n".join(page_texts)
        document = ParsedDocument(
            filename=path.name,
            pages=pages,
            language=detect_language(full_text),
            extraction_method=method,
            ocr_applied=ocr_applied,
            warnings=warnings,
        )
        logger.info(
            "document_ingested",
            file=path.name,
            doc_id=document.doc_id,
            pages=document.page_count,
            language=document.language,
            method=method.value,
            warnings=len(warnings),
        )
        return document
