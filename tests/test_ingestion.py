"""Tests for the ingestion pipeline (offline, no OCR binary required)."""

from pathlib import Path

import fitz
import pytest

from app.ingestion.pdf_reader import extract_text
from app.ingestion.service import DocumentIngestionService
from app.schemas.document import ExtractionMethod

PT_TEXT = (
    "EXCELENTISSIMO SENHOR DOUTOR JUIZ DE DIREITO DA 3a VARA CIVEL. "
    "Maria Silva, brasileira, portadora do CPF 000.000.000-00, vem, "
    "por seu advogado, propor a presente acao de indenizacao por danos "
    "morais e materiais em face de Banco Exemplo S.A., pelos fatos e "
    "fundamentos juridicos a seguir expostos. Da tutela de urgencia. "
    "Do valor da causa: R$ 50.000,00."
)


def _make_pdf(path: Path, page_texts: list[str]) -> Path:
    doc = fitz.open()
    for text in page_texts:
        page = doc.new_page()
        if text:
            # insert_textbox wraps lines; insert_text would clip at the
            # page edge and silently truncate the fixture content.
            rect = fitz.Rect(72, 72, page.rect.width - 72, page.rect.height - 72)
            page.insert_textbox(rect, text, fontsize=11)
    doc.save(path)
    doc.close()
    return path


class FakeOcr:
    """Deterministic OCR double."""

    def ocr_image(self, image_png: bytes) -> str:
        return "texto reconhecido via OCR"


@pytest.fixture
def text_pdf(tmp_path: Path) -> Path:
    return _make_pdf(tmp_path / "lawsuit.pdf", [PT_TEXT, PT_TEXT])


@pytest.fixture
def scanned_pdf(tmp_path: Path) -> Path:
    # No text layer at all -> looks like a scan
    return _make_pdf(tmp_path / "scan.pdf", ["", ""])


async def test_ingests_native_text_pdf(text_pdf: Path) -> None:
    service = DocumentIngestionService()
    doc = await service.ingest(text_pdf)

    assert doc.page_count == 2
    assert doc.extraction_method is ExtractionMethod.NATIVE_TEXT
    assert not doc.ocr_applied
    assert doc.language == "pt"
    assert "danos" in doc.full_text
    assert doc.warnings == []


async def test_scanned_pdf_uses_ocr_when_engine_available(scanned_pdf: Path) -> None:
    service = DocumentIngestionService(ocr_engine=FakeOcr())
    doc = await service.ingest(scanned_pdf)

    assert doc.extraction_method is ExtractionMethod.OCR
    assert doc.ocr_applied
    assert "OCR" in doc.full_text


async def test_scanned_pdf_without_ocr_engine_warns(scanned_pdf: Path) -> None:
    service = DocumentIngestionService(ocr_engine=None)
    doc = await service.ingest(scanned_pdf)

    assert doc.extraction_method is ExtractionMethod.NATIVE_TEXT
    assert not doc.ocr_applied
    assert len(doc.warnings) == 1


def test_needs_ocr_heuristic(text_pdf: Path, scanned_pdf: Path) -> None:
    assert extract_text(text_pdf).needs_ocr is False
    assert extract_text(scanned_pdf).needs_ocr is True


def test_doc_id_is_content_addressed(text_pdf: Path, tmp_path: Path) -> None:
    copy = _make_pdf(tmp_path / "copy.pdf", [PT_TEXT, PT_TEXT])
    service = DocumentIngestionService()
    import asyncio

    doc1 = asyncio.run(service.ingest(text_pdf))
    doc2 = asyncio.run(service.ingest(copy))
    assert doc1.doc_id == doc2.doc_id  # same content, same id (idempotency)


def test_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        extract_text(Path("does/not/exist.pdf"))
