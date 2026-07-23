"""PDF text extraction with PyMuPDF.

See ADR 0005 for why PyMuPDF and how the OCR decision works.
"""

from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF

# Below this average of extractable characters per page we assume the PDF is
# a scan (image-only) and needs OCR. A page of legal text has 1500-3500
# chars; scans yield ~0. The generous margin tolerates cover pages/stamps.
MIN_AVG_CHARS_PER_PAGE = 50


@dataclass(frozen=True)
class PdfExtraction:
    """Raw result of native text extraction."""

    page_texts: list[str]
    needs_ocr: bool


def extract_text(path: Path) -> PdfExtraction:
    """Extract the text layer of each page and decide whether OCR is needed.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if the file is not a readable PDF.
    """
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        doc = fitz.open(path)
    except Exception as exc:  # fitz raises generic RuntimeError subclasses
        raise ValueError(f"Not a readable PDF: {path.name}") from exc

    with doc:
        page_texts = [page.get_text("text") for page in doc]

    if not page_texts:
        return PdfExtraction(page_texts=[], needs_ocr=False)

    avg_chars = sum(len(t.strip()) for t in page_texts) / len(page_texts)
    return PdfExtraction(
        page_texts=page_texts, needs_ocr=avg_chars < MIN_AVG_CHARS_PER_PAGE
    )


def render_page_images(path: Path, dpi: int = 200) -> list[bytes]:
    """Render each page as a PNG (input for OCR engines)."""
    with fitz.open(path) as doc:
        return [page.get_pixmap(dpi=dpi).tobytes("png") for page in doc]
