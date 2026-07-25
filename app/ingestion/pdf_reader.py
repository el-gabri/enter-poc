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
    page_needs_ocr: list[bool]

    @property
    def needs_ocr(self) -> bool:
        """Whether at least one page needs OCR."""
        return any(self.page_needs_ocr)

    @property
    def ocr_page_indexes(self) -> list[int]:
        """Zero-based indexes of pages that need OCR."""
        return [index for index, needed in enumerate(self.page_needs_ocr) if needed]


def extract_text(path: Path, max_pages: int | None = None) -> PdfExtraction:
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
        if max_pages is not None and len(doc) > max_pages:
            raise ValueError(
                f"PDF exceeds the {max_pages}-page limit: {path.name} has {len(doc)} pages"
            )
        page_texts = [page.get_text("text") for page in doc]

    if not page_texts:
        return PdfExtraction(page_texts=[], page_needs_ocr=[])

    return PdfExtraction(
        page_texts=page_texts,
        page_needs_ocr=[
            len(page_text.strip()) < MIN_AVG_CHARS_PER_PAGE for page_text in page_texts
        ],
    )


def render_page_images(
    path: Path, page_indexes: list[int] | None = None, dpi: int = 200
) -> list[bytes]:
    """Render selected pages as PNGs (input for OCR engines)."""
    with fitz.open(path) as doc:
        indexes = page_indexes if page_indexes is not None else list(range(len(doc)))
        return [doc.load_page(index).get_pixmap(dpi=dpi).tobytes("png") for index in indexes]
