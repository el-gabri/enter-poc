"""OCR engine port + Tesseract adapter.

OCR is an optional capability: the app degrades gracefully (with an explicit
warning on the document) when Tesseract is not installed. Install with:
``pip install -e ".[ocr]"`` plus the Tesseract binary + 'por' language pack.
"""

from typing import Protocol, runtime_checkable

from app.core.logging import get_logger

logger = get_logger(__name__)


@runtime_checkable
class OcrEngine(Protocol):
    """Turns a page image into text."""

    def ocr_image(self, image_png: bytes) -> str: ...


class TesseractOcr:
    """OcrEngine backed by pytesseract (lazy import)."""

    def __init__(self, language: str = "por") -> None:
        self._language = language

    def ocr_image(self, image_png: bytes) -> str:
        import io

        import pytesseract
        from PIL import Image

        image = Image.open(io.BytesIO(image_png))
        return pytesseract.image_to_string(image, lang=self._language)


def create_default_ocr_engine(language: str = "por") -> OcrEngine | None:
    """Return a Tesseract engine if the stack is available, else None."""
    try:
        import pytesseract

        pytesseract.get_tesseract_version()
    except Exception:
        logger.warning("ocr_unavailable", hint="install tesseract + pip install .[ocr]")
        return None
    return TesseractOcr(language=language)
