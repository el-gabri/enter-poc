# ADR 0005: PyMuPDF for extraction, heuristic OCR fallback

Status: accepted · Date: 2026-07-23

## Context

Brazilian court PDFs vary wildly: born-digital filings with clean text
layers, scanned exhibits with no text at all, and hybrids. Candidates for
extraction: pypdf (BSD), pdfplumber (MIT), PyMuPDF (AGPL). OCR candidates:
Tesseract (local, free), cloud OCR APIs (Google Vision, Textract).

## Decision

1. **PyMuPDF** for native text extraction and page rendering: best
   extraction quality on court documents and 10-50x faster than
   alternatives, which matters for 100+ page filings.
2. **OCR decision heuristic**: if average extractable characters per page
   < 50, treat the PDF as scanned. A legal text page has 1500-3500 chars;
   scans yield ~0. Threshold is a named constant, tuned per corpus.
3. **Tesseract (por) as optional OCR adapter** behind an `OcrEngine`
   protocol. When unavailable, the document carries an explicit warning
   instead of failing - degraded output is flagged, never silent.

## Consequences

- (+) Fast, robust extraction; OCR only when needed (OCR is 100x slower).
- (+) Cloud OCR later = one new `OcrEngine` adapter.
- (-) **PyMuPDF is AGPL**: acceptable for a portfolio/internal tool; a
  commercial closed-source deployment needs the Artifex commercial license
  or a swap to pdfplumber (MIT). Isolated in `pdf_reader.py` (~60 lines) so
  the swap is cheap. This trade-off is deliberate and documented.
- (-) Tesseract quality on low-resolution scans is mediocre; mitigated by
  200 DPI rendering and surfaced via document warnings.
