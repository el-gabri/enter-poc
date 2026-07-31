"""Create a safe LLM/RAG projection while preserving original evidence."""

import unicodedata

from app.schemas.document import ParsedDocument
from app.schemas.security import PromptInjectionAssessment

SECURITY_MASK = "[TRECHO SINALIZADO E REMOVIDO PELO CONTROLE DE SEGURANCA]"

_IGNORABLE_FORMAT_CHARS = frozenset(
    "\u00ad\u200b\u200c\u200d\u2060\ufeff"
    "\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069"
)
_HOMOGLYPHS = str.maketrans(
    {
        "а": "a",
        "е": "e",
        "і": "i",
        "ο": "o",
        "о": "o",
        "р": "p",
        "с": "c",
        "х": "x",
        "у": "y",
        "Α": "a",
        "Β": "b",
        "Ε": "e",
        "Ι": "i",
        "Ο": "o",
        "Ρ": "p",
        "Χ": "x",
    }
)


def canonical_with_positions(
    text: str, *, collapse_whitespace: bool = False
) -> tuple[str, list[int]]:
    """Normalize adversarial text while retaining offsets into the source."""
    canonical: list[str] = []
    positions: list[int] = []
    previous_was_space = False
    for index, original_char in enumerate(text):
        if original_char in _IGNORABLE_FORMAT_CHARS:
            continue
        translated = original_char.translate(_HOMOGLYPHS)
        for char in unicodedata.normalize("NFKD", translated):
            if unicodedata.combining(char):
                continue
            if collapse_whitespace and char.isspace():
                if not previous_was_space:
                    canonical.append(" ")
                    positions.append(index)
                previous_was_space = True
                continue
            previous_was_space = False
            canonical.append(char.lower())
            positions.append(index)
    return "".join(canonical), positions


def canonicalize(text: str, *, collapse_whitespace: bool = False) -> str:
    return canonical_with_positions(
        text, collapse_whitespace=collapse_whitespace
    )[0]


def mask_flagged_text(
    text: str, assessment: PromptInjectionAssessment | None
) -> str:
    """Mask findings across whitespace/accent normalization and line wrapping."""
    if assessment is None or not assessment.findings:
        return text

    masked = text
    quotes = sorted(
        {finding.quote for finding in assessment.findings},
        key=lambda quote: len(canonicalize(quote, collapse_whitespace=True)),
        reverse=True,
    )
    for quote in quotes:
        target = canonicalize(quote, collapse_whitespace=True).strip()
        if not target:
            continue
        canonical, positions = canonical_with_positions(
            masked, collapse_whitespace=True
        )
        spans: list[tuple[int, int]] = []
        offset = 0
        while (start := canonical.find(target, offset)) >= 0:
            end = start + len(target)
            spans.append((positions[start], positions[end - 1] + 1))
            offset = end
        for start, end in reversed(spans):
            masked = masked[:start] + SECURITY_MASK + masked[end:]
    return masked


def sanitized_document(
    document: ParsedDocument, assessment: PromptInjectionAssessment | None
) -> ParsedDocument:
    """Return a masked projection; the input document remains untouched."""
    if assessment is None or not assessment.findings:
        return document
    pages = [
        page.model_copy(update={"text": mask_flagged_text(page.text, assessment)})
        for page in document.pages
    ]
    return document.model_copy(update={"pages": pages})
