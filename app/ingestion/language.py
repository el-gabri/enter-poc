"""Language detection for ingested documents."""

from langdetect import DetectorFactory, LangDetectException, detect

# langdetect is probabilistic; a fixed seed makes results deterministic,
# which matters for tests and reproducible pipeline runs.
DetectorFactory.seed = 0

FALLBACK_LANGUAGE = "pt"  # product operates in the Brazilian market
MIN_TEXT_LENGTH = 40  # below this, detection is noise


def detect_language(text: str) -> str:
    """Return the ISO 639-1 language code of ``text``.

    Falls back to Portuguese when the sample is too short or detection
    fails - the safe default for our market.
    """
    sample = text.strip()
    if len(sample) < MIN_TEXT_LENGTH:
        return FALLBACK_LANGUAGE
    try:
        return detect(sample[:5000])  # first pages are enough; keep it fast
    except LangDetectException:
        return FALLBACK_LANGUAGE
