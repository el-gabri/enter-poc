"""Quality metrics.

Groundedness and hallucination are computed MECHANICALLY: a conclusion's
citations either quote the source document or they do not. No LLM opinion
involved (see ADR 0008). Completeness and accuracy compare extraction
output against golden labels.
"""

import re
import unicodedata

from app.schemas.common import ConfidentConclusion
from app.schemas.evaluation import MetricResult
from app.schemas.lawsuit import LawsuitExtraction, PartyRole
from app.schemas.report import LitigationReport


def _normalize(text: str) -> str:
    """Lowercase, strip accents, collapse whitespace and punctuation."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^\w\s]", " ", text.casefold())
    return re.sub(r"\s+", " ", text).strip()


def citation_supported(quote: str, document_text: str) -> bool:
    """True if the quoted passage really occurs in the document."""
    normalized_quote = _normalize(quote)
    return bool(normalized_quote) and normalized_quote in _normalize(document_text)


def _all_conclusions(report: LitigationReport) -> list[ConfidentConclusion]:
    conclusions: list[ConfidentConclusion] = []
    if report.classification:
        conclusions.append(report.classification.conclusion)
    conclusions.extend(c.assessment for c in report.main_claims)
    conclusions.extend(report.evidence_found)
    if report.legal_risks:
        conclusions.append(report.legal_risks.overall)
        conclusions.extend(r.conclusion for r in report.legal_risks.risks)
    if report.suggested_strategy:
        conclusions.append(report.suggested_strategy.overall_approach)
        conclusions.append(report.suggested_strategy.settlement)
        conclusions.extend(d.assessment for d in report.suggested_strategy.defenses)
    return conclusions


def groundedness(report: LitigationReport, document_text: str) -> MetricResult:
    """Fraction of cited quotes that verify against the source document."""
    quotes = [
        citation.quote
        for conclusion in _all_conclusions(report)
        for citation in conclusion.citations
    ]
    if not quotes:
        return MetricResult(
            name="groundedness", score=0.0, details="no citations produced"
        )
    supported = sum(1 for q in quotes if citation_supported(q, document_text))
    return MetricResult(
        name="groundedness",
        score=round(supported / len(quotes), 3),
        details=f"{supported}/{len(quotes)} citations verified in source",
    )


def hallucination_rate(report: LitigationReport, document_text: str) -> MetricResult:
    """1 - groundedness: fraction of citations that do NOT verify.

    A fabricated quote is the strongest observable signal of hallucination.
    """
    grounded = groundedness(report, document_text)
    score = 1.0 - grounded.score if "no citations" not in grounded.details else 1.0
    return MetricResult(
        name="hallucination_rate",
        score=round(score, 3),
        details=f"complement of groundedness ({grounded.details})",
    )


def citation_coverage(report: LitigationReport) -> MetricResult:
    """Fraction of conclusions that carry at least one citation."""
    conclusions = _all_conclusions(report)
    if not conclusions:
        return MetricResult(name="citation_coverage", score=0.0, details="no conclusions")
    cited = sum(1 for c in conclusions if c.citations)
    return MetricResult(
        name="citation_coverage",
        score=round(cited / len(conclusions), 3),
        details=f"{cited}/{len(conclusions)} conclusions cite the document",
    )


def extraction_accuracy(
    extraction: LawsuitExtraction | None, expected: dict
) -> MetricResult:
    """Field-level agreement with golden labels."""
    if extraction is None:
        return MetricResult(name="extraction_accuracy", score=0.0, details="no extraction")

    checks: list[tuple[str, bool]] = []

    def _contains(haystack: str | None, needle: str) -> bool:
        return haystack is not None and _normalize(needle) in _normalize(haystack)

    if "lawsuit_type" in expected:
        pass  # scored separately in classification_accuracy
    if "case_number" in expected and expected["case_number"] is not None:
        checks.append(("case_number", extraction.case_number == expected["case_number"]))
    if "claim_value_amount" in expected and expected["claim_value_amount"] is not None:
        actual = extraction.claim_value.amount if extraction.claim_value else None
        checks.append(("claim_value", actual == expected["claim_value_amount"]))
    if "plaintiff" in expected:
        names = [p.name for p in extraction.parties if p.role is PartyRole.PLAINTIFF]
        checks.append(
            ("plaintiff", any(_contains(n, expected["plaintiff"]) for n in names))
        )
    if "defendant" in expected:
        names = [p.name for p in extraction.parties if p.role is PartyRole.DEFENDANT]
        checks.append(
            ("defendant", any(_contains(n, expected["defendant"]) for n in names))
        )
    for request in expected.get("main_requests_contains", []):
        checks.append(
            (
                f"request:{request}",
                any(_contains(r, request) for r in extraction.main_requests),
            )
        )

    if not checks:
        return MetricResult(name="extraction_accuracy", score=0.0, details="no golden labels")
    passed = sum(1 for _, ok in checks if ok)
    failed = [name for name, ok in checks if not ok]
    return MetricResult(
        name="extraction_accuracy",
        score=round(passed / len(checks), 3),
        details=f"{passed}/{len(checks)} checks passed"
        + (f"; failed: {', '.join(failed)}" if failed else ""),
    )


def completeness(extraction: LawsuitExtraction | None, expected: dict) -> MetricResult:
    """Fraction of expected-present fields the pipeline actually filled."""
    if extraction is None:
        return MetricResult(name="completeness", score=0.0, details="no extraction")
    expected_fields = [
        key
        for key in ("case_number", "claim_value_amount", "plaintiff", "defendant")
        if expected.get(key) is not None
    ]
    if expected.get("main_requests_contains"):
        expected_fields.append("main_requests")
    if not expected_fields:
        return MetricResult(name="completeness", score=1.0, details="nothing expected")

    found = 0
    for field in expected_fields:
        if field == "claim_value_amount":
            found += extraction.claim_value is not None
        elif field == "plaintiff":
            found += any(p.role is PartyRole.PLAINTIFF for p in extraction.parties)
        elif field == "defendant":
            found += any(p.role is PartyRole.DEFENDANT for p in extraction.parties)
        elif field == "main_requests":
            found += bool(extraction.main_requests)
        else:
            found += getattr(extraction, field) is not None
    return MetricResult(
        name="completeness",
        score=round(found / len(expected_fields), 3),
        details=f"{found}/{len(expected_fields)} expected fields present",
    )


def classification_accuracy(
    report: LitigationReport, expected: dict
) -> MetricResult | None:
    if "lawsuit_type" not in expected:
        return None
    actual = (
        report.classification.lawsuit_type.value if report.classification else None
    )
    correct = actual == expected["lawsuit_type"]
    return MetricResult(
        name="classification_accuracy",
        score=1.0 if correct else 0.0,
        details=f"expected={expected['lawsuit_type']} actual={actual}",
    )
