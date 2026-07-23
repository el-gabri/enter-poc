"""Tests for domain schema behavior (not just field presence)."""

from app.schemas.common import ConfidentConclusion
from app.schemas.lawsuit import LawsuitExtraction, Party, PartyRole


def test_confidence_is_clamped_not_rejected() -> None:
    high = ConfidentConclusion(statement="s", confidence=1.7, reasoning="r")
    low = ConfidentConclusion(statement="s", confidence=-0.2, reasoning="r")
    assert high.confidence == 1.0
    assert low.confidence == 0.0
    assert high.confidence_pct == 100


def test_missing_fields_reports_absent_information() -> None:
    empty = LawsuitExtraction()
    assert "case_number" in empty.missing_fields()
    assert "parties" in empty.missing_fields()

    partial = LawsuitExtraction(
        case_number="0001234-56.2026.8.26.0100",
        parties=[Party(name="Maria Silva", role=PartyRole.PLAINTIFF)],
        main_requests=["indenizacao por danos morais"],
    )
    missing = partial.missing_fields()
    assert "case_number" not in missing
    assert "parties" not in missing
    assert "judge" in missing
