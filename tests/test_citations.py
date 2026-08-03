"""Tests for runtime citation validation before report delivery."""

from app.schemas.analysis import LawsuitClassification, TimelineEvent
from app.schemas.common import Citation, ConfidentConclusion
from app.schemas.document import DocumentPage, ExtractionMethod, ParsedDocument
from app.schemas.lawsuit import LawsuitType
from app.schemas.rag import Chunk
from app.schemas.report import LitigationReport
from app.services.citations import validate_report_citations


def _document() -> ParsedDocument:
    return ParsedDocument(
        filename="peticao.pdf",
        pages=[
            DocumentPage(number=1, text="A autora relata cobrancas indevidas."),
            DocumentPage(number=2, text="Requer indenizacao por danos morais."),
        ],
        language="pt",
        extraction_method=ExtractionMethod.NATIVE_TEXT,
    )


def _report(document: ParsedDocument) -> LitigationReport:
    return LitigationReport(
        doc_id=document.doc_id,
        filename=document.filename,
        language=document.language,
        executive_summary="Resumo.",
        classification=LawsuitClassification(
            lawsuit_type=LawsuitType.CONSUMER,
            conclusion=ConfidentConclusion(
                statement="Acao de consumo.",
                confidence=0.8,
                reasoning="Cobrancas bancarias.",
                citations=[
                    Citation(
                        quote="cobrancas indevidas", page=1, chunk_id=f"{document.doc_id}:0000"
                    ),
                    Citation(quote="indenizacao", page=1),
                ],
            ),
        ),
    )


def test_keeps_only_citations_with_a_valid_source_location() -> None:
    document = _document()
    chunk = Chunk(
        chunk_id=f"{document.doc_id}:0000",
        doc_id=document.doc_id,
        text="A autora relata cobrancas indevidas.",
        page_start=1,
        page_end=1,
    )
    report = _report(document)

    result = validate_report_citations(report, document, [chunk])

    assert result.total_citations == 2
    assert result.verified_citations == 1
    assert result.rejected_citations == 1
    assert report.classification is not None
    assert report.classification.conclusion.citations == [
        Citation(
            quote="cobrancas indevidas", page=1, chunk_id=f"{document.doc_id}:0000"
        )
    ]


def test_rejects_a_quote_on_the_wrong_page_even_if_it_exists_elsewhere() -> None:
    document = _document()
    report = _report(document)
    assert report.classification is not None
    report.classification.conclusion.citations = [Citation(quote="indenizacao", page=1)]

    result = validate_report_citations(report, document, [])

    assert result.rejected_citations == 1
    assert result.conclusions_without_verified_citation == 1
    assert report.classification.conclusion.citations == []


def test_rejects_an_invalid_timeline_citation() -> None:
    document = _document()
    report = _report(document)
    assert report.classification is not None
    report.classification.conclusion.citations = []
    report.timeline = [
        TimelineEvent(
            date="2025-01-10",
            description="Pedido de indenizacao",
            citation=Citation(quote="indenizacao", page=1),
        )
    ]

    result = validate_report_citations(report, document, [])

    assert result.total_citations == 1
    assert result.rejected_citations == 1
    assert report.timeline[0].citation is None
