"""Tests for metrics, golden loading and the evaluation runner."""

from pathlib import Path

from app.evaluation import metrics
from app.evaluation.golden import load_dataset
from app.evaluation.runner import EvaluationRunner
from app.llm.mock_client import MockLLMClient
from app.orchestration.graph import build_analysis_graph
from app.rag.embeddings import MockEmbeddingClient
from app.rag.pipeline import RagPipeline
from app.rag.vector_store import InMemoryVectorStore
from app.schemas.analysis import LawsuitClassification
from app.schemas.common import Citation, ConfidentConclusion
from app.schemas.lawsuit import (
    LawsuitExtraction,
    LawsuitType,
    MonetaryValue,
    Party,
    PartyRole,
)
from app.schemas.report import LitigationReport

DOC_TEXT = (
    "A autora e correntista do banco reu. Passou a identificar cobrancas "
    "mensais indevidas de tarifa, no valor de R$ 89,90."
)


def _report(quotes: list[str]) -> LitigationReport:
    return LitigationReport(
        doc_id="d1",
        filename="x.pdf",
        language="pt",
        executive_summary="resumo",
        classification=LawsuitClassification(
            lawsuit_type=LawsuitType.CONSUMER,
            conclusion=ConfidentConclusion(
                statement="consumerista",
                confidence=0.9,
                reasoning="r",
                citations=[Citation(quote=q) for q in quotes],
            ),
        ),
    )


def test_citation_verification_normalizes_accents_and_case() -> None:
    assert metrics.citation_supported("COBRANCAS mensais INDEVIDAS", DOC_TEXT)
    assert metrics.citation_supported("cobranças mensais indevidas", DOC_TEXT)
    assert not metrics.citation_supported("clausula de arbitragem", DOC_TEXT)


def test_groundedness_and_hallucination_are_complements() -> None:
    report = _report(["cobrancas mensais indevidas", "clausula inexistente"])
    grounded = metrics.groundedness(report, DOC_TEXT)
    hallucinated = metrics.hallucination_rate(report, DOC_TEXT)
    assert grounded.score == 0.5
    assert hallucinated.score == 0.5


def test_no_citations_means_zero_groundedness_full_hallucination_risk() -> None:
    report = _report([])
    assert metrics.groundedness(report, DOC_TEXT).score == 0.0
    assert metrics.hallucination_rate(report, DOC_TEXT).score == 1.0


def test_extraction_accuracy_and_completeness() -> None:
    extraction = LawsuitExtraction(
        claim_value=MonetaryValue(amount=16978.0),
        parties=[
            Party(name="Maria Silva", role=PartyRole.PLAINTIFF),
            Party(name="Banco Exemplo S.A.", role=PartyRole.DEFENDANT),
        ],
        main_requests=["indenizacao por danos morais"],
    )
    expected = {
        "claim_value_amount": 16978.0,
        "plaintiff": "Maria Silva",
        "defendant": "Banco Exemplo",
        "main_requests_contains": ["danos morais"],
    }
    assert metrics.extraction_accuracy(extraction, expected).score == 1.0
    assert metrics.completeness(extraction, expected).score == 1.0

    wrong = {"claim_value_amount": 99.0, "plaintiff": "Outra Pessoa"}
    assert metrics.extraction_accuracy(extraction, wrong).score == 0.0


async def test_runner_over_golden_dataset_offline() -> None:
    cases = load_dataset(Path("eval_data"))
    assert {c.name for c in cases} == {"consumer_billing", "labor_overtime"}

    graph = build_analysis_graph(
        MockLLMClient(),
        RagPipeline(embedder=MockEmbeddingClient(), store=InMemoryVectorStore()),
    )
    summary = await EvaluationRunner(graph).run(cases)

    assert len(summary.cases) == 2
    # every metric present for every case
    for case in summary.cases:
        names = {m.name for m in case.metrics}
        assert {
            "groundedness",
            "hallucination_rate",
            "citation_coverage",
            "extraction_accuracy",
            "completeness",
            "classification_accuracy",
        } <= names
    assert "groundedness" in summary.averages
