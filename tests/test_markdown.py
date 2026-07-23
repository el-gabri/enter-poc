"""Tests for the Markdown renderer."""

from app.reporting.markdown import render_markdown
from app.schemas.analysis import LawsuitClassification
from app.schemas.common import Citation, ConfidentConclusion
from app.schemas.lawsuit import LawsuitType
from app.schemas.report import LitigationReport, RunMetrics
from app.schemas.risk import RiskAssessment, RiskItem, RiskLevel


def _report() -> LitigationReport:
    return LitigationReport(
        doc_id="abc123",
        filename="peticao.pdf",
        language="pt",
        executive_summary="Acao consumerista por cobranca indevida.",
        classification=LawsuitClassification(
            lawsuit_type=LawsuitType.CONSUMER,
            conclusion=ConfidentConclusion(
                statement="Acao consumerista",
                confidence=0.92,
                reasoning="Relacao de consumo bancaria",
                citations=[Citation(quote="cobrancas indevidas", page=1)],
            ),
        ),
        legal_risks=RiskAssessment(
            overall_level=RiskLevel.HIGH,
            overall=ConfidentConclusion(
                statement="Risco alto", confidence=0.8, reasoning="CDC favoravel"
            ),
            risks=[
                RiskItem(
                    title="Inversao do onus",
                    level=RiskLevel.HIGH,
                    conclusion=ConfidentConclusion(
                        statement="Provavel", confidence=0.85, reasoning="Sumula"
                    ),
                    financial_exposure="R$ 20.000,00",
                )
            ],
        ),
        missing_information=["judge", "contrato assinado"],
        confidence_level=0.84,
        ai_reasoning="Como esta analise foi produzida: ...",
        metrics=RunMetrics(agents_run=5, total_tokens=1234, total_cost_usd=0.0421),
    )


def test_renders_all_present_sections_with_confidence_and_citations() -> None:
    md = render_markdown(_report())

    assert "# Relatorio de Analise - peticao.pdf" in md
    assert "**Nivel de confianca agregado: 84%**" in md
    assert "## Resumo Executivo" in md
    assert "Confianca: **92%**" in md
    assert '"cobrancas indevidas", p. 1' in md
    assert "## Riscos Juridicos" in md
    assert "Nivel geral: **Alto**" in md
    assert "Exposicao financeira: R$ 20.000,00" in md
    assert "## Informacoes Ausentes" in md
    assert "Custo estimado: US$ 0.0421" in md
    assert "Nao substitui a analise de um advogado" in md


def test_absent_sections_are_omitted() -> None:
    md = render_markdown(_report())
    assert "## Linha do Tempo" not in md  # no timeline provided
    assert "## Estrategia Sugerida" not in md  # no strategy provided


def test_rendering_is_deterministic() -> None:
    report = _report()
    assert render_markdown(report) == render_markdown(report)
