"""End-to-end tests of the analysis graph with mock providers (offline)."""


from app.llm.mock_client import MockLLMClient
from app.orchestration.graph import build_analysis_graph
from app.orchestration.state import AnalysisState
from app.rag.embeddings import MockEmbeddingClient
from app.rag.pipeline import RagPipeline
from app.rag.vector_store import InMemoryVectorStore
from app.schemas.analysis import (
    ClaimAnalysis,
    LawsuitClassification,
    LegalAnalysis,
)
from app.schemas.common import Citation, ConfidentConclusion
from app.schemas.document import DocumentPage, ExtractionMethod, ParsedDocument
from app.schemas.lawsuit import (
    LawsuitExtraction,
    LawsuitType,
    MonetaryValue,
    Party,
    PartyRole,
)
from app.schemas.risk import RiskAssessment, RiskItem, RiskLevel
from app.schemas.strategy import ActionPriority, RecommendedAction, StrategyPlan
from app.schemas.trace import AgentStatus


def _document() -> ParsedDocument:
    return ParsedDocument(
        filename="peticao.pdf",
        pages=[
            DocumentPage(
                number=1,
                text=(
                    "DOS FATOS\n\nA autora verificou cobrancas indevidas "
                    "em sua fatura de cartao de credito do Banco Exemplo S.A."
                ),
            ),
            DocumentPage(
                number=2,
                text=(
                    "DOS PEDIDOS\n\nRequer indenizacao por danos morais de "
                    "R$ 20.000,00 e restituicao em dobro."
                ),
            ),
        ],
        language="pt",
        extraction_method=ExtractionMethod.NATIVE_TEXT,
    )


def _canned_responses() -> dict:
    classification = LawsuitClassification(
        lawsuit_type=LawsuitType.CONSUMER,
        conclusion=ConfidentConclusion(
            statement="Acao de natureza consumerista",
            confidence=0.92,
            reasoning="Cobranca indevida em relacao de consumo bancaria",
            citations=[Citation(quote="cobrancas indevidas", page=1)],
        ),
        secondary_types=[LawsuitType.BANKING],
    )
    extraction = LawsuitExtraction(
        court="3a Vara Civel",
        state="SP",
        claim_value=MonetaryValue(amount=20000.0, as_written="R$ 20.000,00"),
        parties=[
            Party(name="Maria Silva", role=PartyRole.PLAINTIFF),
            Party(name="Banco Exemplo S.A.", role=PartyRole.DEFENDANT),
        ],
        main_requests=["danos morais", "restituicao em dobro"],
    )
    analysis = LegalAnalysis(
        executive_summary="Acao consumerista por cobranca indevida.",
        claims=[
            ClaimAnalysis(
                claim="danos morais",
                legal_basis="CDC art. 42",
                assessment=ConfidentConclusion(
                    statement="Pedido plausivel",
                    confidence=0.75,
                    reasoning="Ha indicio documental de cobranca indevida",
                ),
            )
        ],
    )
    risk = RiskAssessment(
        overall_level=RiskLevel.MEDIUM,
        overall=ConfidentConclusion(
            statement="Risco medio de condenacao",
            confidence=0.7,
            reasoning="Pedido documentado, mas valor moderado",
        ),
        risks=[
            RiskItem(
                title="Inversao do onus da prova",
                level=RiskLevel.HIGH,
                conclusion=ConfidentConclusion(
                    statement="Provavel inversao (CDC)",
                    confidence=0.8,
                    reasoning="Relacao de consumo caracterizada",
                ),
                financial_exposure="R$ 20.000,00",
            )
        ],
    )
    strategy = StrategyPlan(
        overall_approach=ConfidentConclusion(
            statement="Postura hibrida: contestar e negociar",
            confidence=0.65,
            reasoning="Risco medio com exposicao limitada",
        ),
        settlement=ConfidentConclusion(
            statement="Acordo recomendado ate R$ 8.000,00",
            confidence=0.6,
            reasoning="Evita sucumbencia e custo processual",
        ),
        next_actions=[
            RecommendedAction(
                action="Verificar prazo de contestacao",
                priority=ActionPriority.URGENT,
                rationale="Prazo processual improrrogavel",
            )
        ],
        missing_information=["contrato assinado", "faturas do periodo"],
    )
    return {
        LawsuitClassification: classification,
        LawsuitExtraction: extraction,
        LegalAnalysis: analysis,
        RiskAssessment: risk,
        StrategyPlan: strategy,
    }


def _rag() -> RagPipeline:
    return RagPipeline(
        embedder=MockEmbeddingClient(), store=InMemoryVectorStore(), default_k=3
    )


async def test_graph_runs_end_to_end() -> None:
    llm = MockLLMClient(responses=_canned_responses())
    graph = build_analysis_graph(llm, _rag())

    result = await graph.ainvoke(AnalysisState(document=_document()))
    state = AnalysisState(**result)

    assert state.errors == []
    assert state.chunks, "document must be indexed"
    assert state.classification.lawsuit_type is LawsuitType.CONSUMER
    assert state.extraction.claim_value.amount == 20000.0
    assert state.legal_analysis.claims[0].legal_basis == "CDC art. 42"
    assert state.risk.overall_level is RiskLevel.MEDIUM
    assert state.strategy.next_actions[0].priority is ActionPriority.URGENT

    # observability: one trace per agent (risk/strategy ran in parallel)
    assert {t.agent for t in state.traces} == {
        "classifier",
        "entity_extraction",
        "legal_analysis",
        "risk_assessment",
        "strategy",
    }
    assert all(t.status is AgentStatus.SUCCESS for t in state.traces)
    assert all(t.llm_meta is not None for t in state.traces)
    assert all(t.llm_meta.prompt_version for t in state.traces)

    # composed report
    report = state.report
    assert report is not None
    assert report.executive_summary
    assert report.possible_settlement.statement.startswith("Acordo")
    assert "contrato assinado" in report.missing_information
    assert "judge" in report.missing_information  # from extraction schema
    assert 0.0 < report.confidence_level <= 1.0
    assert report.metrics.agents_run == 5
    assert report.metrics.total_tokens > 0
    assert "classifier:v1.0" in report.metrics.prompt_versions


async def test_graph_halts_on_agent_failure() -> None:
    class ExplodingLLM(MockLLMClient):
        async def parse(self, **kwargs):  # type: ignore[override]
            raise RuntimeError("provider unavailable")

    graph = build_analysis_graph(ExplodingLLM(), _rag())
    result = await graph.ainvoke(AnalysisState(document=_document()))
    state = AnalysisState(**result)

    assert len(state.errors) == 1  # classifier failed...
    assert state.errors[0].startswith("classifier:")
    assert state.extraction is None  # ...and downstream agents never ran
    assert state.legal_analysis is None
    assert state.report is None  # compose never reached
    assert state.traces[0].status is AgentStatus.FAILED
    assert state.traces[0].error is not None


async def test_partial_report_when_one_branch_fails() -> None:
    """If only the risk branch fails, we still deliver a partial report."""

    class RiskFailsLLM(MockLLMClient):
        async def parse(self, *, schema, **kwargs):  # type: ignore[override]
            if schema is RiskAssessment:
                raise RuntimeError("risk provider timeout")
            return await super().parse(schema=schema, **kwargs)

    llm = RiskFailsLLM(responses=_canned_responses())
    graph = build_analysis_graph(llm, _rag())
    result = await graph.ainvoke(AnalysisState(document=_document()))
    state = AnalysisState(**result)

    assert state.errors and state.errors[0].startswith("risk_assessment:")
    assert state.report is not None  # partial delivery, not total failure
    assert state.report.legal_risks is None
    assert state.report.suggested_strategy is not None
    assert "ATENCAO" in state.report.ai_reasoning  # failure disclosed


async def test_prompt_placeholders_render() -> None:
    """Prompts must not raise KeyError on their declared placeholders."""
    from app.prompts.classifier import CLASSIFIER_PROMPT
    from app.prompts.extraction import EXTRACTION_PROMPT
    from app.prompts.legal_analysis import LEGAL_ANALYSIS_PROMPT

    for template in (CLASSIFIER_PROMPT, EXTRACTION_PROMPT):
        rendered = template.render_user(language="pt", context="CTX")
        assert "CTX" in rendered

    assert "consumer" in LEGAL_ANALYSIS_PROMPT.system.format(lawsuit_type="consumer")
    assert "CTX" in LEGAL_ANALYSIS_PROMPT.render_user(language="pt", context="CTX")
