"""Analysis graph: wires agents into a LangGraph state machine.

Topology:

    START -> index -> classify -> extract -> analyze -+-> risk -----+-> compose -> END
                         |            |          |     +-> strategy -+
                         +----- on error: halt --+

Risk and strategy are independent given the legal analysis, so they run in
PARALLEL branches (the additive reducers on traces/errors make concurrent
writes safe). Compose is deterministic (no LLM, see ADR 0007) and always
runs, producing a partial report when a branch failed.

Every agent node is wrapped so that failures are recorded in state
(errors + a failed AgentTrace) instead of crashing the run.
"""

import time
from collections.abc import Awaitable, Callable

from langgraph.graph import END, START, StateGraph

from app.agents.base import BaseAgent
from app.agents.classifier import ClassifierAgent
from app.agents.extraction import EntityExtractionAgent
from app.agents.legal_analysis import LegalAnalysisAgent
from app.agents.risk import RiskAssessmentAgent
from app.agents.strategy import StrategyAgent
from app.core.logging import get_logger
from app.enrichment.datajud import DataJudClient
from app.enrichment.node import make_enrich_node
from app.llm.base import LLMClient
from app.orchestration.state import AnalysisState
from app.rag.pipeline import RagPipeline
from app.services.composer import compose_report

logger = get_logger(__name__)

NodeFn = Callable[[AnalysisState], Awaitable[dict]]


def _agent_node(agent: BaseAgent, state_field: str) -> NodeFn:
    """Wrap an agent as a graph node with uniform error handling."""

    async def node(state: AnalysisState) -> dict:
        start = time.perf_counter()
        try:
            output, trace = await agent.run(state)
            return {state_field: output, "traces": [trace]}
        except Exception as exc:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.exception("agent_failed", agent=agent.name)
            return {
                "errors": [f"{agent.name}: {type(exc).__name__}: {exc}"],
                "traces": [agent.failure_trace(exc, duration_ms)],
            }

    return node


def _halt_on_error(state: AnalysisState) -> str:
    return "halt" if state.errors else "continue"


def build_analysis_graph(
    llm: LLMClient, rag: RagPipeline, datajud: DataJudClient | None = None
):
    """Compose the analysis pipeline. Dependencies injected at the root."""

    async def index_node(state: AnalysisState) -> dict:
        try:
            chunks = await rag.index_document(state.document)
            return {"chunks": chunks}
        except Exception as exc:
            logger.exception("indexing_failed", doc_id=state.document.doc_id)
            return {"errors": [f"index: {type(exc).__name__}: {exc}"]}

    async def compose_node(state: AnalysisState) -> dict:
        # Deterministic assembly - no LLM call, cannot hallucinate.
        return {"report": compose_report(state)}

    classifier = ClassifierAgent(llm)
    extractor = EntityExtractionAgent(llm, rag)
    analyst = LegalAnalysisAgent(llm, rag)
    risk_assessor = RiskAssessmentAgent(llm, rag)
    strategist = StrategyAgent(llm, rag)

    builder = StateGraph(AnalysisState)
    builder.add_node("index", index_node)
    builder.add_node("classify", _agent_node(classifier, "classification"))
    builder.add_node("extract", _agent_node(extractor, "extraction"))
    builder.add_node("analyze", _agent_node(analyst, "legal_analysis"))
    builder.add_node("risk", _agent_node(risk_assessor, "risk"))
    builder.add_node("strategy", _agent_node(strategist, "strategy"))
    builder.add_node("enrich", make_enrich_node(datajud))
    builder.add_node("compose", compose_node)

    builder.add_edge(START, "index")
    for source, target in [("index", "classify"), ("classify", "extract")]:
        builder.add_conditional_edges(
            source, _halt_on_error, {"continue": target, "halt": END}
        )
    # fan-out 1: legal analysis and DataJud enrichment are independent
    builder.add_conditional_edges(
        "extract",
        lambda s: ["analyze", "enrich"] if not s.errors else "halt",
        {"analyze": "analyze", "enrich": "enrich", "halt": END},
    )
    # fan-out 2: risk and strategy run in parallel after the legal analysis
    builder.add_conditional_edges(
        "analyze",
        lambda s: ["risk", "strategy"] if not s.errors else "halt",
        {"risk": "risk", "strategy": "strategy", "halt": END},
    )
    # fan-in: compose waits for all branches; runs even on partial failure
    builder.add_edge(["risk", "strategy", "enrich"], "compose")
    builder.add_edge("compose", END)

    return builder.compile()
