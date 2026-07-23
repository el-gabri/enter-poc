"""Analysis graph: wires agents into a LangGraph state machine.

Topology (M3a):

    START -> index -> classify -> extract -> analyze -> END
                         |            |          |
                         +---- on error: halt ---+

Every agent node is wrapped so that failures are recorded in state
(errors + a failed AgentTrace) instead of crashing the run; conditional
edges route to END as soon as an error appears. M3b extends the graph with
risk, strategy and report composition.
"""

import time
from collections.abc import Awaitable, Callable

from langgraph.graph import END, START, StateGraph

from app.agents.base import BaseAgent
from app.agents.classifier import ClassifierAgent
from app.agents.extraction import EntityExtractionAgent
from app.agents.legal_analysis import LegalAnalysisAgent
from app.core.logging import get_logger
from app.llm.base import LLMClient
from app.orchestration.state import AnalysisState
from app.rag.pipeline import RagPipeline

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


def build_analysis_graph(llm: LLMClient, rag: RagPipeline):
    """Compose the analysis pipeline. Dependencies injected at the root."""

    async def index_node(state: AnalysisState) -> dict:
        try:
            chunks = await rag.index_document(state.document)
            return {"chunks": chunks}
        except Exception as exc:
            logger.exception("indexing_failed", doc_id=state.document.doc_id)
            return {"errors": [f"index: {type(exc).__name__}: {exc}"]}

    classifier = ClassifierAgent(llm)
    extractor = EntityExtractionAgent(llm, rag)
    analyst = LegalAnalysisAgent(llm, rag)

    builder = StateGraph(AnalysisState)
    builder.add_node("index", index_node)
    builder.add_node("classify", _agent_node(classifier, "classification"))
    builder.add_node("extract", _agent_node(extractor, "extraction"))
    builder.add_node("analyze", _agent_node(analyst, "legal_analysis"))

    builder.add_edge(START, "index")
    for source, target in [("index", "classify"), ("classify", "extract"), ("extract", "analyze")]:
        builder.add_conditional_edges(
            source, _halt_on_error, {"continue": target, "halt": END}
        )
    builder.add_edge("analyze", END)

    return builder.compile()
