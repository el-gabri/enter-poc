"""Strategy agent."""

from app.agents.base import BaseAgent
from app.agents.context import format_context, retrieve_for_queries
from app.agents.risk import _dump
from app.llm.base import LLMClient
from app.prompts.strategy import STRATEGY_PROMPT
from app.rag.pipeline import RagPipeline
from app.schemas.strategy import StrategyPlan

STRATEGY_QUERIES = [
    "pedidos requerimentos condenacao",
    "fundamentos juridicos artigos leis",
    "provas documentos anexos",
    "prazos audiencia citacao contestacao",
]


class StrategyAgent(BaseAgent[StrategyPlan]):
    """Proposes initial defense strategy, settlement posture and actions."""

    name = "strategy"
    prompt = STRATEGY_PROMPT
    output_schema = StrategyPlan

    def __init__(self, llm: LLMClient, rag: RagPipeline) -> None:
        super().__init__(llm)
        self._rag = rag

    async def build_user_prompt(self, state: object) -> str:
        document = state.document  # type: ignore[attr-defined]
        retrieved = await retrieve_for_queries(
            self._rag, doc_id=document.doc_id, queries=STRATEGY_QUERIES
        )
        return self.prompt.render_user(
            language=document.language,
            context=format_context(document, retrieved),
            extraction_json=_dump(state.extraction),  # type: ignore[attr-defined]
            analysis_json=_dump(state.legal_analysis),  # type: ignore[attr-defined]
        )
