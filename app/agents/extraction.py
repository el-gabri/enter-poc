"""Structured entity extraction agent."""

from app.agents.base import BaseAgent, BuiltAgentPrompt
from app.agents.context import format_retrieval_bundle, retrieve_for_queries_with_trace
from app.llm.base import LLMClient
from app.prompts.extraction import EXTRACTION_PROMPT
from app.rag.pipeline import RagPipeline
from app.schemas.lawsuit import LawsuitExtraction

# Targeted queries: each maps to fields of LawsuitExtraction. Retrieval is
# query-driven so a 200-page filing costs the same context as a 10-page one.
EXTRACTION_QUERIES = [
    "numero do processo vara comarca tribunal juiz",
    "valor da causa",
    "autor reu partes qualificacao CPF CNPJ advogado OAB",
    "prazos datas audiencia citacao intimacao",
    "pedidos requerimentos condenacao",
    "fundamentos juridicos artigos leis codigo",
]


class EntityExtractionAgent(BaseAgent[LawsuitExtraction]):
    """Extracts parties, court, values, dates and requests."""

    name = "entity_extraction"
    prompt = EXTRACTION_PROMPT
    output_schema = LawsuitExtraction

    def __init__(self, llm: LLMClient, rag: RagPipeline) -> None:
        super().__init__(llm)
        self._rag = rag

    async def build_user_prompt(self, state: object) -> BuiltAgentPrompt:
        document = state.document  # type: ignore[attr-defined]
        bundle = await retrieve_for_queries_with_trace(
            self._rag,
            doc_id=document.doc_id,
            queries=EXTRACTION_QUERIES,
            agent=self.name,
        )
        context, retrievals = format_retrieval_bundle(
            document,
            bundle,
            security_assessment=getattr(state, "security_assessment", None),
        )
        return BuiltAgentPrompt(
            text=self.prompt.render_user(language=document.language, context=context),
            retrievals=retrievals,
        )
