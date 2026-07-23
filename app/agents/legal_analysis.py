"""Legal analysis agent."""

from app.agents.base import BaseAgent
from app.agents.context import format_context, retrieve_for_queries
from app.llm.base import LLMClient
from app.prompts.legal_analysis import LEGAL_ANALYSIS_PROMPT
from app.rag.pipeline import RagPipeline
from app.schemas.analysis import LegalAnalysis

ANALYSIS_QUERIES = [
    "fatos ocorridos historico narrativa",
    "pedidos requerimentos condenacao tutela",
    "provas documentos anexos comprovantes testemunhas",
    "fundamentos juridicos artigos leis jurisprudencia sumula",
    "danos morais materiais prejuizo indenizacao",
]


class LegalAnalysisAgent(BaseAgent[LegalAnalysis]):
    """Produces summary, timeline, claim-by-claim assessment and evidence."""

    name = "legal_analysis"
    prompt = LEGAL_ANALYSIS_PROMPT
    output_schema = LegalAnalysis

    def __init__(self, llm: LLMClient, rag: RagPipeline) -> None:
        super().__init__(llm)
        self._rag = rag

    def system_prompt(self, state: object) -> str:
        classification = state.classification  # type: ignore[attr-defined]
        lawsuit_type = (
            classification.lawsuit_type.value if classification else "desconhecida"
        )
        return self.prompt.system.format(lawsuit_type=lawsuit_type)

    async def build_user_prompt(self, state: object) -> str:
        document = state.document  # type: ignore[attr-defined]
        retrieved = await retrieve_for_queries(
            self._rag, doc_id=document.doc_id, queries=ANALYSIS_QUERIES, k=5
        )
        context = format_context(document, retrieved)
        return self.prompt.render_user(language=document.language, context=context)
