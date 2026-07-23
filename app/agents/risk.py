"""Risk assessment agent."""

from app.agents.base import BaseAgent
from app.agents.context import format_context, retrieve_for_queries
from app.llm.base import LLMClient
from app.prompts.risk import RISK_PROMPT
from app.rag.pipeline import RagPipeline
from app.schemas.risk import RiskAssessment

RISK_QUERIES = [
    "valor da causa condenacao indenizacao multa",
    "tutela de urgencia liminar antecipacao",
    "inversao do onus da prova hipossuficiencia",
    "danos morais materiais lucros cessantes",
]

MAX_JSON_CHARS = 6_000


class RiskAssessmentAgent(BaseAgent[RiskAssessment]):
    """Estimates legal and financial exposure for the defendant."""

    name = "risk_assessment"
    prompt = RISK_PROMPT
    output_schema = RiskAssessment

    def __init__(self, llm: LLMClient, rag: RagPipeline) -> None:
        super().__init__(llm)
        self._rag = rag

    async def build_user_prompt(self, state: object) -> str:
        document = state.document  # type: ignore[attr-defined]
        retrieved = await retrieve_for_queries(
            self._rag, doc_id=document.doc_id, queries=RISK_QUERIES
        )
        extraction = state.extraction  # type: ignore[attr-defined]
        analysis = state.legal_analysis  # type: ignore[attr-defined]
        return self.prompt.render_user(
            language=document.language,
            context=format_context(document, retrieved),
            extraction_json=_dump(extraction),
            analysis_json=_dump(analysis),
        )


def _dump(model: object) -> str:
    if model is None:
        return "(indisponivel)"
    return model.model_dump_json()[:MAX_JSON_CHARS]  # type: ignore[attr-defined]
