"""Lawsuit type classifier agent."""

from app.agents.base import BaseAgent
from app.agents.context import format_context
from app.llm.base import LLMClient
from app.prompts.classifier import CLASSIFIER_PROMPT
from app.schemas.analysis import LawsuitClassification


class ClassifierAgent(BaseAgent[LawsuitClassification]):
    """Determines the area of law from the opening of the petition.

    No retrieval needed: Brazilian petitions announce their nature in the
    first pages (addressed court, cause of action, invoked statutes).
    """

    name = "classifier"
    prompt = CLASSIFIER_PROMPT
    output_schema = LawsuitClassification

    def __init__(self, llm: LLMClient) -> None:
        super().__init__(llm)

    async def build_user_prompt(self, state: object) -> str:
        document = state.document  # type: ignore[attr-defined]
        context = format_context(document, retrieved=[])
        return self.prompt.render_user(language=document.language, context=context)
