"""Prompt template primitive.

Prompts are versioned artifacts: the version string travels with every LLM
call (see ``LLMCallMetadata.prompt_version``), so quality regressions can be
correlated with prompt changes in the observability layer.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptTemplate:
    """An immutable, versioned prompt."""

    name: str
    version: str
    system: str
    user_template: str

    def render_user(self, **kwargs: str) -> str:
        return self.user_template.format(**kwargs)
