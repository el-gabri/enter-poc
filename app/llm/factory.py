"""Composition root for LLM clients.

The ONLY module allowed to know which concrete providers exist. Everything
else depends on the ``LLMClient`` protocol.
"""

from app.core.config import LLMProvider, Settings
from app.llm.base import LLMClient
from app.llm.mock_client import MockLLMClient
from app.llm.openai_client import OpenAIClient


def create_llm_client(settings: Settings) -> LLMClient:
    """Build the configured LLM client.

    Raises:
        ValueError: if the provider needs credentials that are missing.
    """
    if settings.llm_provider is LLMProvider.MOCK:
        return MockLLMClient()

    if settings.llm_provider is LLMProvider.OPENAI:
        if not settings.openai_api_key:
            raise ValueError(
                "LITIGATION_OPENAI_API_KEY is required when "
                "LITIGATION_LLM_PROVIDER=openai. Set it in .env or switch "
                "the provider to 'mock'."
            )
        return OpenAIClient(api_key=settings.openai_api_key, model=settings.llm_model)

    raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")
