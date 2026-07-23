"""LLM provider abstraction.

Public surface:
- ``LLMClient`` protocol (what the rest of the app depends on)
- ``create_llm_client`` factory (the only place that knows concrete providers)
"""

from app.llm.base import LLMCallMetadata, LLMClient, ParsedResult, TextResult, TokenUsage
from app.llm.factory import create_llm_client

__all__ = [
    "LLMCallMetadata",
    "LLMClient",
    "ParsedResult",
    "TextResult",
    "TokenUsage",
    "create_llm_client",
]
