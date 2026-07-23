"""Agent framework.

An agent = role (name) + versioned prompt + output schema + context policy.
``run`` handles the invariant part: render prompt, call the LLM with the
schema, capture an AgentTrace either way. Subclasses only decide WHAT
context to build - they cannot forget observability or validation.
"""

import time
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from pydantic import BaseModel

from app.core.logging import get_logger
from app.llm.base import LLMClient
from app.prompts.base import PromptTemplate
from app.schemas.trace import AgentStatus, AgentTrace

OutputT = TypeVar("OutputT", bound=BaseModel)

logger = get_logger(__name__)


class BaseAgent(ABC, Generic[OutputT]):
    """Template method base for all analysis agents."""

    name: str
    prompt: PromptTemplate
    output_schema: type[OutputT]

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    @abstractmethod
    async def build_user_prompt(self, state: "object") -> str:
        """Assemble this agent's context (may retrieve from RAG)."""

    def system_prompt(self, state: "object") -> str:
        """System prompt; override when it depends on state."""
        return self.prompt.system

    async def run(self, state: "object") -> tuple[OutputT, AgentTrace]:
        start = time.perf_counter()
        user_prompt = await self.build_user_prompt(state)
        result = await self._llm.parse(
            system=self.system_prompt(state),
            user=user_prompt,
            schema=self.output_schema,
            prompt_version=f"{self.prompt.name}:{self.prompt.version}",
        )
        trace = AgentTrace(
            agent=self.name,
            status=AgentStatus.SUCCESS,
            duration_ms=(time.perf_counter() - start) * 1000,
            llm_meta=result.meta,
        )
        logger.info(
            "agent_completed",
            agent=self.name,
            duration_ms=round(trace.duration_ms, 1),
            cost_usd=result.meta.cost_usd,
        )
        return result.data, trace

    def failure_trace(self, error: Exception, duration_ms: float) -> AgentTrace:
        return AgentTrace(
            agent=self.name,
            status=AgentStatus.FAILED,
            duration_ms=duration_ms,
            error=f"{type(error).__name__}: {error}",
        )
