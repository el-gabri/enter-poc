"""Execution traces - the observability record of a pipeline run."""

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field

from app.llm.base import LLMCallMetadata


class AgentStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"


class AgentTrace(BaseModel):
    """What one agent did during a run: timing, cost, outcome."""

    agent: str
    status: AgentStatus
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    duration_ms: float = 0.0
    llm_meta: LLMCallMetadata | None = None
    error: str | None = None
