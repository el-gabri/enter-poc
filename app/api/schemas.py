"""API DTOs - the HTTP contract, separate from domain schemas."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class JobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class StageState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"


class StageStatus(BaseModel):
    name: str
    state: StageState


class JobStatus(BaseModel):
    """Progress snapshot returned by GET /analyses/{job_id}."""

    job_id: str
    filename: str
    state: JobState
    stages: list[StageStatus] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    created_at: datetime
    finished_at: datetime | None = None


class JobCreated(BaseModel):
    job_id: str
    status_url: str
