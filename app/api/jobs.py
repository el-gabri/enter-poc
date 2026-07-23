"""In-process async job manager.

Why in-process (see ADR 0009): a single-node deployment needs no broker.
The manager's public surface (submit / get) is broker-agnostic, so moving
to Redis + workers later replaces this class, not its callers.

Progress tracking uses LangGraph value streaming: after each superstep we
inspect which state fields became non-null and mark the corresponding
pipeline stage as done - the UI polls this to animate agent execution.
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from app.api.schemas import JobState, StageState, StageStatus
from app.core.logging import get_logger
from app.ingestion.service import DocumentIngestionService
from app.observability.store import RunRecord, RunStore
from app.orchestration.state import AnalysisState
from app.schemas.report import RunMetrics

logger = get_logger(__name__)

# Ordered pipeline stages and the state predicate that marks each as done.
STAGE_PREDICATES: list[tuple[str, str]] = [
    ("index", "chunks"),
    ("classify", "classification"),
    ("extract", "extraction"),
    ("analyze", "legal_analysis"),
    ("risk", "risk"),
    ("strategy", "strategy"),
    ("compose", "report"),
]


@dataclass
class Job:
    job_id: str
    filename: str
    state: JobState = JobState.QUEUED
    done_stages: set[str] = field(default_factory=set)
    errors: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None
    result: AnalysisState | None = None

    def stages(self) -> list[StageStatus]:
        statuses: list[StageStatus] = []
        running_assigned = False
        for name, _ in STAGE_PREDICATES:
            if name in self.done_stages:
                state = StageState.DONE
            elif self.state is JobState.RUNNING and not running_assigned:
                state = StageState.RUNNING
                running_assigned = True
            else:
                state = StageState.PENDING
            statuses.append(StageStatus(name=name, state=state))
        return statuses


class AnalysisJobManager:
    """Owns job lifecycle: accept upload, run pipeline, expose status."""

    def __init__(
        self,
        ingestion: DocumentIngestionService,
        graph: object,
        run_store: RunStore,
        uploads_dir: Path,
    ) -> None:
        self._ingestion = ingestion
        self._graph = graph
        self._run_store = run_store
        self._uploads_dir = uploads_dir
        self._uploads_dir.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, Job] = {}

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    async def submit(self, filename: str, content: bytes) -> Job:
        job = Job(job_id=uuid.uuid4().hex, filename=filename)
        self._jobs[job.job_id] = job
        pdf_path = self._uploads_dir / f"{job.job_id}.pdf"
        await asyncio.to_thread(pdf_path.write_bytes, content)
        asyncio.create_task(self._execute(job, pdf_path))
        logger.info("job_submitted", job_id=job.job_id, filename=filename)
        return job

    async def _execute(self, job: Job, pdf_path: Path) -> None:
        job.state = JobState.RUNNING
        try:
            document = await self._ingestion.ingest(pdf_path)
            last_state: AnalysisState | None = None
            async for chunk in self._graph.astream(  # type: ignore[attr-defined]
                AnalysisState(document=document), stream_mode="values"
            ):
                last_state = AnalysisState(**chunk) if isinstance(chunk, dict) else chunk
                self._update_stages(job, last_state)
            job.result = last_state
            job.errors = list(last_state.errors) if last_state else ["no result"]
            job.state = JobState.SUCCEEDED if not job.errors else JobState.FAILED
        except Exception as exc:
            logger.exception("job_crashed", job_id=job.job_id)
            job.errors.append(f"{type(exc).__name__}: {exc}")
            job.state = JobState.FAILED
        finally:
            job.finished_at = datetime.now(timezone.utc)
            self._persist_run(job)

    def _update_stages(self, job: Job, state: AnalysisState) -> None:
        for stage_name, state_field in STAGE_PREDICATES:
            value = getattr(state, state_field)
            if value:
                job.done_stages.add(stage_name)

    def _persist_run(self, job: Job) -> None:
        metrics = (
            job.result.report.metrics
            if job.result and job.result.report
            else RunMetrics()
        )
        traces = job.result.traces if job.result else []
        doc_id = job.result.document.doc_id if job.result else ""
        try:
            self._run_store.append(
                RunRecord(
                    run_id=job.job_id,
                    doc_id=doc_id,
                    filename=job.filename,
                    success=job.state is JobState.SUCCEEDED,
                    errors=job.errors,
                    metrics=metrics,
                    traces=traces,
                )
            )
        except Exception:
            logger.exception("run_persist_failed", job_id=job.job_id)
