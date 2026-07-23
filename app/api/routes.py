"""REST routes."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile

from app.api.jobs import AnalysisJobManager, Job
from app.api.schemas import JobCreated, JobState, JobStatus
from app.observability.store import RunStore
from app.schemas.report import LitigationReport

router = APIRouter()

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB


def get_job_manager(request: Request) -> AnalysisJobManager:
    return request.app.state.job_manager


def get_run_store(request: Request) -> RunStore:
    return request.app.state.run_store


JobManagerDep = Annotated[AnalysisJobManager, Depends(get_job_manager)]
RunStoreDep = Annotated[RunStore, Depends(get_run_store)]


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@router.post("/analyses", response_model=JobCreated, status_code=202)
async def create_analysis(file: UploadFile, manager: JobManagerDep) -> JobCreated:
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=422, detail="Only PDF files are accepted")
    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds 20 MB limit")
    if not content.startswith(b"%PDF"):
        raise HTTPException(status_code=422, detail="File is not a valid PDF")

    job = await manager.submit(file.filename or "upload.pdf", content)
    return JobCreated(job_id=job.job_id, status_url=f"/analyses/{job.job_id}")


def _get_job_or_404(manager: AnalysisJobManager, job_id: str) -> Job:
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.get("/analyses/{job_id}", response_model=JobStatus)
async def get_analysis(job_id: str, manager: JobManagerDep) -> JobStatus:
    job = _get_job_or_404(manager, job_id)
    return JobStatus(
        job_id=job.job_id,
        filename=job.filename,
        state=job.state,
        stages=job.stages(),
        errors=job.errors,
        created_at=job.created_at,
        finished_at=job.finished_at,
    )


@router.get("/analyses/{job_id}/report", response_model=LitigationReport)
async def get_report(job_id: str, manager: JobManagerDep) -> LitigationReport:
    job = _get_job_or_404(manager, job_id)
    if job.state in (JobState.QUEUED, JobState.RUNNING):
        raise HTTPException(status_code=409, detail="Analysis still in progress")
    if job.result is None or job.result.report is None:
        raise HTTPException(status_code=422, detail={"errors": job.errors})
    return job.result.report


@router.get("/runs")
async def list_runs(store: RunStoreDep, limit: int = 20) -> list[dict]:
    return [
        record.model_dump(mode="json", exclude={"traces"})
        for record in store.list_runs(limit=limit)
    ]


@router.get("/runs/totals")
async def run_totals(store: RunStoreDep) -> dict:
    return store.totals()
