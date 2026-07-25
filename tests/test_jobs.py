"""Tests for bounded upload persistence and partial job statuses."""

from io import BytesIO
from pathlib import Path

import pytest
from fastapi import UploadFile

from app.api.jobs import Job, UploadTooLargeError, _write_upload_in_chunks
from app.api.schemas import JobState, StageState


async def test_upload_is_written_in_chunks(tmp_path: Path) -> None:
    path = tmp_path / "upload.pdf"
    upload = UploadFile(filename="upload.pdf", file=BytesIO(b"%PDF-1.7 content"))

    header = await _write_upload_in_chunks(upload, path, max_upload_bytes=1024)

    assert header == b"%PDF"
    assert path.read_bytes() == b"%PDF-1.7 content"


async def test_upload_over_limit_is_rejected_before_writing(tmp_path: Path) -> None:
    path = tmp_path / "too-large.pdf"
    upload = UploadFile(filename="too-large.pdf", file=BytesIO(b"%PDF-1.7 content"))

    with pytest.raises(UploadTooLargeError):
        await _write_upload_in_chunks(upload, path, max_upload_bytes=4)

    assert not path.exists()


def test_partial_job_exposes_failed_stage_without_hiding_completed_work() -> None:
    job = Job(job_id="job", filename="x.pdf", state=JobState.PARTIAL)
    job.done_stages.update({"index", "compose"})
    job.failed_stages.add("classify")

    states = {stage.name: stage.state for stage in job.stages()}

    assert states["index"] is StageState.DONE
    assert states["classify"] is StageState.FAILED
    assert states["compose"] is StageState.DONE
