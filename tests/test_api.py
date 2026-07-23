"""API tests: full lifecycle against the ASGI app with mock providers."""

import asyncio
from pathlib import Path

import fitz
import httpx
import pytest

from app.api.main import create_app
from app.core.config import LLMProvider, Settings, VectorStoreBackend


def _pdf_bytes() -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    rect = fitz.Rect(72, 72, page.rect.width - 72, page.rect.height - 72)
    page.insert_textbox(
        rect,
        "DOS FATOS\n\nCobrancas indevidas do Banco Exemplo S.A.\n\n"
        "DOS PEDIDOS\n\nDanos morais de R$ 20.000,00.",
        fontsize=11,
    )
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture
async def client(tmp_path: Path):
    settings = Settings(
        llm_provider=LLMProvider.MOCK,
        vector_store=VectorStoreBackend.MEMORY,
        data_dir=tmp_path / "data",
        _env_file=None,
    )
    app = create_app(settings)
    # httpx.ASGITransport does not run startup events; enter the lifespan
    # explicitly so app.state is populated exactly as in production.
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            yield client


async def _wait_done(client: httpx.AsyncClient, job_id: str, timeout: float = 15.0) -> dict:
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        response = await client.get(f"/analyses/{job_id}")
        payload = response.json()
        if payload["state"] in ("succeeded", "failed"):
            return payload
        await asyncio.sleep(0.05)
    raise TimeoutError("job did not finish in time")


async def test_health(client: httpx.AsyncClient) -> None:
    response = await client.get("/health")
    assert response.status_code == 200


async def test_full_analysis_lifecycle(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/analyses", files={"file": ("peticao.pdf", _pdf_bytes(), "application/pdf")}
    )
    assert response.status_code == 202
    job_id = response.json()["job_id"]

    # report not ready yet -> 409 (or already done if very fast; poll first)
    status = await _wait_done(client, job_id)
    assert status["state"] == "succeeded"
    assert all(s["state"] == "done" for s in status["stages"])
    assert [s["name"] for s in status["stages"]][0] == "index"

    report_response = await client.get(f"/analyses/{job_id}/report")
    assert report_response.status_code == 200
    report = report_response.json()
    assert report["metrics"]["agents_run"] == 5
    assert report["ai_reasoning"].startswith("Como esta analise")

    # run history persisted
    runs = (await client.get("/runs")).json()
    assert len(runs) == 1
    assert runs[0]["run_id"] == job_id
    totals = (await client.get("/runs/totals")).json()
    assert totals["runs"] == 1
    assert totals["failures"] == 0


async def test_rejects_non_pdf_upload(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/analyses", files={"file": ("notes.txt", b"hello", "text/plain")}
    )
    assert response.status_code == 422

    response = await client.post(
        "/analyses", files={"file": ("fake.pdf", b"not a pdf", "application/pdf")}
    )
    assert response.status_code == 422


async def test_unknown_job_is_404(client: httpx.AsyncClient) -> None:
    assert (await client.get("/analyses/nope")).status_code == 404
    assert (await client.get("/analyses/nope/report")).status_code == 404
