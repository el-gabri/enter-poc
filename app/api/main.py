"""FastAPI application factory.

The lifespan is the composition root of the running service: every
dependency is built once from Settings and attached to app.state - routes
never construct infrastructure.

Run locally:
    uvicorn app.api.main:app --reload
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.jobs import AnalysisJobManager
from app.api.routes import router
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging
from app.ingestion.ocr import create_default_ocr_engine
from app.ingestion.service import DocumentIngestionService
from app.llm.factory import create_llm_client
from app.observability.store import RunStore
from app.orchestration.graph import build_analysis_graph
from app.rag.factory import create_rag_pipeline


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configure_logging(settings.log_level)
        llm = create_llm_client(settings)
        rag = create_rag_pipeline(settings)
        run_store = RunStore(settings.data_dir / "runs.jsonl")
        app.state.run_store = run_store
        app.state.job_manager = AnalysisJobManager(
            ingestion=DocumentIngestionService(ocr_engine=create_default_ocr_engine()),
            graph=build_analysis_graph(llm, rag),
            run_store=run_store,
            uploads_dir=settings.uploads_dir,
        )
        yield

    app = FastAPI(
        title="AI Litigation Copilot API",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:8501"],  # Streamlit frontend
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    return app


app = create_app()
