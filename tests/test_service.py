"""End-to-end test: real PDF file -> LawsuitAnalysisService -> report.

Uses the mock provider via Settings, exactly as a developer without an API
key (or CI) would run the product.
"""

from pathlib import Path

import fitz

from app.core.config import LLMProvider, Settings, VectorStoreBackend
from app.services.analysis import create_analysis_service


def _write_pdf(path: Path) -> Path:
    doc = fitz.open()
    page = doc.new_page()
    rect = fitz.Rect(72, 72, page.rect.width - 72, page.rect.height - 72)
    page.insert_textbox(
        rect,
        "DOS FATOS\n\nA autora sofreu cobrancas indevidas do Banco Exemplo "
        "S.A. em sua fatura.\n\nDOS PEDIDOS\n\nRequer indenizacao por danos "
        "morais de R$ 20.000,00 e restituicao em dobro dos valores.",
        fontsize=11,
    )
    doc.save(path)
    doc.close()
    return path


async def test_service_analyzes_pdf_offline(tmp_path: Path) -> None:
    settings = Settings(
        llm_provider=LLMProvider.MOCK,
        vector_store=VectorStoreBackend.MEMORY,
        data_dir=tmp_path / "data",
        _env_file=None,
    )
    service = create_analysis_service(settings)

    state = await service.analyze(_write_pdf(tmp_path / "peticao.pdf"))

    assert state.errors == []
    assert state.report is not None
    assert state.report.doc_id == state.document.doc_id
    assert state.report.metrics.agents_run == 5
    # mock provider synthesizes placeholder outputs; the pipeline shape holds
    assert state.report.ai_reasoning.startswith("Como esta analise")
