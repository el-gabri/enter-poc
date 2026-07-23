"""Tests for chunking, embeddings, vector stores and retrieval."""

import pytest

from app.rag.chunking import SectionAwareChunker, is_heading
from app.rag.embeddings import MockEmbeddingClient
from app.rag.pipeline import RagPipeline
from app.rag.vector_store import InMemoryVectorStore, _cosine
from app.schemas.document import DocumentPage, ExtractionMethod, ParsedDocument

FACTS = (
    "A autora contratou os servicos do banco reu em janeiro de 2024. "
    "Apos o contrato, verificou cobrancas indevidas em sua fatura mensal, "
    "totalizando prejuizo material significativo ao longo de seis meses."
)
LAW = (
    "Aplica-se o Codigo de Defesa do Consumidor, artigo 42, paragrafo unico, "
    "que garante a repeticao do indebito em dobro nas cobrancas indevidas."
)
REQUESTS = (
    "Requer a condenacao do reu ao pagamento de indenizacao por danos morais "
    "no valor de R$ 20.000,00 e a restituicao em dobro dos valores cobrados."
)


def _petition() -> ParsedDocument:
    return ParsedDocument(
        filename="peticao.pdf",
        pages=[
            DocumentPage(number=1, text=f"DOS FATOS\n\n{FACTS}"),
            DocumentPage(number=2, text=f"DO DIREITO\n\n{LAW}"),
            DocumentPage(number=3, text=f"DOS PEDIDOS\n\n{REQUESTS}"),
        ],
        language="pt",
        extraction_method=ExtractionMethod.NATIVE_TEXT,
    )


def test_heading_detection() -> None:
    assert is_heading("DOS FATOS")
    assert is_heading("II - DO DIREITO")
    assert not is_heading("A autora contratou os servicos do banco reu.")
    assert not is_heading("")
    assert not is_heading("R$ 50.000,00")  # too few letters to qualify


def test_chunker_respects_sections_and_provenance() -> None:
    chunks = SectionAwareChunker(target_chars=1200, overlap_chars=100).chunk(_petition())

    sections = {c.section for c in chunks}
    assert sections == {"DOS FATOS", "DO DIREITO", "DOS PEDIDOS"}
    # no chunk mixes sections; provenance points at the right page
    pedidos = next(c for c in chunks if c.section == "DOS PEDIDOS")
    assert pedidos.page_start == 3
    assert "danos morais" in pedidos.text
    # ids are stable and content-scoped
    assert all(c.chunk_id.startswith(c.doc_id) for c in chunks)


def test_chunker_splits_oversized_sections_with_overlap() -> None:
    long_text = " ".join(f"paragrafo numero {i} do documento juridico." for i in range(200))
    doc = ParsedDocument(
        filename="x.pdf",
        pages=[DocumentPage(number=1, text=f"DOS FATOS\n\n{long_text}")],
        language="pt",
        extraction_method=ExtractionMethod.NATIVE_TEXT,
    )
    chunker = SectionAwareChunker(target_chars=500, overlap_chars=80)
    chunks = chunker.chunk(doc)

    assert len(chunks) > 1
    assert all(len(c.text) <= 500 + 100 for c in chunks)  # target + heading prefix slack


def test_chunker_rejects_invalid_overlap() -> None:
    with pytest.raises(ValueError):
        SectionAwareChunker(target_chars=100, overlap_chars=100)


async def test_mock_embeddings_are_deterministic_and_semanticish() -> None:
    embedder = MockEmbeddingClient()
    [a1] = await embedder.embed(["cobranca indevida danos morais"])
    [a2] = await embedder.embed(["cobranca indevida danos morais"])
    [related] = await embedder.embed(["indenizacao por danos morais"])
    [unrelated] = await embedder.embed(["contrato de trabalho ferias salario"])

    assert a1 == a2  # deterministic
    assert _cosine(a1, related) > _cosine(a1, unrelated)  # shared vocab ranks higher


async def test_pipeline_indexes_and_retrieves_relevant_section() -> None:
    pipeline = RagPipeline(
        embedder=MockEmbeddingClient(), store=InMemoryVectorStore(), default_k=2
    )
    doc = _petition()
    chunks = await pipeline.index_document(doc)
    assert len(chunks) == 3

    results = await pipeline.retrieve(
        "indenizacao por danos morais valor pedido", doc_id=doc.doc_id
    )
    assert results
    assert results[0].chunk.section == "DOS PEDIDOS"


async def test_retrieval_is_isolated_per_document() -> None:
    pipeline = RagPipeline(
        embedder=MockEmbeddingClient(), store=InMemoryVectorStore(), default_k=5
    )
    doc = _petition()
    await pipeline.index_document(doc)

    results = await pipeline.retrieve("danos morais", doc_id="other-doc-id")
    assert results == []  # never leak chunks across documents


async def test_chroma_adapter_roundtrip(tmp_path) -> None:
    chromadb = pytest.importorskip("chromadb")  # noqa: F841
    from app.rag.vector_store import ChromaVectorStore

    pipeline = RagPipeline(
        embedder=MockEmbeddingClient(),
        store=ChromaVectorStore(persist_dir=tmp_path / "chroma"),
        default_k=2,
    )
    doc = _petition()
    await pipeline.index_document(doc)

    results = await pipeline.retrieve("restituicao em dobro danos morais", doc_id=doc.doc_id)
    assert results
    assert results[0].chunk.doc_id == doc.doc_id
    assert results[0].chunk.page_start >= 1
