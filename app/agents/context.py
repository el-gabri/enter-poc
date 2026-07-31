"""Context assembly helpers shared by all agents.

Agents never receive the raw document: they receive a curated context block
built from targeted retrievals (plus the document head, which carries the
case header in Brazilian petitions). Chunk ids and page numbers are embedded
in the block so the LLM can produce verifiable citations.
"""

from collections.abc import Iterable
from html import escape

from app.rag.pipeline import RagPipeline
from app.schemas.document import ParsedDocument
from app.schemas.rag import RetrievedChunk
from app.schemas.security import PromptInjectionAssessment
from app.security.sanitization import mask_flagged_text

MAX_CONTEXT_CHARS = 12_000
DOC_HEAD_CHARS = 2_500


def merge_retrievals(
    result_lists: Iterable[list[RetrievedChunk]],
) -> list[RetrievedChunk]:
    """Merge multiple retrieval result lists, keeping the best score per chunk."""
    best: dict[str, RetrievedChunk] = {}
    for results in result_lists:
        for item in results:
            current = best.get(item.chunk.chunk_id)
            if current is None or item.score > current.score:
                best[item.chunk.chunk_id] = item
    return sorted(best.values(), key=lambda r: r.score, reverse=True)


def format_context(
    document: ParsedDocument,
    retrieved: list[RetrievedChunk],
    max_chars: int = MAX_CONTEXT_CHARS,
    security_assessment: PromptInjectionAssessment | None = None,
) -> str:
    """Build the context block: document head + cited retrieval results."""
    first_page_text = document.pages[0].text if document.pages else ""
    first_page_text = mask_flagged_text(first_page_text, security_assessment)[
        :DOC_HEAD_CHARS
    ]
    source = document.doc_id
    parts: list[str] = [
        f"<document_excerpt source=\"{source}\" page=\"1\">\n"
        f"{first_page_text}\n"
        "</document_excerpt>"
    ]
    used = len(parts[0])
    for item in retrieved:
        chunk = item.chunk
        chunk_id = escape(chunk.chunk_id, quote=True)
        raw_section = mask_flagged_text(
            chunk.section or "sem secao", security_assessment
        )
        section = escape(raw_section, quote=True)
        chunk_text = mask_flagged_text(chunk.text, security_assessment)
        block = (
            f"<document_excerpt chunk_id=\"{chunk_id}\" section=\"{section}\" "
            f"pages=\"{chunk.page_start}-{chunk.page_end}\">\n{chunk_text}\n"
            "</document_excerpt>"
        )
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)
    return "\n\n---\n\n".join(parts)


async def retrieve_for_queries(
    rag: RagPipeline, doc_id: str, queries: list[str], k: int = 4
) -> list[RetrievedChunk]:
    """Run several targeted retrievals and merge the results."""
    results = await rag.retrieve_many(queries, doc_id=doc_id, k=k)
    return merge_retrievals(results)
