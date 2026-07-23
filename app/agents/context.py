"""Context assembly helpers shared by all agents.

Agents never receive the raw document: they receive a curated context block
built from targeted retrievals (plus the document head, which carries the
case header in Brazilian petitions). Chunk ids and page numbers are embedded
in the block so the LLM can produce verifiable citations.
"""

from collections.abc import Iterable

from app.rag.pipeline import RagPipeline
from app.schemas.document import ParsedDocument
from app.schemas.rag import RetrievedChunk

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
) -> str:
    """Build the context block: document head + cited retrieval results."""
    parts: list[str] = [
        f"[INICIO DO DOCUMENTO | {document.filename} | p.1]\n"
        f"{document.full_text[:DOC_HEAD_CHARS]}"
    ]
    used = len(parts[0])
    for item in retrieved:
        chunk = item.chunk
        section = chunk.section or "sem secao"
        block = (
            f"[{chunk.chunk_id} | {section} | p.{chunk.page_start}-{chunk.page_end}]\n"
            f"{chunk.text}"
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
    results = [await rag.retrieve(query, doc_id=doc_id, k=k) for query in queries]
    return merge_retrievals(results)
