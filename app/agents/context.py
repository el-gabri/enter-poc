"""Context assembly helpers shared by all agents.

Agents never receive the raw document: they receive a curated context block
built from targeted retrievals (plus the document head, which carries the
case header in Brazilian petitions). Chunk ids and page numbers are embedded
in the block so the LLM can produce verifiable citations.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from html import escape

from app.rag.pipeline import RagPipeline
from app.schemas.document import ParsedDocument
from app.schemas.rag import RetrievedChunk
from app.schemas.security import PromptInjectionAssessment
from app.schemas.trace import RetrievalTrace
from app.security.sanitization import mask_flagged_text

MAX_CONTEXT_CHARS = 12_000
DOC_HEAD_CHARS = 2_500


@dataclass(frozen=True)
class RetrievalBundle:
    """Merged results plus the raw query-level retrieval audit."""

    chunks: list[RetrievedChunk]
    traces: list[RetrievalTrace]


@dataclass(frozen=True)
class FormattedContext:
    """Prompt context and the exact retrieval chunks that reached it."""

    text: str
    included_chunk_ids: frozenset[str]
    truncated: bool


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
    return sorted(best.values(), key=lambda item: (-item.score, item.chunk.chunk_id))


def format_context(
    document: ParsedDocument,
    retrieved: list[RetrievedChunk],
    max_chars: int = MAX_CONTEXT_CHARS,
    security_assessment: PromptInjectionAssessment | None = None,
) -> str:
    """Build the context block: document head + cited retrieval results."""
    return format_context_with_audit(
        document,
        retrieved,
        max_chars=max_chars,
        security_assessment=security_assessment,
    ).text


def format_context_with_audit(
    document: ParsedDocument,
    retrieved: list[RetrievedChunk],
    max_chars: int = MAX_CONTEXT_CHARS,
    security_assessment: PromptInjectionAssessment | None = None,
) -> FormattedContext:
    """Build context and record which retrieved chunks survive truncation."""
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
    separator = "\n\n---\n\n"
    included_chunk_ids: set[str] = set()
    truncated = False
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
        added_chars = len(separator) + len(block)
        if used + added_chars > max_chars:
            truncated = True
            break
        parts.append(block)
        used += added_chars
        included_chunk_ids.add(chunk.chunk_id)
    return FormattedContext(
        text=separator.join(parts),
        included_chunk_ids=frozenset(included_chunk_ids),
        truncated=truncated,
    )


def annotate_retrieval_selection(
    traces: list[RetrievalTrace],
    merged: list[RetrievedChunk],
    formatted: FormattedContext,
) -> list[RetrievalTrace]:
    """Mark deduplication winners and chunks included in the final prompt."""
    merged_ranks = {
        item.chunk.chunk_id: rank for rank, item in enumerate(merged, start=1)
    }
    winners: dict[str, tuple[int, int, float]] = {}
    for trace_index, trace in enumerate(traces):
        for item in trace.results:
            current = winners.get(item.chunk_id)
            if current is None or item.score > current[2]:
                winners[item.chunk_id] = (trace_index, item.rank, item.score)
    annotated: list[RetrievalTrace] = []
    for trace_index, trace in enumerate(traces):
        results = []
        for item in trace.results:
            merged_rank = merged_ranks.get(item.chunk_id)
            winner = winners.get(item.chunk_id)
            selected = winner is not None and winner[:2] == (trace_index, item.rank)
            results.append(
                item.model_copy(
                    update={
                        "selected_for_merge": selected,
                        "merged_rank": merged_rank,
                        "included_in_context": (
                            selected
                            and item.chunk_id in formatted.included_chunk_ids
                        ),
                    }
                )
            )
        annotated.append(
            trace.model_copy(
                update={
                    "context_truncated": formatted.truncated,
                    "results": results,
                }
            )
        )
    return annotated


def format_retrieval_bundle(
    document: ParsedDocument,
    bundle: RetrievalBundle,
    *,
    security_assessment: PromptInjectionAssessment | None = None,
    max_chars: int = MAX_CONTEXT_CHARS,
) -> tuple[str, list[RetrievalTrace]]:
    """Format a retrieval bundle and finalize its context-selection audit."""
    formatted = format_context_with_audit(
        document,
        bundle.chunks,
        max_chars=max_chars,
        security_assessment=security_assessment,
    )
    return (
        formatted.text,
        annotate_retrieval_selection(bundle.traces, bundle.chunks, formatted),
    )


async def retrieve_for_queries(
    rag: RagPipeline, doc_id: str, queries: list[str], k: int | None = None
) -> list[RetrievedChunk]:
    """Run several targeted retrievals and merge the results."""
    results = await rag.retrieve_many(queries, doc_id=doc_id, k=k)
    return merge_retrievals(results)


async def retrieve_for_queries_with_trace(
    rag: RagPipeline,
    doc_id: str,
    queries: list[str],
    *,
    agent: str,
    k: int | None = None,
) -> RetrievalBundle:
    """Run targeted retrieval and retain every query's raw ranked results."""
    results, traces = await rag.retrieve_many_with_traces(
        queries, doc_id=doc_id, agent=agent, k=k
    )
    return RetrievalBundle(chunks=merge_retrievals(results), traces=traces)
