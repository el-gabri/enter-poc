# ADR 0011: Retrieval traceability and ranking evaluation

Status: accepted · Date: 2026-07-31

## Context

Page-attributed citations prove that a quote exists in the uploaded document,
but they do not prove which query retrieved a chunk, its original rank and
score, or whether deduplication and prompt limits removed it. Similarity scores
alone also do not measure retrieval quality; Recall@K and related metrics need
human relevance judgments.

## Decision

1. Store a typed retrieval trace under each agent execution trace. This reuses
   LangGraph's additive trace reducer and preserves attribution across parallel
   risk and strategy branches.
2. Record each query's effective `k`, raw ranks and scores, chunk/document IDs,
   section and page span, embedding/vector/index versions, latency, failure,
   consuming-agent outcome/prompt version, and SHA-256 of the exact indexed
   text. Preserve successful sibling lookups when one query in a batch fails.
3. Never persist embeddings or full chunk text in the JSONL audit record.
   Bounded whitespace-normalized previews are disabled by default and require
   an explicit configuration opt-in covered by a retention/access policy.
4. Mark deduplication winners and the chunks that actually reached the prompt
   after the context character limit is applied.
5. Expose current and historical audits through dedicated API endpoints and a
   native Streamlit table. Keep the compact `/runs` listing free of trace data.
6. Label golden retrieval queries using relevant page ranges and short passage
   anchors rather than chunk IDs. Resolve labels against the current chunker and
   report Precision@K, Recall@K, HitRate@K, MRR@K and NDCG@K.

## Consequences

- (+) A reviewer can reconstruct query -> ranking -> merged context -> citation.
- (+) Chunk and query hashes make configuration or content drift detectable.
- (+) Relevance labels survive most chunk-size and overlap changes.
- (+) Old JSONL records remain readable because retrieval lists default empty.
- (-) Run records grow with `queries × k`; the configured `k` bounds that
  increase, while optional previews add further storage and privacy cost.
- (-) Runtime traces provide observability, not relevance truth. Formal ranking
  metrics remain an offline operation requiring reviewed labels.
- (-) The mandatory first-page document-head excerpt is prompt provenance but
  not a retrieval result, so it is intentionally outside the retrieval audit.
