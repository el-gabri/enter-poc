# ADR 0004: Async-first LLM and I/O layer

Status: accepted · Date: 2026-07-23

## Context

A full analysis makes 6+ LLM calls plus embeddings and external API calls
(DataJud). FastAPI and LangGraph are async-native. Retrofitting async into a
sync codebase is one of the most expensive refactors in LLM applications.

## Decision

All I/O-bound interfaces (`LLMClient`, vector store, external APIs) are
`async` from day one. Sync entry points (Streamlit) bridge via
`asyncio.run`.

## Consequences

- (+) Independent agents (e.g. risk + strategy retrieval) can run
  concurrently; the API stays non-blocking under load.
- (+) No future migration cost.
- (-) Slightly more ceremony in tests (pytest-asyncio) and in Streamlit.
