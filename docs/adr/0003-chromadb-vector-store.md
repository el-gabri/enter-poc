# ADR 0003: ChromaDB behind a VectorStore port

Status: accepted · Date: 2026-07-23

## Context

RAG needs a vector database. Candidates: ChromaDB (embedded), FAISS
(library), Pinecone/Qdrant (managed).

## Decision

ChromaDB with local persistence for development and single-node deployment,
accessed exclusively through a `VectorStore` protocol defined by us.
Pinecone becomes a new adapter when scale demands it.

## Consequences

- (+) Zero-infrastructure start; runs inside Docker Compose.
- (+) Metadata filtering (per-document isolation) built in - FAISS would
  need extra bookkeeping for this.
- (+) Migration path to managed stores without touching agent code.
- (-) Not horizontally scalable; acceptable for the current stage and
  explicitly addressed by the port.
