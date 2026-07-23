# AI Litigation Copilot

AI assistant that analyzes lawsuit PDFs and prepares an initial legal
strategy for legal teams. Built to demonstrate production-grade AI
engineering: multi-agent orchestration (LangGraph), RAG, structured outputs,
observability, evaluation and explainability.

> Not a replacement for lawyers. Every output includes confidence scores,
> reasoning and source citations so a human can verify each conclusion.

## Problem

Legal teams receiving a new lawsuit spend hours on triage: identifying
parties, deadlines, claims, estimating risk and drafting an initial
strategy. This copilot compresses that to minutes while keeping every
conclusion auditable.

## Architecture

See [docs/architecture.md](docs/architecture.md) and the ADRs in
[docs/adr/](docs/adr/).

```
Browser -> Streamlit -> FastAPI -> LangGraph Orchestrator
        -> [Parser -> Classifier -> Extraction -> Legal Analysis
            -> Risk -> Strategy -> Report Composer]
        -> RAG (ChromaDB) + DataJud (CNJ) enrichment
        -> Structured report (MD / PDF / DOCX)
```

## Features

- PDF ingestion with OCR fallback and language detection (PT-BR first)
- Lawsuit classification (consumer, labor, civil, banking, insurance, health)
- Structured entity extraction (parties, court, claim value, deadlines...)
- Executive summary, timeline, risk assessment, suggested strategy
- Confidence score + reasoning + source citations on every conclusion
- DataJud (CNJ) case lookup for enrichment and validation
- Per-agent observability: latency, tokens, cost, model, prompt version
- Evaluation harness: faithfulness, groundedness, hallucination, completeness
- Mock LLM mode: run the full pipeline offline, free and deterministic

## Quickstart

```bash
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -e ".[dev]"
copy .env.example .env                            # then fill in your key
pytest                                            # runs offline (mock mode)
```

## Project status

Built milestone by milestone; each design decision recorded as an ADR.

- [x] M0 - Foundation: config, logging, LLM provider abstraction, ADRs
- [x] M1 - Ingestion: PDF -> text, OCR, language detection, domain schemas
- [x] M2 - RAG: section-aware chunking, embeddings, ChromaDB behind a port
- [x] M3 - Agents: LangGraph pipeline (5 agents, parallel risk/strategy,
      deterministic report composer)
- [x] M4 - Observability (run store) + evaluation harness
      (`python -m app.evaluation`)
- [ ] M5 - FastAPI backend (async jobs)
- [ ] M6 - Streamlit frontend
- [ ] M7 - DataJud enrichment
- [ ] M8 - Docker, docs, screenshots
