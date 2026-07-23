# AI Litigation Copilot

AI assistant that analyzes Brazilian lawsuit PDFs and prepares an initial
legal strategy for legal teams - built as a production-grade demonstration
of modern AI engineering: multi-agent orchestration (LangGraph), RAG,
structured outputs, observability, evaluation and explainability.

> **Not a replacement for lawyers.** Every conclusion carries a confidence
> score, explicit reasoning and verbatim citations from the source
> document, so a human can audit every claim.

## The problem

When a company is sued, the legal team spends hours on triage: reading the
petition, identifying parties and deadlines, estimating exposure, drafting
an initial strategy. This copilot compresses that to minutes - and keeps
every conclusion verifiable.

## What it does

Upload a lawsuit PDF and get a structured report: executive summary,
lawsuit classification, extracted entities (parties, court, claim value,
deadlines), timeline, claim-by-claim assessment, risk analysis with
financial exposure, suggested defense strategy, settlement posture,
missing information, and validation of the case number against DataJud -
the official CNJ court-records API. Export as Markdown, PDF, DOCX or JSON.

## Architecture

```
Browser -> Streamlit -> FastAPI (202 + async job)
    -> LangGraph state machine:
       index -> classify -> extract --+--> analyze --+--> risk     --+
                                      |              +--> strategy --+--> compose
                                      +--> DataJud enrichment -------+
    -> RAG: section-aware chunking -> embeddings -> ChromaDB (per-doc isolation)
    -> LLM port: OpenAI adapter | Mock adapter (offline mode)
    -> Deterministic report composer -> MD / PDF / DOCX / JSON
```

Full diagram and layer map: [docs/architecture.md](docs/architecture.md).

### Design decisions (ADRs)

| ADR | Decision |
|---|---|
| [0001](docs/adr/0001-use-langgraph.md) | LangGraph over LangChain chains (conditional routing, parallelism, introspectable graph) |
| [0002](docs/adr/0002-llm-provider-abstraction.md) | Own 2-method LLM port instead of LiteLLM - observability and structured outputs are mandatory by type |
| [0003](docs/adr/0003-chromadb-vector-store.md) | ChromaDB behind a VectorStore port (Pinecone = one new adapter) |
| [0004](docs/adr/0004-async-first.md) | Async-first I/O from day one |
| [0005](docs/adr/0005-pymupdf-ocr-fallback.md) | PyMuPDF + heuristic OCR fallback (AGPL trade-off documented) |
| [0006](docs/adr/0006-section-aware-chunking.md) | Section-aware chunking for Brazilian petitions (DOS FATOS / DO DIREITO / DOS PEDIDOS) |
| [0007](docs/adr/0007-deterministic-report-composer.md) | No LLM at the last mile - the report is assembled by code |
| [0008](docs/adr/0008-citation-based-groundedness.md) | Hallucination detection by mechanical citation verification |
| [0009](docs/adr/0009-in-process-async-jobs.md) | In-process async jobs with a broker-ready interface |

### Explainability model

Every important conclusion is a `ConfidentConclusion`:

```json
{
  "statement": "Recomendado buscar acordo ate R$ 8.000,00",
  "confidence": 0.87,
  "reasoning": "O documento comprova a cobranca indevida e o CDC preve...",
  "citations": [{"quote": "cobrancas mensais indevidas", "page": 3}]
}
```

The evaluation harness verifies each citation actually occurs in the
source document - a fabricated quote is caught mechanically, not by
another LLM's opinion.

### Observability

Every LLM call returns typed metadata (provider, model, latency, tokens,
cost, prompt version) - agents physically cannot make untracked calls.
Per-run aggregates persist to a JSONL run store surfaced in the API
(`/runs`, `/runs/totals`) and the UI's cost panel.

### Evaluation

```bash
python -m app.evaluation           # golden dataset in eval_data/
```

Metrics: groundedness, hallucination rate, citation coverage, extraction
accuracy, completeness, classification accuracy, and LLM-as-judge response
quality (real provider only). Runs offline in CI with the mock provider as
a pipeline health check.

## Quickstart

### Docker (recommended)

```bash
copy .env.example .env      # fill LITIGATION_OPENAI_API_KEY (or use mock)
docker compose up --build
# UI:  http://localhost:8501
# API: http://localhost:8000/docs
```

### Local development

```bash
python -m venv .venv && .venv\Scripts\activate    # Windows
pip install -e ".[dev,frontend]"
copy .env.example .env
pytest                                            # 50+ tests, fully offline

uvicorn app.api.main:app --reload                 # terminal 1
streamlit run frontend/streamlit_app.py           # terminal 2
```

No API key? Set `LITIGATION_LLM_PROVIDER=mock` - the entire product runs
offline with deterministic outputs (also how CI works).

## Project structure

```
app/
├── core/           config (pydantic-settings), structured logging
├── llm/            LLMClient port · OpenAI + Mock adapters · pricing
├── schemas/        typed contracts for every layer (the domain model)
├── ingestion/      PDF -> text, OCR fallback, language detection
├── rag/            section-aware chunker · embeddings port · vector store port
├── agents/         classifier · extraction · legal analysis · risk · strategy
├── prompts/        versioned PT-BR prompt templates
├── orchestration/  LangGraph state machine
├── enrichment/     DataJud (CNJ) client + graph node
├── services/       analysis use case · deterministic report composer
├── evaluation/     metrics · golden runner · LLM judge · CLI
├── observability/  JSONL run store
├── reporting/      Markdown (canonical) -> PDF / DOCX converters
└── api/            FastAPI app · async job manager · routes
frontend/           Streamlit UI (pure API client)
eval_data/          golden dataset
docs/               architecture + 9 ADRs + demo script
tests/              52 tests, all offline
```

## Screenshots

<!-- After running locally: add screenshots of upload flow, agent stepper,
     risk cards, explainability tab and cost panel here. -->

## Future improvements

- Redis-backed job queue + horizontal workers (ADR 0009 documents the path)
- Anthropic/Gemini adapters for the LLM port
- Pinecone adapter for multi-tenant scale
- Case-law retrieval (jurisprudence RAG) as a second corpus
- Human feedback loop: lawyer corrections feeding the golden dataset
- AuthN/AuthZ and per-tenant data isolation at the API layer

## Disclaimer

Reports are AI-generated decision support with explicit confidence levels
and citations. They do not constitute legal advice and must be reviewed by
a qualified lawyer.
