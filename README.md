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
the official CNJ court-records API. Before any document content reaches the
RAG or LLM layers, a prompt-injection gate scans every page and includes any
security findings in the report. Export as Markdown, PDF, DOCX or JSON.
Every RAG query also produces a durable audit trail with the ranked chunk
IDs, pages, sections, similarity scores, hashes, latency, and an explicit
marker for the chunks that actually reached each model prompt.

## Two product journeys

The existing **business** journey remains the lawsuit-analysis workflow above.
Legal teams upload an incoming petition, assess exposure and prepare an initial
defense and settlement posture.

The **consumer** journey is a separate workflow for complaints against
suppliers, companies or institutions. A guided chat structures the complaint
and accepts supporting PDF, PNG and JPEG evidence.
Every upload passes through the prompt-injection security gate. Images and
scanned PDFs are accepted only when OCR produces reviewable text; unreadable
files are rejected instead of being treated as supporting evidence.
The assistant then retrieves relevant, versioned summaries of provisions from
the [Brazilian Constitution](https://www.planalto.gov.br/ccivil_03/constituicao/constituicao.htm)
and the [Consumer Defense Code (CDC)](https://www.planalto.gov.br/ccivil_03/leis/l8078compilado.htm),
keeps legal-source citations separate from evidence citations, and prepares an
auditable draft for export.

The generated artifact is a **notificação extrajudicial com proposta de
acordo** (extrajudicial notice with a settlement proposal), not a lawsuit or a
court filing. Its financial section is a transparent scenario calculation
based on confirmed losses, optional no-agreement costs and explicit assumptions.
The user does not choose an additional compensation amount. Explicit values
found in evidence are shown as candidates and enter the calculation only after
the consumer confirms which one is the factual loss. Amounts mentioned in chat
remain part of the untrusted narrative and are never promoted automatically to
confirmed losses. The interface exposes each financial component's source,
excerpt and hashes for audit. The calculator then produces an auditable
negotiation proposal without claiming calibrated odds.
The system never sends or files the notice. A qualified Brazilian lawyer must
review it before use.

## Architecture

```
Browser -> Streamlit -> FastAPI (202 + async job)
    -> LangGraph state machine:
       security_scan --+--> index -> classify -> extract --+--> analyze --+--> risk     --+
                       |                                  |              +--> strategy --+--> compose
                       |                                  +--> DataJud enrichment -------+
                       +--> halt for human review / block -------------------------------+
    -> RAG: section-aware chunking -> embeddings -> ChromaDB (per-doc isolation)
       -> retrieval audit: query -> top-k ranks/scores -> prompt inclusion
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
| [0010](docs/adr/0010-prompt-injection-security-gate.md) | Scan untrusted document content before indexing or LLM analysis |
| [0011](docs/adr/0011-retrieval-traceability-and-evaluation.md) | Persist query-to-context provenance and evaluate retrieval rankings |
| [0012](docs/adr/0012-bounded-consumer-extrajudicial-notice.md) | Bound the consumer workflow to auditable, human-reviewed extrajudicial notice drafts |

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

Retrieval traces are nested under each agent trace and persist the raw top-k
rankings without storing full chunk text: query/hash, requested `k`, rank,
similarity score, chunk/document IDs, section, page span, indexed-text hash,
latency, embedding/index versions, and whether the chunk survived deduplication
and context truncation. Failed embedding/search attempts retain one event per
query, successful sibling lookups, and the consuming agent's status, error, and
prompt version. Text previews are disabled by default; enable
`LITIGATION_RETRIEVAL_TRACE_INCLUDE_PREVIEWS=true` only under an explicit data
retention and access-control policy. The full audit is available at
`/analyses/{job_id}/retrievals`; historical traces remain available at
`/runs/{run_id}/retrievals`. The Streamlit explainability tab provides the same
data as a filterable table. `citation_retrieval_coverage` counts a citation only
when its chunk was included in the context of the agent that produced it.

### Prompt-injection defense

Every page is checked with deterministic Portuguese and English rules before
indexing. In the default `balanced` mode, suspicious excerpts also receive a
bounded semantic review; document text is always treated as untrusted data.
The routing policy is deterministic: `none`/`low` proceeds, `medium` proceeds
with a warning and masks flagged excerpts from downstream prompts, `high`
halts with job state `review_required`, and `critical` ends with job state
`blocked`. These outcomes are not counted as completed legal analyses. The
original document is never altered, and page-attributed findings remain
visible in the report.

Set `LITIGATION_PROMPT_INJECTION_SCAN_MODE=rules` for deterministic checks
only, keep the default `balanced` mode for targeted semantic review, or use
`strict` to semantically review all page text (with higher latency and cost).
Strict mode is capped by `LITIGATION_PROMPT_INJECTION_STRICT_MAX_CHARS` and
`LITIGATION_PROMPT_INJECTION_STRICT_MAX_BATCHES`; exceeding either budget
fails closed instead of creating an unbounded number of model calls.

### Evaluation

```bash
python -m app.evaluation           # golden dataset in eval_data/
```

Metrics: groundedness, hallucination rate, citation coverage, extraction
accuracy, completeness, classification accuracy, Precision@K, Recall@K,
HitRate@K, MRR@K, NDCG@K, and LLM-as-judge response quality (real provider
only). Retrieval labels use relevant page ranges and passage anchors so they
remain valid when chunk sizes change. Overlapping chunks that satisfy the same
passage label count as one relevance unit, and unresolved labels fail loudly.
Runs offline in CI with the mock provider as a pipeline health check.

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
pip install -e ".[dev,frontend,ocr]"
copy .env.example .env
pytest                                            # fully offline

uvicorn app.api.main:app --reload                 # terminal 1
streamlit run frontend/streamlit_app.py           # terminal 2
```
Local OCR for images and scanned PDFs also requires the Tesseract system binary
and Portuguese language data. The API Docker image already installs both.

No API key? Set `LITIGATION_LLM_PROVIDER=mock` - the entire product runs
offline with deterministic outputs (also how CI works).

### Document handling

Uploads are streamed with a 20 MB API limit. The business journey remains
PDF-only and rejects files above `LITIGATION_MAX_DOCUMENT_PAGES` (250 by
default). Consumer evidence accepts PDF, PNG and JPEG; images above 40
megapixels are rejected before raster decoding. Images and scanned PDFs without
usable OCR text are also rejected.
Raw consumer evidence is always deleted immediately after ingestion. Business
PDFs are deleted after analysis by default; set
`LITIGATION_RETAIN_UPLOADS=true` only when an explicit retention policy
requires it. ChromaDB and run history remain persistent storage and should be
protected with authentication, tenant isolation and an operational retention
policy before real case data is used.

## Project structure

```
app/
├── core/           config (pydantic-settings), structured logging
├── consumer/       guided intake · legal corpus · evidence · notice composer
├── llm/            LLMClient port · OpenAI + Mock adapters · pricing
├── schemas/        typed contracts for every layer (the domain model)
├── ingestion/      PDF/image -> text, OCR fallback, language detection
├── security/       prompt-injection scanning, policy and safe context masking
├── rag/            section-aware chunker · embeddings port · vector store port
├── agents/         classifier · extraction · legal analysis · risk · strategy
├── prompts/        versioned PT-BR prompt templates
├── orchestration/  LangGraph state machine
├── enrichment/     DataJud (CNJ) client + graph node
├── services/       analysis use case · deterministic report composer
├── evaluation/     metrics · golden runner · LLM judge · CLI
├── observability/  JSONL run store with durable agent/retrieval traces
├── reporting/      Markdown (canonical) -> PDF / DOCX converters
└── api/            FastAPI app · async job manager · routes
frontend/           Streamlit UI (pure API client)
eval_data/          golden dataset
docs/               architecture + 12 ADRs + demo script
tests/              offline unit, integration and security tests
```

## Screenshots

<!-- After running locally: add screenshots of upload flow, agent stepper,
     risk cards, explainability tab and cost panel here. -->

## Future improvements

- Add brazilian jurisprudence
- Redis-backed job queue + horizontal workers (ADR 0009 documents the path)
- Anthropic/Gemini adapters for the LLM port
- Pinecone adapter for multi-tenant scale
- Case-law retrieval (jurisprudence RAG) as a second corpus
- Human feedback loop: lawyer corrections feeding the golden dataset
- AuthN/AuthZ and per-tenant data isolation at the API layer
- Calibrated consumer outcome models from reviewed, representative settlement
  and judgment data; statutory text alone cannot supply win probabilities

## Disclaimer

Reports and consumer notices are AI-generated decision support with explicit
provenance. They do not constitute legal advice and must be reviewed by a
qualified lawyer. The consumer flow does not create an attorney-client
relationship, submit a complaint, interrupt a limitation period or replace
urgent help from a lawyer or the competent authorities.
