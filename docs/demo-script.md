# Interview demo script (~5 minutes)

## Setup (before the interview)

1. `.env` with a real OpenAI key; `docker compose up --build` (or the two
   local processes).
2. Have a real-ish lawsuit PDF ready (or print one of the `eval_data`
   cases to PDF).
3. Open three tabs: Streamlit (8501), Swagger (8000/docs), the repo.

## Narrative

**1. Problem (30s).** "Legal teams triage incoming lawsuits by hand. This
copilot turns a petition PDF into an auditable strategy report in about a
minute - built the way I'd build it for production."

**2. Live run (90s).** Upload the PDF in the UI. While the agent stepper
animates, narrate the pipeline: ingestion with OCR fallback ->
section-aware RAG indexing -> six agents on a LangGraph state machine,
with risk/strategy and DataJud enrichment running in parallel branches.
Point at the stepper: "this progress is the graph's own value stream,
polled through an async job API - no blocking HTTP calls."

**3. Explainability (60s).** Open a risk card -> expand "Por que?" ->
show confidence %, reasoning, and the verbatim citation with page number.
"No black boxes: every conclusion cites the document, and the eval harness
verifies citations mechanically - a fabricated quote fails the metric."
Show the Explicabilidade tab: per-agent latency, tokens, prompt versions.

**4. Engineering (90s).** In the repo, show:
- `docs/adr/` - nine decision records ("this is how I document trade-offs");
- `app/llm/base.py` - the 2-method LLM port ("Anthropic support is one new
  file; observability is enforced by the return type");
- `app/services/composer.py` - deterministic last mile ("the report cannot
  hallucinate at assembly");
- `tests/` + CI - 52 offline tests via the mock provider ("CI costs zero
  tokens").

**5. Honest limits (30s).** In-process jobs (ADR 0009 documents the Redis
path), PyMuPDF licensing (ADR 0005), verbatim citation matching
undercounts paraphrases (ADR 0008). "I'd rather show you the trade-offs I
chose than pretend there aren't any."

## Likely questions

- **Why LangGraph?** Conditional routing (OCR, halt-on-error), two
  parallel fan-outs, typed state, introspectable execution -> the UI
  stepper is free. ADR 0001.
- **How do you know it works?** Golden dataset + mechanical groundedness +
  accuracy/completeness vs labels + LLM judge as secondary. `python -m
  app.evaluation`.
- **Cost control?** Per-call metering in the type system, per-run
  aggregates in the run store, retrieval-driven context (200-page filing
  costs ~the same as 10 pages).
- **How would you scale it?** Swap job manager (ADR 0009), Pinecone
  adapter (ADR 0003), stateless API replicas behind a load balancer.
