# ADR 0008: Citation verification as the groundedness metric

Status: accepted · Date: 2026-07-23

## Context

Standard RAG evaluation (RAGAS-style) uses an LLM to judge whether answers
are faithful to context. That is expensive, non-deterministic, and
circular: an LLM grading an LLM with no ground truth.

## Decision

Our primary groundedness/hallucination signal is mechanical: every
conclusion carries citations (verbatim quotes); the metric checks whether
each quote actually occurs in the source document (after normalization -
case, accents, whitespace, punctuation). Complementary metrics:

- citation_coverage: fraction of conclusions that cite anything at all;
- extraction_accuracy / completeness: exact comparison against golden
  labels;
- response_quality: LLM-as-judge, kept as a SECONDARY metric because prose
  quality genuinely needs judgment.

## Consequences

- (+) Deterministic, free, runs in CI on every commit (mock provider
  exercises the plumbing; real provider measures real quality).
- (+) A fabricated quote - the most dangerous hallucination in a legal
  product - is caught with certainty, not with a judge's opinion.
- (-) Verbatim matching misses paraphrased-but-faithful citations
  (undercounts groundedness) and cannot detect a true quote used to
  support a wrong conclusion; the judge metric partially covers that gap.
