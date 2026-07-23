# ADR 0002: Own LLM port instead of LiteLLM/proxy

Status: accepted · Date: 2026-07-23

## Context

We must support OpenAI today and make Anthropic/Gemini a cheap switch.
Options: LiteLLM (uniform API over 100+ providers), LangChain chat models,
or our own thin protocol.

## Decision

Define our own `LLMClient` protocol (`app/llm/base.py`) with exactly the two
operations the product needs (`complete`, `parse`) and rich call metadata.
OpenAI and Mock are the first adapters; a factory selects the provider from
config.

## Consequences

- (+) The interface matches OUR domain (structured outputs + observability
  are mandatory, not optional extras).
- (+) Zero lock-in: a new provider is one file implementing two methods.
- (+) The mock adapter makes tests/CI/demos free and deterministic.
- (-) We write ~50 lines per new provider that LiteLLM would give for free.
  Acceptable: we expect 2-3 providers, not 100.
