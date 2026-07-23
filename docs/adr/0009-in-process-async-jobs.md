# ADR 0009: In-process async jobs (broker-ready interface)

Status: accepted · Date: 2026-07-23

## Context

A full analysis takes 30-90s of LLM calls; blocking an HTTP request that
long is unacceptable. Standard options: task queue (Celery/RQ + Redis),
cloud queue, or in-process asyncio tasks.

## Decision

`POST /analyses` returns 202 + job id immediately; an asyncio task streams
the LangGraph execution, updating per-stage progress that the UI polls.
Jobs live in process memory; completed runs are persisted to the RunStore.

The `AnalysisJobManager` surface (submit/get) is broker-agnostic: adopting
Redis + workers later replaces this one class, not its callers.

## Consequences

- (+) Zero infrastructure for single-node deployment (fits Docker Compose).
- (+) Progress comes from LangGraph value-streaming - no bespoke callback
  plumbing inside agents.
- (-) Jobs are lost on process restart and don't scale horizontally.
  Accepted for current scale; the RunStore preserves history, and the
  isolation boundary makes the upgrade path mechanical.
