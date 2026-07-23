# ADR 0001: Use LangGraph for agent orchestration

Status: accepted · Date: 2026-07-23

## Context

The pipeline is not linear: OCR only runs when native text extraction fails,
analysis prompts depend on the lawsuit classification, and failed nodes need
retries without restarting the whole run. We considered plain LangChain
chains, a hand-rolled orchestrator, and LangGraph.

## Decision

Use LangGraph. The pipeline is modeled as a typed state graph; each agent is
a node, routing decisions are edges.

## Consequences

- (+) Conditional routing, retries and checkpointing are framework features.
- (+) The graph is introspectable -> powers the agent-execution UI for free.
- (+) Typed shared state (`AnalysisState`) documents the data flow.
- (-) Framework dependency; mitigated by keeping agents pure functions that
  receive/return Pydantic models - the graph is thin wiring we could replace.

## Alternatives rejected

- **LangChain chains**: linear, weak fit for conditional flow.
- **Custom orchestrator**: full control but we would rebuild checkpointing,
  streaming and visualization - not where a startup should spend time.
