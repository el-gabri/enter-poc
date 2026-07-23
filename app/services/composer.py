"""Deterministic report composer (see ADR 0007).

The composer is intentionally NOT an LLM agent. Agents produce typed,
cited conclusions; this module assembles them into the final report with
plain code. The last mile of a legal product must not be able to
hallucinate: aggregate confidence is computed, the AI-reasoning section is
built from real traces, and missing information comes from the extraction
schema itself.
"""

from app.schemas.common import ConfidentConclusion
from app.schemas.report import LitigationReport, RunMetrics
from app.schemas.trace import AgentStatus, AgentTrace


def compose_report(state: object) -> LitigationReport:
    """Assemble the LitigationReport from a completed AnalysisState."""
    document = state.document  # type: ignore[attr-defined]
    classification = state.classification  # type: ignore[attr-defined]
    extraction = state.extraction  # type: ignore[attr-defined]
    analysis = state.legal_analysis  # type: ignore[attr-defined]
    risk = state.risk  # type: ignore[attr-defined]
    strategy = state.strategy  # type: ignore[attr-defined]
    traces: list[AgentTrace] = state.traces  # type: ignore[attr-defined]

    conclusions = _collect_conclusions(classification, analysis, risk, strategy)
    missing = _missing_information(extraction, strategy)

    return LitigationReport(
        doc_id=document.doc_id,
        filename=document.filename,
        language=document.language,
        executive_summary=analysis.executive_summary if analysis else "",
        classification=classification,
        parties=extraction,
        timeline=analysis.timeline if analysis else [],
        main_claims=analysis.claims if analysis else [],
        evidence_found=analysis.evidence_found if analysis else [],
        missing_information=missing,
        legal_risks=risk,
        suggested_strategy=strategy,
        possible_settlement=strategy.settlement if strategy else None,
        datajud=getattr(state, "enrichment", None),
        confidence_level=_aggregate_confidence(conclusions),
        ai_reasoning=_build_ai_reasoning(state, traces),
        warnings=list(document.warnings),
        metrics=_build_metrics(traces),
        traces=traces,
    )


def _collect_conclusions(
    classification: object, analysis: object, risk: object, strategy: object
) -> list[ConfidentConclusion]:
    conclusions: list[ConfidentConclusion] = []
    if classification is not None:
        conclusions.append(classification.conclusion)  # type: ignore[attr-defined]
    if analysis is not None:
        conclusions.extend(c.assessment for c in analysis.claims)  # type: ignore[attr-defined]
        conclusions.extend(analysis.evidence_found)  # type: ignore[attr-defined]
    if risk is not None:
        conclusions.append(risk.overall)  # type: ignore[attr-defined]
        conclusions.extend(r.conclusion for r in risk.risks)  # type: ignore[attr-defined]
    if strategy is not None:
        conclusions.append(strategy.overall_approach)  # type: ignore[attr-defined]
        conclusions.append(strategy.settlement)  # type: ignore[attr-defined]
        conclusions.extend(d.assessment for d in strategy.defenses)  # type: ignore[attr-defined]
    return conclusions


def _aggregate_confidence(conclusions: list[ConfidentConclusion]) -> float:
    """Unweighted mean of all conclusion confidences.

    Deliberately simple and transparent: easy to explain, easy to audit.
    A weighted scheme would imply precision we do not have.
    """
    if not conclusions:
        return 0.0
    return round(sum(c.confidence for c in conclusions) / len(conclusions), 3)


def _missing_information(extraction: object, strategy: object) -> list[str]:
    missing: list[str] = []
    if extraction is not None:
        missing.extend(extraction.missing_fields())  # type: ignore[attr-defined]
    if strategy is not None:
        missing.extend(strategy.missing_information)  # type: ignore[attr-defined]
    # dedupe, preserve order
    return list(dict.fromkeys(missing))


def _build_metrics(traces: list[AgentTrace]) -> RunMetrics:
    metered = [t.llm_meta for t in traces if t.llm_meta is not None]
    return RunMetrics(
        total_duration_ms=round(sum(t.duration_ms for t in traces), 1),
        total_tokens=sum(m.usage.total_tokens for m in metered),
        total_cost_usd=round(sum(m.cost_usd or 0.0 for m in metered), 6),
        models_used=sorted({m.model for m in metered}),
        prompt_versions=sorted(
            {m.prompt_version for m in metered if m.prompt_version}
        ),
        agents_run=len(traces),
    )


def _build_ai_reasoning(state: object, traces: list[AgentTrace]) -> str:
    """Human-readable account of HOW the analysis was produced."""
    document = state.document  # type: ignore[attr-defined]
    chunks = state.chunks  # type: ignore[attr-defined]
    lines = [
        "Como esta analise foi produzida:",
        f"1. O documento '{document.filename}' ({document.page_count} paginas, "
        f"idioma '{document.language}', extracao "
        f"{document.extraction_method.value}) foi dividido em "
        f"{len(chunks)} trechos indexados por secao.",
        "2. Agentes especializados analisaram o documento em sequencia, cada "
        "um recuperando apenas os trechos relevantes para sua tarefa "
        "(RAG), citando as fontes usadas em cada conclusao.",
    ]
    for i, trace in enumerate(traces, start=3):
        if trace.llm_meta is None:
            lines.append(
                f"{i}. Agente '{trace.agent}': {trace.status.value}"
                + (f" ({trace.error})" if trace.error else "")
            )
        else:
            lines.append(
                f"{i}. Agente '{trace.agent}' "
                f"(prompt {trace.llm_meta.prompt_version}, modelo "
                f"{trace.llm_meta.model}): {trace.status.value} em "
                f"{trace.duration_ms:.0f}ms, "
                f"{trace.llm_meta.usage.total_tokens} tokens."
            )
    failed = [t for t in traces if t.status is AgentStatus.FAILED]
    lines.append(
        "Cada conclusao inclui nivel de confianca, justificativa e citacoes "
        "do documento original. Conclusoes sem citacao devem ser verificadas "
        "com atencao redobrada."
    )
    if failed:
        lines.append(
            f"ATENCAO: {len(failed)} agente(s) falharam nesta execucao; "
            "o relatorio pode estar incompleto."
        )
    return "\n".join(lines)
