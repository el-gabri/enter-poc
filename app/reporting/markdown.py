"""Deterministic Markdown rendering of a LitigationReport.

Pure function of the report model: same report, same Markdown. This is the
canonical export; PDF and DOCX (M7) are derived formats.
"""

from app.schemas.common import ConfidentConclusion
from app.schemas.report import LitigationReport

RISK_LABELS = {"low": "Baixo", "medium": "Medio", "high": "Alto", "critical": "Critico"}
PRIORITY_LABELS = {"urgent": "URGENTE", "high": "Alta", "medium": "Media", "low": "Baixa"}


def _conclusion(c: ConfidentConclusion, indent: str = "") -> list[str]:
    lines = [
        f"{indent}{c.statement}",
        f"{indent}- Confianca: **{c.confidence_pct}%**",
        f"{indent}- Justificativa: {c.reasoning}",
    ]
    for citation in c.citations:
        page = f", p. {citation.page}" if citation.page else ""
        lines.append(f'{indent}- Fonte: "{citation.quote}"{page}')
    return lines


def render_markdown(report: LitigationReport) -> str:  # noqa: PLR0912, PLR0915
    md: list[str] = [
        f"# Relatorio de Analise - {report.filename}",
        "",
        f"Documento `{report.doc_id}` · idioma {report.language} · gerado em "
        f"{report.generated_at:%Y-%m-%d %H:%M} UTC",
        "",
        f"**Nivel de confianca agregado: {round(report.confidence_level * 100)}%**",
        "",
    ]
    if report.warnings:
        md += ["> **Avisos:** " + " / ".join(report.warnings), ""]

    md += ["## Resumo Executivo", "", report.executive_summary or "(indisponivel)", ""]

    if report.classification:
        md += ["## Classificacao", ""]
        md += [f"Tipo de acao: **{report.classification.lawsuit_type.value}**", ""]
        md += _conclusion(report.classification.conclusion)
        md += [""]

    if report.parties:
        extraction = report.parties
        md += ["## Partes e Dados do Processo", ""]
        for party in extraction.parties:
            lawyer = f" (adv.: {party.lawyer})" if party.lawyer else ""
            md.append(f"- **{party.role.value}**: {party.name}{lawyer}")
        details = [
            ("Numero do processo", extraction.case_number),
            ("Juizo", extraction.court),
            ("UF", extraction.state),
            ("Juiz(a)", extraction.judge),
            ("Distribuicao", extraction.filing_date),
            (
                "Valor da causa",
                extraction.claim_value.as_written
                if extraction.claim_value
                else None,
            ),
        ]
        md += [f"- {label}: {value}" for label, value in details if value]
        md += [""]

    if report.timeline:
        md += ["## Linha do Tempo", ""]
        for event in report.timeline:
            date = event.date or "data indeterminada"
            md.append(f"- **{date}**: {event.description}")
        md += [""]

    if report.main_claims:
        md += ["## Pedidos e Avaliacao", ""]
        for claim in report.main_claims:
            basis = f" (base legal: {claim.legal_basis})" if claim.legal_basis else ""
            md += [f"### {claim.claim}{basis}", ""]
            md += _conclusion(claim.assessment)
            md += [""]

    if report.evidence_found:
        md += ["## Provas Identificadas", ""]
        for evidence in report.evidence_found:
            md += _conclusion(evidence)
            md += [""]

    if report.missing_information:
        md += ["## Informacoes Ausentes", ""]
        md += [f"- {item}" for item in report.missing_information]
        md += [""]

    if report.legal_risks:
        risk = report.legal_risks
        overall = RISK_LABELS.get(risk.overall_level.value, risk.overall_level.value)
        md += ["## Riscos Juridicos", "", f"Nivel geral: **{overall}**", ""]
        md += _conclusion(risk.overall)
        md += [""]
        for item in risk.risks:
            level = RISK_LABELS.get(item.level.value, item.level.value)
            md += [f"### {item.title} - risco {level}", ""]
            md += _conclusion(item.conclusion)
            if item.financial_exposure:
                md.append(f"- Exposicao financeira: {item.financial_exposure}")
            md += [""]

    if report.suggested_strategy:
        strategy = report.suggested_strategy
        md += ["## Estrategia Sugerida", ""]
        md += _conclusion(strategy.overall_approach)
        md += [""]
        if strategy.defenses:
            md += ["### Linhas de Defesa", ""]
            for defense in strategy.defenses:
                basis = f" (base: {defense.legal_basis})" if defense.legal_basis else ""
                md += [f"**{defense.argument}**{basis}", ""]
                md += _conclusion(defense.assessment)
                md += [""]
        if strategy.next_actions:
            md += ["### Proximas Acoes", ""]
            for action in strategy.next_actions:
                priority = PRIORITY_LABELS.get(action.priority.value, action.priority.value)
                md.append(f"- [{priority}] {action.action} - {action.rationale}")
            md += [""]

    if report.possible_settlement:
        md += ["## Possibilidade de Acordo", ""]
        md += _conclusion(report.possible_settlement)
        md += [""]

    md += [
        "## Como a IA Chegou a Estas Conclusoes",
        "",
        report.ai_reasoning or "(indisponivel)",
        "",
        "## Metricas da Execucao",
        "",
        f"- Agentes executados: {report.metrics.agents_run}",
        f"- Tokens: {report.metrics.total_tokens}",
        f"- Custo estimado: US$ {report.metrics.total_cost_usd:.4f}",
        f"- Duracao total: {report.metrics.total_duration_ms:.0f} ms",
        f"- Modelos: {', '.join(report.metrics.models_used) or '-'}",
        f"- Prompts: {', '.join(report.metrics.prompt_versions) or '-'}",
        "",
        "---",
        "",
        "*Relatorio gerado por IA como apoio a decisao. Nao substitui a "
        "analise de um advogado.*",
    ]
    return "\n".join(md)
