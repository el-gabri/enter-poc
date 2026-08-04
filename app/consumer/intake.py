"""Deterministic, provenance-friendly consumer intake helpers.

Chat messages are untrusted allegations.  This module only extracts a small
allowlist of explicit facts and never turns a user's instruction into legal
advice or documentary evidence.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

from app.consumer.schemas import (
    ConsumerCaseFacts,
    ConsumerIntakeExtraction,
    ConsumerIssueCategory,
)

_MONEY_RE = re.compile(
    r"R\$\s*(\d{1,3}(?:\.\d{3})*(?:,\d{1,2})?|\d+(?:[.,]\d{1,2})?)",
    re.IGNORECASE,
)
_DATE_RE = re.compile(
    r"\b(?:\d{1,2}/\d{1,2}/\d{2,4}|"
    r"(?:janeiro|fevereiro|março|abril|maio|junho|julho|agosto|setembro|"
    r"outubro|novembro|dezembro)\s+de\s+\d{4}|\d{4})\b",
    re.IGNORECASE,
)
_BANK_RE = re.compile(
    r"\b((?:banco\s+)?(?:nubank|ita[uú]|bradesco|santander|inter|c6(?:\s+bank)?|"
    r"caixa(?:\s+econ[oô]mica\s+federal)?|banco\s+do\s+brasil|picpay|neon))\b",
    re.IGNORECASE,
)
_GENERIC_BANK_RE = re.compile(
    r"\b(?:banco|instituiç[aã]o financeira)\s+([A-ZÁÀÂÃÉÊÍÓÔÕÚÇ][\wÀ-ÿ.-]*"
    r"(?:\s+[A-ZÁÀÂÃÉÊÍÓÔÕÚÇ][\wÀ-ÿ.-]*){0,3})"
)

_CATEGORY_TERMS: tuple[tuple[ConsumerIssueCategory, tuple[str, ...]], ...] = (
    (
        ConsumerIssueCategory.NEGATIVE_CREDIT_RECORD,
        ("serasa", "spc", "negativ", "cadastro de inadimpl"),
    ),
    (
        ConsumerIssueCategory.ACCOUNT_BLOCK,
        ("conta bloque", "saldo bloque", "conta encerr", "acesso bloque"),
    ),
    (
        ConsumerIssueCategory.FRAUD,
        ("fraude", "golpe", "pix que não", "pix que nao", "cartão clonado"),
    ),
    (
        ConsumerIssueCategory.LOAN_OR_INTEREST,
        ("emprést", "emprest", "juros", "financiamento", "consignado"),
    ),
    (
        ConsumerIssueCategory.OVER_INDEBTEDNESS,
        ("superendivid", "não consigo pagar", "nao consigo pagar"),
    ),
    (
        ConsumerIssueCategory.UNAUTHORIZED_CHARGE,
        ("cobrança", "cobranca", "débito", "debito", "não reconheço", "nao reconheco"),
    ),
    (
        ConsumerIssueCategory.SERVICE_FAILURE,
        ("falha", "serviço", "servico", "atendimento", "indisponível", "indisponivel"),
    ),
)

_RECOMMENDED_DOCUMENTS: dict[ConsumerIssueCategory, list[str]] = {
    ConsumerIssueCategory.UNAUTHORIZED_CHARGE: [
        "extrato ou fatura com a cobrança destacada",
        "comprovante do pagamento, se houve",
        "protocolos e respostas do banco",
    ],
    ConsumerIssueCategory.FRAUD: [
        "extrato com as transações contestadas",
        "boletim de ocorrência, se disponível",
        "protocolos de bloqueio e contestação",
    ],
    ConsumerIssueCategory.ACCOUNT_BLOCK: [
        "comunicação ou tela que mostre o bloqueio",
        "extrato do saldo afetado",
        "protocolos e respostas do banco",
    ],
    ConsumerIssueCategory.NEGATIVE_CREDIT_RECORD: [
        "consulta do cadastro restritivo com data e credor",
        "comprovantes de pagamento ou inexistência da dívida",
        "protocolos de contestação",
    ],
    ConsumerIssueCategory.LOAN_OR_INTEREST: [
        "contrato e demonstrativo do custo efetivo total",
        "extratos ou boletos pagos",
        "oferta, simulação ou publicidade recebida",
    ],
    ConsumerIssueCategory.SERVICE_FAILURE: [
        "contrato, oferta ou termos do serviço",
        "telas, mensagens e protocolos que demonstrem a falha",
        "comprovantes do prejuízo direto",
    ],
    ConsumerIssueCategory.OVER_INDEBTEDNESS: [
        "contratos e faturas das dívidas de consumo",
        "comprovantes de renda e despesas essenciais",
        "propostas e protocolos de renegociação",
    ],
    ConsumerIssueCategory.OTHER: [
        "contrato, fatura ou extrato relacionado",
        "protocolos e respostas do banco",
        "comprovantes do prejuízo alegado",
    ],
}


def extract_explicit_facts(text: str, current: ConsumerCaseFacts) -> ConsumerIntakeExtraction:
    """Extract only facts directly present in a message.

    The first message becomes the complaint summary. Later edits remain the
    consumer's responsibility through the confirmation form; this extractor
    intentionally does not silently replace already collected facts.
    """
    normalized = " ".join(text.split())
    lowered = normalized.casefold()
    category = _classify(lowered) if current.issue_category is None else None
    bank_name = _extract_bank(normalized) if current.bank_name is None else None
    period = None
    if current.incident_date_or_period is None and (match := _DATE_RE.search(normalized)):
        period = match.group(0)
    amount = None
    if current.direct_loss_amount is None and (match := _MONEY_RE.search(normalized)):
        amount = _parse_brl(match.group(1))

    return ConsumerIntakeExtraction(
        bank_name=bank_name,
        issue_category=category,
        complaint_summary=(normalized if current.complaint_summary is None else None),
        incident_date_or_period=period,
        direct_loss_amount=amount,
        desired_resolution=(
            normalized
            if current.desired_resolution is None
            and any(term in lowered for term in ("quero ", "desejo ", "solicito "))
            else None
        ),
    )


def merge_explicit_facts(
    current: ConsumerCaseFacts, extraction: ConsumerIntakeExtraction
) -> ConsumerCaseFacts:
    """Fill missing allowlisted fields without overwriting confirmed values."""
    updates: dict[str, object] = {}
    for field_name in ConsumerIntakeExtraction.model_fields:
        value = getattr(extraction, field_name)
        existing = getattr(current, field_name)
        if value is not None and (existing is None or existing == []):
            updates[field_name] = value
    return current.model_copy(update=updates)


def recommended_documents(category: ConsumerIssueCategory | None) -> list[str]:
    return list(_RECOMMENDED_DOCUMENTS[category or ConsumerIssueCategory.OTHER])


def next_assistant_message(facts: ConsumerCaseFacts, *, has_evidence: bool) -> str:
    """Select one concise next question from deterministic readiness state."""
    questions = {
        "bank_name": "Qual é o nome do banco ou da instituição financeira?",
        "consumer_name": "Qual é o seu nome completo para identificar a parte notificante?",
        "issue_category": (
            "Qual tipo de problema ocorreu: cobrança, fraude, bloqueio, "
            "negativação, crédito ou outro?"
        ),
        "complaint_summary": (
            "Conte, em ordem cronológica, o que aconteceu e o que o banco "
            "fez ou deixou de fazer."
        ),
        "incident_date_or_period": (
            "Quando o problema ocorreu? Informe a data ou o período aproximado."
        ),
        "desired_resolution": "Que solução você quer pedir ao banco?",
    }
    missing = facts.missing_fields()
    if missing:
        return questions[missing[0]]
    if not has_evidence:
        return (
            "Os fatos essenciais estão preenchidos. Agora envie ao menos um PDF que "
            "comprove a ocorrência, o valor ou uma tentativa anterior de solução."
        )
    return (
        "Já há fatos e evidência suficientes para um rascunho. Confira o resumo, "
        "confirme os dados e então gere a notificação extrajudicial."
    )


def _classify(lowered: str) -> ConsumerIssueCategory:
    for category, terms in _CATEGORY_TERMS:
        if any(term in lowered for term in terms):
            return category
    return ConsumerIssueCategory.OTHER


def _extract_bank(text: str) -> str | None:
    if match := _BANK_RE.search(text):
        return match.group(1).strip().title()
    if match := _GENERIC_BANK_RE.search(text):
        return f"Banco {match.group(1).strip()}"
    return None


def _parse_brl(raw: str) -> Decimal | None:
    if "," in raw:
        value = raw.replace(".", "").replace(",", ".")
    elif raw.count(".") == 1 and len(raw.rsplit(".", maxsplit=1)[1]) <= 2:
        value = raw
    else:
        value = raw.replace(".", "")
    try:
        return Decimal(value)
    except InvalidOperation:
        return None
