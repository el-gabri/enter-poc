"""Deterministic, provenance-friendly consumer intake helpers.

Chat messages are untrusted allegations.  This module only extracts a small
allowlist of explicit facts and never turns a user's instruction into legal
advice or documentary evidence.
"""

from __future__ import annotations

import re

from app.consumer.schemas import (
    ConsumerCaseFacts,
    ConsumerIntakeExtraction,
    ConsumerIssueCategory,
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
_GENERIC_SUPPLIER_PREFIX_RE = re.compile(
    r"\b(?:empresa|loja|fornecedor(?:a)?|operadora|plataforma|site|aplicativo)\s+",
    re.IGNORECASE,
)
_SUPPLIER_TOKEN_RE = re.compile(r"[0-9A-Za-zÀ-ÿ&]+(?:[.-][0-9A-Za-zÀ-ÿ&]+)*")
_SUPPLIER_NAME_CONNECTORS = frozenset({"da", "das", "de", "do", "dos", "e"})
_SUPPLIER_STOP_WORDS = frozenset(
    [
        "a",
        "agora",
        "após",
        "atrasou",
        "bloqueou",
        "cancelou",
        "cobrou",
        "com",
        "debitou",
        "desde",
        "deve",
        "deveria",
        "disse",
        "ela",
        "ele",
        "em",
        "entregou",
        "enviou",
        "era",
        "está",
        "estornou",
        "é",
        "fez",
        "foi",
        "informou",
        "isso",
        "já",
        "mas",
        "me",
        "não",
        "nao",
        "negou",
        "nos",
        "novamente",
        "o",
        "onde",
        "para",
        "pois",
        "por",
        "porque",
        "prometeu",
        "quando",
        "que",
        "realizou",
        "recusou",
        "resolveu",
        "respondeu",
        "sem",
        "simplesmente",
        "suspendeu",
        "tem",
        "uma",
        "vendeu",
    ]
)
_MAX_SUPPLIER_NAME_TOKENS = 4


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
        (
            "falha",
            "serviço",
            "servico",
            "atendimento",
            "indisponível",
            "indisponivel",
            "não entreg",
            "nao entreg",
            "produto com defeito",
        ),
    ),
)

_RECOMMENDED_DOCUMENTS: dict[ConsumerIssueCategory, list[str]] = {
    ConsumerIssueCategory.UNAUTHORIZED_CHARGE: [
        "extrato ou fatura com a cobrança destacada",
        "comprovante do pagamento, se houve",
        "protocolos e respostas da empresa ou instituição",
    ],
    ConsumerIssueCategory.FRAUD: [
        "extrato com as transações contestadas",
        "boletim de ocorrência, se disponível",
        "protocolos de bloqueio e contestação",
    ],
    ConsumerIssueCategory.ACCOUNT_BLOCK: [
        "comunicação ou tela que mostre o bloqueio",
        "extrato do saldo afetado",
        "protocolos e respostas da empresa ou instituição",
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
        "protocolos e respostas da empresa ou instituição",
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
    bank_name = _extract_supplier(normalized) if current.bank_name is None else None
    period = None
    if current.incident_date_or_period is None and (match := _DATE_RE.search(normalized)):
        period = match.group(0)
    return ConsumerIntakeExtraction(
        bank_name=bank_name,
        issue_category=category,
        complaint_summary=(normalized if current.complaint_summary is None else None),
        incident_date_or_period=period,
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
        "bank_name": "Qual é o nome da empresa, fornecedor ou instituição?",
        "consumer_name": "Qual é o seu nome completo para identificar a parte notificante?",
        "issue_category": (
            "Qual tipo de problema ocorreu: cobrança, fraude, bloqueio, "
            "negativação, crédito ou outro?"
        ),
        "complaint_summary": (
            "Conte, em ordem cronológica, o que aconteceu e o que a empresa "
            "ou fornecedor fez ou deixou de fazer."
        ),
        "incident_date_or_period": (
            "Quando o problema ocorreu? Informe a data ou o período aproximado."
        ),
        "desired_resolution": "Qual solução você espera da empresa ou fornecedor?",
    }
    missing = facts.missing_fields()
    if missing:
        return questions[missing[0]]
    if not has_evidence:
        return (
            "Os fatos essenciais estão preenchidos. Agora envie ao menos um PDF ou "
            "imagem que comprove a ocorrência, o valor ou uma tentativa de solução."
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


def _extract_supplier(text: str) -> str | None:
    if match := _BANK_RE.search(text):
        return match.group(1).strip().title()
    if match := _GENERIC_BANK_RE.search(text):
        return f"Banco {match.group(1).strip()}"
    if match := _GENERIC_SUPPLIER_PREFIX_RE.search(text):
        return _supplier_name_after_prefix(text, match.end())
    return None


def _supplier_name_after_prefix(text: str, start: int) -> str | None:
    tokens: list[str] = []
    previous_end = start

    for match in _SUPPLIER_TOKEN_RE.finditer(text, start):
        if text[previous_end : match.start()] and not text[previous_end : match.start()].isspace():
            break

        token = match.group(0)
        folded = token.casefold()
        if folded in _SUPPLIER_STOP_WORDS:
            break
        if not tokens and folded in _SUPPLIER_NAME_CONNECTORS:
            break

        tokens.append(token)
        previous_end = match.end()
        if len(tokens) == _MAX_SUPPLIER_NAME_TOKENS:
            break

    while tokens and tokens[-1].casefold() in _SUPPLIER_NAME_CONNECTORS:
        tokens.pop()
    if not tokens:
        return None

    name = " ".join(tokens)
    if name.islower() or name.isupper():
        return " ".join(
            part.casefold() if part.casefold() in _SUPPLIER_NAME_CONNECTORS else part.title()
            for part in tokens
        )
    return name
