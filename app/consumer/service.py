"""Consumer case application service.

The service keeps the consumer workflow separate from the defendant-side
analysis graph. Facts are user allegations, evidence is scanned before RAG,
legal authorities come only from the reviewed corpus, and the final artifact
is assembled deterministically.
"""

from __future__ import annotations

import asyncio
import hashlib
import uuid
from decimal import Decimal
from pathlib import Path
from typing import Any

from app.consumer.intake import (
    extract_explicit_facts,
    merge_explicit_facts,
    next_assistant_message,
    recommended_documents,
)
from app.consumer.legal_corpus import LegalCorpus, get_default_legal_corpus
from app.consumer.schemas import (
    ConsumerCaseFacts,
    ConsumerCaseSnapshot,
    ConsumerCaseStatus,
    ConsumerEvidence,
    ConsumerMessage,
    ConsumerMessageRole,
    ConsumerNotice,
    EvidenceCitation,
    EvidenceStatus,
    LegalAuthorityCitation,
    LegalGround,
    ProvisionStatus,
    SettlementInputs,
)
from app.consumer.settlement import SettlementCalculator
from app.consumer.store import ConsumerCaseRecord, ConsumerCaseStore, StoredEvidence
from app.ingestion.service import DocumentIngestionService
from app.rag.pipeline import RagPipeline
from app.schemas.document import DocumentPage, ExtractionMethod, ParsedDocument
from app.schemas.rag import RetrievedChunk
from app.schemas.security import SecurityAction
from app.schemas.trace import AgentStatus, RetrievalTrace
from app.security.prompt_injection import PromptInjectionDetector
from app.security.sanitization import sanitized_document

CONSUMER_NOTICE_WARNING = (
    "Rascunho informativo para revisão humana. Não é petição judicial nem substitui "
    "orientação jurídica individualizada. Confira fatos, documentos, destinatário e prazos."
)

_CATEGORY_QUERY = {
    "unauthorized_charge": "cobrança indevida pagamento repetição indébito oferta banco",
    "fraud": "fraude bancária falha segurança serviço responsabilidade reparação consumidor",
    "account_block": "bloqueio conta serviço bancário informação reparação consumidor",
    "negative_credit_record": "cadastro consumidor negativação cobrança informação correção",
    "loan_or_interest": "crédito empréstimo juros informação contrato consumidor banco",
    "service_failure": "falha prestação serviço banco responsabilidade reparação",
    "over_indebtedness": "superendividamento crédito responsável conciliação consumidor",
    "other": "proteção consumidor banco informação reparação boa-fé",
}
_CATEGORY_LABEL = {
    "unauthorized_charge": "cobrança não reconhecida ou indevida",
    "fraud": "fraude ou movimentação não reconhecida",
    "account_block": "bloqueio de conta ou valores",
    "negative_credit_record": "registro negativo de crédito",
    "loan_or_interest": "empréstimo, financiamento ou juros",
    "service_failure": "falha na prestação do serviço bancário",
    "over_indebtedness": "superendividamento",
    "other": "controvérsia bancária de consumo",
}


class ConsumerCaseNotReadyError(ValueError):
    def __init__(self, missing: list[str]) -> None:
        self.missing = missing
        super().__init__("consumer case is not ready for a notice")


class ConsumerRetrievalError(RuntimeError):
    """Required evidence or authority retrieval failed."""


class ConsumerCaseService:
    def __init__(
        self,
        *,
        ingestion: DocumentIngestionService,
        detector: PromptInjectionDetector,
        rag: RagPipeline,
        store: ConsumerCaseStore | None = None,
        legal_corpus: LegalCorpus | None = None,
        settlement_calculator: SettlementCalculator | None = None,
    ) -> None:
        self._ingestion = ingestion
        self._detector = detector
        self._rag = rag
        self._store = store or ConsumerCaseStore()
        self._legal_corpus = legal_corpus or get_default_legal_corpus()
        self._settlement = settlement_calculator or SettlementCalculator()
        self._legal_index_lock = asyncio.Lock()
        self._legal_indexed = False

    def create_case(self) -> tuple[ConsumerCaseSnapshot, str, str]:
        record, token = self._store.create()
        greeting = (
            "Conte o que aconteceu com o banco, incluindo datas, valores e tentativas "
            "anteriores de solução. Suas mensagens serão tratadas como alegações até você "
            "confirmar os fatos e enviar documentos."
        )
        record.messages.append(
            ConsumerMessage(role=ConsumerMessageRole.ASSISTANT, content=greeting)
        )
        record.touch()
        return self._snapshot(record), token, greeting

    def get_case(self, case_id: str, token: str) -> ConsumerCaseSnapshot:
        return self._snapshot(self._store.get_authorized(case_id, token))

    def add_message(
        self,
        case_id: str,
        token: str,
        text: str,
        *,
        client_message_id: str | None = None,
    ) -> tuple[ConsumerCaseSnapshot, str]:
        record = self._store.get_authorized(case_id, token)
        if client_message_id and client_message_id in record.idempotent_messages:
            return self._snapshot(record), record.idempotent_messages[client_message_id]

        user_message = ConsumerMessage(role=ConsumerMessageRole.USER, content=text.strip())
        record.messages.append(user_message)
        extraction = extract_explicit_facts(user_message.content, record.facts)
        merged = merge_explicit_facts(record.facts, extraction)
        if merged != record.facts:
            record.facts = merged
        # Every new allegation requires a fresh human review, even when the
        # conservative extractor cannot map it into one structured field.
        record.facts_confirmed = False
        record.notice = None

        assistant = next_assistant_message(
            record.facts, has_evidence=self._has_accepted_evidence(record)
        )
        record.messages.append(
            ConsumerMessage(role=ConsumerMessageRole.ASSISTANT, content=assistant)
        )
        if client_message_id:
            record.idempotent_messages[client_message_id] = assistant
        record.touch()
        return self._snapshot(record), assistant

    def update_facts(
        self,
        case_id: str,
        token: str,
        updates: dict[str, Any],
        *,
        facts_confirmed: bool | None = None,
    ) -> ConsumerCaseSnapshot:
        record = self._store.get_authorized(case_id, token)
        allowed = set(ConsumerCaseFacts.model_fields)
        unknown = set(updates) - allowed
        if unknown:
            raise ValueError(f"unsupported fact fields: {', '.join(sorted(unknown))}")
        payload = record.facts.model_dump()
        payload.update(updates)
        updated = ConsumerCaseFacts.model_validate(payload)
        if updated != record.facts:
            record.facts = updated
            record.notice = None
            record.facts_confirmed = False
        if facts_confirmed is not None:
            record.facts_confirmed = facts_confirmed
        record.touch()
        return self._snapshot(record)

    async def add_document(
        self, case_id: str, token: str, *, filename: str, path: Path
    ) -> tuple[ConsumerCaseSnapshot, ConsumerEvidence]:
        record = self._store.get_authorized(case_id, token)
        document = await self._ingestion.ingest(path)
        content_sha256 = hashlib.sha256(document.full_text.encode()).hexdigest()
        duplicate = next(
            (
                item.public
                for item in record.documents
                if item.public.content_sha256 == content_sha256
            ),
            None,
        )
        if duplicate is not None:
            return self._snapshot(record), duplicate
        assessment, _ = await self._detector.scan(document)
        status = _evidence_status(assessment.recommended_action)
        safe = (
            sanitized_document(document, assessment)
            if assessment.recommended_action.allows_automated_analysis
            else None
        )
        warnings = [*document.warnings, *assessment.warnings]
        if assessment.detected:
            warnings.append(
                "O controle de segurança sinalizou instruções dirigidas à IA neste arquivo."
            )
        if not assessment.recommended_action.allows_automated_analysis:
            warnings.append(
                "O conteúdo não foi disponibilizado ao RAG; revise o arquivo manualmente."
            )
        public = ConsumerEvidence(
            evidence_id=uuid.uuid4().hex,
            filename=Path(filename).name[:255] or "evidencia.pdf",
            page_count=document.page_count,
            status=status,
            content_sha256=content_sha256,
            security_assessment=assessment,
            warnings=warnings,
        )
        record.documents.append(StoredEvidence(public=public, safe_document=safe))
        record.notice = None
        record.facts_confirmed = False
        record.touch()
        assistant = (
            f"Documento {public.filename} aceito e vinculado ao caso."
            if safe is not None
            else f"Documento {public.filename} requer revisão e não será usado automaticamente."
        )
        record.messages.append(
            ConsumerMessage(role=ConsumerMessageRole.ASSISTANT, content=assistant)
        )
        return self._snapshot(record), public

    async def generate_notice(self, case_id: str, token: str) -> ConsumerNotice:
        record = self._store.get_authorized(case_id, token)
        snapshot = self._snapshot(record)
        if not snapshot.ready_for_notice:
            raise ConsumerCaseNotReadyError(self._readiness_missing(record))

        await self._ensure_legal_corpus_indexed()
        evidence_document, page_sources = self._combined_evidence(record)
        evidence_chunks = await self._rag.index_document(evidence_document)
        if not evidence_chunks:
            raise ConsumerRetrievalError("accepted evidence produced no retrievable text")
        record.indexed_document_ids.add(evidence_document.doc_id)

        category = record.facts.issue_category
        category_value = category.value if category is not None else "other"
        legal_queries = [
            "relação de consumo serviço bancário direitos básicos reparação",
            _CATEGORY_QUERY[category_value],
        ]
        evidence_queries = [
            "documento que comprova ocorrência data valor protocolo comunicação banco",
            "comprovante do prejuízo e tentativa de solução",
        ]
        try:
            legal_results, legal_traces = await self._rag.retrieve_many_with_traces(
                legal_queries,
                doc_id=self._legal_corpus.as_parsed_document().doc_id,
                agent="consumer_legal_authorities",
                k=8,
            )
            evidence_results, evidence_traces = await self._rag.retrieve_many_with_traces(
                evidence_queries,
                doc_id=evidence_document.doc_id,
                agent="consumer_case_evidence",
                k=6,
            )
        except Exception as exc:
            raise ConsumerRetrievalError(
                "required retrieval failed; no unsupported notice was generated"
            ) from exc

        legal_grounds = self._legal_grounds(legal_results, record.facts)
        evidence_references = self._evidence_references(
            evidence_results, page_sources
        )
        if not legal_grounds or not evidence_references:
            raise ConsumerRetrievalError(
                "retrieval returned insufficient grounded support for a notice"
            )
        legal_merged = _merge_results(legal_results)
        evidence_merged = _merge_results(evidence_results)
        legal_traces = _annotate_composer_selection(
            legal_traces,
            legal_merged,
            {
                ground.authority.chunk_id
                for ground in legal_grounds
                if ground.authority.chunk_id is not None
            },
        )
        evidence_traces = _annotate_composer_selection(
            evidence_traces,
            evidence_merged,
            {citation.chunk_id for citation in evidence_references},
        )

        settlement = self._settlement.calculate(
            SettlementInputs(
                direct_loss_amount=record.facts.direct_loss_amount or Decimal("0"),
                improper_payment_amount=(
                    record.facts.improper_payment_amount or Decimal("0")
                ),
                requested_compensation_amount=(
                    record.facts.requested_compensation_amount or Decimal("0")
                ),
                downside_cost_amount=(
                    record.facts.unsuccessful_scenario_cost_amount or Decimal("0")
                ),
                article_42_double_repayment_supported=(
                    record.facts.article_42_double_repayment_requested
                    and (record.facts.improper_payment_amount or Decimal("0")) > 0
                ),
                evidence_strength=min(
                    Decimal("0.85"),
                    Decimal("0.45")
                    + Decimal("0.10") * len(self._accepted_documents(record)),
                ),
                factual_completeness=Decimal("1"),
            )
        )
        requests = _requests(record.facts)
        full_text = _render_notice_markdown(
            facts=record.facts,
            evidence=evidence_references,
            legal_grounds=legal_grounds,
            requests=requests,
            public_proposal=settlement.public_proposal_amount,
        )
        notice = ConsumerNotice(
            notice_id=uuid.uuid4().hex,
            case_id=record.case_id,
            addressee=record.facts.bank_name or "[INSTITUIÇÃO FINANCEIRA]",
            facts_summary=record.facts.complaint_summary or "",
            evidence_references=evidence_references,
            legal_grounds=legal_grounds,
            requests=requests,
            response_deadline_business_days=record.facts.response_deadline_business_days,
            settlement=settlement,
            full_text=full_text,
            corpus_release_id=self._legal_corpus.release_id,
            retrievals=[*legal_traces, *evidence_traces],
            warnings=[CONSUMER_NOTICE_WARNING],
        )
        record.notice = notice
        record.touch()
        return notice

    def get_notice(self, case_id: str, token: str) -> ConsumerNotice:
        record = self._store.get_authorized(case_id, token)
        if record.notice is None:
            raise ConsumerCaseNotReadyError(["notice_not_generated"])
        return record.notice

    async def delete_case(self, case_id: str, token: str) -> None:
        record = self._store.delete_authorized(case_id, token)
        for doc_id in record.indexed_document_ids:
            await self._rag.delete_document(doc_id)

    async def _ensure_legal_corpus_indexed(self) -> None:
        if self._legal_indexed:
            return
        async with self._legal_index_lock:
            if not self._legal_indexed:
                await self._rag.index_document(self._legal_corpus.as_parsed_document())
                self._legal_indexed = True

    def _snapshot(self, record: ConsumerCaseRecord) -> ConsumerCaseSnapshot:
        missing = record.facts.missing_fields()
        ready = not self._readiness_missing(record)
        if record.notice is not None:
            status = ConsumerCaseStatus.NOTICE_GENERATED
        elif ready:
            status = ConsumerCaseStatus.READY_FOR_NOTICE
        elif missing:
            status = ConsumerCaseStatus.COLLECTING_FACTS
        else:
            status = ConsumerCaseStatus.COLLECTING_EVIDENCE
        return ConsumerCaseSnapshot(
            case_id=record.case_id,
            status=status,
            messages=record.messages,
            facts=record.facts,
            missing_fields=missing,
            recommended_documents=recommended_documents(record.facts.issue_category),
            documents=[item.public for item in record.documents],
            ready_for_notice=ready,
            facts_confirmed=record.facts_confirmed,
            notice_available=record.notice is not None,
            created_at=record.created_at,
            updated_at=record.updated_at,
        )

    def _readiness_missing(self, record: ConsumerCaseRecord) -> list[str]:
        missing = list(record.facts.missing_fields())
        if not self._has_accepted_evidence(record):
            missing.append("accepted_evidence")
        if not record.facts_confirmed:
            missing.append("facts_confirmation")
        return missing

    @staticmethod
    def _accepted_documents(record: ConsumerCaseRecord) -> list[StoredEvidence]:
        return [item for item in record.documents if item.safe_document is not None]

    def _has_accepted_evidence(self, record: ConsumerCaseRecord) -> bool:
        return bool(self._accepted_documents(record))

    def _combined_evidence(
        self, record: ConsumerCaseRecord
    ) -> tuple[ParsedDocument, dict[int, tuple[StoredEvidence, int]]]:
        pages: list[DocumentPage] = []
        sources: dict[int, tuple[StoredEvidence, int]] = {}
        for evidence in self._accepted_documents(record):
            assert evidence.safe_document is not None
            for original in evidence.safe_document.pages:
                global_page = len(pages) + 1
                heading = (
                    f"CASO {record.case_id[:8]} EVIDENCIA "
                    f"{evidence.public.evidence_id[:8]} PAGINA {original.number}"
                )
                pages.append(
                    DocumentPage(number=global_page, text=f"{heading}\n\n{original.text}")
                )
                sources[global_page] = (evidence, original.number)
        return (
            ParsedDocument(
                filename=f"consumer_case_{record.case_id}_evidence.pdf",
                pages=pages,
                language="pt",
                extraction_method=ExtractionMethod.NATIVE_TEXT,
                warnings=[],
            ),
            sources,
        )

    def _legal_grounds(
        self,
        result_sets: list[list[RetrievedChunk]],
        facts: ConsumerCaseFacts,
    ) -> list[LegalGround]:
        merged = _merge_results(result_sets)
        grounds: list[LegalGround] = []
        seen: set[str] = set()
        issue = _CATEGORY_LABEL[
            facts.issue_category.value if facts.issue_category else "other"
        ]
        for rank, result in enumerate(merged, start=1):
            for provision in self._legal_corpus.provisions_for_chunk(result):
                if provision.status is not ProvisionStatus.ACTIVE:
                    continue
                if provision.provision_id in seen:
                    continue
                seen.add(provision.provision_id)
                authority = LegalAuthorityCitation.from_provision(
                    provision,
                    chunk_id=result.chunk.chunk_id,
                    retrieval_rank=rank,
                    retrieval_score=result.score,
                )
                grounds.append(
                    LegalGround(
                        authority=authority,
                        application_to_facts=(
                            f"A regra resumida em {provision.citation_label} é pertinente "
                            f"à alegação de {issue} "
                            "e deve ser confrontada com os documentos citados."
                        ),
                    )
                )
                if len(grounds) >= 8:
                    return grounds
        return grounds

    @staticmethod
    def _evidence_references(
        result_sets: list[list[RetrievedChunk]],
        page_sources: dict[int, tuple[StoredEvidence, int]],
    ) -> list[EvidenceCitation]:
        citations: list[EvidenceCitation] = []
        for result in _merge_results(result_sets):
            source = page_sources.get(result.chunk.page_start)
            if source is None:
                continue
            evidence, original_page = source
            quote = _clean_chunk_quote(result.chunk.text)
            if not quote:
                continue
            citations.append(
                EvidenceCitation(
                    evidence_id=evidence.public.evidence_id,
                    filename=evidence.public.filename,
                    page=original_page,
                    quote=quote,
                    chunk_id=result.chunk.chunk_id,
                    content_sha256=hashlib.sha256(
                        result.chunk.text.encode("utf-8")
                    ).hexdigest(),
                )
            )
            if len(citations) >= 8:
                break
        return citations


def _evidence_status(action: SecurityAction) -> EvidenceStatus:
    return {
        SecurityAction.PROCEED: EvidenceStatus.ACCEPTED,
        SecurityAction.PROCEED_WITH_WARNING: EvidenceStatus.ACCEPTED_WITH_WARNING,
        SecurityAction.HUMAN_REVIEW: EvidenceStatus.REVIEW_REQUIRED,
        SecurityAction.BLOCK: EvidenceStatus.BLOCKED,
    }[action]


def _merge_results(result_sets: list[list[RetrievedChunk]]) -> list[RetrievedChunk]:
    best: dict[str, RetrievedChunk] = {}
    for results in result_sets:
        for result in results:
            current = best.get(result.chunk.chunk_id)
            if current is None or result.score > current.score:
                best[result.chunk.chunk_id] = result
    return sorted(best.values(), key=lambda item: (-item.score, item.chunk.chunk_id))


def _annotate_composer_selection(
    traces: list[RetrievalTrace],
    merged: list[RetrievedChunk],
    included_chunk_ids: set[str],
) -> list[RetrievalTrace]:
    """Record which ranked hits became grounded composer inputs."""
    merged_ranks = {
        item.chunk.chunk_id: rank for rank, item in enumerate(merged, start=1)
    }
    winners: dict[str, tuple[int, int, float]] = {}
    for trace_index, trace in enumerate(traces):
        for item in trace.results:
            current = winners.get(item.chunk_id)
            if current is None or item.score > current[2]:
                winners[item.chunk_id] = (trace_index, item.rank, item.score)

    annotated: list[RetrievalTrace] = []
    truncated = any(item.chunk.chunk_id not in included_chunk_ids for item in merged)
    for trace_index, trace in enumerate(traces):
        results = []
        for item in trace.results:
            winner = winners.get(item.chunk_id)
            selected = winner is not None and winner[:2] == (trace_index, item.rank)
            results.append(
                item.model_copy(
                    update={
                        "selected_for_merge": selected,
                        "merged_rank": merged_ranks.get(item.chunk_id),
                        "included_in_context": (
                            selected and item.chunk_id in included_chunk_ids
                        ),
                    }
                )
            )
        annotated.append(
            trace.model_copy(
                update={
                    "context_truncated": truncated,
                    "agent_status": AgentStatus.SUCCESS,
                    "prompt_version": "consumer-grounded-composer:v1",
                    "results": results,
                }
            )
        )
    return annotated


def _clean_chunk_quote(text: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].startswith("[") and lines[0].endswith("]"):
        lines = lines[1:]
    return " ".join(" ".join(lines).split())[:700]


def _requests(facts: ConsumerCaseFacts) -> list[str]:
    requests = [facts.desired_resolution or "solução integral do problema relatado"]
    if facts.direct_loss_amount and facts.direct_loss_amount > 0:
        requests.append(
            "restituição do prejuízo direto alegado, após conferência dos comprovantes"
        )
    if facts.prior_protocols:
        requests.append("resposta escrita e fundamentada aos protocolos já registrados")
    requests.append("confirmação escrita das providências adotadas dentro do prazo indicado")
    return requests


def _render_notice_markdown(
    *,
    facts: ConsumerCaseFacts,
    evidence: list[EvidenceCitation],
    legal_grounds: list[LegalGround],
    requests: list[str],
    public_proposal: Decimal | None,
) -> str:
    name = facts.consumer_name or "[PREENCHER NOME DO(A) CONSUMIDOR(A)]"
    bank = facts.bank_name or "[PREENCHER INSTITUIÇÃO FINANCEIRA]"
    subject = _CATEGORY_LABEL[
        facts.issue_category.value if facts.issue_category else "other"
    ]
    protocols = ", ".join(facts.prior_protocols) or "nenhum protocolo informado"
    lines = [
        "# NOTIFICAÇÃO EXTRAJUDICIAL COM PROPOSTA DE ACORDO",
        "",
        f"**Notificante:** {name}",
        f"**Notificada:** {bank}",
        f"**Assunto:** {subject}",
        "",
        "## 1. Finalidade",
        "",
        "Esta notificação busca solução consensual de uma controvérsia de consumo. "
        "Não se trata de ação judicial nem de reconhecimento definitivo de responsabilidade.",
        "",
        "## 2. Fatos declarados pelo(a) consumidor(a)",
        "",
        facts.complaint_summary or "[PREENCHER RELATO]",
        "",
        f"**Data ou período:** {facts.incident_date_or_period or '[PREENCHER]'}",
        f"**Protocolos anteriores:** {protocols}",
        "",
        "## 3. Documentos de suporte",
        "",
    ]
    for item in evidence:
        lines.append(
            f"- **{item.filename}, p. {item.page}** — {item.quote} "
            f"(chunk `{item.chunk_id}`)"
        )
    lines.extend(["", "## 4. Fundamentos jurídicos", ""])
    for ground in legal_grounds:
        authority = ground.authority
        lines.append(
            f"- **[{authority.citation_label}]({authority.official_url})** — "
            f"{authority.summary} Aplicação: {ground.application_to_facts} "
            f"(corpus `{authority.corpus_release_id}`, SHA-256 `{authority.content_sha256}`)"
        )
    lines.extend(["", "## 5. Providências solicitadas", ""])
    lines.extend(f"- {request}" for request in requests)
    lines.extend(["", "## 6. Proposta para composição", ""])
    if public_proposal is None:
        lines.append(
            "Neste momento, propõe-se solução não monetária nos termos dos pedidos acima, "
            "sem atribuição automática de indenização."
        )
    else:
        lines.append(
            f"Para tentativa de composição, propõe-se o valor de **R$ {_brl(public_proposal)}**, "
            "sujeito à conferência dos comprovantes e à revisão humana. O valor é uma âncora "
            "de negociação calculada por cenário, não uma previsão de decisão judicial."
        )
    lines.extend(
        [
            "",
            "## 7. Prazo e encerramento",
            "",
            "Solicita-se resposta escrita em até "
            f"**{facts.response_deadline_business_days} dias úteis**. "
            "A ausência de acordo não altera direitos, defesas ou prazos legais de qualquer parte.",
            "",
            "---",
            "",
            f"> {CONSUMER_NOTICE_WARNING}",
        ]
    )
    markdown = "\n".join(lines)
    # The private reservation value is intentionally unavailable to this
    # renderer, so it cannot leak into an exported notice by accident.
    return markdown


def _brl(value: Decimal) -> str:
    rendered = f"{value:,.2f}"
    return rendered.replace(",", "_").replace(".", ",").replace("_", ".")
