"""Typed contracts for the consumer-side extrajudicial-notice workflow.

Evidence citations and legal-authority citations deliberately use different
models. A document excerpt can establish a fact; it cannot become a legal
authority merely because it was retrieved by the same RAG pipeline.
"""

from __future__ import annotations

import hashlib
from datetime import date, datetime, timezone
from decimal import Decimal
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator

from app.schemas.security import PromptInjectionAssessment
from app.schemas.trace import RetrievalTrace


class ConsumerIssueCategory(str, Enum):
    """Bank-first complaint categories supported by the initial intake."""

    UNAUTHORIZED_CHARGE = "unauthorized_charge"
    FRAUD = "fraud"
    ACCOUNT_BLOCK = "account_block"
    NEGATIVE_CREDIT_RECORD = "negative_credit_record"
    LOAN_OR_INTEREST = "loan_or_interest"
    SERVICE_FAILURE = "service_failure"
    OVER_INDEBTEDNESS = "over_indebtedness"
    OTHER = "other"


class ConsumerMessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


class EvidenceStatus(str, Enum):
    ACCEPTED = "accepted"
    ACCEPTED_WITH_WARNING = "accepted_with_warning"
    REVIEW_REQUIRED = "review_required"
    BLOCKED = "blocked"


class ConsumerCaseStatus(str, Enum):
    COLLECTING_FACTS = "collecting_facts"
    COLLECTING_EVIDENCE = "collecting_evidence"
    READY_FOR_NOTICE = "ready_for_notice"
    NOTICE_GENERATED = "notice_generated"


class LegalSource(str, Enum):
    FEDERAL_CONSTITUTION = "federal_constitution"
    CONSUMER_DEFENSE_CODE = "consumer_defense_code"


class ProvisionStatus(str, Enum):
    ACTIVE = "active"
    VETOED = "vetoed"


class ConsumerMessage(BaseModel):
    role: ConsumerMessageRole
    content: str = Field(min_length=1, max_length=20_000)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ConsumerCaseFacts(BaseModel):
    """Facts supplied or explicitly confirmed by the consumer.

    Monetary fields have different meanings and must not be silently merged:
    direct loss is evidence-backed loss, while requested compensation is the
    consumer's own negotiation input.
    """

    consumer_name: str | None = Field(default=None, max_length=200)
    bank_name: str | None = Field(default=None, max_length=200)
    issue_category: ConsumerIssueCategory | None = None
    complaint_summary: str | None = Field(default=None, max_length=10_000)
    incident_date_or_period: str | None = Field(default=None, max_length=500)
    prior_protocols: list[str] = Field(default_factory=list)
    direct_loss_amount: Decimal | None = Field(default=None, ge=0)
    improper_payment_amount: Decimal | None = Field(
        default=None,
        ge=0,
        description=(
            "Part of direct_loss_amount actually paid under an allegedly improper "
            "charge; used only for the conditional CDC art. 42 scenario"
        ),
    )
    article_42_double_repayment_requested: bool = False
    requested_compensation_amount: Decimal | None = Field(
        default=None,
        ge=0,
        description="Consumer-supplied negotiation input; never inferred by the system",
    )
    unsuccessful_scenario_cost_amount: Decimal | None = Field(
        default=None,
        ge=0,
        description=(
            "Consumer-supplied estimate of explicit cost if no agreement is reached"
        ),
    )
    desired_resolution: str | None = Field(default=None, max_length=2_000)
    response_deadline_business_days: int = Field(default=10, ge=1, le=60)

    @field_validator("prior_protocols")
    @classmethod
    def _clean_protocols(cls, values: list[str]) -> list[str]:
        return [value.strip()[:200] for value in values if value.strip()]

    @model_validator(mode="after")
    def _validate_improper_payment_subset(self) -> ConsumerCaseFacts:
        if (
            self.improper_payment_amount is not None
            and self.direct_loss_amount is not None
            and self.improper_payment_amount > self.direct_loss_amount
        ):
            raise ValueError("improper_payment_amount cannot exceed direct_loss_amount")
        return self

    def missing_fields(self) -> list[str]:
        """Return information needed before drafting a useful notice."""
        required = {
            "bank_name": self.bank_name,
            "consumer_name": self.consumer_name,
            "issue_category": self.issue_category,
            "complaint_summary": self.complaint_summary,
            "incident_date_or_period": self.incident_date_or_period,
            "desired_resolution": self.desired_resolution,
        }
        return [name for name, value in required.items() if value is None or value == ""]


class ConsumerIntakeExtraction(BaseModel):
    """Schema-constrained extraction result; every field is optional."""

    consumer_name: str | None = None
    bank_name: str | None = None
    issue_category: ConsumerIssueCategory | None = None
    complaint_summary: str | None = None
    incident_date_or_period: str | None = None
    prior_protocols: list[str] | None = None
    direct_loss_amount: Decimal | None = Field(default=None, ge=0)
    improper_payment_amount: Decimal | None = Field(default=None, ge=0)
    article_42_double_repayment_requested: bool | None = None
    requested_compensation_amount: Decimal | None = Field(default=None, ge=0)
    unsuccessful_scenario_cost_amount: Decimal | None = Field(default=None, ge=0)
    desired_resolution: str | None = None


class ConsumerEvidence(BaseModel):
    evidence_id: str
    filename: str
    page_count: int = Field(ge=0)
    status: EvidenceStatus
    content_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    security_assessment: PromptInjectionAssessment | None = None
    warnings: list[str] = Field(default_factory=list)


class EvidenceCitation(BaseModel):
    """A source-verifiable fact citation from consumer-supplied evidence."""

    evidence_id: str
    filename: str
    page: int = Field(ge=1)
    page_end: int | None = Field(default=None, ge=1)
    quote: str = Field(min_length=1, max_length=1_000)
    chunk_id: str
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _valid_page_range(self) -> EvidenceCitation:
        if self.page_end is not None and self.page_end < self.page:
            raise ValueError("page_end cannot precede page")
        return self


class LegalProvision(BaseModel):
    """Reviewed original summary of a provision, not a statutory quotation."""

    provision_id: str = Field(pattern=r"^br-(cf|cdc)-[a-z0-9-]+$")
    source: LegalSource
    source_name: str
    article: str
    citation_label: str
    summary: str = Field(min_length=1)
    official_url: str = Field(pattern=r"^https://www\.planalto\.gov\.br/")
    tags: list[str] = Field(default_factory=list)
    corpus_release_id: str
    verified_on: date
    status: ProvisionStatus = ProvisionStatus.ACTIVE
    content_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _set_and_validate_hash(self) -> LegalProvision:
        expected = hashlib.sha256(self.summary.encode("utf-8")).hexdigest()
        if self.content_sha256 is None:
            self.content_sha256 = expected
        elif self.content_sha256 != expected:
            raise ValueError("content_sha256 does not match summary")
        return self


class LegalAuthorityCitation(BaseModel):
    """Citation to reviewed law metadata, kept separate from case evidence."""

    provision_id: str
    source_name: str
    article: str
    citation_label: str
    summary: str
    official_url: str
    corpus_release_id: str
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: ProvisionStatus = ProvisionStatus.ACTIVE
    chunk_id: str | None = None
    retrieval_rank: int | None = Field(default=None, ge=1)
    retrieval_score: float | None = None

    @classmethod
    def from_provision(
        cls,
        provision: LegalProvision,
        *,
        chunk_id: str | None = None,
        retrieval_rank: int | None = None,
        retrieval_score: float | None = None,
    ) -> LegalAuthorityCitation:
        if provision.content_sha256 is None:  # pragma: no cover - validator guarantees it
            raise ValueError("provision has no content hash")
        return cls(
            provision_id=provision.provision_id,
            source_name=provision.source_name,
            article=provision.article,
            citation_label=provision.citation_label,
            summary=provision.summary,
            official_url=provision.official_url,
            corpus_release_id=provision.corpus_release_id,
            content_sha256=provision.content_sha256,
            status=provision.status,
            chunk_id=chunk_id,
            retrieval_rank=retrieval_rank,
            retrieval_score=retrieval_score,
        )


class LegalGround(BaseModel):
    authority: LegalAuthorityCitation
    application_to_facts: str = Field(min_length=1, max_length=4_000)


class SettlementInputs(BaseModel):
    """Explicit inputs to a deterministic, non-predictive negotiation scenario."""

    direct_loss_amount: Decimal = Field(default=Decimal("0"), ge=0)
    improper_payment_amount: Decimal = Field(default=Decimal("0"), ge=0)
    requested_compensation_amount: Decimal = Field(default=Decimal("0"), ge=0)
    downside_cost_amount: Decimal = Field(
        default=Decimal("0"),
        ge=0,
        description="Explicit cost assigned to the unsuccessful scenario",
    )
    article_42_double_repayment_supported: bool = False
    evidence_strength: Decimal = Field(default=Decimal("0.5"), ge=0, le=1)
    factual_completeness: Decimal = Field(default=Decimal("0.5"), ge=0, le=1)
    public_proposal_override: Decimal | None = Field(default=None, ge=0)
    private_reservation_override: Decimal | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _validate_inputs(self) -> SettlementInputs:
        if self.improper_payment_amount > self.direct_loss_amount:
            raise ValueError("improper_payment_amount cannot exceed direct_loss_amount")
        if (
            self.public_proposal_override is not None
            and self.private_reservation_override is not None
            and self.public_proposal_override < self.private_reservation_override
        ):
            raise ValueError("public proposal cannot be lower than private reservation")
        return self


class SettlementScenario(BaseModel):
    """Illustrative negotiation math; weights are not predicted legal odds."""

    methodology_version: str
    calibrated: bool = False
    is_legal_outcome_prediction: bool = False
    direct_loss_amount: Decimal = Field(ge=0)
    improper_payment_amount: Decimal = Field(ge=0)
    requested_compensation_amount: Decimal = Field(ge=0)
    downside_cost_amount: Decimal = Field(ge=0)
    unsuccessful_outcome_value: Decimal
    conditional_article_42_increment_amount: Decimal = Field(ge=0)
    low_outcome_value: Decimal = Field(ge=0)
    high_outcome_value: Decimal = Field(ge=0)
    exploratory_weight_low: Decimal = Field(ge=0, le=1)
    exploratory_weight_high: Decimal = Field(ge=0, le=1)
    illustrative_expected_value_low: Decimal
    illustrative_expected_value_high: Decimal
    public_proposal_amount: Decimal | None = Field(default=None, ge=0)
    private_reservation_amount: Decimal | None = Field(default=None, ge=0)
    article_42_assumption: str
    methodology: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)


class ConsumerCaseSnapshot(BaseModel):
    case_id: str
    status: ConsumerCaseStatus = ConsumerCaseStatus.COLLECTING_FACTS
    messages: list[ConsumerMessage] = Field(default_factory=list)
    facts: ConsumerCaseFacts = Field(default_factory=ConsumerCaseFacts)
    missing_fields: list[str] = Field(default_factory=list)
    recommended_documents: list[str] = Field(default_factory=list)
    documents: list[ConsumerEvidence] = Field(default_factory=list)
    ready_for_notice: bool = False
    facts_confirmed: bool = False
    notice_available: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ConsumerNotice(BaseModel):
    """Auditable extrajudicial notice draft, not a filed lawsuit."""

    notice_id: str
    case_id: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    title: str = "Notificação extrajudicial com proposta de acordo"
    addressee: str
    facts_summary: str
    evidence_references: list[EvidenceCitation] = Field(default_factory=list)
    legal_grounds: list[LegalGround] = Field(default_factory=list)
    requests: list[str] = Field(default_factory=list)
    response_deadline_business_days: int = Field(ge=1, le=60)
    settlement: SettlementScenario
    full_text: str
    corpus_release_id: str
    retrievals: list[RetrievalTrace] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
