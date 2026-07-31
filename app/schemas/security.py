"""Typed results produced by the document prompt-injection gate."""

from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator


class InjectionCategory(str, Enum):
    """Kinds of attempts to influence an LLM through document content."""

    INSTRUCTION_OVERRIDE = "instruction_override"
    ROLE_MANIPULATION = "role_manipulation"
    DATA_EXFILTRATION = "data_exfiltration"
    TOOL_MANIPULATION = "tool_manipulation"
    OUTPUT_MANIPULATION = "output_manipulation"
    OBFUSCATED_PAYLOAD = "obfuscated_payload"


class SecurityRiskLevel(str, Enum):
    """Aggregate prompt-injection risk for one document."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityAction(str, Enum):
    """Deterministic action selected from the aggregate risk level."""

    PROCEED = "proceed"
    PROCEED_WITH_WARNING = "proceed_with_warning"
    HUMAN_REVIEW = "human_review"
    BLOCK = "block"

    @property
    def allows_automated_analysis(self) -> bool:
        return self in {SecurityAction.PROCEED, SecurityAction.PROCEED_WITH_WARNING}


class FindingSource(str, Enum):
    """Detector that produced a finding."""

    RULE = "rule"
    SEMANTIC = "semantic"


class PromptInjectionFinding(BaseModel):
    """One suspicious, source-verifiable passage from a document."""

    category: InjectionCategory
    severity: SecurityRiskLevel
    page: int
    page_end: int | None = None
    quote: str
    reasoning: str
    confidence: float
    source: FindingSource = FindingSource.SEMANTIC
    rule_id: str | None = None

    @field_validator("page")
    @classmethod
    def _positive_page(cls, value: int) -> int:
        if value < 1:
            raise ValueError("page must be at least 1")
        return value

    @field_validator("quote")
    @classmethod
    def _bounded_quote(cls, value: str) -> str:
        if not value:
            raise ValueError("quote cannot be empty")
        return value[:1_000]

    @field_validator("reasoning")
    @classmethod
    def _bounded_reasoning(cls, value: str) -> str:
        if not value:
            raise ValueError("reasoning cannot be empty")
        return value[:500]

    @field_validator("confidence")
    @classmethod
    def _clamp_confidence(cls, value: float) -> float:
        return max(0.0, min(1.0, value))

    @model_validator(mode="after")
    def _valid_page_range(self) -> "PromptInjectionFinding":
        if self.page_end is not None and self.page_end < self.page:
            raise ValueError("page_end cannot precede page")
        return self


class SemanticPromptInjectionReview(BaseModel):
    """Schema-constrained output of the optional semantic reviewer."""

    findings: list[PromptInjectionFinding] = Field(default_factory=list)


class PromptInjectionAssessment(BaseModel):
    """Complete security-gate result attached to state and final reports."""

    detected: bool = False
    risk_level: SecurityRiskLevel = SecurityRiskLevel.NONE
    recommended_action: SecurityAction = SecurityAction.PROCEED
    findings: list[PromptInjectionFinding] = Field(default_factory=list)
    scanned_pages: int = Field(default=0, ge=0)
    scan_complete: bool = True
    scan_mode: str = "balanced"
    semantic_reviewed: bool = False
    warnings: list[str] = Field(default_factory=list)
