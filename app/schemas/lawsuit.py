"""Schemas describing the structured content of a lawsuit.

These models double as the ``response_format`` for the extraction agent, so
field descriptions are written for the LLM as much as for humans. Every
field that may be absent from a document is Optional - the extractor must
say "not found" instead of guessing (anti-hallucination by schema design).
"""

from enum import Enum

from pydantic import BaseModel, Field


class LawsuitType(str, Enum):
    CONSUMER = "consumer"
    LABOR = "labor"
    CIVIL = "civil"
    BANKING = "banking"
    INSURANCE = "insurance"
    HEALTHCARE = "healthcare"
    OTHER = "other"


class PartyRole(str, Enum):
    PLAINTIFF = "plaintiff"  # autor / reclamante / requerente
    DEFENDANT = "defendant"  # reu / reclamada / requerido
    THIRD_PARTY = "third_party"


class Party(BaseModel):
    """A party to the lawsuit."""

    name: str = Field(description="Full name or corporate name as written")
    role: PartyRole
    document_id: str | None = Field(
        default=None, description="CPF or CNPJ if present in the document"
    )
    lawyer: str | None = Field(
        default=None, description="Lead counsel name (with OAB number if shown)"
    )
    law_firm: str | None = None


class MonetaryValue(BaseModel):
    """An amount of money as stated in the document."""

    amount: float = Field(description="Numeric amount, e.g. 50000.00")
    currency: str = Field(default="BRL", description="ISO 4217 code")
    as_written: str | None = Field(
        default=None,
        description="Original text, e.g. 'R$ 50.000,00 (cinquenta mil reais)'",
    )


class Deadline(BaseModel):
    """A deadline or key date mentioned in the document."""

    description: str = Field(description="What the deadline refers to")
    date: str | None = Field(
        default=None, description="ISO date (YYYY-MM-DD) if determinable"
    )
    page: int | None = Field(default=None, description="Page where it appears")


class LawsuitExtraction(BaseModel):
    """Structured information extracted from a lawsuit document.

    A None / empty value means "not found in the document" - never invent.
    """

    case_number: str | None = Field(
        default=None,
        description="CNJ unified numbering, e.g. 0000000-00.0000.0.00.0000",
    )
    court: str | None = Field(
        default=None, description="Court / vara, e.g. '3a Vara Civel de Sao Paulo'"
    )
    state: str | None = Field(default=None, description="Brazilian state (UF)")
    judge: str | None = None
    filing_date: str | None = Field(
        default=None, description="ISO date (YYYY-MM-DD) if determinable"
    )
    claim_value: MonetaryValue | None = Field(
        default=None, description="Valor da causa"
    )
    parties: list[Party] = Field(default_factory=list)
    main_requests: list[str] = Field(
        default_factory=list, description="Pedidos - what the plaintiff asks for"
    )
    legal_grounds: list[str] = Field(
        default_factory=list,
        description="Legal bases invoked (statutes, articles, sumulas)",
    )
    deadlines: list[Deadline] = Field(default_factory=list)

    def missing_fields(self) -> list[str]:
        """Names of scalar fields not found in the document.

        Feeds the 'Missing Information' section of the final report.
        """
        scalars = ["case_number", "court", "state", "judge", "filing_date", "claim_value"]
        missing = [name for name in scalars if getattr(self, name) is None]
        if not self.parties:
            missing.append("parties")
        if not self.main_requests:
            missing.append("main_requests")
        return missing
