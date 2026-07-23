"""DataJud (CNJ) enrichment schemas."""

from pydantic import BaseModel, Field


class DataJudMovement(BaseModel):
    code: int | None = None
    name: str
    date: str | None = None


class DataJudCaseInfo(BaseModel):
    """Official court-record metadata for a case."""

    case_number: str
    tribunal: str | None = None
    court_class: str | None = Field(default=None, description="Classe processual")
    subjects: list[str] = Field(default_factory=list, description="Assuntos")
    court_body: str | None = Field(default=None, description="Orgao julgador")
    degree: str | None = Field(default=None, description="Grau (G1/G2...)")
    filing_date: str | None = None
    last_update: str | None = None
    movement_count: int = 0
    latest_movement: DataJudMovement | None = None


class DataJudEnrichment(BaseModel):
    """Result of validating the extracted case number against DataJud.

    ``attempted=False`` means we had no case number or no API key -
    enrichment is always optional and never fails the pipeline.
    """

    attempted: bool
    found: bool = False
    tribunal_alias: str | None = None
    info: DataJudCaseInfo | None = None
    notes: list[str] = Field(default_factory=list)
