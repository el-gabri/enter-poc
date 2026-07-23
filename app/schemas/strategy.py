"""Legal strategy schemas."""

from enum import Enum

from pydantic import BaseModel, Field

from app.schemas.common import ConfidentConclusion


class ActionPriority(str, Enum):
    URGENT = "urgent"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class DefenseOption(BaseModel):
    """A possible line of defense."""

    argument: str = Field(description="The defense argument")
    legal_basis: str | None = Field(
        default=None, description="Statute/article/precedent supporting it"
    )
    assessment: ConfidentConclusion = Field(
        description="Viability of this defense, with reasoning"
    )


class RecommendedAction(BaseModel):
    """A concrete next step for the legal team."""

    action: str
    priority: ActionPriority
    rationale: str = Field(description="Why this action, why this priority")


class StrategyPlan(BaseModel):
    """Suggested initial strategy. Decision support, never a decision."""

    overall_approach: ConfidentConclusion = Field(
        description="Recommended posture (contest / negotiate / hybrid) with reasoning"
    )
    defenses: list[DefenseOption] = Field(default_factory=list)
    settlement: ConfidentConclusion = Field(
        description="Settlement recommendation: whether to pursue, plausible "
        "range if derivable from the document, and why"
    )
    next_actions: list[RecommendedAction] = Field(default_factory=list)
    missing_information: list[str] = Field(
        default_factory=list,
        description="Information not in the document that the legal team "
        "should obtain before finalizing the strategy",
    )
