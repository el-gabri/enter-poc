"""Evaluation schemas."""

from pydantic import BaseModel, Field


class MetricResult(BaseModel):
    """One metric computed for one case."""

    name: str
    score: float = Field(description="0.0 (worst) to 1.0 (best)")
    details: str = Field(default="", description="Human-readable explanation")


class CaseResult(BaseModel):
    """All metrics for one golden case."""

    case_name: str
    metrics: list[MetricResult] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    def score(self, metric_name: str) -> float | None:
        for metric in self.metrics:
            if metric.name == metric_name:
                return metric.score
        return None


class EvaluationSummary(BaseModel):
    """Aggregate over all cases in a run."""

    cases: list[CaseResult] = Field(default_factory=list)
    averages: dict[str, float] = Field(default_factory=dict)

    @classmethod
    def from_cases(cls, cases: list[CaseResult]) -> "EvaluationSummary":
        totals: dict[str, list[float]] = {}
        for case in cases:
            for metric in case.metrics:
                totals.setdefault(metric.name, []).append(metric.score)
        averages = {
            name: round(sum(scores) / len(scores), 3)
            for name, scores in totals.items()
        }
        return cls(cases=cases, averages=averages)
