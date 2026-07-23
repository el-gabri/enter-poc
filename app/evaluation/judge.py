"""LLM-as-judge for response quality.

Judgment goes through the same LLMClient port as everything else, so it
works with any provider - including the mock (useful to test the plumbing;
meaningless as a score, which the mock's 0.0 makes obvious).
"""

from pydantic import BaseModel, Field, field_validator

from app.llm.base import LLMClient
from app.prompts.judge import JUDGE_PROMPT
from app.schemas.evaluation import MetricResult
from app.schemas.report import LitigationReport

MAX_EXCERPT = 6_000


class JudgeVerdict(BaseModel):
    quality: float = Field(description="0.0 to 1.0")
    reasoning: str

    @field_validator("quality")
    @classmethod
    def _clamp(cls, v: float) -> float:
        return max(0.0, min(1.0, v))


async def judge_response_quality(
    llm: LLMClient, report: LitigationReport, document_text: str
) -> MetricResult:
    report_excerpt = report.model_dump_json(
        include={
            "executive_summary",
            "main_claims",
            "legal_risks",
            "suggested_strategy",
            "missing_information",
        }
    )[:MAX_EXCERPT]
    result = await llm.parse(
        system=JUDGE_PROMPT.system,
        user=JUDGE_PROMPT.render_user(
            document_excerpt=document_text[:MAX_EXCERPT],
            report_excerpt=report_excerpt,
        ),
        schema=JudgeVerdict,
        prompt_version=f"{JUDGE_PROMPT.name}:{JUDGE_PROMPT.version}",
    )
    return MetricResult(
        name="response_quality",
        score=round(result.data.quality, 3),
        details=result.data.reasoning,
    )
