"""Evaluation runner: golden cases -> pipeline -> metrics -> summary."""

from app.core.logging import get_logger
from app.evaluation import metrics
from app.evaluation.golden import GoldenCase
from app.evaluation.judge import judge_response_quality
from app.llm.base import LLMClient
from app.orchestration.state import AnalysisState
from app.schemas.evaluation import CaseResult, EvaluationSummary

logger = get_logger(__name__)


class EvaluationRunner:
    """Runs the analysis graph over golden cases and scores the results."""

    def __init__(
        self, graph: object, judge_llm: LLMClient | None = None
    ) -> None:
        self._graph = graph
        self._judge_llm = judge_llm

    async def run(self, cases: list[GoldenCase]) -> EvaluationSummary:
        results = [await self._run_case(case) for case in cases]
        summary = EvaluationSummary.from_cases(results)
        logger.info("evaluation_finished", cases=len(results), averages=summary.averages)
        return summary

    async def _run_case(self, case: GoldenCase) -> CaseResult:
        raw = await self._graph.ainvoke(AnalysisState(document=case.document))  # type: ignore[attr-defined]
        state = AnalysisState(**raw)
        if state.report is None:
            return CaseResult(
                case_name=case.name,
                errors=state.errors or ["pipeline produced no report"],
            )

        report = state.report
        text = case.document.full_text
        case_metrics = [
            metrics.groundedness(report, text),
            metrics.hallucination_rate(report, text),
            metrics.citation_coverage(report),
            metrics.extraction_accuracy(state.extraction, case.expected),
            metrics.completeness(state.extraction, case.expected),
        ]
        if (cls_metric := metrics.classification_accuracy(report, case.expected)):
            case_metrics.append(cls_metric)
        if self._judge_llm is not None:
            case_metrics.append(
                await judge_response_quality(self._judge_llm, report, text)
            )
        return CaseResult(case_name=case.name, metrics=case_metrics, errors=state.errors)
