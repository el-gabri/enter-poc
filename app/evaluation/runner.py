"""Evaluation runner: golden cases -> pipeline -> metrics -> summary."""

from app.core.logging import get_logger
from app.evaluation import metrics
from app.evaluation.golden import GoldenCase
from app.evaluation.judge import judge_response_quality
from app.llm.base import LLMClient
from app.orchestration.state import AnalysisState
from app.rag.pipeline import RagPipeline
from app.schemas.evaluation import CaseResult, EvaluationSummary

logger = get_logger(__name__)


class EvaluationRunner:
    """Runs the analysis graph over golden cases and scores the results."""

    def __init__(
        self,
        graph: object,
        judge_llm: LLMClient | None = None,
        rag: RagPipeline | None = None,
    ) -> None:
        self._graph = graph
        self._judge_llm = judge_llm
        self._rag = rag

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
        if case.retrieval_judgments and self._rag is not None:
            rankings = await self._rag.retrieve_many(
                [judgment.query for judgment in case.retrieval_judgments],
                doc_id=case.document.doc_id,
                k=case.retrieval_k,
            )
            relevant_groups = [
                metrics.relevant_chunk_groups(
                    state.chunks,
                    page_ranges=judgment.relevant_page_ranges,
                    passages=judgment.relevant_passages,
                )
                for judgment in case.retrieval_judgments
            ]
            case_metrics.extend(
                metrics.retrieval_metrics_at_k(
                    rankings,
                    relevant_groups,
                    k=case.retrieval_k,
                )
            )
        elif case.retrieval_judgments:
            logger.warning(
                "retrieval_evaluation_skipped",
                case=case.name,
                reason="EvaluationRunner was not configured with a RagPipeline",
            )
        if self._judge_llm is not None:
            case_metrics.append(
                await judge_response_quality(self._judge_llm, report, text)
            )
        return CaseResult(case_name=case.name, metrics=case_metrics, errors=state.errors)
