"""CLI entry point: ``python -m app.evaluation [golden_dir]``.

Runs the evaluation suite with the configured provider and prints a
summary table. With LITIGATION_LLM_PROVIDER=mock it exercises the full
pipeline offline (pipeline health check); with a real provider it measures
actual quality, including the LLM judge.
"""

import asyncio
import sys
from pathlib import Path

from app.core.config import LLMProvider, get_settings
from app.core.logging import configure_logging
from app.evaluation.golden import load_dataset
from app.evaluation.runner import EvaluationRunner
from app.llm.factory import create_llm_client
from app.orchestration.graph import build_analysis_graph
from app.rag.factory import create_rag_pipeline


async def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    golden_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("eval_data")

    llm = create_llm_client(settings)
    graph = build_analysis_graph(llm, create_rag_pipeline(settings))
    judge_llm = llm if settings.llm_provider is not LLMProvider.MOCK else None

    runner = EvaluationRunner(graph, judge_llm=judge_llm)
    summary = await runner.run(load_dataset(golden_dir))

    print("\n=== Evaluation summary ===")
    for name, score in sorted(summary.averages.items()):
        print(f"{name:>24}: {score:.3f}")
    print(f"{'cases':>24}: {len(summary.cases)}")
    for case in summary.cases:
        status = "ERRORS" if case.errors else "ok"
        print(f"\n[{status}] {case.case_name}")
        for metric in case.metrics:
            print(f"    {metric.name:>24}: {metric.score:.3f}  {metric.details}")


if __name__ == "__main__":
    asyncio.run(main())
