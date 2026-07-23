"""Tests for the run store."""

from pathlib import Path

from app.observability.store import RunRecord, RunStore
from app.schemas.report import RunMetrics


def _record(run_id: str, cost: float, success: bool = True) -> RunRecord:
    return RunRecord(
        run_id=run_id,
        doc_id=f"doc-{run_id}",
        filename="x.pdf",
        success=success,
        metrics=RunMetrics(total_cost_usd=cost, total_tokens=100, agents_run=5),
    )


def test_run_store_roundtrip_and_totals(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs.jsonl")
    store.append(_record("a", 0.01))
    store.append(_record("b", 0.02, success=False))

    runs = store.list_runs()
    assert len(runs) == 2
    assert {r.run_id for r in runs} == {"a", "b"}

    totals = store.totals()
    assert totals["runs"] == 2
    assert totals["failures"] == 1
    assert totals["total_cost_usd"] == 0.03
    assert totals["total_tokens"] == 200


def test_empty_store(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs.jsonl")
    assert store.list_runs() == []
    assert store.totals()["runs"] == 0
