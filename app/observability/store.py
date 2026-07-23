"""Persistent record of pipeline runs (JSONL).

Append-only JSONL keeps this dependency-free and greppable; the API and UI
read it to show run history and cost dashboards. Swapping to Postgres or a
telemetry backend later means implementing these three methods elsewhere.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from app.schemas.report import RunMetrics
from app.schemas.trace import AgentTrace


class RunRecord(BaseModel):
    """One pipeline execution, as persisted."""

    run_id: str
    doc_id: str
    filename: str
    finished_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    success: bool
    errors: list[str] = Field(default_factory=list)
    metrics: RunMetrics
    traces: list[AgentTrace] = Field(default_factory=list)


class RunStore:
    """Append-only JSONL store of RunRecords."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: RunRecord) -> None:
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(record.model_dump_json() + "\n")

    def list_runs(self, limit: int = 50) -> list[RunRecord]:
        if not self._path.exists():
            return []
        lines = self._path.read_text(encoding="utf-8").splitlines()
        records = [RunRecord(**json.loads(line)) for line in lines if line.strip()]
        return sorted(records, key=lambda r: r.finished_at, reverse=True)[:limit]

    def totals(self) -> dict:
        """Aggregate cost/token totals across all runs."""
        runs = self.list_runs(limit=10_000)
        return {
            "runs": len(runs),
            "total_cost_usd": round(sum(r.metrics.total_cost_usd for r in runs), 6),
            "total_tokens": sum(r.metrics.total_tokens for r in runs),
            "failures": sum(1 for r in runs if not r.success),
        }
