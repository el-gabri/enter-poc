"""Golden dataset loading.

A golden case is a JSON file with the document text (per page) and the
expected labels. Text-based (not PDF) so cases are reviewable in a diff and
cheap to author - the ingestion layer has its own tests.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from app.schemas.document import DocumentPage, ExtractionMethod, ParsedDocument


@dataclass(frozen=True)
class GoldenCase:
    name: str
    document: ParsedDocument
    expected: dict


def load_case(path: Path) -> GoldenCase:
    payload = json.loads(path.read_text(encoding="utf-8"))
    document = ParsedDocument(
        filename=f"{payload['name']}.pdf",
        pages=[
            DocumentPage(number=i + 1, text=text)
            for i, text in enumerate(payload["pages"])
        ],
        language=payload.get("language", "pt"),
        extraction_method=ExtractionMethod.NATIVE_TEXT,
    )
    return GoldenCase(
        name=payload["name"], document=document, expected=payload["expected"]
    )


def load_dataset(directory: Path) -> list[GoldenCase]:
    cases = [load_case(p) for p in sorted(directory.glob("*.json"))]
    if not cases:
        raise FileNotFoundError(f"No golden cases (*.json) found in {directory}")
    return cases
