"""Maintainer CLI for refreshing the pinned official CDC snapshot.

Runtime code never calls this module.  A refresh is an explicit, reviewable
operation: download, parse/coverage validation, hash, then atomic replacement
of the snapshot and manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import urllib.request
from datetime import date
from pathlib import Path

from app.consumer.cdc_snapshot import (
    CDC_PARSER_VERSION,
    CDC_SOURCE_URL,
    DEFAULT_SNAPSHOT_DIR,
    parse_cdc_html,
)

SNAPSHOT_FILENAME = "l8078compilado.html"
SOURCE_ENCODING = "windows-1252"
REFRESH_TOOL_VERSION = "cdc-snapshot-refresh-v2"


def refresh_snapshot(
    output_dir: Path,
    retrieved_on: date,
    *,
    source_file: Path | None = None,
    acquisition_note: str | None = None,
) -> tuple[Path, Path]:
    if source_file is None:
        request = urllib.request.Request(
            CDC_SOURCE_URL,
            headers={"User-Agent": "give-exit-legal-corpus-maintainer/1.0"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
            source = response.read()
            final_url = response.geturl()
            http_etag = response.headers.get("ETag")
            http_last_modified = response.headers.get("Last-Modified")
        acquisition_method = "download_https"
    else:
        if not acquisition_note or not acquisition_note.strip():
            raise ValueError("--source-file requires a non-empty --acquisition-note")
        source = source_file.read_bytes()
        acquisition_method = "local_file"
        final_url = None
        http_etag = None
        http_last_modified = None
    if not source:
        raise RuntimeError("Planalto returned an empty CDC document")

    html = source.decode(SOURCE_ENCODING)
    articles = parse_cdc_html(html)
    if len(articles) != 130:
        raise RuntimeError(f"unexpected compiled CDC article count: {len(articles)}")

    digest = hashlib.sha256(source).hexdigest()
    release_id = f"br-cdc-official-{retrieved_on.isoformat()}-v1"
    manifest = {
        "schema_version": 2,
        "release_id": release_id,
        "law_id": "br-cdc",
        "source_url": CDC_SOURCE_URL,
        "retrieved_on": retrieved_on.isoformat(),
        "encoding": SOURCE_ENCODING,
        "snapshot_file": SNAPSHOT_FILENAME,
        "snapshot_sha256": digest,
        "parser_version": CDC_PARSER_VERSION,
        "acquisition_method": acquisition_method,
        "acquisition_note": acquisition_note.strip() if acquisition_note else None,
        "final_url": final_url,
        "http_etag": http_etag,
        "http_last_modified": http_last_modified,
        "refresh_tool_version": REFRESH_TOOL_VERSION,
        "review_status": "pending_review",
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = output_dir / SNAPSHOT_FILENAME
    manifest_path = output_dir / "manifest.json"
    _atomic_write(snapshot_path, source)
    _atomic_write(
        manifest_path,
        (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )
    return snapshot_path, manifest_path


def _atomic_write(path: Path, content: bytes) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        temporary.write(content)
        temporary.flush()
        temporary_path = Path(temporary.name)
    temporary_path.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_SNAPSHOT_DIR,
        help="directory that will contain the HTML snapshot and manifest",
    )
    parser.add_argument(
        "--acquisition-note",
        help="required provenance note when --source-file is used",
    )
    parser.add_argument(
        "--retrieved-on",
        type=date.fromisoformat,
        default=date.today(),
        help="review date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--source-file",
        type=Path,
        help="validate and pin an already downloaded copy instead of using the network",
    )
    arguments = parser.parse_args()
    snapshot, manifest = refresh_snapshot(
        arguments.output_dir,
        arguments.retrieved_on,
        source_file=arguments.source_file,
        acquisition_note=arguments.acquisition_note,
    )
    print(snapshot)
    print(manifest)


if __name__ == "__main__":
    main()
