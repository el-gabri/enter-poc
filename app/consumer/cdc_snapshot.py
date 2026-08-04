"""Offline, integrity-checked access to the official compiled CDC snapshot.

The Planalto page is legacy HTML encoded as Windows-1252 and does not expose a
stable machine-readable API.  This module deliberately parses a pinned local
snapshot: production retrieval never depends on the government website being
available at request time.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from app.consumer.schemas import LegalTextUnit, LegalUnitKind, ProvisionStatus

CDC_SOURCE_URL = "https://www.planalto.gov.br/ccivil_03/leis/l8078compilado.htm"
CDC_PARSER_VERSION = "cdc-html-parser-v2"
DEFAULT_SNAPSHOT_DIR = Path(__file__).with_name("data") / "cdc"
DEFAULT_MANIFEST_PATH = DEFAULT_SNAPSHOT_DIR / "manifest.json"

_ARTICLE_RE = re.compile(
    r"^Art\.\s*(?P<number>\d+)(?:[º°])?(?:-(?P<suffix>[A-Z]))?\.?(?=\s)",
    re.IGNORECASE,
)
_STRUCTURAL_HEADING_RE = re.compile(
    r"^(?P<level>TÍTULO|CAPÍTULO|SEÇÃO|SUBSEÇÃO)\s+"
    r"(?:[IVXLCDM]+(?:-[A-Z])?|ÚNIC[AO]|GERAL)(?:\s+|$)",
    re.IGNORECASE,
)
_PARAGRAPH_RE = re.compile(
    r"^(?:(?P<unique>Parágrafo\s+único)|§\s*(?P<number>\d+(?:-[A-Z])?)"
    r"(?:\s*(?:[º°oª]\.?|\.\s*[º°oª]?))?)\.?(?=\s)",
    re.IGNORECASE,
)
_INCISO_RE = re.compile(r"^(?P<label>[IVXLCDM]+)\s*[-–—](?=\s)")
_ALINEA_RE = re.compile(r"^(?P<label>[a-z])\s*[)](?=\s)", re.IGNORECASE)
_PENALTY_RE = re.compile(r"^Pena\b", re.IGNORECASE)
_QUOTED_AMENDMENT_RE = re.compile(
    r'^["“”«]\s*(?:Art\.\s*\d+|§\s*\d+|[IVXLCDM]+\s*[-–—])',
    re.IGNORECASE,
)
_EDITORIAL_NOTE_RE = re.compile(
    r"^\((?:redação\s+dada|incluíd[oa]|acrescentad[oa]|revogad[oa]|vide|vigência|"
    r"produção\s+de\s+efeito|regulamentad[oa])\b",
    re.IGNORECASE,
)
_VETOED_RE = re.compile(r"\(\s*VETAD[OA]\s*\)", re.IGNORECASE)
_REVOKED_RE = re.compile(r"\(\s*REVOGAD[OA][^)]*\)", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class CdcSnapshotManifest:
    """Metadata that makes the downloaded source independently auditable."""

    schema_version: int
    release_id: str
    law_id: str
    source_url: str
    retrieved_on: date
    encoding: str
    snapshot_file: str
    snapshot_sha256: str
    parser_version: str
    acquisition_method: str
    acquisition_note: str | None
    final_url: str | None
    http_etag: str | None
    http_last_modified: str | None
    refresh_tool_version: str
    review_status: str

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> CdcSnapshotManifest:
        required = {
            "schema_version",
            "release_id",
            "law_id",
            "source_url",
            "retrieved_on",
            "encoding",
            "snapshot_file",
            "snapshot_sha256",
            "parser_version",
            "acquisition_method",
            "acquisition_note",
            "final_url",
            "http_etag",
            "http_last_modified",
            "refresh_tool_version",
            "review_status",
        }
        missing = required - value.keys()
        if missing:
            raise ValueError(f"CDC manifest is missing fields: {sorted(missing)}")
        manifest = cls(
            schema_version=int(value["schema_version"]),
            release_id=str(value["release_id"]),
            law_id=str(value["law_id"]),
            source_url=str(value["source_url"]),
            retrieved_on=date.fromisoformat(str(value["retrieved_on"])),
            encoding=str(value["encoding"]),
            snapshot_file=str(value["snapshot_file"]),
            snapshot_sha256=str(value["snapshot_sha256"]),
            parser_version=str(value["parser_version"]),
            acquisition_method=str(value["acquisition_method"]),
            acquisition_note=(
                str(value["acquisition_note"]) if value["acquisition_note"] is not None else None
            ),
            final_url=(str(value["final_url"]) if value["final_url"] is not None else None),
            http_etag=(str(value["http_etag"]) if value["http_etag"] is not None else None),
            http_last_modified=(
                str(value["http_last_modified"])
                if value["http_last_modified"] is not None
                else None
            ),
            refresh_tool_version=str(value["refresh_tool_version"]),
            review_status=str(value["review_status"]),
        )
        if manifest.schema_version != 2:
            raise ValueError(f"unsupported CDC manifest schema: {manifest.schema_version}")
        if manifest.law_id != "br-cdc":
            raise ValueError("CDC manifest has an unexpected law_id")
        if manifest.source_url != CDC_SOURCE_URL:
            raise ValueError("CDC manifest does not point to the official Planalto URL")
        if not re.fullmatch(r"[0-9a-f]{64}", manifest.snapshot_sha256):
            raise ValueError("CDC manifest has an invalid snapshot SHA-256")
        if manifest.parser_version != CDC_PARSER_VERSION:
            raise ValueError("CDC manifest parser version does not match runtime parser")
        if manifest.acquisition_method not in {"download_https", "local_file"}:
            raise ValueError("CDC manifest has an invalid acquisition method")
        if manifest.acquisition_method == "local_file" and not manifest.acquisition_note:
            raise ValueError("local CDC snapshots require an acquisition note")
        if manifest.review_status not in {
            "pending_review",
            "engineering_validated",
            "legal_reviewed",
        }:
            raise ValueError("CDC manifest has an invalid review status")
        if manifest.review_status == "pending_review":
            raise ValueError("CDC snapshot has not been promoted after review")
        return manifest


@dataclass(frozen=True, slots=True)
class ParsedCdcArticle:
    """One top-level CDC article and its addressable normative units."""

    number: int
    suffix: str | None
    article_key: str
    article_label: str
    provision_id: str
    title: str | None
    chapter: str | None
    section: str | None
    official_text: str
    status: ProvisionStatus
    units: tuple[LegalTextUnit, ...]


@dataclass(slots=True)
class _ParagraphBuilder:
    fragments: list[str] = field(default_factory=list)


class _OfficialHtmlTextParser(HTMLParser):
    """Extract rendered paragraphs from the legacy Planalto markup."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.paragraphs: list[str] = []
        self._stack: list[_ParagraphBuilder] = []
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        lowered = tag.lower()
        if lowered in {"script", "style"}:
            self._ignored_depth += 1
            return
        if self._ignored_depth:
            return
        if lowered == "p":
            self._stack.append(_ParagraphBuilder())
        elif lowered == "br" and self._stack:
            self._stack[-1].fragments.append("\n")

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.lower()
        if lowered in {"script", "style"} and self._ignored_depth:
            self._ignored_depth -= 1
            return
        if self._ignored_depth or lowered != "p" or not self._stack:
            return
        builder = self._stack.pop()
        paragraph = _normalize_text("".join(builder.fragments))
        if paragraph:
            self.paragraphs.append(paragraph)

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth and self._stack:
            self._stack[-1].fragments.append(data)


def load_manifest(path: Path = DEFAULT_MANIFEST_PATH) -> CdcSnapshotManifest:
    """Load and validate the pinned snapshot manifest."""

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"CDC snapshot manifest not found: {path}") from exc
    if not isinstance(raw, dict):
        raise ValueError("CDC manifest root must be an object")
    return CdcSnapshotManifest.from_mapping(raw)


def load_official_cdc(
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
) -> tuple[CdcSnapshotManifest, tuple[ParsedCdcArticle, ...]]:
    """Verify the local source bytes and parse all compiled CDC articles."""

    manifest = load_manifest(manifest_path)
    snapshot_path = (manifest_path.parent / manifest.snapshot_file).resolve()
    if snapshot_path.parent != manifest_path.parent.resolve():
        raise ValueError("CDC snapshot_file must stay inside the manifest directory")
    try:
        source = snapshot_path.read_bytes()
    except FileNotFoundError as exc:
        raise RuntimeError(f"CDC snapshot not found: {snapshot_path}") from exc
    actual_sha256 = hashlib.sha256(source).hexdigest()
    if actual_sha256 != manifest.snapshot_sha256:
        raise ValueError(
            "CDC snapshot integrity check failed: "
            f"expected {manifest.snapshot_sha256}, got {actual_sha256}"
        )
    try:
        html = source.decode(manifest.encoding)
    except (LookupError, UnicodeDecodeError) as exc:
        raise ValueError(f"cannot decode CDC snapshot as {manifest.encoding}") from exc
    return manifest, parse_cdc_html(html)


def parse_cdc_html(html: str) -> tuple[ParsedCdcArticle, ...]:
    """Parse the top-level articles and hierarchy from an official CDC HTML page."""

    parser = _OfficialHtmlTextParser()
    parser.feed(html)
    parser.close()

    hierarchy: dict[str, str | None] = {
        "title": None,
        "chapter": None,
        "section": None,
    }
    article_specs: list[tuple[int, str | None, dict[str, str | None], list[str]]] = []
    current_number: int | None = None
    current_suffix: str | None = None
    current_hierarchy: dict[str, str | None] | None = None
    current_blocks: list[str] = []

    def finish_current() -> None:
        nonlocal current_number, current_suffix, current_hierarchy, current_blocks
        if current_number is None or current_hierarchy is None:
            return
        article_specs.append(
            (
                current_number,
                current_suffix,
                dict(current_hierarchy),
                list(current_blocks),
            )
        )
        current_number = None
        current_suffix = None
        current_hierarchy = None
        current_blocks = []

    previous_number: int | None = None
    previous_suffix: str | None = None
    pending_heading_level: str | None = None
    for paragraph in parser.paragraphs:
        if pending_heading_level and _is_heading_continuation(paragraph):
            _append_heading_continuation(hierarchy, pending_heading_level, paragraph)
            pending_heading_level = None
            continue
        pending_heading_level = None
        heading = _STRUCTURAL_HEADING_RE.match(paragraph)
        if heading:
            finish_current()
            _update_hierarchy(hierarchy, heading.group("level"), paragraph)
            if heading.end() == len(paragraph):
                pending_heading_level = heading.group("level")
            continue

        article_match = _ARTICLE_RE.match(paragraph)
        if article_match:
            number = int(article_match.group("number"))
            suffix = article_match.group("suffix")
            if _is_next_cdc_article(number, suffix, previous_number, previous_suffix):
                finish_current()
                current_number = number
                current_suffix = suffix.upper() if suffix else None
                current_hierarchy = dict(hierarchy)
                current_blocks = [paragraph]
                previous_number = number
                previous_suffix = current_suffix
                continue

        if current_number == 119 and paragraph.casefold().startswith("brasília,"):
            finish_current()
            break
        if current_number is not None:
            current_blocks.append(paragraph)
    finish_current()

    articles = tuple(
        _build_article(number, suffix, article_hierarchy, blocks)
        for number, suffix, article_hierarchy, blocks in article_specs
    )
    _validate_complete_cdc(articles)
    return articles


def _is_next_cdc_article(
    number: int,
    suffix: str | None,
    previous_number: int | None,
    previous_suffix: str | None,
) -> bool:
    if previous_number is None:
        return number == 1 and suffix is None
    if number == previous_number + 1 and suffix is None:
        return True
    if number != previous_number or suffix is None:
        return False
    if previous_suffix is None:
        return suffix.upper() == "A"
    return ord(suffix.upper()) == ord(previous_suffix.upper()) + 1


def _update_hierarchy(hierarchy: dict[str, str | None], level: str, heading: str) -> None:
    normalized_level = _strip_accents(level).upper()
    if normalized_level == "TITULO":
        hierarchy.update(title=heading, chapter=None, section=None)
    elif normalized_level == "CAPITULO":
        hierarchy.update(chapter=heading, section=None)
    else:
        hierarchy["section"] = heading


def _is_heading_continuation(paragraph: str) -> bool:
    letters = [character for character in paragraph if character.isalpha()]
    return bool(letters) and all(character.isupper() for character in letters)


def _append_heading_continuation(
    hierarchy: dict[str, str | None], level: str, continuation: str
) -> None:
    normalized_level = _strip_accents(level).upper()
    key = (
        "title"
        if normalized_level == "TITULO"
        else "chapter"
        if normalized_level == "CAPITULO"
        else "section"
    )
    heading = hierarchy[key]
    hierarchy[key] = f"{heading} {continuation}" if heading else continuation


def _build_article(
    number: int,
    suffix: str | None,
    hierarchy: dict[str, str | None],
    blocks: list[str],
) -> ParsedCdcArticle:
    article_key = str(number) if suffix is None else f"{number}-{suffix.lower()}"
    article_label = f"art. {number}" if suffix is None else f"art. {number}-{suffix}"
    provision_id = f"br-cdc-art-{article_key}"
    units = _build_units(provision_id, blocks)
    statuses = {unit.status for unit in units if unit.kind is not LegalUnitKind.NOTE}
    if statuses == {ProvisionStatus.VETOED}:
        status = ProvisionStatus.VETOED
    elif statuses == {ProvisionStatus.REVOKED}:
        status = ProvisionStatus.REVOKED
    else:
        status = ProvisionStatus.ACTIVE
    return ParsedCdcArticle(
        number=number,
        suffix=suffix,
        article_key=article_key,
        article_label=article_label,
        provision_id=provision_id,
        title=hierarchy["title"],
        chapter=hierarchy["chapter"],
        section=hierarchy["section"],
        official_text="\n\n".join(blocks),
        status=status,
        units=units,
    )


def _build_units(provision_id: str, blocks: list[str]) -> tuple[LegalTextUnit, ...]:
    units: list[LegalTextUnit] = []
    used_ids: set[str] = set()
    current_paragraph: str | None = None
    current_inciso: str | None = None
    unstructured_counts: dict[LegalUnitKind, int] = {}

    for position, text in enumerate(blocks):
        paragraph_match = _PARAGRAPH_RE.match(text)
        inciso_match = _INCISO_RE.match(text)
        alinea_match = _ALINEA_RE.match(text)
        paragraph: str | None = current_paragraph
        inciso: str | None = current_inciso
        alinea: str | None = None

        if position == 0:
            kind = LegalUnitKind.CAPUT
            label = "caput"
            fragment = "caput"
            current_paragraph = None
            current_inciso = None
            paragraph = None
            inciso = None
        elif paragraph_match:
            kind = LegalUnitKind.PARAGRAPH
            number = paragraph_match.group("number")
            paragraph = "unico" if paragraph_match.group("unique") else str(number).lower()
            label = "parágrafo único" if paragraph == "unico" else f"§ {number}"
            fragment = f"paragrafo-{paragraph}"
            current_paragraph = paragraph
            current_inciso = None
            inciso = None
        elif inciso_match:
            kind = LegalUnitKind.INCISO
            inciso = inciso_match.group("label").lower()
            label = f"inciso {inciso.upper()}"
            prefix = f"paragrafo-{paragraph}-" if paragraph else ""
            fragment = f"{prefix}inciso-{inciso}"
            current_inciso = inciso
        elif alinea_match:
            kind = LegalUnitKind.ALINEA
            alinea = alinea_match.group("label").lower()
            label = f"alínea {alinea}"
            paragraph_prefix = f"paragrafo-{paragraph}-" if paragraph else ""
            inciso_prefix = f"inciso-{inciso}-" if inciso else ""
            fragment = f"{paragraph_prefix}{inciso_prefix}alinea-{alinea}"
        else:
            kind = _unstructured_unit_kind(text)
            unstructured_counts[kind] = unstructured_counts.get(kind, 0) + 1
            ordinal = unstructured_counts[kind]
            label_prefix, fragment_prefix = {
                LegalUnitKind.PENALTY: ("pena", "pena"),
                LegalUnitKind.QUOTED_AMENDMENT: (
                    "alteração legal transcrita",
                    "alteracao-citada",
                ),
                LegalUnitKind.NOTE: ("nota editorial", "nota"),
                LegalUnitKind.NORMATIVE_OTHER: (
                    "bloco normativo",
                    "normativo",
                ),
            }[kind]
            label = f"{label_prefix} {ordinal}"
            fragment = f"{fragment_prefix}-{ordinal:03d}"
            if kind is LegalUnitKind.QUOTED_AMENDMENT:
                paragraph, inciso = _quoted_unit_hierarchy(text)

        base_unit_id = f"{provision_id}-{fragment}"
        unit_id = base_unit_id
        duplicate = 2
        while unit_id in used_ids:
            unit_id = f"{base_unit_id}-{duplicate}"
            duplicate += 1
        used_ids.add(unit_id)
        units.append(
            LegalTextUnit(
                unit_id=unit_id,
                kind=kind,
                label=label,
                text=text,
                paragraph=paragraph,
                inciso=inciso,
                alinea=alinea,
                status=_status_for_text(text),
            )
        )
    return tuple(units)


def _unstructured_unit_kind(text: str) -> LegalUnitKind:
    """Classify official normative blocks before falling back to editorial notes."""

    if _PENALTY_RE.match(text):
        return LegalUnitKind.PENALTY
    if _QUOTED_AMENDMENT_RE.match(text):
        return LegalUnitKind.QUOTED_AMENDMENT
    if _EDITORIAL_NOTE_RE.match(text):
        return LegalUnitKind.NOTE
    return LegalUnitKind.NORMATIVE_OTHER


def _quoted_unit_hierarchy(text: str) -> tuple[str | None, str | None]:
    unquoted = text.lstrip('"“”« ').strip()
    paragraph_match = _PARAGRAPH_RE.match(unquoted)
    if paragraph_match:
        number = paragraph_match.group("number")
        paragraph = "unico" if paragraph_match.group("unique") else str(number).lower()
        return paragraph, None
    inciso_match = _INCISO_RE.match(unquoted)
    if inciso_match:
        return None, inciso_match.group("label").lower()
    return None, None


def _status_for_text(text: str) -> ProvisionStatus:
    if _REVOKED_RE.search(text):
        return ProvisionStatus.REVOKED
    if _VETOED_RE.search(text):
        return ProvisionStatus.VETOED
    return ProvisionStatus.ACTIVE


def _validate_complete_cdc(articles: tuple[ParsedCdcArticle, ...]) -> None:
    base_numbers = {article.number for article in articles}
    expected_numbers = set(range(1, 120))
    if base_numbers != expected_numbers:
        missing = sorted(expected_numbers - base_numbers)
        unexpected = sorted(base_numbers - expected_numbers)
        raise ValueError(
            f"CDC snapshot does not cover articles 1-119; missing={missing}, "
            f"unexpected={unexpected}"
        )
    ids = [article.provision_id for article in articles]
    if len(ids) != len(set(ids)):
        raise ValueError("CDC snapshot produced duplicate provision ids")
    required_extensions = {
        "br-cdc-art-42-a",
        *(f"br-cdc-art-54-{suffix}" for suffix in "abcdefg"),
        *(f"br-cdc-art-104-{suffix}" for suffix in "abc"),
    }
    if not required_extensions.issubset(ids):
        missing_extensions = sorted(required_extensions - set(ids))
        raise ValueError(f"CDC snapshot is missing compiled articles: {missing_extensions}")


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFC", value.replace("\xa0", " "))
    return " ".join(normalized.split())


def _strip_accents(value: str) -> str:
    return "".join(
        character
        for character in unicodedata.normalize("NFD", value)
        if unicodedata.category(character) != "Mn"
    )
