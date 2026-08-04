"""Tests for the versioned consumer-law reference corpus."""

import hashlib

import pytest

from app.consumer.legal_corpus import (
    CONSUMER_LAW_CORPUS_RELEASE_ID,
    LegalCorpus,
    get_default_legal_corpus,
)
from app.consumer.schemas import LegalAuthorityCitation, ProvisionStatus
from app.rag.chunking import SectionAwareChunker
from app.schemas.rag import Chunk, RetrievedChunk


def test_default_corpus_has_reviewed_stable_metadata() -> None:
    corpus = get_default_legal_corpus()

    assert corpus.release_id == CONSUMER_LAW_CORPUS_RELEASE_ID
    assert len(corpus.provisions) == 30
    assert len({item.provision_id for item in corpus.provisions}) == 30
    assert {item.verified_on.isoformat() for item in corpus.provisions} == {"2026-08-04"}
    assert all(
        item.official_url.startswith("https://www.planalto.gov.br/")
        for item in corpus.provisions
    )
    for item in corpus.provisions:
        assert item.content_sha256 == hashlib.sha256(item.summary.encode()).hexdigest()


def test_corpus_covers_requested_constitution_and_cdc_provisions() -> None:
    corpus = get_default_legal_corpus()
    ids = {item.provision_id for item in corpus.provisions}

    assert {
        "br-cf-art-1-iii",
        "br-cf-art-5-v",
        "br-cf-art-5-x",
        "br-cf-art-5-xxxii",
        "br-cf-art-5-xxxv",
        "br-cf-art-5-lxxix",
        "br-cf-art-170-v",
        "br-cdc-art-2",
        "br-cdc-art-3-p2",
        "br-cdc-art-4",
        "br-cdc-art-6",
        "br-cdc-art-14",
        "br-cdc-art-20",
        "br-cdc-art-26",
        "br-cdc-art-27",
        "br-cdc-art-30",
        "br-cdc-art-35",
        "br-cdc-art-42",
        "br-cdc-art-43",
        "br-cdc-art-46",
        "br-cdc-art-47",
        "br-cdc-art-51",
        "br-cdc-art-52",
        "br-cdc-art-54-a",
        "br-cdc-art-54-b",
        "br-cdc-art-54-c",
        "br-cdc-art-54-d",
        "br-cdc-art-54-e",
        "br-cdc-art-54-f",
        "br-cdc-art-54-g",
    } == ids


def test_vetoed_article_is_present_but_not_an_active_authority() -> None:
    corpus = get_default_legal_corpus()
    article_54_e = corpus.get("br-cdc-art-54-e")

    assert article_54_e.status is ProvisionStatus.VETOED
    assert "vetado" in article_54_e.summary.lower()
    assert article_54_e not in corpus.active_provisions
    assert len(corpus.active_provisions) == 29


def test_parsed_document_preserves_page_to_provision_mapping() -> None:
    corpus = get_default_legal_corpus()
    document = corpus.as_parsed_document()

    assert document.page_count == len(corpus.provisions)
    assert document.language == "pt"
    assert "resumos editoriais" in document.warnings[0].lower()
    for page, provision in zip(document.pages, corpus.provisions, strict=True):
        assert corpus.provision_for_page(page.number) == provision
        assert provision.provision_id.upper() in page.text
        assert provision.content_sha256 in page.text


def test_retrieved_chunk_maps_to_legal_authority_not_evidence() -> None:
    corpus = get_default_legal_corpus()
    document = corpus.as_parsed_document()
    chunks = SectionAwareChunker(target_chars=1_200, overlap_chars=100).chunk(document)
    article_42_chunk = next(
        chunk
        for chunk in chunks
        if corpus.provision_for_chunk(chunk).provision_id == "br-cdc-art-42"
    )
    retrieved = RetrievedChunk(chunk=article_42_chunk, score=0.87)

    citation = corpus.authority_for_chunk(retrieved, retrieval_rank=1)

    assert isinstance(citation, LegalAuthorityCitation)
    assert citation.provision_id == "br-cdc-art-42"
    assert citation.chunk_id == article_42_chunk.chunk_id
    assert citation.retrieval_rank == 1
    assert citation.retrieval_score == 0.87
    assert citation.official_url.startswith("https://www.planalto.gov.br/")


def test_chunk_from_another_document_cannot_be_mapped_as_law() -> None:
    corpus = get_default_legal_corpus()
    foreign = Chunk(
        chunk_id="foreign:0000",
        doc_id="foreign",
        text="not law",
        page_start=1,
        page_end=1,
    )

    with pytest.raises(ValueError, match="does not belong"):
        corpus.provision_for_chunk(foreign)


def test_corpus_rejects_duplicate_ids_and_has_stable_manifest_hash() -> None:
    corpus = get_default_legal_corpus()

    with pytest.raises(ValueError, match="duplicate"):
        LegalCorpus([corpus.provisions[0], corpus.provisions[0]])
    assert len(corpus.corpus_sha256) == 64
    assert corpus.corpus_sha256 == get_default_legal_corpus().corpus_sha256
