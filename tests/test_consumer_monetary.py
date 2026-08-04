"""Tests for conservative monetary-candidate extraction."""

from decimal import Decimal

from app.consumer.monetary import extract_brl_mentions


def test_explicit_values_remain_candidates_without_loss_classification() -> None:
    mentions = extract_brl_mentions(
        "A loja cobrou R$ 120,00, estornou o valor e eu quero R$ 5.000,00."
    )

    assert [item.amount for item in mentions] == [Decimal("120.00"), Decimal("5000.00")]


def test_brl_mentions_preserve_excerpt_hashes_without_summing_values() -> None:
    mentions = extract_brl_mentions(
        "O produto custou R$ 100,00. A fatura também mostra limite de R$ 2.000,00."
    )

    assert [item.amount for item in mentions] == [Decimal("100.00"), Decimal("2000.00")]
    assert all(len(item.quote_sha256) == 64 for item in mentions)
    assert all(len(item.quote) <= 300 for item in mentions)
