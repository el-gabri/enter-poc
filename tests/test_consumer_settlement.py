"""Tests for deterministic, explicitly uncalibrated settlement scenarios."""

from decimal import Decimal

import pytest
from pydantic import ValidationError

from app.consumer.schemas import SettlementInputs
from app.consumer.settlement import SettlementCalculator


def test_scenario_uses_only_confirmed_loss_and_conditional_legal_increment() -> None:
    scenario = SettlementCalculator().calculate(
        SettlementInputs(
            direct_loss_amount=Decimal("100"),
            improper_payment_amount=Decimal("50"),
            article_42_double_repayment_supported=True,
            evidence_strength=Decimal("0.8"),
            factual_completeness=Decimal("0.75"),
        )
    )

    assert scenario.direct_loss_amount == Decimal("100.00")
    assert scenario.improper_payment_amount == Decimal("50.00")
    assert scenario.conditional_article_42_increment_amount == Decimal("50.00")
    assert scenario.low_outcome_value == Decimal("100.00")
    assert scenario.high_outcome_value == Decimal("150.00")
    assert scenario.exploratory_weight_low == Decimal("0.5800")
    assert scenario.exploratory_weight_high == Decimal("0.7800")
    assert scenario.illustrative_expected_value_low == Decimal("58.00")
    assert scenario.illustrative_expected_value_high == Decimal("117.00")
    assert scenario.private_reservation_amount == Decimal("100.00")
    assert scenario.public_proposal_amount == Decimal("150.00")
    assert [component.kind for component in scenario.components] == [
        "direct_loss",
        "conditional_article_42",
    ]
    assert len(scenario.calculation_sha256) == 64


def test_article_42_increment_is_excluded_without_explicit_support() -> None:
    scenario = SettlementCalculator().calculate(
        SettlementInputs(
            direct_loss_amount=Decimal("100"),
            improper_payment_amount=Decimal("50"),
            article_42_double_repayment_supported=False,
        )
    )

    assert scenario.conditional_article_42_increment_amount == Decimal("0.00")
    assert scenario.high_outcome_value == Decimal("100.00")
    assert "não inclui devolução em dobro" in scenario.article_42_assumption.lower()


def test_explicit_unsuccessful_scenario_cost_is_part_of_expected_value() -> None:
    scenario = SettlementCalculator().calculate(
        SettlementInputs(
            direct_loss_amount=Decimal("100"),
            downside_cost_amount=Decimal("20"),
            evidence_strength=Decimal("0.5"),
            factual_completeness=Decimal("0.5"),
        )
    )

    assert scenario.downside_cost_amount == Decimal("20.00")
    assert scenario.unsuccessful_outcome_value == Decimal("-20.00")
    assert scenario.illustrative_expected_value_low == Decimal("31.00")
    assert scenario.illustrative_expected_value_high == Decimal("55.00")


def test_article_42_uses_only_amount_actually_paid_as_increment() -> None:
    scenario = SettlementCalculator().calculate(
        SettlementInputs(
            direct_loss_amount=Decimal("180"),
            improper_payment_amount=Decimal("80"),
            article_42_double_repayment_supported=True,
        )
    )

    assert scenario.direct_loss_amount == Decimal("180.00")
    assert scenario.conditional_article_42_increment_amount == Decimal("80.00")
    assert scenario.high_outcome_value == Decimal("260.00")
    assert "engano justificável" in scenario.article_42_assumption


def test_zero_value_complaint_does_not_invent_a_financial_proposal() -> None:
    scenario = SettlementCalculator().calculate(SettlementInputs())

    assert scenario.public_proposal_amount is None
    assert scenario.private_reservation_amount is None
    assert scenario.low_outcome_value == 0
    assert scenario.high_outcome_value == 0


def test_arbitrary_public_and_private_overrides_are_rejected() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        SettlementInputs.model_validate(
            {
                "direct_loss_amount": "1000",
                "public_proposal_override": "1750",
                "private_reservation_override": "1200",
            }
        )


def test_scenario_never_labels_exploratory_weights_as_predicted_odds() -> None:
    scenario = SettlementCalculator().calculate(SettlementInputs(direct_loss_amount=Decimal("100")))

    assert scenario.calibrated is False
    assert scenario.is_legal_outcome_prediction is False
    joined_caveats = " ".join(scenario.caveats).lower()
    assert "não estima chance de vitória" in joined_caveats
    assert "não foram treinados nem calibrados" in joined_caveats


def test_improper_payment_must_be_part_of_direct_loss() -> None:
    with pytest.raises(ValidationError, match="cannot exceed direct_loss_amount"):
        SettlementInputs(
            direct_loss_amount=Decimal("50"),
            improper_payment_amount=Decimal("75"),
        )
