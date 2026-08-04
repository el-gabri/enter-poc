"""Transparent settlement scenarios for consumer extrajudicial notices."""

import hashlib
import json
from decimal import ROUND_HALF_UP, Decimal

from app.consumer.schemas import (
    SettlementComponent,
    SettlementInputs,
    SettlementScenario,
)


class SettlementCalculator:
    """Build an auditable negotiation range without pretending to predict a court.

    The exploratory weights are deterministic completeness/evidence heuristics.
    They are not trained, calibrated or legally validated probabilities.
    """

    METHODOLOGY_VERSION = "consumer-settlement-scenario-v2"
    _CENT = Decimal("0.01")

    def calculate(self, inputs: SettlementInputs) -> SettlementScenario:
        direct_loss = self._money(inputs.direct_loss_amount)
        improper_paid = self._money(inputs.improper_payment_amount)
        downside_cost = self._money(inputs.downside_cost_amount)

        article_42_increment = (
            improper_paid if inputs.article_42_double_repayment_supported else Decimal("0")
        )
        low_outcome = direct_loss
        high_outcome = direct_loss + article_42_increment

        center = (
            Decimal("0.25")
            + Decimal("0.35") * inputs.evidence_strength
            + Decimal("0.20") * inputs.factual_completeness
        )
        low_weight = max(Decimal("0.05"), center - Decimal("0.10"))
        high_weight = min(Decimal("0.90"), center + Decimal("0.10"))
        expected_low = self._money(
            low_outcome * low_weight - downside_cost * (Decimal("1") - low_weight)
        )
        expected_high = self._money(
            high_outcome * high_weight - downside_cost * (Decimal("1") - high_weight)
        )

        has_monetary_input = high_outcome > 0
        if has_monetary_input:
            private_reservation = self._money(direct_loss)
            public_proposal = self._money(high_outcome)
        else:
            private_reservation = None
            public_proposal = None

        if inputs.article_42_double_repayment_supported:
            article_42_assumption = (
                "O cenário inclui apenas como hipótese condicionada um acréscimo igual ao "
                "valor indevido efetivamente pago. A devolução em dobro depende da incidência "
                "do art. 42, parágrafo único, e da ausência de engano justificável."
            )
        else:
            article_42_assumption = (
                "O cenário não inclui devolução em dobro. Cobrança não paga ou ausência de "
                "suporte para os requisitos do art. 42 não gera esse acréscimo automático."
            )

        components: list[SettlementComponent] = []
        if direct_loss > 0:
            components.append(
                SettlementComponent(
                    kind="direct_loss",
                    amount=direct_loss,
                    included_in_public_proposal=True,
                    formula="prejuízo direto explicitamente confirmado",
                    sources=inputs.direct_loss_sources,
                )
            )
        if article_42_increment > 0:
            components.append(
                SettlementComponent(
                    kind="conditional_article_42",
                    amount=article_42_increment,
                    included_in_public_proposal=True,
                    formula="acréscimo condicional igual ao valor indevidamente pago",
                    sources=inputs.improper_payment_sources,
                )
            )
        if downside_cost > 0:
            components.append(
                SettlementComponent(
                    kind="downside_cost",
                    amount=downside_cost,
                    included_in_public_proposal=False,
                    formula="custo explícito do cenário sem acordo",
                    sources=inputs.downside_cost_sources,
                )
            )
        calculation_payload = {
            "methodology_version": self.METHODOLOGY_VERSION,
            "components": [item.model_dump(mode="json") for item in components],
            "public_proposal": str(public_proposal) if public_proposal is not None else None,
            "private_reservation": (
                str(private_reservation) if private_reservation is not None else None
            ),
            "weights": [str(low_weight), str(high_weight)],
        }
        calculation_sha256 = hashlib.sha256(
            json.dumps(
                calculation_payload,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()

        return SettlementScenario(
            methodology_version=self.METHODOLOGY_VERSION,
            calibrated=False,
            is_legal_outcome_prediction=False,
            direct_loss_amount=direct_loss,
            improper_payment_amount=improper_paid,
            downside_cost_amount=downside_cost,
            unsuccessful_outcome_value=-downside_cost,
            conditional_article_42_increment_amount=article_42_increment,
            low_outcome_value=self._money(low_outcome),
            high_outcome_value=self._money(high_outcome),
            exploratory_weight_low=low_weight,
            exploratory_weight_high=high_weight,
            illustrative_expected_value_low=expected_low,
            illustrative_expected_value_high=expected_high,
            public_proposal_amount=public_proposal,
            private_reservation_amount=private_reservation,
            article_42_assumption=article_42_assumption,
            components=components,
            calculation_sha256=calculation_sha256,
            methodology=[
                "A proposta pública soma apenas o prejuízo explicitamente confirmado "
                "e acréscimos legais condicionados; não inclui indenização inventada.",
                "Valores encontrados em documentos são candidatos e só entram no cálculo "
                "depois da confirmação do consumidor.",
                "Os valores esperados multiplicam os cenários por pesos exploratórios "
                "derivados da força da prova e da completude dos fatos e descontam "
                "o custo explícito atribuído ao cenário sem acordo.",
                "A reserva privada corresponde ao prejuízo confirmado e não deve "
                "constar da notificação enviada ao fornecedor.",
            ],
            caveats=[
                "Os pesos não foram treinados nem calibrados com resultados judiciais.",
                "Na etapa extrajudicial, o custo do cenário sem acordo é zero por padrão; "
                "custas, honorários, demora e riscos de eventual processo não são inferidos.",
                "O cálculo não estima chance de vitória, indenização judicial ou valor devido.",
                "A Constituição e o CDC não fornecem, por si sós, probabilidades ou tabelas "
                "de indenização para o caso concreto.",
                "Revise fatos, provas, prazos e estratégia com profissional habilitado antes "
                "do envio.",
            ],
        )

    @classmethod
    def _money(cls, value: Decimal) -> Decimal:
        return value.quantize(cls._CENT, rounding=ROUND_HALF_UP)
