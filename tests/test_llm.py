"""Tests for the LLM abstraction layer."""

from enum import Enum

import pytest
from pydantic import BaseModel

from app.core.config import LLMProvider, Settings
from app.llm.base import LLMClient, TokenUsage
from app.llm.factory import create_llm_client
from app.llm.mock_client import MockLLMClient, synthesize_instance
from app.llm.pricing import estimate_cost_usd


class RiskLevel(str, Enum):
    LOW = "low"
    HIGH = "high"


class _Party(BaseModel):
    name: str


class _Extraction(BaseModel):
    plaintiff: _Party
    claim_value: float | None
    risk: RiskLevel
    claims: list[str]


async def test_mock_parse_returns_valid_schema_and_metadata() -> None:
    client = MockLLMClient()
    result = await client.parse(system="s", user="u", schema=_Extraction)

    assert isinstance(result.data, _Extraction)
    assert result.meta.provider == "mock"
    assert result.meta.usage.total_tokens == 150
    assert result.meta.cost_usd == 0.0


async def test_mock_uses_canned_response_when_registered() -> None:
    canned = _Extraction(
        plaintiff=_Party(name="Maria Silva"),
        claim_value=50_000.0,
        risk=RiskLevel.HIGH,
        claims=["danos morais"],
    )
    client = MockLLMClient(responses={_Extraction: canned})
    result = await client.parse(system="s", user="u", schema=_Extraction)

    assert result.data.plaintiff.name == "Maria Silva"
    assert client.calls[0]["schema"] == "_Extraction"


def test_synthesizer_handles_nested_optional_enum_and_list_fields() -> None:
    instance = synthesize_instance(_Extraction)
    assert instance.plaintiff.name == "[mock]"
    assert instance.claim_value is None
    assert instance.risk is RiskLevel.LOW
    assert instance.claims == []


def test_factory_returns_mock_client() -> None:
    settings = Settings(llm_provider=LLMProvider.MOCK, _env_file=None)
    client = create_llm_client(settings)
    assert isinstance(client, MockLLMClient)
    assert isinstance(client, LLMClient)  # protocol conformance


def test_factory_rejects_openai_without_key() -> None:
    settings = Settings(
        llm_provider=LLMProvider.OPENAI, openai_api_key=None, _env_file=None
    )
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        create_llm_client(settings)


def test_pricing_known_and_unknown_models() -> None:
    usage = TokenUsage(prompt_tokens=1_000_000, completion_tokens=1_000_000)
    assert estimate_cost_usd("gpt-4o-mini", usage) == pytest.approx(0.75)
    assert estimate_cost_usd("some-future-model", usage) is None
