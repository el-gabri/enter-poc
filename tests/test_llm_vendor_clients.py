"""Offline contract tests for the native Anthropic and Gemini adapters."""

from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from app.llm.anthropic_client import AnthropicClient
from app.llm.gemini_client import GeminiClient


class _Answer(BaseModel):
    summary: str
    score: int


def _anthropic_response(**overrides: Any) -> SimpleNamespace:
    values: dict[str, Any] = {
        "content": [
            SimpleNamespace(type="text", text="primeiro"),
            SimpleNamespace(type="text", text="segundo"),
        ],
        "usage": SimpleNamespace(input_tokens=11, output_tokens=7),
        "model": "claude-sonnet-5",
        "stop_reason": "end_turn",
        "parsed_output": _Answer(summary="ok", score=9),
        "_request_id": "req-test",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class _FakeAnthropicMessages:
    def __init__(self, response: SimpleNamespace) -> None:
        self.response = response
        self.create_calls: list[dict[str, Any]] = []
        self.parse_calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> SimpleNamespace:
        self.create_calls.append(kwargs)
        return self.response

    async def parse(self, **kwargs: Any) -> SimpleNamespace:
        self.parse_calls.append(kwargs)
        return self.response


def _fake_anthropic(response: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(messages=_FakeAnthropicMessages(response))


async def test_anthropic_complete_maps_messages_usage_and_metadata() -> None:
    fake = _fake_anthropic(_anthropic_response())
    client = AnthropicClient("test-key", "claude-sonnet-5", max_output_tokens=4096, client=fake)

    result = await client.complete(
        system="system policy",
        user="user evidence",
        temperature=0.7,
        prompt_version="agent-v2",
    )

    assert result.text == "primeiro\nsegundo"
    assert result.meta.provider == "anthropic"
    assert result.meta.usage.prompt_tokens == 11
    assert result.meta.usage.completion_tokens == 7
    assert result.meta.prompt_version == "agent-v2"
    [call] = fake.messages.create_calls
    assert call == {
        "model": "claude-sonnet-5",
        "max_tokens": 4096,
        "system": "system policy",
        "messages": [{"role": "user", "content": "user evidence"}],
    }


async def test_anthropic_parse_uses_native_pydantic_output() -> None:
    fake = _fake_anthropic(_anthropic_response())
    client = AnthropicClient("test-key", "claude-sonnet-5", client=fake)

    result = await client.parse(system="s", user="u", schema=_Answer)

    assert result.data == _Answer(summary="ok", score=9)
    [call] = fake.messages.parse_calls
    assert call["output_format"] is _Answer
    assert "temperature" not in call


@pytest.mark.parametrize("stop_reason", ["refusal", "max_tokens"])
async def test_anthropic_rejects_refused_or_truncated_output(stop_reason: str) -> None:
    fake = _fake_anthropic(_anthropic_response(stop_reason=stop_reason))
    client = AnthropicClient("test-key", "claude-sonnet-5", client=fake)

    with pytest.raises(ValueError, match=stop_reason):
        await client.parse(system="s", user="u", schema=_Answer)


class _FakeGeminiModels:
    def __init__(self, responses: list[SimpleNamespace]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    async def generate_content(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        return self.responses.pop(0)


def _fake_gemini(*responses: SimpleNamespace) -> SimpleNamespace:
    models = _FakeGeminiModels(list(responses))
    return SimpleNamespace(aio=SimpleNamespace(models=models), models=models)


def _gemini_response(**overrides: Any) -> SimpleNamespace:
    values: dict[str, Any] = {
        "text": "resposta",
        "parsed": _Answer(summary="typed", score=8),
        "usage_metadata": SimpleNamespace(
            prompt_token_count=13,
            candidates_token_count=5,
            total_token_count=25,
        ),
        "model_version": "gemini-3.6-flash",
        "response_id": "gemini-response-test",
        "prompt_feedback": None,
        "candidates": [],
    }
    values.update(overrides)
    return SimpleNamespace(**values)


async def test_gemini_complete_uses_async_sdk_and_counts_thinking_tokens() -> None:
    fake = _fake_gemini(_gemini_response(model_version="gemini-3.6-flash-001"))
    client = GeminiClient("test-key", "gemini-3.6-flash", client=fake)

    result = await client.complete(
        system="system policy",
        user="user evidence",
        temperature=0.9,
        prompt_version="agent-v3",
    )

    assert result.text == "resposta"
    assert result.meta.provider == "gemini"
    assert result.meta.usage.prompt_tokens == 13
    assert result.meta.usage.completion_tokens == 12
    assert result.meta.cost_usd is not None
    assert result.meta.prompt_version == "agent-v3"
    [call] = fake.models.calls
    assert call["model"] == "gemini-3.6-flash"
    assert call["contents"] == "user evidence"
    assert call["config"].system_instruction == "system policy"
    assert call["config"].temperature is None


async def test_gemini_parse_uses_native_parsed_pydantic_model() -> None:
    fake = _fake_gemini(_gemini_response())
    client = GeminiClient("test-key", "gemini-3.6-flash", client=fake)

    result = await client.parse(system="s", user="u", schema=_Answer)

    assert result.data == _Answer(summary="typed", score=8)
    [call] = fake.models.calls
    assert call["config"].response_mime_type == "application/json"
    assert call["config"].response_schema is _Answer


async def test_gemini_parse_validates_json_when_sdk_parsed_value_is_absent() -> None:
    response = _gemini_response(parsed=None, text='{"summary":"fallback","score":6}')
    client = GeminiClient("test-key", "gemini-3.6-flash", client=_fake_gemini(response))

    result = await client.parse(system="s", user="u", schema=_Answer)

    assert result.data == _Answer(summary="fallback", score=6)


async def test_gemini_reports_blocked_empty_response() -> None:
    response = _gemini_response(
        parsed=None,
        text="",
        prompt_feedback=SimpleNamespace(block_reason="SAFETY"),
    )
    client = GeminiClient("test-key", "gemini-3.6-flash", client=_fake_gemini(response))

    with pytest.raises(ValueError, match="SAFETY"):
        await client.parse(system="s", user="u", schema=_Answer)
