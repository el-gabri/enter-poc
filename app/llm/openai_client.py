"""OpenAI implementation of the LLMClient protocol."""

import time

from openai import AsyncOpenAI

from app.core.logging import get_logger
from app.llm.base import (
    LLMCallMetadata,
    ParsedResult,
    SchemaT,
    TextResult,
    TokenUsage,
)
from app.llm.pricing import estimate_cost_usd

logger = get_logger(__name__)

PROVIDER_NAME = "openai"


class OpenAIClient:
    """LLMClient backed by the OpenAI API with native structured outputs."""

    def __init__(self, api_key: str, model: str) -> None:
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model

    async def complete(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.0,
        prompt_version: str | None = None,
    ) -> TextResult:
        start = time.perf_counter()
        response = await self._client.chat.completions.create(
            model=self._model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        meta = self._build_meta(response.usage, start, prompt_version)
        text = response.choices[0].message.content or ""
        logger.info("llm_complete", **meta.model_dump())
        return TextResult(text=text, meta=meta)

    async def parse(
        self,
        *,
        system: str,
        user: str,
        schema: type[SchemaT],
        temperature: float = 0.0,
        prompt_version: str | None = None,
    ) -> ParsedResult[SchemaT]:
        start = time.perf_counter()
        response = await self._client.beta.chat.completions.parse(
            model=self._model,
            temperature=temperature,
            response_format=schema,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        message = response.choices[0].message
        if message.parsed is None:  # refusal or parsing failure
            raise ValueError(
                f"Model did not return valid {schema.__name__}: "
                f"{message.refusal or 'unparseable output'}"
            )
        meta = self._build_meta(response.usage, start, prompt_version)
        logger.info("llm_parse", schema=schema.__name__, **meta.model_dump())
        return ParsedResult[schema](data=message.parsed, meta=meta)  # type: ignore[valid-type]

    def _build_meta(
        self, usage: object, start: float, prompt_version: str | None
    ) -> LLMCallMetadata:
        token_usage = TokenUsage(
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )
        return LLMCallMetadata(
            provider=PROVIDER_NAME,
            model=self._model,
            latency_ms=(time.perf_counter() - start) * 1000,
            usage=token_usage,
            cost_usd=estimate_cost_usd(self._model, token_usage),
            prompt_version=prompt_version,
        )
