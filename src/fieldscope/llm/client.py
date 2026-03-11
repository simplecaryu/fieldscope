"""OpenAI-compatible LLM client for fieldscope."""

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass

import httpx

from fieldscope.config import LLMConfig

logger = logging.getLogger(__name__)

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class LLMError(Exception):
    """Base exception for LLM client errors."""


class LLMNotConfiguredError(LLMError):
    """Raised when LLM is used but not configured."""


@dataclass
class ChatResult:
    """Result from a chat completion request."""

    content: str
    usage: dict[str, int]
    finish_reason: str = "stop"


def resolve_stage_config(config: LLMConfig, stage: str) -> dict:
    """Resolve effective LLM config for a stage, merging base + override."""
    base = {
        "base_url": config.base_url,
        "model": config.model,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }

    override_map = {
        "keyword_expansion": config.keyword_expansion,
        "topic_labeling": config.topic_labeling,
        "reporting": config.reporting,
    }

    override = override_map.get(stage)
    if override is None:
        return base

    for field in ("base_url", "model", "temperature", "max_tokens"):
        val = getattr(override, field, None)
        if val is not None:
            base[field] = val

    return base


class LLMClient:
    """Async OpenAI-compatible chat completions client with retry."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        retry_max_attempts: int = 3,
        retry_backoff_base: float = 2.0,
    ) -> None:
        if not base_url or not model or not api_key:
            raise LLMNotConfiguredError(
                "LLM is not configured. Set base_url, model, and API key."
            )
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retry_max_attempts = retry_max_attempts
        self.retry_backoff_base = retry_backoff_base

    @classmethod
    def from_config(
        cls,
        config: LLMConfig,
        api_key: str,
        stage: str | None = None,
    ) -> LLMClient:
        """Create an LLMClient from config, optionally with stage overrides."""
        if stage:
            resolved = resolve_stage_config(config, stage)
        else:
            resolved = {
                "base_url": config.base_url,
                "model": config.model,
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
            }
        return cls(
            base_url=resolved["base_url"],
            model=resolved["model"],
            api_key=api_key,
            temperature=resolved["temperature"],
            max_tokens=resolved["max_tokens"],
        )

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult:
        """Send a chat completion request with retry logic."""
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_error: Exception | None = None
        for attempt in range(self.retry_max_attempts):
            try:
                async with httpx.AsyncClient() as http:
                    response = await http.post(
                        url, json=payload, headers=headers, timeout=60.0
                    )

                if response.status_code == 200:
                    data = response.json()
                    choice = data["choices"][0]
                    return ChatResult(
                        content=choice["message"]["content"],
                        usage=data.get("usage", {}),
                        finish_reason=choice.get("finish_reason", "stop"),
                    )

                if response.status_code not in RETRYABLE_STATUS_CODES:
                    raise LLMError(
                        f"LLM request failed with status {response.status_code}: "
                        f"{response.text}"
                    )

                last_error = LLMError(
                    f"LLM request failed with status {response.status_code}"
                )
                logger.warning(
                    "LLM request failed (attempt %d/%d): %s",
                    attempt + 1,
                    self.retry_max_attempts,
                    response.status_code,
                )

            except httpx.TimeoutException as e:
                last_error = LLMError(f"LLM request timed out: {e}")
                logger.warning("LLM request timed out (attempt %d/%d)", attempt + 1, self.retry_max_attempts)
            except LLMError:
                raise
            except httpx.HTTPError as e:
                last_error = LLMError(f"HTTP error: {e}")
                logger.warning("HTTP error (attempt %d/%d): %s", attempt + 1, self.retry_max_attempts, e)

            if attempt < self.retry_max_attempts - 1:
                backoff = self.retry_backoff_base * (2**attempt) + random.uniform(0, 1)
                await asyncio.sleep(backoff)

        raise LLMError(f"LLM request failed after {self.retry_max_attempts} retries: {last_error}")
