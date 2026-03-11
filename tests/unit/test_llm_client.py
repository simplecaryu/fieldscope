"""Tests for fieldscope LLM client."""

import json

import httpx
import pytest
from pytest_httpx import HTTPXMock

from fieldscope.config import LLMConfig, LLMStageOverride
from fieldscope.llm.client import (
    LLMClient,
    LLMError,
    LLMNotConfiguredError,
    resolve_stage_config,
)


# ---------------------------------------------------------------------------
# resolve_stage_config
# ---------------------------------------------------------------------------


class TestResolveStageConfig:
    def test_no_override_returns_base(self):
        base = LLMConfig(base_url="http://localhost/v1", model="base-model")
        resolved = resolve_stage_config(base, stage="keyword_expansion")
        assert resolved["base_url"] == "http://localhost/v1"
        assert resolved["model"] == "base-model"
        assert resolved["temperature"] == 0.3

    def test_override_merges_fields(self):
        base = LLMConfig(
            base_url="http://localhost/v1",
            model="base-model",
            temperature=0.3,
            keyword_expansion=LLMStageOverride(model="override-model", temperature=0.5),
        )
        resolved = resolve_stage_config(base, stage="keyword_expansion")
        assert resolved["model"] == "override-model"
        assert resolved["temperature"] == 0.5
        assert resolved["base_url"] == "http://localhost/v1"  # inherited

    def test_override_only_overrides_set_fields(self):
        base = LLMConfig(
            base_url="http://localhost/v1",
            model="base-model",
            max_tokens=2048,
            reporting=LLMStageOverride(max_tokens=4096),
        )
        resolved = resolve_stage_config(base, stage="reporting")
        assert resolved["max_tokens"] == 4096
        assert resolved["model"] == "base-model"  # inherited

    def test_unknown_stage_returns_base(self):
        base = LLMConfig(base_url="http://localhost/v1", model="m")
        resolved = resolve_stage_config(base, stage="nonexistent")
        assert resolved["model"] == "m"


# ---------------------------------------------------------------------------
# LLMClient - unit tests with mocked HTTP
# ---------------------------------------------------------------------------


class TestLLMClient:
    def test_create_client(self):
        client = LLMClient(
            base_url="https://openrouter.ai/api/v1",
            model="google/gemini-2.0-flash-001",
            api_key="test-key",
        )
        assert client.model == "google/gemini-2.0-flash-001"

    def test_not_configured_error(self):
        with pytest.raises(LLMNotConfiguredError):
            LLMClient(base_url="", model="", api_key="")

    @pytest.mark.asyncio
    async def test_chat_completion(self, httpx_mock: HTTPXMock):
        httpx_mock.add_response(
            url="https://api.example.com/v1/chat/completions",
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "Hello world"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        )

        client = LLMClient(
            base_url="https://api.example.com/v1",
            model="test-model",
            api_key="test-key",
        )
        result = await client.chat(
            messages=[{"role": "user", "content": "Say hello"}],
        )
        assert result.content == "Hello world"
        assert result.usage["total_tokens"] == 15

    @pytest.mark.asyncio
    async def test_chat_with_temperature_and_max_tokens(self, httpx_mock: HTTPXMock):
        httpx_mock.add_response(
            url="https://api.example.com/v1/chat/completions",
            json={
                "choices": [{"message": {"role": "assistant", "content": "response"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            },
        )

        client = LLMClient(
            base_url="https://api.example.com/v1",
            model="test-model",
            api_key="test-key",
            temperature=0.7,
            max_tokens=512,
        )
        result = await client.chat(messages=[{"role": "user", "content": "test"}])
        assert result.content == "response"

        # Verify request payload
        request = httpx_mock.get_requests()[0]
        body = json.loads(request.content)
        assert body["temperature"] == 0.7
        assert body["max_tokens"] == 512
        assert body["model"] == "test-model"

    @pytest.mark.asyncio
    async def test_retry_on_server_error(self, httpx_mock: HTTPXMock):
        # First call: 500, second call: success
        httpx_mock.add_response(status_code=500)
        httpx_mock.add_response(
            json={
                "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

        client = LLMClient(
            base_url="https://api.example.com/v1",
            model="m",
            api_key="k",
            retry_max_attempts=3,
            retry_backoff_base=0.01,  # fast for tests
        )
        result = await client.chat(messages=[{"role": "user", "content": "test"}])
        assert result.content == "ok"
        assert len(httpx_mock.get_requests()) == 2

    @pytest.mark.asyncio
    async def test_non_retryable_error_fails_immediately(self, httpx_mock: HTTPXMock):
        httpx_mock.add_response(status_code=401)

        client = LLMClient(
            base_url="https://api.example.com/v1",
            model="m",
            api_key="k",
            retry_max_attempts=3,
            retry_backoff_base=0.01,
        )
        with pytest.raises(LLMError, match="401"):
            await client.chat(messages=[{"role": "user", "content": "test"}])
        assert len(httpx_mock.get_requests()) == 1

    @pytest.mark.asyncio
    async def test_exhausted_retries_raises(self, httpx_mock: HTTPXMock):
        for _ in range(3):
            httpx_mock.add_response(status_code=500)

        client = LLMClient(
            base_url="https://api.example.com/v1",
            model="m",
            api_key="k",
            retry_max_attempts=3,
            retry_backoff_base=0.01,
        )
        with pytest.raises(LLMError, match="retries"):
            await client.chat(messages=[{"role": "user", "content": "test"}])


# ---------------------------------------------------------------------------
# LLMClient.from_config helper
# ---------------------------------------------------------------------------


class TestLLMClientFromConfig:
    def test_from_config_base(self):
        cfg = LLMConfig(base_url="http://localhost/v1", model="test-model")
        client = LLMClient.from_config(cfg, api_key="key")
        assert client.model == "test-model"

    def test_from_config_with_stage_override(self):
        cfg = LLMConfig(
            base_url="http://localhost/v1",
            model="base",
            reporting=LLMStageOverride(model="big-model", max_tokens=4096),
        )
        client = LLMClient.from_config(cfg, api_key="key", stage="reporting")
        assert client.model == "big-model"
        assert client.max_tokens == 4096
