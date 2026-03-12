"""Integration tests with OpenRouter API.

These tests make real API calls and require:
  - OPENROUTER_API_KEY environment variable set

Run with: pytest tests/integration/test_openrouter.py -v
Skip with: pytest -m "not integration"
"""

import os

import pytest

from fieldscope.config import LLMConfig, LLMStageOverride
from fieldscope.llm.client import LLMClient
from fieldscope.stages.keyword_expansion import expand_keywords

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# Cost-effective model for testing
TEST_MODEL = "google/gemini-2.0-flash-001"

pytestmark = pytest.mark.skipif(
    not OPENROUTER_API_KEY,
    reason="OPENROUTER_API_KEY not set",
)


class TestOpenRouterLLMClient:
    @pytest.mark.asyncio
    async def test_basic_chat_completion(self):
        client = LLMClient(
            base_url=OPENROUTER_BASE_URL,
            model=TEST_MODEL,
            api_key=OPENROUTER_API_KEY,
            temperature=0.0,
            max_tokens=50,
        )
        result = await client.chat(
            messages=[{"role": "user", "content": "Reply with exactly one word: hello"}],
        )
        assert result.content
        assert len(result.content) < 200
        assert result.finish_reason in ("stop", "end_turn")

    @pytest.mark.asyncio
    async def test_keyword_expansion_prompt(self):
        """Test a prompt similar to what keyword_expansion stage would use."""
        client = LLMClient(
            base_url=OPENROUTER_BASE_URL,
            model=TEST_MODEL,
            api_key=OPENROUTER_API_KEY,
            temperature=0.3,
            max_tokens=256,
        )
        result = await client.chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a research librarian helping define search queries for "
                        "bibliometric analysis. Given a research field, suggest 10 related "
                        "search keywords. Return ONLY a JSON array of strings."
                    ),
                },
                {
                    "role": "user",
                    "content": "Research field: altermagnetism",
                },
            ],
        )
        assert result.content
        # Should contain something that looks like keywords
        content_lower = result.content.lower()
        assert any(
            term in content_lower
            for term in ["altermag", "magnet", "spin", "symmetry"]
        )

    @pytest.mark.asyncio
    async def test_from_config(self):
        cfg = LLMConfig(
            base_url=OPENROUTER_BASE_URL,
            model=TEST_MODEL,
            temperature=0.0,
            max_tokens=30,
        )
        client = LLMClient.from_config(cfg, api_key=OPENROUTER_API_KEY)
        result = await client.chat(
            messages=[{"role": "user", "content": "Say 'ok'"}],
        )
        assert result.content

    @pytest.mark.asyncio
    async def test_stage_override_config(self):
        cfg = LLMConfig(
            base_url=OPENROUTER_BASE_URL,
            model=TEST_MODEL,
            temperature=0.3,
            keyword_expansion=LLMStageOverride(temperature=0.5, max_tokens=100),
        )
        client = LLMClient.from_config(
            cfg, api_key=OPENROUTER_API_KEY, stage="keyword_expansion"
        )
        assert client.temperature == 0.5
        assert client.max_tokens == 100

        result = await client.chat(
            messages=[{"role": "user", "content": "Say 'test passed'"}],
        )
        assert result.content


class TestOpenRouterKeywordExpansion:
    @pytest.mark.asyncio
    async def test_expand_keywords_real_api(self):
        """End-to-end keyword expansion via OpenRouter."""
        config = LLMConfig(
            base_url=OPENROUTER_BASE_URL,
            model=TEST_MODEL,
            temperature=0.3,
            max_tokens=512,
        )
        keywords = await expand_keywords(
            query="altermagnetism",
            config=config,
            api_key=OPENROUTER_API_KEY,
        )
        assert len(keywords) >= 5
        assert "altermagnetism" in [k.lower() for k in keywords]
        # Should contain domain-relevant terms
        all_lower = " ".join(k.lower() for k in keywords)
        assert any(
            term in all_lower
            for term in ["magnet", "spin", "symmetry", "antiferro", "order"]
        )
