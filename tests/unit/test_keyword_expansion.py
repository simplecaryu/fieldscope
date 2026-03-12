"""Tests for keyword expansion stage."""

import json

import pytest
from pytest_httpx import HTTPXMock

from fieldscope.config import LLMConfig
from fieldscope.stages.keyword_expansion import expand_keywords, parse_keywords_response


# ---------------------------------------------------------------------------
# parse_keywords_response
# ---------------------------------------------------------------------------


class TestParseKeywordsResponse:
    def test_parse_json_array(self):
        raw = '["quantum magnetism", "spin dynamics", "magnetic ordering"]'
        result = parse_keywords_response(raw)
        assert result == ["quantum magnetism", "spin dynamics", "magnetic ordering"]

    def test_parse_json_array_in_markdown_block(self):
        raw = '```json\n["quantum magnetism", "spin dynamics"]\n```'
        result = parse_keywords_response(raw)
        assert result == ["quantum magnetism", "spin dynamics"]

    def test_parse_newline_separated(self):
        raw = "quantum magnetism\nspin dynamics\nmagnetic ordering"
        result = parse_keywords_response(raw)
        assert result == ["quantum magnetism", "spin dynamics", "magnetic ordering"]

    def test_parse_numbered_list(self):
        raw = "1. quantum magnetism\n2. spin dynamics\n3. magnetic ordering"
        result = parse_keywords_response(raw)
        assert result == ["quantum magnetism", "spin dynamics", "magnetic ordering"]

    def test_parse_bullet_list(self):
        raw = "- quantum magnetism\n- spin dynamics\n- magnetic ordering"
        result = parse_keywords_response(raw)
        assert result == ["quantum magnetism", "spin dynamics", "magnetic ordering"]

    def test_strips_whitespace(self):
        raw = '["  quantum magnetism  ", " spin dynamics "]'
        result = parse_keywords_response(raw)
        assert result == ["quantum magnetism", "spin dynamics"]

    def test_filters_empty_strings(self):
        raw = '["quantum magnetism", "", "spin dynamics"]'
        result = parse_keywords_response(raw)
        assert result == ["quantum magnetism", "spin dynamics"]

    def test_includes_original_query(self):
        raw = '["spin dynamics", "magnetic ordering"]'
        result = parse_keywords_response(raw, original_query="quantum magnetism")
        assert "quantum magnetism" in result
        assert "spin dynamics" in result

    def test_deduplicates(self):
        raw = '["quantum magnetism", "quantum magnetism", "spin dynamics"]'
        result = parse_keywords_response(raw)
        assert result == ["quantum magnetism", "spin dynamics"]


# ---------------------------------------------------------------------------
# expand_keywords (mocked LLM)
# ---------------------------------------------------------------------------


class TestExpandKeywords:
    @pytest.mark.asyncio
    async def test_expand_keywords(self, httpx_mock: HTTPXMock):
        httpx_mock.add_response(
            url="https://api.example.com/v1/chat/completions",
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": '["altermagnetism", "altermagnetic order", "spin-split bands", "collinear antiferromagnet"]',
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            },
        )

        config = LLMConfig(
            base_url="https://api.example.com/v1",
            model="test-model",
            temperature=0.5,
        )
        result = await expand_keywords(
            query="altermagnetism",
            config=config,
            api_key="test-key",
        )
        assert "altermagnetism" in result
        assert "altermagnetic order" in result
        assert len(result) >= 4

    @pytest.mark.asyncio
    async def test_expand_keywords_always_includes_query(self, httpx_mock: HTTPXMock):
        httpx_mock.add_response(
            url="https://api.example.com/v1/chat/completions",
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": '["spin dynamics", "magnetic ordering"]',
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70},
            },
        )

        config = LLMConfig(
            base_url="https://api.example.com/v1",
            model="test-model",
        )
        result = await expand_keywords(
            query="altermagnetism",
            config=config,
            api_key="test-key",
        )
        # Original query should always be included
        assert "altermagnetism" in result
