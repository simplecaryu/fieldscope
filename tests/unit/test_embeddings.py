"""Tests for fieldscope embedding providers."""

import json

import numpy as np
import pytest
from pytest_httpx import HTTPXMock

from fieldscope.config import EmbeddingConfig
from fieldscope.embeddings.base import EmbeddingProvider, create_embedding_provider, prepare_text
from fieldscope.embeddings.sentence_transformers import SentenceTransformerProvider
from fieldscope.embeddings.openai_compatible import OpenAICompatibleProvider
from fieldscope.models import Author, Paper, Provenance


# ---------------------------------------------------------------------------
# Text preparation
# ---------------------------------------------------------------------------


def _make_paper(**overrides):
    defaults = dict(
        title="Test Paper",
        source="openalex",
        provenance=Provenance(method="initial_retrieval", depth=0),
        doi="10.1234/test",
    )
    defaults.update(overrides)
    return Paper(**defaults)


class TestPrepareText:
    def test_title_and_abstract(self):
        p = _make_paper(title="My Title", abstract="My abstract text")
        text = prepare_text(p, text_fields=["title", "abstract"])
        assert text == "My Title. My abstract text"

    def test_title_only_when_no_abstract(self):
        p = _make_paper(title="My Title", abstract=None)
        text = prepare_text(p, text_fields=["title", "abstract"])
        assert text == "My Title"

    def test_title_only_field(self):
        p = _make_paper(title="My Title", abstract="Some abstract")
        text = prepare_text(p, text_fields=["title"])
        assert text == "My Title"


# ---------------------------------------------------------------------------
# Provider factory
# ---------------------------------------------------------------------------


class TestCreateEmbeddingProvider:
    def test_sentence_transformers_provider(self):
        cfg = EmbeddingConfig(provider="sentence-transformers")
        provider = create_embedding_provider(cfg)
        assert isinstance(provider, SentenceTransformerProvider)

    def test_openai_compatible_provider(self):
        cfg = EmbeddingConfig(
            provider="openai-compatible",
            base_url="https://api.example.com/v1",
        )
        provider = create_embedding_provider(cfg, api_key="test-key")
        assert isinstance(provider, OpenAICompatibleProvider)

    def test_unknown_provider_raises(self):
        cfg = EmbeddingConfig(provider="unknown")
        with pytest.raises(ValueError, match="Unknown embedding provider"):
            create_embedding_provider(cfg)


# ---------------------------------------------------------------------------
# SentenceTransformerProvider
# ---------------------------------------------------------------------------


class TestSentenceTransformerProvider:
    def test_embed_returns_correct_shape(self):
        cfg = EmbeddingConfig(
            provider="sentence-transformers",
            model="all-MiniLM-L6-v2",
            dimensions=384,
        )
        provider = SentenceTransformerProvider(cfg)
        texts = ["Hello world", "Another text"]
        embeddings = provider.embed(texts)
        assert embeddings.shape == (2, 384)

    def test_embed_single_text(self):
        cfg = EmbeddingConfig(
            provider="sentence-transformers",
            model="all-MiniLM-L6-v2",
            dimensions=384,
        )
        provider = SentenceTransformerProvider(cfg)
        embeddings = provider.embed(["Hello"])
        assert embeddings.shape == (1, 384)

    def test_embed_normalized(self):
        cfg = EmbeddingConfig(
            provider="sentence-transformers",
            model="all-MiniLM-L6-v2",
            dimensions=384,
            normalize=True,
        )
        provider = SentenceTransformerProvider(cfg)
        embeddings = provider.embed(["Test text"])
        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_embed_papers(self):
        cfg = EmbeddingConfig(
            provider="sentence-transformers",
            model="all-MiniLM-L6-v2",
            dimensions=384,
        )
        provider = SentenceTransformerProvider(cfg)
        papers = [
            _make_paper(title="Quantum magnetism", abstract="Study of spin systems"),
            _make_paper(title="Spintronics", abstract=None, doi="10.1234/other"),
        ]
        embeddings = provider.embed_papers(papers)
        assert embeddings.shape == (2, 384)

    def test_embed_empty_raises(self):
        cfg = EmbeddingConfig(provider="sentence-transformers")
        provider = SentenceTransformerProvider(cfg)
        with pytest.raises(ValueError, match="empty"):
            provider.embed([])


# ---------------------------------------------------------------------------
# OpenAICompatibleProvider (mocked HTTP)
# ---------------------------------------------------------------------------


class TestOpenAICompatibleProvider:
    @pytest.mark.asyncio
    async def test_embed_async(self, httpx_mock: HTTPXMock):
        httpx_mock.add_response(
            url="https://api.example.com/v1/embeddings",
            json={
                "data": [
                    {"embedding": [0.1] * 384, "index": 0},
                    {"embedding": [0.2] * 384, "index": 1},
                ],
                "usage": {"prompt_tokens": 10, "total_tokens": 10},
            },
        )

        cfg = EmbeddingConfig(
            provider="openai-compatible",
            base_url="https://api.example.com/v1",
            model="text-embedding-3-small",
            dimensions=384,
        )
        provider = OpenAICompatibleProvider(cfg, api_key="test-key")
        embeddings = await provider.embed_async(["text one", "text two"])
        assert embeddings.shape == (2, 384)

    @pytest.mark.asyncio
    async def test_embed_async_normalized(self, httpx_mock: HTTPXMock):
        # Return unnormalized vectors
        httpx_mock.add_response(
            url="https://api.example.com/v1/embeddings",
            json={
                "data": [{"embedding": [3.0, 4.0, 0.0], "index": 0}],
                "usage": {"prompt_tokens": 5, "total_tokens": 5},
            },
        )

        cfg = EmbeddingConfig(
            provider="openai-compatible",
            base_url="https://api.example.com/v1",
            dimensions=3,
            normalize=True,
        )
        provider = OpenAICompatibleProvider(cfg, api_key="k")
        embeddings = await provider.embed_async(["test"])
        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    @pytest.mark.asyncio
    async def test_embed_async_retries_on_server_error(self, httpx_mock: HTTPXMock):
        httpx_mock.add_response(status_code=500)
        httpx_mock.add_response(
            json={
                "data": [{"embedding": [0.1] * 3, "index": 0}],
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            },
        )

        cfg = EmbeddingConfig(
            provider="openai-compatible",
            base_url="https://api.example.com/v1",
            dimensions=3,
        )
        provider = OpenAICompatibleProvider(
            cfg, api_key="k", retry_max_attempts=3, retry_backoff_base=0.01
        )
        embeddings = await provider.embed_async(["test"])
        assert embeddings.shape == (1, 3)

    def test_embed_sync_wrapper(self, httpx_mock: HTTPXMock):
        httpx_mock.add_response(
            url="https://api.example.com/v1/embeddings",
            json={
                "data": [{"embedding": [0.5] * 3, "index": 0}],
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            },
        )

        cfg = EmbeddingConfig(
            provider="openai-compatible",
            base_url="https://api.example.com/v1",
            dimensions=3,
        )
        provider = OpenAICompatibleProvider(cfg, api_key="k")
        embeddings = provider.embed(["test"])
        assert embeddings.shape == (1, 3)
