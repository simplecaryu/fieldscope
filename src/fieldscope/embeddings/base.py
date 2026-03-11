"""Embedding provider abstraction for fieldscope."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from fieldscope.config import EmbeddingConfig
    from fieldscope.models import Paper


def prepare_text(paper: Paper, text_fields: list[str]) -> str:
    """Prepare text for embedding from a paper's fields."""
    parts = []
    for field in text_fields:
        value = getattr(paper, field, None)
        if value is not None:
            parts.append(value)
    return ". ".join(parts)


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts, returning an (N, D) array."""
        ...

    def embed_papers(self, papers: list[Paper], text_fields: list[str] | None = None) -> np.ndarray:
        """Embed a list of papers."""
        fields = text_fields or self.config.text_fields
        texts = [prepare_text(p, fields) for p in papers]
        return self.embed(texts)

    @property
    @abstractmethod
    def config(self) -> EmbeddingConfig:
        ...


def create_embedding_provider(
    config: EmbeddingConfig,
    api_key: str | None = None,
) -> EmbeddingProvider:
    """Factory to create the correct embedding provider."""
    if config.provider == "sentence-transformers":
        from fieldscope.embeddings.sentence_transformers import SentenceTransformerProvider
        return SentenceTransformerProvider(config)
    elif config.provider == "openai-compatible":
        from fieldscope.embeddings.openai_compatible import OpenAICompatibleProvider
        return OpenAICompatibleProvider(config, api_key=api_key or "")
    else:
        raise ValueError(f"Unknown embedding provider: {config.provider}")
