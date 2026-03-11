"""Sentence-transformers embedding provider."""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from fieldscope.config import EmbeddingConfig
from fieldscope.embeddings.base import EmbeddingProvider


class SentenceTransformerProvider(EmbeddingProvider):
    """Local sentence-transformers embedding provider."""

    def __init__(self, cfg: EmbeddingConfig) -> None:
        self._config = cfg
        self._model = SentenceTransformer(cfg.model)

    @property
    def config(self) -> EmbeddingConfig:
        return self._config

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            raise ValueError("Cannot embed empty text list")

        embeddings = self._model.encode(
            texts,
            batch_size=self._config.batch_size,
            normalize_embeddings=self._config.normalize,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)
