"""OpenAI-compatible embedding provider."""

from __future__ import annotations

import asyncio
import logging
import random

import httpx
import numpy as np

from fieldscope.config import EmbeddingConfig
from fieldscope.embeddings.base import EmbeddingProvider

logger = logging.getLogger(__name__)

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class OpenAICompatibleProvider(EmbeddingProvider):
    """Remote OpenAI-compatible embedding provider."""

    def __init__(
        self,
        cfg: EmbeddingConfig,
        api_key: str,
        retry_max_attempts: int = 3,
        retry_backoff_base: float = 2.0,
    ) -> None:
        self._config = cfg
        self._api_key = api_key
        self._retry_max_attempts = retry_max_attempts
        self._retry_backoff_base = retry_backoff_base

    @property
    def config(self) -> EmbeddingConfig:
        return self._config

    async def embed_async(self, texts: list[str]) -> np.ndarray:
        """Embed texts via API with retry logic."""
        if not texts:
            raise ValueError("Cannot embed empty text list")

        url = f"{self._config.base_url.rstrip('/')}/embeddings"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._config.model,
            "input": texts,
        }

        last_error: Exception | None = None
        for attempt in range(self._retry_max_attempts):
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        url, json=payload, headers=headers, timeout=60.0
                    )

                if response.status_code == 200:
                    data = response.json()
                    # Sort by index to guarantee order
                    sorted_data = sorted(data["data"], key=lambda x: x["index"])
                    embeddings = np.array(
                        [d["embedding"] for d in sorted_data], dtype=np.float32
                    )
                    if self._config.normalize:
                        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                        norms = np.where(norms == 0, 1, norms)
                        embeddings = embeddings / norms
                    return embeddings

                if response.status_code not in RETRYABLE_STATUS_CODES:
                    raise RuntimeError(
                        f"Embedding request failed with status {response.status_code}: "
                        f"{response.text}"
                    )

                last_error = RuntimeError(f"Status {response.status_code}")
                logger.warning(
                    "Embedding request failed (attempt %d/%d): %s",
                    attempt + 1, self._retry_max_attempts, response.status_code,
                )
            except RuntimeError:
                raise
            except httpx.HTTPError as e:
                last_error = e
                logger.warning("HTTP error (attempt %d/%d): %s", attempt + 1, self._retry_max_attempts, e)

            if attempt < self._retry_max_attempts - 1:
                backoff = self._retry_backoff_base * (2**attempt) + random.uniform(0, 1)
                await asyncio.sleep(backoff)

        raise RuntimeError(
            f"Embedding request failed after {self._retry_max_attempts} retries: {last_error}"
        )

    def embed(self, texts: list[str]) -> np.ndarray:
        """Synchronous wrapper around embed_async."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(lambda: asyncio.run(self.embed_async(texts))).result()
        return asyncio.run(self.embed_async(texts))
