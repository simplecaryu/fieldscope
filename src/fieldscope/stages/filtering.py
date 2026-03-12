"""Stage 6: Dataset filtering."""

from __future__ import annotations

import logging

import numpy as np

from fieldscope.config import FilteringConfig
from fieldscope.models import Paper

logger = logging.getLogger(__name__)


def filter_by_metadata(papers: list[Paper], config: FilteringConfig) -> list[Paper]:
    """Filter papers based on metadata completeness."""
    result = []
    for p in papers:
        if config.require_year and p.year is None:
            continue
        if config.require_abstract and p.abstract is None:
            continue
        result.append(p)
    removed = len(papers) - len(result)
    if removed:
        logger.info("Metadata filter: removed %d papers", removed)
    return result


def filter_by_keyword_overlap(
    papers: list[Paper],
    keywords: list[str],
    config: FilteringConfig,
) -> list[Paper]:
    """Filter papers by keyword overlap with the search keywords."""
    if not keywords:
        return list(papers)

    keywords_lower = {kw.lower() for kw in keywords}
    result = []
    for p in papers:
        text = f"{p.title} {p.abstract or ''}".lower()
        overlap = sum(1 for kw in keywords_lower if kw in text)
        if overlap >= config.keyword_min_overlap:
            result.append(p)
    removed = len(papers) - len(result)
    if removed:
        logger.info("Keyword filter: removed %d papers", removed)
    return result


def filter_by_semantic_similarity(
    papers: list[Paper],
    seed_centroid: np.ndarray,
    config: FilteringConfig,
) -> list[Paper]:
    """Filter papers by cosine similarity to seed centroid."""
    result = []
    for p in papers:
        if p.embedding is None:
            # Keep papers without embeddings (can't evaluate)
            result.append(p)
            continue
        emb = np.array(p.embedding)
        norm_emb = np.linalg.norm(emb)
        norm_cent = np.linalg.norm(seed_centroid)
        if norm_emb == 0 or norm_cent == 0:
            result.append(p)
            continue
        sim = np.dot(emb, seed_centroid) / (norm_emb * norm_cent)
        if sim >= config.semantic_threshold:
            result.append(p)
    removed = len(papers) - len(result)
    if removed:
        logger.info("Semantic filter: removed %d papers", removed)
    return result


def filter_dataset(
    papers: list[Paper],
    keywords: list[str],
    config: FilteringConfig,
    seed_centroid: np.ndarray | None = None,
) -> list[Paper]:
    """Apply all filtering layers in sequence."""
    initial = len(papers)

    result = filter_by_metadata(papers, config)
    result = filter_by_keyword_overlap(result, keywords, config)
    if seed_centroid is not None:
        result = filter_by_semantic_similarity(result, seed_centroid, config)

    logger.info("Dataset filtering: %d → %d papers", initial, len(result))
    return result
