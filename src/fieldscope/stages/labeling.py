"""Stage 10: Topic labeling for clusters."""

from __future__ import annotations

import logging
from collections import Counter

from fieldscope.config import LLMConfig
from fieldscope.models import Cluster, Paper

logger = logging.getLogger(__name__)


def _extract_keywords(titles: list[str], top_n: int = 5) -> list[str]:
    """Extract top keywords from a list of titles."""
    stopwords = {
        "the", "a", "an", "of", "in", "on", "for", "and", "or", "to", "is",
        "are", "was", "were", "by", "with", "from", "at", "as", "its", "it",
        "this", "that", "be", "has", "have", "had", "not", "but", "can",
        "will", "do", "does", "their", "we", "our", "using", "based", "via",
    }
    words: list[str] = []
    for title in titles:
        for w in title.lower().split():
            cleaned = w.strip(".,;:!?()[]{}\"'")
            if cleaned and len(cleaned) > 2 and cleaned not in stopwords:
                words.append(cleaned)
    counter = Counter(words)
    return [w for w, _ in counter.most_common(top_n)]


def _extractive_label(titles: list[str]) -> str:
    """Generate an extractive label from the most common title terms."""
    keywords = _extract_keywords(titles, top_n=3)
    if keywords:
        return " / ".join(keywords)
    return "unlabeled"


def label_clusters(
    clusters: list[Cluster],
    papers: list[Paper],
    llm_config: LLMConfig | None = None,
) -> list[Cluster]:
    """Label clusters with extractive labels and optionally refine with LLM.

    Always generates extractive labels first. LLM refinement is applied
    only if llm_config is provided.
    """
    if not clusters:
        return []

    # Build paper lookup
    paper_map = {p.paper_id: p for p in papers}

    result = []
    for cluster in clusters:
        # Get titles for this cluster's papers
        member_papers = [paper_map[pid] for pid in cluster.member_paper_ids if pid in paper_map]
        titles = [p.title for p in member_papers]

        # Extractive label
        label = _extractive_label(titles) if titles else cluster.label_extractive
        keywords = _extract_keywords(titles) if titles else cluster.top_keywords

        updated = cluster.model_copy(
            update={
                "label_extractive": label,
                "top_keywords": keywords,
                "label_refined": None,  # LLM refinement would go here
            }
        )
        result.append(updated)

    # TODO: LLM refinement when llm_config is provided
    if llm_config is not None:
        logger.info("LLM label refinement requested but not yet implemented; using extractive labels")

    logger.info("Labeled %d clusters", len(result))
    return result
