"""Stage 9: Adaptive clustering of papers."""

from __future__ import annotations

import logging
from collections import Counter

import numpy as np

from fieldscope.config import ClusteringConfig
from fieldscope.models import Cluster, FieldMaturity, Paper

logger = logging.getLogger(__name__)


def _adapt_resolution(base_resolution: float, maturity: FieldMaturity) -> float:
    """Adjust Leiden resolution based on field maturity."""
    multipliers = {
        "emerging": 0.8,   # fewer, broader clusters
        "growing": 1.0,    # default
        "mature": 1.3,     # more, finer clusters
    }
    return base_resolution * multipliers.get(maturity.classification, 1.0)


def _build_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Build cosine similarity matrix from embeddings."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normed = embeddings / norms
    return normed @ normed.T


def _extract_top_keywords(papers: list[Paper], top_n: int = 5) -> list[str]:
    """Extract top keywords from paper titles."""
    # Simple extractive: most common title words (excluding stopwords)
    stopwords = {
        "the", "a", "an", "of", "in", "on", "for", "and", "or", "to", "is",
        "are", "was", "were", "by", "with", "from", "at", "as", "its", "it",
        "this", "that", "be", "has", "have", "had", "not", "but", "can",
        "will", "do", "does", "their", "we", "our", "using", "based", "via",
    }
    words: list[str] = []
    for p in papers:
        for w in p.title.lower().split():
            cleaned = w.strip(".,;:!?()[]{}\"'")
            if cleaned and len(cleaned) > 2 and cleaned not in stopwords:
                words.append(cleaned)
    counter = Counter(words)
    return [w for w, _ in counter.most_common(top_n)]


def _extractive_label(papers: list[Paper]) -> str:
    """Generate an extractive label from the most common title terms."""
    keywords = _extract_top_keywords(papers, top_n=3)
    if keywords:
        return " / ".join(keywords)
    return "unlabeled"


def cluster_papers(
    papers: list[Paper],
    maturity: FieldMaturity,
    config: ClusteringConfig,
) -> list[Cluster]:
    """Cluster papers using graph-based community detection on embedding similarity."""
    if not papers:
        return []

    # Separate papers with and without embeddings
    papers_with_emb = [p for p in papers if p.embedding is not None]
    papers_without_emb = [p for p in papers if p.embedding is None]

    if not papers_with_emb:
        # All papers lack embeddings — put them all in one cluster
        return [
            Cluster(
                cluster_id=0,
                member_paper_ids=[p.paper_id for p in papers],
                label_extractive=_extractive_label(papers),
                size=len(papers),
                top_keywords=_extract_top_keywords(papers),
            )
        ]

    embeddings = np.array([p.embedding for p in papers_with_emb])
    sim_matrix = _build_similarity_matrix(embeddings)
    threshold = config.embedding_similarity_threshold
    resolution = _adapt_resolution(config.leiden_resolution, maturity)

    # Try Leiden (via leidenalg + igraph) first, fall back to simple spectral
    labels = _cluster_with_leiden(sim_matrix, threshold, resolution)
    if labels is None:
        labels = _cluster_simple(sim_matrix, threshold)

    # Build clusters
    cluster_map: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        cluster_map.setdefault(label, []).append(idx)

    clusters = []
    for cluster_id, indices in sorted(cluster_map.items()):
        member_papers = [papers_with_emb[i] for i in indices]
        member_ids = [p.paper_id for p in member_papers]
        member_embeddings = embeddings[indices]
        centroid = member_embeddings.mean(axis=0).tolist()

        clusters.append(
            Cluster(
                cluster_id=cluster_id,
                member_paper_ids=member_ids,
                label_extractive=_extractive_label(member_papers),
                centroid=centroid,
                size=len(member_ids),
                top_keywords=_extract_top_keywords(member_papers),
            )
        )

    # Assign papers without embeddings to nearest cluster by title keyword overlap
    if papers_without_emb and clusters:
        for p in papers_without_emb:
            # Add to first cluster as fallback
            clusters[0].member_paper_ids.append(p.paper_id)
            clusters[0].size += 1

    logger.info("Clustering: %d papers → %d clusters", len(papers), len(clusters))
    return clusters


def _cluster_with_leiden(
    sim_matrix: np.ndarray,
    threshold: float,
    resolution: float,
) -> list[int] | None:
    """Try Leiden community detection. Returns None if leidenalg not available."""
    try:
        import igraph as ig
        import leidenalg
    except ImportError:
        logger.debug("leidenalg/igraph not available, falling back to simple clustering")
        return None

    n = sim_matrix.shape[0]
    # Build graph from similarity matrix
    edges = []
    weights = []
    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i, j] >= threshold:
                edges.append((i, j))
                weights.append(float(sim_matrix[i, j]))

    if not edges:
        return list(range(n))  # each paper is its own cluster

    g = ig.Graph(n=n, edges=edges, directed=False)
    g.es["weight"] = weights

    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights=weights,
        resolution_parameter=resolution,
    )
    return list(partition.membership)


def _cluster_simple(
    sim_matrix: np.ndarray,
    threshold: float,
) -> list[int]:
    """Simple connected-components clustering based on similarity threshold."""
    n = sim_matrix.shape[0]
    # Union-Find
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i, j] >= threshold:
                union(i, j)

    # Normalize labels
    label_map: dict[int, int] = {}
    labels = []
    for i in range(n):
        root = find(i)
        if root not in label_map:
            label_map[root] = len(label_map)
        labels.append(label_map[root])

    return labels
