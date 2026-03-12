"""Stage 3: Seed candidate detection."""

from __future__ import annotations

import logging

import networkx as nx

from fieldscope.config import SeedsConfig
from fieldscope.models import Paper, SeedCandidate

logger = logging.getLogger(__name__)


def score_by_citation_count(papers: list[Paper]) -> dict[str, float]:
    """Score papers by normalized citation count."""
    if not papers:
        return {}
    max_count = max(p.citation_count for p in papers)
    if max_count == 0:
        return {p.paper_id: 1.0 for p in papers}
    return {p.paper_id: p.citation_count / max_count for p in papers}


def score_by_pagerank(papers: list[Paper]) -> dict[str, float]:
    """Score papers by PageRank on the citation graph."""
    G = nx.DiGraph()
    paper_ids = {p.paper_id for p in papers}

    for p in papers:
        G.add_node(p.paper_id)

    for p in papers:
        for ref in p.references:
            ref_lower = ref.lower() if ref else ref
            if ref_lower in paper_ids:
                G.add_edge(p.paper_id, ref_lower)
            elif ref in paper_ids:
                G.add_edge(p.paper_id, ref)

    pr = nx.pagerank(G)
    max_pr = max(pr.values()) if pr else 1.0
    if max_pr == 0:
        max_pr = 1.0
    return {pid: pr.get(pid, 0.0) / max_pr for pid in paper_ids}


SCORING_METHODS = {
    "citation_count": score_by_citation_count,
    "pagerank": score_by_pagerank,
}


def detect_seed_candidates(
    papers: list[Paper],
    config: SeedsConfig,
) -> list[SeedCandidate]:
    """Detect seed candidate papers using configured scoring methods."""
    if not papers:
        return []

    # Compute scores for each method
    method_scores: dict[str, dict[str, float]] = {}
    for method in config.methods:
        if method in SCORING_METHODS:
            method_scores[method] = SCORING_METHODS[method](papers)
        else:
            logger.warning("Unknown scoring method '%s', skipping", method)

    if not method_scores:
        return []

    # Compute composite scores (average across methods)
    paper_ids = {p.paper_id for p in papers}
    composite: dict[str, float] = {}
    per_method: dict[str, dict[str, float]] = {}

    for pid in paper_ids:
        scores_for_pid = {}
        for method, scores in method_scores.items():
            scores_for_pid[method] = scores.get(pid, 0.0)
        per_method[pid] = scores_for_pid
        composite[pid] = sum(scores_for_pid.values()) / len(scores_for_pid)

    # Sort by composite score descending
    ranked = sorted(composite.items(), key=lambda x: x[1], reverse=True)
    top_k = min(config.top_k, len(ranked))

    # Build paper_id -> Paper lookup
    paper_map = {p.paper_id: p for p in papers}

    candidates = []
    for pid, score in ranked[:top_k]:
        paper = paper_map[pid]
        methods_detail = per_method[pid]
        top_method = max(methods_detail, key=methods_detail.get)
        rationale = (
            f"Composite score {score:.3f} "
            f"(top method: {top_method}={methods_detail[top_method]:.3f}, "
            f"citations={paper.citation_count})"
        )
        candidates.append(
            SeedCandidate(
                paper_id=pid,
                score=score,
                methods=methods_detail,
                rationale=rationale,
            )
        )

    logger.info("Detected %d seed candidates from %d papers", len(candidates), len(papers))
    return candidates
