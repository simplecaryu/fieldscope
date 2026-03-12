"""Tests for seed candidate detection stage."""

import numpy as np
import pytest

from fieldscope.config import SeedsConfig
from fieldscope.models import Author, Paper, Provenance, SeedCandidate
from fieldscope.stages.seeds import (
    detect_seed_candidates,
    score_by_centroid_proximity,
    score_by_citation_count,
    score_by_pagerank,
)


def _make_paper(doi: str, citation_count: int = 0, references: list[str] | None = None, **kw):
    defaults = dict(
        title=f"Paper {doi}",
        source="openalex",
        provenance=Provenance(method="initial_retrieval", depth=0),
        doi=doi,
        citation_count=citation_count,
        cited_by_count=citation_count,
        references=references or [],
    )
    defaults.update(kw)
    return Paper(**defaults)


@pytest.fixture
def sample_papers():
    """Create a small corpus with known citation structure."""
    # Paper A: highly cited, references B and C
    # Paper B: moderately cited, references C
    # Paper C: foundational, referenced by A and B
    # Paper D: isolated, low citation
    # Paper E: cites A
    return [
        _make_paper("10.1/a", citation_count=100, references=["10.1/b", "10.1/c"]),
        _make_paper("10.1/b", citation_count=50, references=["10.1/c"]),
        _make_paper("10.1/c", citation_count=200, references=[]),
        _make_paper("10.1/d", citation_count=5, references=[]),
        _make_paper("10.1/e", citation_count=30, references=["10.1/a"]),
    ]


# ---------------------------------------------------------------------------
# Citation count scoring
# ---------------------------------------------------------------------------


class TestScoreByCitationCount:
    def test_scores_normalized(self, sample_papers):
        scores = score_by_citation_count(sample_papers)
        assert len(scores) == 5
        # Paper C has the highest citation count
        assert scores["10.1/c"] == 1.0
        # All scores between 0 and 1
        assert all(0 <= v <= 1 for v in scores.values())

    def test_single_paper(self):
        papers = [_make_paper("10.1/x", citation_count=42)]
        scores = score_by_citation_count(papers)
        assert scores["10.1/x"] == 1.0

    def test_all_same_citations(self):
        papers = [_make_paper(f"10.1/{i}", citation_count=10) for i in range(3)]
        scores = score_by_citation_count(papers)
        assert all(v == 1.0 for v in scores.values())


# ---------------------------------------------------------------------------
# PageRank scoring
# ---------------------------------------------------------------------------


class TestScoreByPagerank:
    def test_scores_returned(self, sample_papers):
        scores = score_by_pagerank(sample_papers)
        assert len(scores) == 5
        assert all(0 <= v <= 1 for v in scores.values())

    def test_cited_papers_rank_higher(self, sample_papers):
        scores = score_by_pagerank(sample_papers)
        # C is referenced by both A and B, should rank high
        assert scores["10.1/c"] > scores["10.1/d"]

    def test_single_paper(self):
        papers = [_make_paper("10.1/x")]
        scores = score_by_pagerank(papers)
        assert len(scores) == 1


# ---------------------------------------------------------------------------
# detect_seed_candidates
# ---------------------------------------------------------------------------


class TestDetectSeedCandidates:
    def test_returns_correct_count(self, sample_papers):
        config = SeedsConfig(
            methods=["citation_count", "pagerank"],
            top_k=3,
        )
        candidates = detect_seed_candidates(sample_papers, config)
        assert len(candidates) == 3
        assert all(isinstance(c, SeedCandidate) for c in candidates)

    def test_sorted_by_score_descending(self, sample_papers):
        config = SeedsConfig(methods=["citation_count", "pagerank"], top_k=5)
        candidates = detect_seed_candidates(sample_papers, config)
        scores = [c.score for c in candidates]
        assert scores == sorted(scores, reverse=True)

    def test_methods_recorded(self, sample_papers):
        config = SeedsConfig(methods=["citation_count", "pagerank"], top_k=3)
        candidates = detect_seed_candidates(sample_papers, config)
        for c in candidates:
            assert "citation_count" in c.methods
            assert "pagerank" in c.methods

    def test_rationale_not_empty(self, sample_papers):
        config = SeedsConfig(methods=["citation_count"], top_k=2)
        candidates = detect_seed_candidates(sample_papers, config)
        for c in candidates:
            assert c.rationale
            assert len(c.rationale) > 0

    def test_validated_is_none(self, sample_papers):
        config = SeedsConfig(methods=["citation_count"], top_k=2)
        candidates = detect_seed_candidates(sample_papers, config)
        for c in candidates:
            assert c.validated is None

    def test_top_k_larger_than_papers(self, sample_papers):
        config = SeedsConfig(methods=["citation_count"], top_k=100)
        candidates = detect_seed_candidates(sample_papers, config)
        assert len(candidates) == len(sample_papers)

    def test_citation_count_only(self, sample_papers):
        config = SeedsConfig(methods=["citation_count"], top_k=2)
        candidates = detect_seed_candidates(sample_papers, config)
        # Top 2 by citation count should be C (200) and A (100)
        ids = [c.paper_id for c in candidates]
        assert "10.1/c" in ids
        assert "10.1/a" in ids

    def test_with_centroid_proximity(self):
        papers = [
            _make_paper("10.1/a", citation_count=10, embedding=[0.9, 0.1, 0.0]),
            _make_paper("10.1/b", citation_count=100, embedding=[0.0, 0.0, 1.0]),
            _make_paper("10.1/c", citation_count=50, embedding=[0.8, 0.2, 0.0]),
        ]
        config = SeedsConfig(methods=["citation_count", "centroid_proximity"], top_k=3)
        candidates = detect_seed_candidates(papers, config)
        assert len(candidates) == 3
        for c in candidates:
            assert "centroid_proximity" in c.methods


# ---------------------------------------------------------------------------
# Centroid proximity scoring
# ---------------------------------------------------------------------------


class TestScoreByCentroidProximity:
    def test_closer_to_centroid_scores_higher(self):
        papers = [
            _make_paper("10.1/a", embedding=[1.0, 0.0, 0.0]),
            _make_paper("10.1/b", embedding=[0.0, 1.0, 0.0]),
            _make_paper("10.1/c", embedding=[0.9, 0.1, 0.0]),
        ]
        # Centroid is roughly [0.63, 0.37, 0.0] — a and c are closer
        scores = score_by_centroid_proximity(papers)
        assert scores["10.1/a"] > scores["10.1/b"]
        assert scores["10.1/c"] > scores["10.1/b"]

    def test_skips_papers_without_embedding(self):
        papers = [
            _make_paper("10.1/a", embedding=[1.0, 0.0]),
            _make_paper("10.1/b", embedding=None),
        ]
        scores = score_by_centroid_proximity(papers)
        assert "10.1/a" in scores
        assert scores.get("10.1/b", 0.0) == 0.0

    def test_all_same_embedding(self):
        papers = [
            _make_paper(f"10.1/{i}", embedding=[0.5, 0.5]) for i in range(3)
        ]
        scores = score_by_centroid_proximity(papers)
        # All equally close to centroid
        assert all(v == 1.0 for v in scores.values())
