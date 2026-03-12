"""Tests for seed user validation stage."""

import pytest

from fieldscope.models import Paper, Provenance, SeedCandidate
from fieldscope.stages.seed_validation import validate_seeds


def _make_candidate(paper_id: str, score: float = 0.5) -> SeedCandidate:
    return SeedCandidate(
        paper_id=paper_id,
        score=score,
        methods={"citation_count": score},
        rationale="test",
    )


def _make_paper(doi: str) -> Paper:
    return Paper(
        title=f"Paper {doi}",
        doi=doi,
        source="openalex",
        provenance=Provenance(method="initial_retrieval", depth=0),
    )


class TestValidateSeeds:
    def test_auto_accept_validates_all(self):
        candidates = [_make_candidate("10.1/a"), _make_candidate("10.1/b")]
        papers = [_make_paper("10.1/a"), _make_paper("10.1/b")]
        result = validate_seeds(candidates, papers, auto_accept=True)
        assert len(result) == 2
        assert all(c.validated is True for c in result)

    def test_auto_accept_preserves_scores(self):
        candidates = [_make_candidate("10.1/a", score=0.9)]
        papers = [_make_paper("10.1/a")]
        result = validate_seeds(candidates, papers, auto_accept=True)
        assert result[0].score == 0.9
        assert result[0].paper_id == "10.1/a"

    def test_auto_accept_empty_list(self):
        result = validate_seeds([], [], auto_accept=True)
        assert result == []

    def test_returns_new_objects(self):
        candidates = [_make_candidate("10.1/a")]
        papers = [_make_paper("10.1/a")]
        result = validate_seeds(candidates, papers, auto_accept=True)
        # Should be new objects, not mutated originals
        assert candidates[0].validated is None
        assert result[0].validated is True
