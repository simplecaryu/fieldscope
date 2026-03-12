"""Tests for dataset filtering stage."""

import numpy as np
import pytest

from fieldscope.config import FilteringConfig
from fieldscope.models import Paper, Provenance
from fieldscope.stages.filtering import (
    filter_by_keyword_overlap,
    filter_by_metadata,
    filter_by_semantic_similarity,
    filter_dataset,
)


def _make_paper(doi: str, year: int | None = 2024, abstract: str | None = "Some abstract",
                embedding: list[float] | None = None, title: str = "Test"):
    return Paper(
        doi=doi,
        title=title,
        abstract=abstract,
        year=year,
        source="openalex",
        provenance=Provenance(method="initial_retrieval", depth=0),
        embedding=embedding,
    )


# ---------------------------------------------------------------------------
# Metadata filtering
# ---------------------------------------------------------------------------


class TestFilterByMetadata:
    def test_filters_papers_without_year(self):
        papers = [
            _make_paper("10.1/a", year=2024),
            _make_paper("10.1/b", year=None),
        ]
        config = FilteringConfig(require_year=True, require_abstract=False)
        result = filter_by_metadata(papers, config)
        assert len(result) == 1
        assert result[0].doi == "10.1/a"

    def test_filters_papers_without_abstract(self):
        papers = [
            _make_paper("10.1/a", abstract="Has abstract"),
            _make_paper("10.1/b", abstract=None),
        ]
        config = FilteringConfig(require_abstract=True, require_year=False)
        result = filter_by_metadata(papers, config)
        assert len(result) == 1

    def test_no_filters(self):
        papers = [
            _make_paper("10.1/a", year=None, abstract=None),
        ]
        config = FilteringConfig(require_year=False, require_abstract=False)
        result = filter_by_metadata(papers, config)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Keyword overlap filtering
# ---------------------------------------------------------------------------


class TestFilterByKeywordOverlap:
    def test_filters_by_keyword_overlap(self):
        papers = [
            _make_paper("10.1/a", title="Quantum magnetism in RuO2", abstract="spin dynamics study"),
            _make_paper("10.1/b", title="Cooking recipes", abstract="delicious food"),
            _make_paper("10.1/c", title="Spin wave theory", abstract=None),
        ]
        keywords = ["quantum", "magnetism", "spin"]
        config = FilteringConfig(keyword_min_overlap=1)
        result = filter_by_keyword_overlap(papers, keywords, config)
        ids = {p.doi for p in result}
        assert "10.1/a" in ids
        assert "10.1/c" in ids
        assert "10.1/b" not in ids

    def test_higher_overlap_threshold(self):
        papers = [
            _make_paper("10.1/a", title="Quantum magnetism", abstract="spin dynamics"),
            _make_paper("10.1/b", title="Quantum computing", abstract="qubits"),
        ]
        keywords = ["quantum", "magnetism", "spin"]
        config = FilteringConfig(keyword_min_overlap=2)
        result = filter_by_keyword_overlap(papers, keywords, config)
        assert len(result) == 1
        assert result[0].doi == "10.1/a"

    def test_empty_keywords_keeps_all(self):
        papers = [_make_paper("10.1/a")]
        result = filter_by_keyword_overlap(papers, [], FilteringConfig(keyword_min_overlap=1))
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Semantic similarity filtering
# ---------------------------------------------------------------------------


class TestFilterBySemanticSimilarity:
    def test_filters_distant_papers(self):
        # Seed centroid direction: [1, 0, 0]
        papers = [
            _make_paper("10.1/a", embedding=[0.9, 0.1, 0.0]),  # close
            _make_paper("10.1/b", embedding=[0.0, 0.0, 1.0]),  # far
            _make_paper("10.1/c", embedding=[0.7, 0.3, 0.0]),  # close-ish
        ]
        seed_centroid = np.array([1.0, 0.0, 0.0])
        config = FilteringConfig(semantic_threshold=0.5)
        result = filter_by_semantic_similarity(papers, seed_centroid, config)
        ids = {p.doi for p in result}
        assert "10.1/a" in ids
        assert "10.1/c" in ids
        assert "10.1/b" not in ids

    def test_skips_papers_without_embedding(self):
        papers = [
            _make_paper("10.1/a", embedding=[1.0, 0.0]),
            _make_paper("10.1/b", embedding=None),
        ]
        seed_centroid = np.array([1.0, 0.0])
        config = FilteringConfig(semantic_threshold=0.5)
        result = filter_by_semantic_similarity(papers, seed_centroid, config)
        # Paper without embedding should be kept (not filtered out)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Full filter_dataset
# ---------------------------------------------------------------------------


class TestFilterDataset:
    def test_applies_all_filters(self):
        papers = [
            _make_paper("10.1/a", year=2024, title="Quantum magnetism",
                        abstract="spin study", embedding=[0.9, 0.1]),
            _make_paper("10.1/b", year=None, title="No year paper",
                        abstract="magnetism", embedding=[0.8, 0.2]),
            _make_paper("10.1/c", year=2023, title="Unrelated topic",
                        abstract="cooking recipe", embedding=[0.0, 1.0]),
        ]
        keywords = ["quantum", "magnetism", "spin"]
        config = FilteringConfig(
            semantic_threshold=0.3,
            keyword_min_overlap=1,
            require_year=True,
            require_abstract=False,
        )
        seed_centroid = np.array([1.0, 0.0])
        result = filter_dataset(papers, keywords, config, seed_centroid=seed_centroid)
        # Paper b: no year -> filtered
        # Paper c: no keyword overlap -> filtered
        assert len(result) == 1
        assert result[0].doi == "10.1/a"

    def test_no_centroid_skips_semantic_filter(self):
        papers = [
            _make_paper("10.1/a", title="Quantum magnetism", embedding=[0.0, 1.0]),
        ]
        keywords = ["quantum", "magnetism"]
        config = FilteringConfig(require_year=False, require_abstract=False)
        result = filter_dataset(papers, keywords, config, seed_centroid=None)
        assert len(result) == 1
