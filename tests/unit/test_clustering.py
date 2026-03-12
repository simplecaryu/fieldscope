"""Tests for adaptive clustering stage."""

import pytest

from fieldscope.config import ClusteringConfig
from fieldscope.models import Cluster, FieldMaturity, Paper, Provenance
from fieldscope.stages.clustering import cluster_papers


def _make_paper(doi: str, year: int = 2024, embedding: list[float] | None = None,
                references: list[str] | None = None, citation_count: int = 10):
    return Paper(
        doi=doi, title=f"Paper {doi}", year=year, citation_count=citation_count,
        cited_by_count=citation_count, source="openalex", embedding=embedding,
        references=references or [],
        provenance=Provenance(method="initial_retrieval", depth=0),
    )


def _make_maturity(classification: str = "growing"):
    return FieldMaturity(
        classification=classification,
        metrics={"growth_rate": 0.5, "citation_density": 0.3, "keyword_burst": 0.2, "median_age": 5.0},
        user_override=False,
    )


class TestClusterPapers:
    def test_returns_clusters(self):
        # Create papers with clear 2-cluster structure
        papers = []
        for i in range(10):
            papers.append(_make_paper(f"10.1/a{i}", embedding=[1.0, 0.0, float(i)*0.01]))
        for i in range(10):
            papers.append(_make_paper(f"10.1/b{i}", embedding=[0.0, 1.0, float(i)*0.01]))

        config = ClusteringConfig(leiden_resolution=1.0)
        maturity = _make_maturity("growing")
        clusters = cluster_papers(papers, maturity, config)

        assert len(clusters) >= 1
        assert all(isinstance(c, Cluster) for c in clusters)
        # All papers should be assigned
        all_ids = set()
        for c in clusters:
            all_ids.update(c.member_paper_ids)
        paper_ids = {p.paper_id for p in papers if p.embedding is not None}
        assert paper_ids == all_ids

    def test_cluster_fields_populated(self):
        papers = [_make_paper(f"10.1/{i}", embedding=[float(i), 0.0]) for i in range(5)]
        config = ClusteringConfig()
        maturity = _make_maturity()
        clusters = cluster_papers(papers, maturity, config)
        for c in clusters:
            assert c.cluster_id >= 0
            assert c.size == len(c.member_paper_ids)
            assert c.label_extractive  # should have an extractive label
            assert len(c.top_keywords) >= 0

    def test_empty_papers(self):
        config = ClusteringConfig()
        maturity = _make_maturity()
        clusters = cluster_papers([], maturity, config)
        assert clusters == []

    def test_papers_without_embeddings(self):
        papers = [_make_paper(f"10.1/{i}", embedding=None) for i in range(5)]
        config = ClusteringConfig()
        maturity = _make_maturity()
        clusters = cluster_papers(papers, maturity, config)
        # Should still work, placing all in one cluster or handling gracefully
        assert isinstance(clusters, list)

    def test_adapts_to_maturity(self):
        papers = [_make_paper(f"10.1/{i}", embedding=[float(i % 3), float(i // 3)]) for i in range(9)]
        config = ClusteringConfig()
        for classification in ["emerging", "growing", "mature"]:
            maturity = _make_maturity(classification)
            clusters = cluster_papers(papers, maturity, config)
            assert isinstance(clusters, list)
