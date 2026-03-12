"""Tests for topic labeling stage."""

import pytest

from fieldscope.models import Cluster, Paper, Provenance
from fieldscope.stages.labeling import label_clusters


def _make_paper(doi: str, title: str = "Test Paper", abstract: str | None = None):
    return Paper(
        doi=doi, title=title, abstract=abstract, year=2024,
        source="openalex",
        provenance=Provenance(method="initial_retrieval", depth=0),
    )


def _make_cluster(cluster_id: int, paper_ids: list[str], label: str = "placeholder"):
    return Cluster(
        cluster_id=cluster_id,
        member_paper_ids=paper_ids,
        label_extractive=label,
        size=len(paper_ids),
        top_keywords=["keyword1", "keyword2"],
    )


class TestLabelClusters:
    def test_generates_extractive_labels(self):
        papers = [
            _make_paper("10.1/a", title="Quantum magnetism in oxide materials"),
            _make_paper("10.1/b", title="Spin dynamics in quantum magnets"),
            _make_paper("10.1/c", title="Machine learning for materials science"),
        ]
        clusters = [
            _make_cluster(0, ["10.1/a", "10.1/b"]),
            _make_cluster(1, ["10.1/c"]),
        ]
        result = label_clusters(clusters, papers, llm_config=None)
        assert len(result) == 2
        for c in result:
            assert c.label_extractive  # non-empty
            assert c.label_refined is None  # no LLM

    def test_empty_clusters(self):
        result = label_clusters([], [], llm_config=None)
        assert result == []

    def test_preserves_cluster_fields(self):
        papers = [_make_paper("10.1/a", title="Test paper about graphs")]
        clusters = [_make_cluster(0, ["10.1/a"])]
        result = label_clusters(clusters, papers, llm_config=None)
        assert result[0].cluster_id == 0
        assert result[0].size == 1
        assert result[0].member_paper_ids == ["10.1/a"]

    def test_updates_top_keywords(self):
        papers = [
            _make_paper("10.1/a", title="Topological insulator band structure"),
            _make_paper("10.1/b", title="Topological insulator surface states"),
        ]
        clusters = [_make_cluster(0, ["10.1/a", "10.1/b"])]
        result = label_clusters(clusters, papers, llm_config=None)
        assert len(result[0].top_keywords) > 0

    def test_handles_missing_papers(self):
        # Cluster references a paper_id not in the papers list
        papers = [_make_paper("10.1/a", title="Real paper")]
        clusters = [_make_cluster(0, ["10.1/a", "10.1/missing"])]
        result = label_clusters(clusters, papers, llm_config=None)
        assert len(result) == 1
        assert result[0].label_extractive  # should still work with available papers
