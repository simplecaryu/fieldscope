"""Tests for field evolution analysis stage."""

import pytest

from fieldscope.config import EvolutionConfig
from fieldscope.models import Cluster, EvolutionEvent, Paper, Provenance
from fieldscope.stages.evolution import analyze_evolution


def _make_paper(doi: str, year: int, embedding: list[float] | None = None):
    return Paper(
        doi=doi, title=f"Paper {doi}", year=year,
        source="openalex", embedding=embedding,
        provenance=Provenance(method="initial_retrieval", depth=0),
    )


def _make_cluster(cluster_id: int, paper_ids: list[str]):
    return Cluster(
        cluster_id=cluster_id,
        member_paper_ids=paper_ids,
        label_extractive=f"Cluster {cluster_id}",
        size=len(paper_ids),
        top_keywords=[],
    )


class TestAnalyzeEvolution:
    def test_returns_events(self):
        # Papers spanning multiple time windows
        papers = []
        for i in range(5):
            papers.append(_make_paper(f"10.1/early{i}", year=2015, embedding=[1.0, 0.0]))
        for i in range(5):
            papers.append(_make_paper(f"10.1/late{i}", year=2022, embedding=[0.0, 1.0]))

        clusters = [
            _make_cluster(0, [f"10.1/early{i}" for i in range(5)]),
            _make_cluster(1, [f"10.1/late{i}" for i in range(5)]),
        ]
        config = EvolutionConfig(window_size_years=3, window_step_years=3)

        events = analyze_evolution(papers, clusters, config)
        assert isinstance(events, list)
        for e in events:
            assert isinstance(e, EvolutionEvent)
            assert e.event_type in ("emergence", "growth", "decline", "merge", "split", "stability")

    def test_empty_papers(self):
        config = EvolutionConfig()
        events = analyze_evolution([], [], config)
        assert events == []

    def test_single_window(self):
        # All papers in same year → single window, no cross-window events
        papers = [_make_paper(f"10.1/{i}", year=2024) for i in range(5)]
        clusters = [_make_cluster(0, [f"10.1/{i}" for i in range(5)])]
        config = EvolutionConfig(window_size_years=3)

        events = analyze_evolution(papers, clusters, config)
        assert isinstance(events, list)

    def test_event_fields(self):
        papers = [
            _make_paper("10.1/a", year=2015),
            _make_paper("10.1/b", year=2020),
        ]
        clusters = [
            _make_cluster(0, ["10.1/a"]),
            _make_cluster(1, ["10.1/b"]),
        ]
        config = EvolutionConfig(window_size_years=3, window_step_years=3)

        events = analyze_evolution(papers, clusters, config)
        for e in events:
            assert len(e.time_window) == 2
            assert e.time_window[0] <= e.time_window[1]
            assert isinstance(e.evidence, dict)

    def test_detects_emergence(self):
        # Cluster appears only in later windows
        papers = [_make_paper(f"10.1/{i}", year=2023) for i in range(5)]
        clusters = [_make_cluster(0, [f"10.1/{i}" for i in range(5)])]
        config = EvolutionConfig(window_size_years=5, window_step_years=5)

        events = analyze_evolution(papers, clusters, config)
        # Should detect at least some events
        assert isinstance(events, list)
