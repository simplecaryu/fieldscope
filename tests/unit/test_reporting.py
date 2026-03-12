"""Tests for report generation stage."""

from pathlib import Path

import pytest

from fieldscope.config import ReportingConfig
from fieldscope.models import (
    Cluster,
    EvolutionEvent,
    FieldMaturity,
    Paper,
    Provenance,
    SeedCandidate,
)
from fieldscope.stages.reporting import generate_reports


def _make_paper(doi: str, year: int = 2024, title: str = "Test Paper",
                citation_count: int = 10):
    return Paper(
        doi=doi, title=title, year=year, citation_count=citation_count,
        cited_by_count=citation_count, source="openalex",
        provenance=Provenance(method="initial_retrieval", depth=0),
    )


def _make_seed(paper_id: str, score: float = 0.9):
    return SeedCandidate(
        paper_id=paper_id, score=score,
        methods={"citation_count": score},
        rationale="High citation count", validated=True,
    )


def _make_cluster(cluster_id: int, paper_ids: list[str]):
    return Cluster(
        cluster_id=cluster_id, member_paper_ids=paper_ids,
        label_extractive=f"Topic {cluster_id}", size=len(paper_ids),
        top_keywords=["keyword1", "keyword2"],
    )


def _make_maturity():
    return FieldMaturity(
        classification="growing",
        metrics={"growth_rate": 0.5, "citation_density": 0.3,
                 "keyword_burst": 0.4, "median_age": 8.0},
        user_override=False,
    )


def _make_event():
    return EvolutionEvent(
        event_type="emergence",
        time_window=(2020, 2023),
        source_cluster_ids=[],
        target_cluster_ids=[0],
        evidence={"presence_before": 0.0, "presence_after": 0.8},
    )


@pytest.fixture
def sample_data():
    papers = [_make_paper(f"10.1/{i}", title=f"Paper about topic {i}") for i in range(5)]
    seeds = [_make_seed("10.1/0"), _make_seed("10.1/1", score=0.8)]
    clusters = [_make_cluster(0, ["10.1/0", "10.1/1", "10.1/2"]),
                _make_cluster(1, ["10.1/3", "10.1/4"])]
    maturity = _make_maturity()
    events = [_make_event()]
    return {
        "query": "quantum magnetism",
        "keywords": ["quantum magnetism", "spin dynamics", "frustrated magnets"],
        "papers": papers,
        "seeds": seeds,
        "maturity": maturity,
        "clusters": clusters,
        "events": events,
    }


class TestGenerateReports:
    def test_generates_markdown(self, tmp_path, sample_data):
        config = ReportingConfig(formats=["markdown"])
        paths = generate_reports(
            **sample_data, config=config, llm_config=None, output_dir=tmp_path,
        )
        assert len(paths) == 1
        assert paths[0].suffix == ".md"
        assert paths[0].exists()
        content = paths[0].read_text()
        assert "quantum magnetism" in content.lower()

    def test_generates_json(self, tmp_path, sample_data):
        config = ReportingConfig(formats=["json"])
        paths = generate_reports(
            **sample_data, config=config, llm_config=None, output_dir=tmp_path,
        )
        assert len(paths) == 1
        assert paths[0].suffix == ".json"
        assert paths[0].exists()
        import json
        data = json.loads(paths[0].read_text())
        assert "query" in data
        assert "papers" in data

    def test_generates_multiple_formats(self, tmp_path, sample_data):
        config = ReportingConfig(formats=["markdown", "json"])
        paths = generate_reports(
            **sample_data, config=config, llm_config=None, output_dir=tmp_path,
        )
        assert len(paths) == 2
        suffixes = {p.suffix for p in paths}
        assert ".md" in suffixes
        assert ".json" in suffixes

    def test_markdown_contains_sections(self, tmp_path, sample_data):
        config = ReportingConfig(formats=["markdown"])
        paths = generate_reports(
            **sample_data, config=config, llm_config=None, output_dir=tmp_path,
        )
        content = paths[0].read_text()
        # Should contain key report sections
        assert "# " in content  # has headers
        assert "seed" in content.lower()
        assert "cluster" in content.lower()
        assert "maturity" in content.lower()

    def test_json_structure(self, tmp_path, sample_data):
        config = ReportingConfig(formats=["json"])
        paths = generate_reports(
            **sample_data, config=config, llm_config=None, output_dir=tmp_path,
        )
        import json
        data = json.loads(paths[0].read_text())
        assert data["query"] == "quantum magnetism"
        assert len(data["papers"]) == 5
        assert len(data["seeds"]) == 2
        assert len(data["clusters"]) == 2
        assert data["maturity"]["classification"] == "growing"

    def test_empty_data(self, tmp_path):
        config = ReportingConfig(formats=["markdown", "json"])
        paths = generate_reports(
            query="empty test",
            keywords=[],
            papers=[],
            seeds=[],
            maturity=_make_maturity(),
            clusters=[],
            events=[],
            config=config,
            llm_config=None,
            output_dir=tmp_path,
        )
        assert len(paths) == 2
        for p in paths:
            assert p.exists()

    def test_output_dir_created(self, tmp_path, sample_data):
        out = tmp_path / "reports" / "sub"
        config = ReportingConfig(formats=["json"])
        paths = generate_reports(
            **sample_data, config=config, llm_config=None, output_dir=out,
        )
        assert out.exists()
        assert len(paths) == 1
