"""Tests for fieldscope data models."""

import pytest
from pydantic import ValidationError

from fieldscope.models import (
    Author,
    Cluster,
    EvolutionEvent,
    FieldMaturity,
    Paper,
    PipelineState,
    Provenance,
    SeedCandidate,
)


# ---------------------------------------------------------------------------
# Author
# ---------------------------------------------------------------------------


class TestAuthor:
    def test_create_with_name_only(self):
        a = Author(name="Alice Smith")
        assert a.name == "Alice Smith"
        assert a.orcid is None

    def test_create_with_orcid(self):
        a = Author(name="Bob", orcid="0000-0001-2345-6789")
        assert a.orcid == "0000-0001-2345-6789"

    def test_name_required(self):
        with pytest.raises(ValidationError):
            Author()


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


class TestProvenance:
    def test_initial_retrieval(self):
        p = Provenance(method="initial_retrieval", depth=0, query="spintronics")
        assert p.method == "initial_retrieval"
        assert p.depth == 0
        assert p.seed_paper_id is None
        assert p.query == "spintronics"

    def test_citation_expansion(self):
        p = Provenance(method="citation_expansion", depth=1, seed_paper_id="10.1234/abc")
        assert p.depth == 1
        assert p.seed_paper_id == "10.1234/abc"

    def test_method_required(self):
        with pytest.raises(ValidationError):
            Provenance(depth=0)


# ---------------------------------------------------------------------------
# Paper
# ---------------------------------------------------------------------------


def _make_paper(**overrides):
    """Helper to create a Paper with sensible defaults."""
    defaults = dict(
        title="Test Paper",
        source="openalex",
        provenance=Provenance(method="initial_retrieval", depth=0),
        doi="10.1234/test",
    )
    defaults.update(overrides)
    return Paper(**defaults)


class TestPaper:
    def test_minimal_paper_with_doi(self):
        p = _make_paper()
        assert p.title == "Test Paper"
        assert p.doi == "10.1234/test"
        assert p.authors == []
        assert p.citation_count == 0
        assert p.references == []
        assert p.embedding is None

    def test_paper_id_from_doi(self):
        p = _make_paper(doi="10.1234/ABC")
        assert p.paper_id == "10.1234/abc"

    def test_paper_id_from_openalex_when_no_doi(self):
        p = _make_paper(doi=None, openalex_id="W2123456789")
        assert p.paper_id == "W2123456789"

    def test_paper_id_prefers_doi_over_openalex(self):
        p = _make_paper(doi="10.1234/abc", openalex_id="W999")
        assert p.paper_id == "10.1234/abc"

    def test_paper_id_raises_without_identifiers(self):
        p = _make_paper(doi=None, openalex_id=None)
        with pytest.raises(ValueError, match="at least one identifier"):
            _ = p.paper_id

    def test_title_required(self):
        with pytest.raises(ValidationError):
            Paper(source="openalex", provenance=Provenance(method="initial_retrieval", depth=0))

    def test_source_required(self):
        with pytest.raises(ValidationError):
            Paper(title="X", provenance=Provenance(method="initial_retrieval", depth=0))

    def test_full_paper(self):
        p = _make_paper(
            openalex_id="W111",
            abstract="Some abstract",
            authors=[Author(name="Alice")],
            year=2024,
            venue="Nature",
            citation_count=42,
            cited_by_count=100,
            references=["10.5678/ref1"],
            embedding=[0.1, 0.2, 0.3],
        )
        assert p.year == 2024
        assert p.venue == "Nature"
        assert len(p.authors) == 1
        assert p.embedding == [0.1, 0.2, 0.3]

    def test_doi_normalized_to_lowercase(self):
        p = _make_paper(doi="10.1234/ABC-DEF")
        assert p.doi == "10.1234/abc-def"


# ---------------------------------------------------------------------------
# SeedCandidate
# ---------------------------------------------------------------------------


class TestSeedCandidate:
    def test_create(self):
        sc = SeedCandidate(
            paper_id="10.1234/test",
            score=0.85,
            methods={"pagerank": 0.9, "citation_count": 0.8},
            rationale="High PageRank and citation count",
        )
        assert sc.score == 0.85
        assert sc.validated is None

    def test_validate(self):
        sc = SeedCandidate(
            paper_id="10.1234/test",
            score=0.5,
            methods={},
            rationale="test",
            validated=True,
        )
        assert sc.validated is True


# ---------------------------------------------------------------------------
# Cluster
# ---------------------------------------------------------------------------


class TestCluster:
    def test_create(self):
        c = Cluster(
            cluster_id=0,
            member_paper_ids=["10.1234/a", "10.1234/b"],
            label_extractive="quantum magnetism",
            size=2,
            top_keywords=["quantum", "magnetism"],
        )
        assert c.cluster_id == 0
        assert c.label_refined is None
        assert c.centroid is None
        assert len(c.member_paper_ids) == 2

    def test_required_fields(self):
        with pytest.raises(ValidationError):
            Cluster(cluster_id=0)


# ---------------------------------------------------------------------------
# EvolutionEvent
# ---------------------------------------------------------------------------


class TestEvolutionEvent:
    def test_create(self):
        e = EvolutionEvent(
            event_type="growth",
            time_window=(2020, 2023),
            source_cluster_ids=[0],
            target_cluster_ids=[0],
            evidence={"overlap": 0.8},
        )
        assert e.event_type == "growth"
        assert e.time_window == (2020, 2023)

    def test_event_type_values(self):
        for et in ["birth", "growth", "split", "merge", "decline"]:
            e = EvolutionEvent(
                event_type=et,
                time_window=(2020, 2022),
                source_cluster_ids=[],
                target_cluster_ids=[],
                evidence={},
            )
            assert e.event_type == et


# ---------------------------------------------------------------------------
# FieldMaturity
# ---------------------------------------------------------------------------


class TestFieldMaturity:
    def test_create(self):
        fm = FieldMaturity(
            classification="emerging",
            metrics={"growth_rate": 0.3, "citation_density": 0.1, "keyword_burst": 0.8, "median_age": 2.5},
            user_override=False,
        )
        assert fm.classification == "emerging"
        assert fm.user_override is False

    def test_classification_values(self):
        for c in ["emerging", "growing", "mature"]:
            fm = FieldMaturity(classification=c, metrics={}, user_override=False)
            assert fm.classification == c


# ---------------------------------------------------------------------------
# PipelineState
# ---------------------------------------------------------------------------


class TestPipelineState:
    def test_create(self):
        ps = PipelineState(
            run_id="20260312_143022_ai_quantum",
            query="AI in quantum magnetism",
            config_snapshot={"retrieval": {"primary_source": "openalex"}},
            completed_stages=["keyword_expansion"],
            stage_outputs={"keyword_expansion": "01_keywords/keywords.json"},
        )
        assert ps.run_id == "20260312_143022_ai_quantum"
        assert ps.current_stage is None
        assert len(ps.completed_stages) == 1

    def test_initial_empty_state(self):
        ps = PipelineState(
            run_id="test_run",
            query="test",
            config_snapshot={},
            completed_stages=[],
            stage_outputs={},
        )
        assert ps.completed_stages == []
        assert ps.current_stage is None
