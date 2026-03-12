"""Tests for field maturity assessment stage."""

import pytest

from fieldscope.models import FieldMaturity, Paper, Provenance
from fieldscope.stages.maturity import assess_maturity, confirm_maturity


def _make_paper(doi: str, year: int, citation_count: int = 10):
    return Paper(
        doi=doi, title=f"Paper {doi}", year=year, citation_count=citation_count,
        cited_by_count=citation_count, source="openalex",
        provenance=Provenance(method="initial_retrieval", depth=0),
    )


class TestAssessMaturity:
    def test_emerging_field(self):
        # Recent papers, low citations, rapid growth
        papers = [_make_paper(f"10.1/{i}", year=2023 + (i % 2), citation_count=5) for i in range(20)]
        result = assess_maturity(papers)
        assert isinstance(result, FieldMaturity)
        assert result.classification in ("emerging", "growing", "mature")
        assert "growth_rate" in result.metrics
        assert "citation_density" in result.metrics
        assert "median_age" in result.metrics
        assert result.user_override is False

    def test_mature_field(self):
        # Old papers with high citations spread across many years
        papers = []
        for y in range(1990, 2020):
            for i in range(5):
                papers.append(_make_paper(f"10.1/{y}_{i}", year=y, citation_count=200))
        result = assess_maturity(papers)
        assert result.classification == "mature"

    def test_empty_papers(self):
        result = assess_maturity([])
        assert result.classification == "emerging"

    def test_metrics_keys(self):
        papers = [_make_paper(f"10.1/{i}", year=2020, citation_count=10) for i in range(10)]
        result = assess_maturity(papers)
        expected_keys = {"growth_rate", "citation_density", "keyword_burst", "median_age"}
        assert expected_keys == set(result.metrics.keys())


class TestConfirmMaturity:
    def test_auto_accept(self):
        fm = FieldMaturity(
            classification="growing",
            metrics={"growth_rate": 0.5, "citation_density": 0.3, "keyword_burst": 0.2, "median_age": 5.0},
            user_override=False,
        )
        result = confirm_maturity(fm, auto_accept=True)
        assert result.classification == "growing"
        assert result.user_override is False

    def test_auto_accept_returns_copy(self):
        fm = FieldMaturity(classification="emerging", metrics={}, user_override=False)
        result = confirm_maturity(fm, auto_accept=True)
        assert result is not fm
