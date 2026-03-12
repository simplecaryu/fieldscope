"""Tests for initial retrieval stage."""

import json
from pathlib import Path

import pytest
from pytest_httpx import HTTPXMock

from fieldscope.config import RetrievalConfig
from fieldscope.models import Paper
from fieldscope.stages.retrieval import (
    normalize_openalex_paper,
    reconstruct_abstract,
    retrieve_papers,
)


FIXTURES = Path(__file__).parent.parent / "fixtures"


# ---------------------------------------------------------------------------
# Abstract reconstruction from inverted index
# ---------------------------------------------------------------------------


class TestReconstructAbstract:
    def test_basic(self):
        inv_idx = {"We": [0], "present": [1], "evidence": [2]}
        assert reconstruct_abstract(inv_idx) == "We present evidence"

    def test_none_returns_none(self):
        assert reconstruct_abstract(None) is None

    def test_empty_returns_none(self):
        assert reconstruct_abstract({}) is None

    def test_longer_text(self):
        inv_idx = {"Hello": [0], "world": [1], "this": [2], "is": [3], "a": [4], "test": [5]}
        assert reconstruct_abstract(inv_idx) == "Hello world this is a test"


# ---------------------------------------------------------------------------
# OpenAlex paper normalization
# ---------------------------------------------------------------------------


class TestNormalizeOpenalexPaper:
    @pytest.fixture
    def sample_data(self):
        with open(FIXTURES / "openalex_search.json") as f:
            return json.load(f)

    def test_normalize_paper_with_doi(self, sample_data):
        raw = sample_data["results"][0]
        paper = normalize_openalex_paper(raw, query="altermagnetism")
        assert isinstance(paper, Paper)
        assert paper.doi == "10.1234/paper1"
        assert paper.openalex_id == "W1111111111"
        assert paper.title == "Altermagnetism in RuO2"
        assert paper.year == 2022
        assert paper.cited_by_count == 150
        assert paper.venue == "Physical Review Letters"
        assert paper.source == "openalex"
        assert len(paper.authors) == 2
        assert paper.authors[0].name == "L. Šmejkal"
        assert paper.abstract is not None
        assert "altermagnetism" in paper.abstract
        assert paper.provenance.method == "initial_retrieval"
        assert paper.provenance.query == "altermagnetism"

    def test_normalize_paper_without_doi(self, sample_data):
        raw = sample_data["results"][2]
        paper = normalize_openalex_paper(raw, query="test")
        assert paper.doi is None
        assert paper.openalex_id == "W3333333333"
        assert paper.paper_id == "W3333333333"

    def test_normalize_paper_null_abstract(self, sample_data):
        raw = sample_data["results"][1]
        paper = normalize_openalex_paper(raw, query="test")
        assert paper.abstract is None

    def test_references_normalized(self, sample_data):
        raw = sample_data["results"][0]
        paper = normalize_openalex_paper(raw, query="test")
        assert paper.references == ["W2222222222", "W3333333333"]

    def test_paper_no_venue(self, sample_data):
        raw = sample_data["results"][2]
        paper = normalize_openalex_paper(raw, query="test")
        assert paper.venue is None


# ---------------------------------------------------------------------------
# retrieve_papers (mocked HTTP)
# ---------------------------------------------------------------------------


class TestRetrievePapers:
    @pytest.mark.asyncio
    async def test_retrieve_single_keyword(self, httpx_mock: HTTPXMock):
        with open(FIXTURES / "openalex_search.json") as f:
            fixture = json.load(f)

        httpx_mock.add_response(json=fixture)

        config = RetrievalConfig(
            primary_source="openalex",
            max_results_per_query=100,
        )
        papers = await retrieve_papers(
            keywords=["altermagnetism"],
            config=config,
        )
        assert len(papers) == 3
        assert all(isinstance(p, Paper) for p in papers)

    @pytest.mark.asyncio
    async def test_retrieve_deduplicates_across_keywords(self, httpx_mock: HTTPXMock):
        with open(FIXTURES / "openalex_search.json") as f:
            fixture = json.load(f)

        # Same results for two different keywords
        httpx_mock.add_response(json=fixture)
        httpx_mock.add_response(json=fixture)

        config = RetrievalConfig(primary_source="openalex", max_results_per_query=100)
        papers = await retrieve_papers(
            keywords=["altermagnetism", "altermagnetic"],
            config=config,
        )
        # Should deduplicate
        assert len(papers) == 3

    @pytest.mark.asyncio
    async def test_retrieve_skips_papers_without_id(self, httpx_mock: HTTPXMock):
        fixture = {
            "meta": {"count": 1, "page": 1, "per_page": 25},
            "results": [
                {
                    "id": None,
                    "doi": None,
                    "title": "Unknown paper",
                    "display_name": "Unknown paper",
                    "publication_year": 2024,
                    "cited_by_count": 0,
                    "primary_location": {"source": None},
                    "authorships": [],
                    "referenced_works": [],
                    "abstract_inverted_index": None,
                }
            ],
        }
        httpx_mock.add_response(json=fixture)

        config = RetrievalConfig(primary_source="openalex", max_results_per_query=100)
        papers = await retrieve_papers(keywords=["test"], config=config)
        assert len(papers) == 0

    @pytest.mark.asyncio
    async def test_retrieve_pagination(self, httpx_mock: HTTPXMock):
        page1 = {
            "meta": {"count": 4, "page": 1, "per_page": 2},
            "results": [
                {
                    "id": "https://openalex.org/W1",
                    "doi": "https://doi.org/10.1/a",
                    "title": "Paper A",
                    "display_name": "Paper A",
                    "publication_year": 2024,
                    "cited_by_count": 10,
                    "primary_location": {"source": None},
                    "authorships": [],
                    "referenced_works": [],
                    "abstract_inverted_index": None,
                },
                {
                    "id": "https://openalex.org/W2",
                    "doi": "https://doi.org/10.1/b",
                    "title": "Paper B",
                    "display_name": "Paper B",
                    "publication_year": 2024,
                    "cited_by_count": 5,
                    "primary_location": {"source": None},
                    "authorships": [],
                    "referenced_works": [],
                    "abstract_inverted_index": None,
                },
            ],
        }
        page2 = {
            "meta": {"count": 4, "page": 2, "per_page": 2},
            "results": [
                {
                    "id": "https://openalex.org/W3",
                    "doi": "https://doi.org/10.1/c",
                    "title": "Paper C",
                    "display_name": "Paper C",
                    "publication_year": 2023,
                    "cited_by_count": 20,
                    "primary_location": {"source": None},
                    "authorships": [],
                    "referenced_works": [],
                    "abstract_inverted_index": None,
                },
                {
                    "id": "https://openalex.org/W4",
                    "doi": "https://doi.org/10.1/d",
                    "title": "Paper D",
                    "display_name": "Paper D",
                    "publication_year": 2023,
                    "cited_by_count": 1,
                    "primary_location": {"source": None},
                    "authorships": [],
                    "referenced_works": [],
                    "abstract_inverted_index": None,
                },
            ],
        }

        httpx_mock.add_response(json=page1)
        httpx_mock.add_response(json=page2)

        config = RetrievalConfig(primary_source="openalex", max_results_per_query=100)
        papers = await retrieve_papers(keywords=["test"], config=config, per_page=2)
        assert len(papers) == 4
