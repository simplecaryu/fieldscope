"""Tests for citation expansion stage."""

import json
from pathlib import Path

import pytest
from pytest_httpx import HTTPXMock

from fieldscope.config import CitationExpansionConfig
from fieldscope.models import Paper, Provenance, SeedCandidate
from fieldscope.stages.citation_expansion import expand_citations


FIXTURES = Path(__file__).parent.parent / "fixtures"


def _make_paper(doi: str, openalex_id: str | None = None, references: list[str] | None = None):
    return Paper(
        title=f"Paper {doi}",
        doi=doi,
        openalex_id=openalex_id,
        source="openalex",
        provenance=Provenance(method="initial_retrieval", depth=0),
        references=references or [],
    )


def _make_seed(paper_id: str):
    return SeedCandidate(
        paper_id=paper_id,
        score=0.9,
        methods={"citation_count": 0.9},
        rationale="test",
        validated=True,
    )


class TestExpandCitations:
    @pytest.mark.asyncio
    async def test_expand_adds_citing_papers(self, httpx_mock: HTTPXMock):
        with open(FIXTURES / "openalex_cited_by.json") as f:
            cited_by_fixture = json.load(f)

        httpx_mock.add_response(json=cited_by_fixture)

        papers = [_make_paper("10.1234/paper1", openalex_id="W1111111111")]
        seeds = [_make_seed("10.1234/paper1")]
        config = CitationExpansionConfig(
            max_depth=1,
            max_papers_per_seed=100,
            directions=["cited_by"],
        )

        result = await expand_citations(papers, seeds, config)
        # Original + 2 citing papers
        assert len(result) >= 3
        ids = {p.paper_id for p in result}
        assert "10.1234/paper1" in ids
        assert "10.1234/citing1" in ids
        assert "10.1234/citing2" in ids

    @pytest.mark.asyncio
    async def test_expand_tracks_provenance(self, httpx_mock: HTTPXMock):
        with open(FIXTURES / "openalex_cited_by.json") as f:
            cited_by_fixture = json.load(f)

        httpx_mock.add_response(json=cited_by_fixture)

        papers = [_make_paper("10.1234/paper1", openalex_id="W1111111111")]
        seeds = [_make_seed("10.1234/paper1")]
        config = CitationExpansionConfig(max_depth=1, directions=["cited_by"])

        result = await expand_citations(papers, seeds, config)
        new_papers = [p for p in result if p.doi != "10.1234/paper1"]
        for p in new_papers:
            assert p.provenance.method == "citation_expansion"
            assert p.provenance.depth == 1
            assert p.provenance.seed_paper_id == "10.1234/paper1"

    @pytest.mark.asyncio
    async def test_expand_deduplicates(self, httpx_mock: HTTPXMock):
        # Both seeds return the same citing paper
        fixture = {
            "meta": {"count": 1, "page": 1, "per_page": 25},
            "results": [{
                "id": "https://openalex.org/W9999",
                "doi": "https://doi.org/10.1234/shared",
                "title": "Shared citing paper",
                "display_name": "Shared citing paper",
                "publication_year": 2024,
                "cited_by_count": 5,
                "primary_location": {"source": None},
                "authorships": [],
                "referenced_works": [],
                "abstract_inverted_index": None,
            }],
        }
        empty = {"meta": {"count": 0}, "results": []}

        # seed1 cited_by, seed2 cited_by
        httpx_mock.add_response(json=fixture)
        httpx_mock.add_response(json=fixture)

        papers = [
            _make_paper("10.1/a", openalex_id="W1"),
            _make_paper("10.1/b", openalex_id="W2"),
        ]
        seeds = [_make_seed("10.1/a"), _make_seed("10.1/b")]
        config = CitationExpansionConfig(max_depth=1, directions=["cited_by"])

        result = await expand_citations(papers, seeds, config)
        ids = [p.paper_id for p in result]
        assert ids.count("10.1234/shared") == 1

    @pytest.mark.asyncio
    async def test_expand_respects_max_papers_per_seed(self, httpx_mock: HTTPXMock):
        # Return many papers but limit should cap it
        many_results = {
            "meta": {"count": 100, "page": 1, "per_page": 25},
            "results": [
                {
                    "id": f"https://openalex.org/W{i}",
                    "doi": f"https://doi.org/10.1/{i}",
                    "title": f"Paper {i}",
                    "display_name": f"Paper {i}",
                    "publication_year": 2024,
                    "cited_by_count": 1,
                    "primary_location": {"source": None},
                    "authorships": [],
                    "referenced_works": [],
                    "abstract_inverted_index": None,
                }
                for i in range(25)
            ],
        }
        httpx_mock.add_response(json=many_results)

        papers = [_make_paper("10.1/seed", openalex_id="W0")]
        seeds = [_make_seed("10.1/seed")]
        config = CitationExpansionConfig(
            max_depth=1,
            max_papers_per_seed=5,
            directions=["cited_by"],
        )

        result = await expand_citations(papers, seeds, config)
        # original + max 5 new
        new_papers = [p for p in result if p.provenance.method == "citation_expansion"]
        assert len(new_papers) <= 5

    @pytest.mark.asyncio
    async def test_expand_only_validated_seeds(self, httpx_mock: HTTPXMock):
        empty = {"meta": {"count": 0}, "results": []}
        # Only 1 API call for seed A (validated), seed B is rejected
        httpx_mock.add_response(json=empty)

        papers = [
            _make_paper("10.1/a", openalex_id="W1"),
            _make_paper("10.1/b", openalex_id="W2"),
        ]
        seeds = [
            _make_seed("10.1/a"),  # validated=True
            SeedCandidate(
                paper_id="10.1/b", score=0.5, methods={},
                rationale="test", validated=False,
            ),
        ]
        config = CitationExpansionConfig(max_depth=1, directions=["cited_by"])

        result = await expand_citations(papers, seeds, config)
        # Only seed A should be expanded, so only 1 API call pair
        assert len(result) == 2  # originals only, since API returned empty
