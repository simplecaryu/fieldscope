"""Stage 2: Initial literature retrieval from scholarly APIs."""

from __future__ import annotations

import asyncio
import logging
import os
import re

import httpx

from fieldscope.config import RetrievalConfig
from fieldscope.models import Author, Paper, Provenance

logger = logging.getLogger(__name__)

OPENALEX_BASE = "https://api.openalex.org/works"


def reconstruct_abstract(inverted_index: dict | None) -> str | None:
    """Reconstruct abstract text from OpenAlex inverted index format."""
    if not inverted_index:
        return None
    positions: list[tuple[int, str]] = []
    for word, indices in inverted_index.items():
        for idx in indices:
            positions.append((idx, word))
    positions.sort(key=lambda x: x[0])
    return " ".join(word for _, word in positions)


def _extract_openalex_id(raw_id: str | None) -> str | None:
    """Extract bare OpenAlex ID from URL like 'https://openalex.org/W123'."""
    if not raw_id:
        return None
    match = re.search(r"(W\d+)", raw_id)
    return match.group(1) if match else None


def _extract_doi(raw_doi: str | None) -> str | None:
    """Extract bare DOI from URL like 'https://doi.org/10.1234/abc'."""
    if not raw_doi:
        return None
    raw_doi = raw_doi.replace("https://doi.org/", "").replace("http://doi.org/", "")
    return raw_doi.lower().strip()


def normalize_openalex_paper(raw: dict, query: str) -> Paper | None:
    """Convert an OpenAlex API result into a Paper model.

    Returns None if the paper has no usable identifier.
    """
    openalex_id = _extract_openalex_id(raw.get("id"))
    doi = _extract_doi(raw.get("doi"))

    if not openalex_id and not doi:
        return None

    # Authors
    authors = []
    for authorship in raw.get("authorships", []):
        author_data = authorship.get("author", {})
        name = author_data.get("display_name")
        if name:
            orcid_raw = author_data.get("orcid")
            orcid = None
            if orcid_raw:
                # Extract ORCID from URL
                orcid = orcid_raw.replace("https://orcid.org/", "")
            authors.append(Author(name=name, orcid=orcid))

    # Venue
    venue = None
    loc = raw.get("primary_location") or {}
    source = loc.get("source") or {}
    venue = source.get("display_name")

    # References
    references = []
    for ref in raw.get("referenced_works", []):
        ref_id = _extract_openalex_id(ref)
        if ref_id:
            references.append(ref_id)

    # Abstract
    abstract = reconstruct_abstract(raw.get("abstract_inverted_index"))

    return Paper(
        doi=doi,
        openalex_id=openalex_id,
        title=raw.get("title") or raw.get("display_name", ""),
        abstract=abstract,
        authors=authors,
        year=raw.get("publication_year"),
        venue=venue,
        citation_count=raw.get("cited_by_count", 0),
        cited_by_count=raw.get("cited_by_count", 0),
        references=references,
        source="openalex",
        provenance=Provenance(method="initial_retrieval", depth=0, query=query),
    )


async def _fetch_openalex_page(
    client: httpx.AsyncClient,
    keyword: str,
    config: RetrievalConfig,
    page: int = 1,
    per_page: int = 25,
) -> dict:
    """Fetch a single page of results from OpenAlex."""
    params = {
        "search": keyword,
        "page": page,
        "per_page": per_page,
    }
    # Polite pool
    email = os.environ.get("OPENALEX_EMAIL")
    if email:
        params["mailto"] = email

    headers = {"Accept": "application/json"}

    for attempt in range(config.retry_max_attempts):
        try:
            resp = await client.get(OPENALEX_BASE, params=params, headers=headers, timeout=30.0)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code in {429, 500, 502, 503, 504}:
                wait = config.retry_backoff_base * (2**attempt)
                logger.warning(
                    "OpenAlex returned %d for '%s' (attempt %d/%d), retrying in %.1fs",
                    resp.status_code, keyword, attempt + 1, config.retry_max_attempts, wait,
                )
                await asyncio.sleep(wait)
                continue
            logger.error("OpenAlex returned %d for '%s': %s", resp.status_code, keyword, resp.text)
            return {"meta": {"count": 0}, "results": []}
        except httpx.HTTPError as e:
            logger.warning("HTTP error fetching '%s' (attempt %d/%d): %s", keyword, attempt + 1, config.retry_max_attempts, e)
            if attempt < config.retry_max_attempts - 1:
                await asyncio.sleep(config.retry_backoff_base * (2**attempt))

    return {"meta": {"count": 0}, "results": []}


async def retrieve_papers(
    keywords: list[str],
    config: RetrievalConfig,
    per_page: int = 25,
) -> list[Paper]:
    """Retrieve papers from OpenAlex for the given keywords.

    Handles pagination and deduplication across keywords.
    """
    seen_ids: set[str] = set()
    papers: list[Paper] = []

    async with httpx.AsyncClient() as client:
        for keyword in keywords:
            page = 1
            retrieved_for_keyword = 0

            while retrieved_for_keyword < config.max_results_per_query:
                data = await _fetch_openalex_page(client, keyword, config, page=page, per_page=per_page)
                results = data.get("results", [])
                if not results:
                    break

                for raw in results:
                    paper = normalize_openalex_paper(raw, query=keyword)
                    if paper is None:
                        logger.debug("Skipping paper without identifier")
                        continue
                    pid = paper.paper_id
                    if pid not in seen_ids:
                        seen_ids.add(pid)
                        papers.append(paper)

                retrieved_for_keyword += len(results)
                total_count = data.get("meta", {}).get("count", 0)
                if retrieved_for_keyword >= total_count:
                    break
                page += 1

            logger.info("Retrieved %d papers for keyword '%s'", retrieved_for_keyword, keyword)

    logger.info("Total: %d unique papers from %d keywords", len(papers), len(keywords))
    return papers
