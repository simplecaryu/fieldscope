"""Stage 5: Citation expansion from validated seed papers."""

from __future__ import annotations

import logging

import httpx

from fieldscope.config import CitationExpansionConfig
from fieldscope.models import Paper, Provenance, SeedCandidate
from fieldscope.stages.retrieval import normalize_openalex_paper

logger = logging.getLogger(__name__)

OPENALEX_BASE = "https://api.openalex.org/works"


async def _fetch_related(
    client: httpx.AsyncClient,
    openalex_id: str,
    direction: str,
    max_papers: int,
) -> list[dict]:
    """Fetch papers related to a given OpenAlex work ID.

    direction: 'cited_by' or 'references'
    """
    if direction == "cited_by":
        params = {"filter": f"cites:{openalex_id}", "per_page": 25, "page": 1}
    else:
        params = {"filter": f"cited_by:{openalex_id}", "per_page": 25, "page": 1}

    results: list[dict] = []
    page = 1
    while len(results) < max_papers:
        params["page"] = page
        try:
            resp = await client.get(OPENALEX_BASE, params=params, timeout=30.0)
            if resp.status_code != 200:
                logger.warning(
                    "OpenAlex %s query for %s returned %d",
                    direction, openalex_id, resp.status_code,
                )
                break
            data = resp.json()
            batch = data.get("results", [])
            if not batch:
                break
            results.extend(batch)
            total = data.get("meta", {}).get("count", 0)
            if len(results) >= total:
                break
            page += 1
        except httpx.HTTPError as e:
            logger.warning("HTTP error during %s expansion: %s", direction, e)
            break

    return results[:max_papers]


async def expand_citations(
    papers: list[Paper],
    validated_seeds: list[SeedCandidate],
    config: CitationExpansionConfig,
) -> list[Paper]:
    """Expand the dataset by retrieving citing/referenced papers for validated seeds.

    Returns the full expanded dataset (original + new, deduplicated).
    """
    # Start with existing papers
    seen_ids: set[str] = set()
    all_papers: list[Paper] = []
    for p in papers:
        pid = p.paper_id
        if pid not in seen_ids:
            seen_ids.add(pid)
            all_papers.append(p)

    # Only expand validated seeds
    active_seeds = [s for s in validated_seeds if s.validated]
    if not active_seeds:
        logger.warning("No validated seeds to expand")
        return all_papers

    # Build paper lookup for openalex_id resolution
    paper_map = {p.paper_id: p for p in all_papers}

    async with httpx.AsyncClient() as client:
        for seed in active_seeds:
            paper = paper_map.get(seed.paper_id)
            oa_id = paper.openalex_id if paper else None
            if not oa_id:
                logger.warning("Seed %s has no OpenAlex ID, skipping expansion", seed.paper_id)
                continue

            seed_new_count = 0
            for direction in config.directions:
                remaining = config.max_papers_per_seed - seed_new_count
                if remaining <= 0:
                    break

                raw_results = await _fetch_related(client, oa_id, direction, remaining)
                for raw in raw_results:
                    if seed_new_count >= config.max_papers_per_seed:
                        break
                    expanded = normalize_openalex_paper(raw, query="")
                    if expanded is None:
                        continue
                    pid = expanded.paper_id
                    if pid in seen_ids:
                        continue
                    # Override provenance for expansion
                    expanded = expanded.model_copy(
                        update={
                            "provenance": Provenance(
                                method="citation_expansion",
                                depth=1,
                                seed_paper_id=seed.paper_id,
                            )
                        }
                    )
                    seen_ids.add(pid)
                    all_papers.append(expanded)
                    seed_new_count += 1

            logger.info(
                "Seed %s: expanded %d new papers",
                seed.paper_id, seed_new_count,
            )

    logger.info(
        "Citation expansion complete: %d → %d papers",
        len(papers), len(all_papers),
    )
    return all_papers
