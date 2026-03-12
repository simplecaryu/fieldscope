"""Demo: keyword expansion → retrieval → embedding → seed detection pipeline.

Tests seed paper quality with centroid_proximity scoring.
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fieldscope.config import LLMConfig, RetrievalConfig, SeedsConfig, EmbeddingConfig
from fieldscope.stages.keyword_expansion import expand_keywords
from fieldscope.stages.retrieval import retrieve_papers
from fieldscope.stages.seeds import detect_seed_candidates
from fieldscope.embeddings.base import create_embedding_provider, prepare_text

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
TEST_MODEL = "google/gemini-2.0-flash-001"

QUERIES = [
    "Condensed matter physics",
    "Quantum magnetism",
    "AI in magnetism",
    "AI in topological material",
]


async def run_query(query: str):
    print(f"\n{'='*70}")
    print(f"  Query: {query}")
    print(f"{'='*70}")

    # 1. Keyword expansion
    llm_config = LLMConfig(
        base_url=OPENROUTER_BASE_URL,
        model=TEST_MODEL,
        temperature=0.3,
        max_tokens=512,
    )
    keywords = await expand_keywords(query, llm_config, OPENROUTER_API_KEY)
    print(f"\n  Keywords ({len(keywords)}): {keywords[:8]}{'...' if len(keywords) > 8 else ''}")

    # 2. Retrieval (use only 3 keywords, 25 per keyword to reduce API load)
    retrieval_config = RetrievalConfig(max_results_per_keyword=25)
    papers = await retrieve_papers(keywords[:3], retrieval_config, per_page=25)
    print(f"  Retrieved: {len(papers)} papers")

    if not papers:
        print("  No papers retrieved, skipping seed detection")
        return

    # 3. Embed papers for centroid_proximity
    emb_config = EmbeddingConfig(provider="sentence-transformers", model="all-MiniLM-L6-v2")
    provider = create_embedding_provider(emb_config)
    texts = [prepare_text(p, emb_config.text_fields) for p in papers]
    embeddings = provider.embed(texts)

    # Assign embeddings back to papers
    for i, p in enumerate(papers):
        p.embedding = embeddings[i].tolist()

    embedded_count = sum(1 for p in papers if p.embedding is not None)
    print(f"  Embedded: {embedded_count}/{len(papers)} papers")

    # 4. Seed detection with centroid_proximity
    seeds_config = SeedsConfig(
        methods=["citation_count", "pagerank", "centroid_proximity"],
        top_k=10,
    )
    candidates = detect_seed_candidates(papers, seeds_config)

    print(f"\n  Top {len(candidates)} Seed Candidates:")
    print(f"  {'Rank':<5} {'Score':<7} {'Year':<6} {'Cites':<7} {'Title'}")
    print(f"  {'-'*5} {'-'*7} {'-'*6} {'-'*7} {'-'*50}")

    paper_map = {p.paper_id: p for p in papers}
    for i, seed in enumerate(candidates, 1):
        p = paper_map.get(seed.paper_id)
        if p:
            title = p.title[:55] + "..." if len(p.title) > 55 else p.title
            print(f"  {i:<5} {seed.score:<7.3f} {p.year or '?':<6} {p.citation_count:<7} {title}")
        else:
            print(f"  {i:<5} {seed.score:<7.3f} {'?':<6} {'?':<7} {seed.paper_id}")

    # Show method breakdown for top 3
    print(f"\n  Method breakdown (top 3):")
    for i, seed in enumerate(candidates[:3], 1):
        methods_str = ", ".join(f"{k}={v:.3f}" for k, v in seed.methods.items())
        print(f"    #{i}: {methods_str}")


async def main():
    if not OPENROUTER_API_KEY:
        print("ERROR: OPENROUTER_API_KEY not set")
        sys.exit(1)

    for idx, query in enumerate(QUERIES):
        if idx > 0:
            print("\n  (waiting 15s for rate limit...)")
            await asyncio.sleep(15)
        try:
            await run_query(query)
        except Exception as e:
            print(f"\n  ERROR for '{query}': {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*70}")
    print("  Demo complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    asyncio.run(main())
