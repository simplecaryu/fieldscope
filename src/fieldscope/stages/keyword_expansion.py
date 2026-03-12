"""Stage 1: Keyword expansion via LLM or manual input."""

from __future__ import annotations

import json
import logging
import re

from fieldscope.config import LLMConfig
from fieldscope.llm.client import LLMClient

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a research librarian helping define search queries for bibliometric analysis. "
    "Given a research field description, suggest 10-15 related search keywords or short phrases "
    "that would help retrieve a comprehensive set of papers on this topic. "
    "Include the original query term, canonical field names, important synonyms, "
    "and adjacent research areas. "
    "Return ONLY a JSON array of strings, no explanation."
)


def parse_keywords_response(raw: str, original_query: str | None = None) -> list[str]:
    """Parse LLM response into a list of keywords.

    Handles JSON arrays, markdown code blocks, numbered/bulleted lists,
    and newline-separated plain text.
    """
    text = raw.strip()

    # Try to extract JSON from markdown code block
    md_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if md_match:
        text = md_match.group(1).strip()

    # Try JSON parse
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            keywords = [str(k).strip() for k in parsed]
        else:
            keywords = _parse_text_list(text)
    except (json.JSONDecodeError, ValueError):
        keywords = _parse_text_list(text)

    # Filter empty and deduplicate (preserving order)
    seen: set[str] = set()
    result: list[str] = []
    for kw in keywords:
        kw_lower = kw.lower()
        if kw and kw_lower not in seen:
            seen.add(kw_lower)
            result.append(kw)

    # Ensure original query is included
    if original_query:
        oq_lower = original_query.lower().strip()
        if oq_lower not in seen:
            result.insert(0, original_query.strip())

    return result


def _parse_text_list(text: str) -> list[str]:
    """Parse numbered lists, bullet lists, or newline-separated text."""
    lines = text.strip().split("\n")
    keywords = []
    for line in lines:
        line = line.strip()
        # Remove numbering: "1. ", "2) ", etc.
        line = re.sub(r"^\d+[\.\)]\s*", "", line)
        # Remove bullet markers
        line = re.sub(r"^[-*•]\s*", "", line)
        # Remove surrounding quotes
        line = line.strip().strip('"').strip("'")
        if line:
            keywords.append(line)
    return keywords


async def expand_keywords(
    query: str,
    config: LLMConfig,
    api_key: str,
) -> list[str]:
    """Expand a research query into search keywords using LLM."""
    client = LLMClient.from_config(config, api_key=api_key, stage="keyword_expansion")

    result = await client.chat(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Research field: {query}"},
        ],
    )

    keywords = parse_keywords_response(result.content, original_query=query)
    logger.info("Expanded query '%s' into %d keywords", query, len(keywords))
    return keywords
