"""Stage 12: Report generation."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from fieldscope.config import LLMConfig, ReportingConfig
from fieldscope.models import (
    Cluster,
    EvolutionEvent,
    FieldMaturity,
    Paper,
    SeedCandidate,
)

logger = logging.getLogger(__name__)


def generate_reports(
    query: str,
    keywords: list[str],
    papers: list[Paper],
    seeds: list[SeedCandidate],
    maturity: FieldMaturity,
    clusters: list[Cluster],
    events: list[EvolutionEvent],
    config: ReportingConfig,
    llm_config: LLMConfig | None,
    output_dir: Path,
) -> list[Path]:
    """Generate report files in the requested formats.

    Returns list of paths to generated files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    for fmt in config.formats:
        if fmt == "markdown":
            p = _write_markdown(
                query, keywords, papers, seeds, maturity, clusters, events, output_dir,
            )
            paths.append(p)
        elif fmt == "json":
            p = _write_json(
                query, keywords, papers, seeds, maturity, clusters, events, output_dir,
            )
            paths.append(p)
        else:
            logger.warning("Unknown report format: %s", fmt)

    logger.info("Generated %d report(s) in %s", len(paths), output_dir)
    return paths


def _write_markdown(
    query: str,
    keywords: list[str],
    papers: list[Paper],
    seeds: list[SeedCandidate],
    maturity: FieldMaturity,
    clusters: list[Cluster],
    events: list[EvolutionEvent],
    output_dir: Path,
) -> Path:
    """Write a Markdown report."""
    paper_map = {p.paper_id: p for p in papers}
    lines: list[str] = []

    # Title
    lines.append(f"# Field Analysis Report: {query}")
    lines.append("")

    # Overview
    lines.append("## Overview")
    lines.append("")
    lines.append(f"- **Query**: {query}")
    lines.append(f"- **Keywords**: {', '.join(keywords)}")
    lines.append(f"- **Total papers**: {len(papers)}")
    lines.append(f"- **Seed papers**: {len(seeds)}")
    lines.append(f"- **Clusters**: {len(clusters)}")
    lines.append("")

    # Field Maturity
    lines.append("## Field Maturity")
    lines.append("")
    lines.append(f"- **Classification**: {maturity.classification}")
    for key, val in maturity.metrics.items():
        lines.append(f"- **{key}**: {val}")
    lines.append("")

    # Seed Papers
    lines.append("## Seed Papers")
    lines.append("")
    if seeds:
        lines.append("| Rank | Score | Title | Year | Citations |")
        lines.append("|------|-------|-------|------|-----------|")
        for i, seed in enumerate(seeds, 1):
            p = paper_map.get(seed.paper_id)
            if p:
                lines.append(f"| {i} | {seed.score:.3f} | {p.title} | {p.year or '-'} | {p.citation_count} |")
            else:
                lines.append(f"| {i} | {seed.score:.3f} | {seed.paper_id} | - | - |")
    else:
        lines.append("No seed papers identified.")
    lines.append("")

    # Clusters
    lines.append("## Clusters")
    lines.append("")
    if clusters:
        for c in clusters:
            lines.append(f"### Cluster {c.cluster_id}: {c.label_extractive}")
            lines.append("")
            lines.append(f"- **Size**: {c.size} papers")
            lines.append(f"- **Keywords**: {', '.join(c.top_keywords)}")
            if c.label_refined:
                lines.append(f"- **Refined label**: {c.label_refined}")
            lines.append("")
    else:
        lines.append("No clusters identified.")
    lines.append("")

    # Evolution Events
    lines.append("## Evolution Events")
    lines.append("")
    if events:
        for e in events:
            lines.append(f"- **{e.event_type}** ({e.time_window[0]}–{e.time_window[1]}): "
                         f"clusters {e.source_cluster_ids} → {e.target_cluster_ids}")
    else:
        lines.append("No evolution events detected.")
    lines.append("")

    path = output_dir / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_json(
    query: str,
    keywords: list[str],
    papers: list[Paper],
    seeds: list[SeedCandidate],
    maturity: FieldMaturity,
    clusters: list[Cluster],
    events: list[EvolutionEvent],
    output_dir: Path,
) -> Path:
    """Write a JSON report."""
    data = {
        "query": query,
        "keywords": keywords,
        "papers": [p.model_dump(mode="json") for p in papers],
        "seeds": [s.model_dump(mode="json") for s in seeds],
        "maturity": maturity.model_dump(mode="json"),
        "clusters": [c.model_dump(mode="json") for c in clusters],
        "events": [e.model_dump(mode="json") for e in events],
        "summary": {
            "total_papers": len(papers),
            "total_seeds": len(seeds),
            "total_clusters": len(clusters),
            "total_events": len(events),
            "field_classification": maturity.classification,
        },
    }

    path = output_dir / "report.json"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
