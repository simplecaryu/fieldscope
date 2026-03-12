"""Stage 11: Field evolution analysis."""

from __future__ import annotations

import logging
from collections import defaultdict

from fieldscope.config import EvolutionConfig
from fieldscope.models import Cluster, EvolutionEvent, Paper

logger = logging.getLogger(__name__)


def _get_year_range(papers: list[Paper]) -> tuple[int, int] | None:
    """Get min/max years from papers."""
    years = [p.year for p in papers if p.year is not None]
    if not years:
        return None
    return min(years), max(years)


def _papers_in_window(
    papers: list[Paper],
    start_year: int,
    end_year: int,
) -> list[Paper]:
    """Filter papers within a time window (inclusive)."""
    return [p for p in papers if p.year is not None and start_year <= p.year <= end_year]


def _cluster_presence_in_window(
    cluster: Cluster,
    window_papers: set[str],
) -> float:
    """Fraction of cluster members present in a time window."""
    if not cluster.member_paper_ids:
        return 0.0
    present = sum(1 for pid in cluster.member_paper_ids if pid in window_papers)
    return present / len(cluster.member_paper_ids)


def analyze_evolution(
    papers: list[Paper],
    clusters: list[Cluster],
    config: EvolutionConfig,
) -> list[EvolutionEvent]:
    """Analyze field evolution using temporal sliding-window analysis.

    Detects emergence, growth, decline, and stability events by tracking
    cluster presence across time windows.
    """
    if not papers or not clusters:
        return []

    year_range = _get_year_range(papers)
    if year_range is None:
        return []

    min_year, max_year = year_range

    # Build time windows
    windows: list[tuple[int, int]] = []
    start = min_year
    while start <= max_year:
        end = start + config.window_size_years - 1
        windows.append((start, min(end, max_year)))
        start += config.window_step_years

    if len(windows) < 1:
        return []

    # Paper ID lookup
    paper_map = {p.paper_id: p for p in papers}

    # Compute cluster presence per window
    window_presence: dict[int, list[float]] = defaultdict(list)
    for window_start, window_end in windows:
        window_paper_ids = {p.paper_id for p in _papers_in_window(papers, window_start, window_end)}
        for cluster in clusters:
            presence = _cluster_presence_in_window(cluster, window_paper_ids)
            window_presence[cluster.cluster_id].append(presence)

    # Detect events
    events: list[EvolutionEvent] = []

    for cluster in clusters:
        presences = window_presence[cluster.cluster_id]
        if len(presences) < 2:
            continue

        for i in range(1, len(presences)):
            prev = presences[i - 1]
            curr = presences[i]
            window = windows[i]

            # Emergence: absent then present
            if prev == 0.0 and curr > 0.0:
                events.append(EvolutionEvent(
                    event_type="emergence",
                    time_window=window,
                    source_cluster_ids=[],
                    target_cluster_ids=[cluster.cluster_id],
                    evidence={"presence_before": prev, "presence_after": curr},
                ))
            # Growth: increasing presence
            elif curr > prev and (curr - prev) >= config.overlap_threshold:
                events.append(EvolutionEvent(
                    event_type="growth",
                    time_window=window,
                    source_cluster_ids=[cluster.cluster_id],
                    target_cluster_ids=[cluster.cluster_id],
                    evidence={"presence_before": prev, "presence_after": curr},
                ))
            # Decline: decreasing presence
            elif curr < prev and (prev - curr) >= config.overlap_threshold:
                events.append(EvolutionEvent(
                    event_type="decline",
                    time_window=window,
                    source_cluster_ids=[cluster.cluster_id],
                    target_cluster_ids=[cluster.cluster_id],
                    evidence={"presence_before": prev, "presence_after": curr},
                ))
            # Stability: roughly same presence
            elif abs(curr - prev) < config.overlap_threshold and curr > 0:
                events.append(EvolutionEvent(
                    event_type="stability",
                    time_window=window,
                    source_cluster_ids=[cluster.cluster_id],
                    target_cluster_ids=[cluster.cluster_id],
                    evidence={"presence_before": prev, "presence_after": curr},
                ))

    logger.info("Evolution analysis: detected %d events across %d windows", len(events), len(windows))
    return events
