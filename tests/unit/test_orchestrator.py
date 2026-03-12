"""Tests for pipeline orchestrator (run_pipeline)."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from fieldscope.config import FieldscopeConfig
from fieldscope.models import (
    Cluster,
    EvolutionEvent,
    FieldMaturity,
    Paper,
    PipelineState,
    Provenance,
    SeedCandidate,
)
from fieldscope.pipeline import Pipeline, run_pipeline, STAGE_ORDER


def _make_paper(doi: str, year: int = 2024, citation_count: int = 10,
                embedding: list[float] | None = None):
    return Paper(
        doi=doi, title=f"Paper {doi}", year=year,
        citation_count=citation_count, cited_by_count=citation_count,
        source="openalex", embedding=embedding or [0.1, 0.2, 0.3],
        provenance=Provenance(method="initial_retrieval", depth=0),
    )


def _make_seed(paper_id: str):
    return SeedCandidate(
        paper_id=paper_id, score=0.9,
        methods={"citation_count": 0.9},
        rationale="test", validated=True,
    )


def _make_cluster(cluster_id: int, paper_ids: list[str]):
    return Cluster(
        cluster_id=cluster_id, member_paper_ids=paper_ids,
        label_extractive=f"Topic {cluster_id}", size=len(paper_ids),
        top_keywords=["kw1"],
    )


MOCK_PAPERS = [_make_paper(f"10.1/{i}") for i in range(5)]
MOCK_SEEDS = [_make_seed("10.1/0")]
MOCK_MATURITY = FieldMaturity(
    classification="growing",
    metrics={"growth_rate": 0.5, "citation_density": 0.3,
             "keyword_burst": 0.4, "median_age": 8.0},
    user_override=False,
)
MOCK_CLUSTERS = [_make_cluster(0, ["10.1/0", "10.1/1"])]
MOCK_EVENTS = [EvolutionEvent(
    event_type="emergence", time_window=(2020, 2024),
    source_cluster_ids=[], target_cluster_ids=[0],
    evidence={"presence": 0.8},
)]


class TestRunPipeline:
    @pytest.mark.asyncio
    async def test_runs_all_stages(self, tmp_path):
        config = FieldscopeConfig()

        with (
            patch("fieldscope.pipeline.expand_keywords", new_callable=AsyncMock,
                  return_value=["quantum magnetism", "spin dynamics"]),
            patch("fieldscope.pipeline.retrieve_papers", new_callable=AsyncMock,
                  return_value=MOCK_PAPERS),
            patch("fieldscope.pipeline.detect_seed_candidates",
                  return_value=MOCK_SEEDS),
            patch("fieldscope.pipeline.validate_seeds",
                  return_value=MOCK_SEEDS),
            patch("fieldscope.pipeline.expand_citations", new_callable=AsyncMock,
                  return_value=MOCK_PAPERS),
            patch("fieldscope.pipeline.filter_dataset",
                  return_value=MOCK_PAPERS),
            patch("fieldscope.pipeline.assess_maturity",
                  return_value=MOCK_MATURITY),
            patch("fieldscope.pipeline.confirm_maturity",
                  return_value=MOCK_MATURITY),
            patch("fieldscope.pipeline.cluster_papers",
                  return_value=MOCK_CLUSTERS),
            patch("fieldscope.pipeline.label_clusters",
                  return_value=MOCK_CLUSTERS),
            patch("fieldscope.pipeline.analyze_evolution",
                  return_value=MOCK_EVENTS),
            patch("fieldscope.pipeline.generate_reports",
                  return_value=[tmp_path / "report.md"]),
        ):
            state = await run_pipeline(
                query="quantum magnetism",
                config=config,
                output_dir=tmp_path,
                auto_accept=True,
            )

        assert isinstance(state, PipelineState)
        assert len(state.completed_stages) == len(STAGE_ORDER)

    @pytest.mark.asyncio
    async def test_state_json_written(self, tmp_path):
        config = FieldscopeConfig()

        with (
            patch("fieldscope.pipeline.expand_keywords", new_callable=AsyncMock,
                  return_value=["kw1"]),
            patch("fieldscope.pipeline.retrieve_papers", new_callable=AsyncMock,
                  return_value=MOCK_PAPERS),
            patch("fieldscope.pipeline.detect_seed_candidates",
                  return_value=MOCK_SEEDS),
            patch("fieldscope.pipeline.validate_seeds",
                  return_value=MOCK_SEEDS),
            patch("fieldscope.pipeline.expand_citations", new_callable=AsyncMock,
                  return_value=MOCK_PAPERS),
            patch("fieldscope.pipeline.filter_dataset",
                  return_value=MOCK_PAPERS),
            patch("fieldscope.pipeline.assess_maturity",
                  return_value=MOCK_MATURITY),
            patch("fieldscope.pipeline.confirm_maturity",
                  return_value=MOCK_MATURITY),
            patch("fieldscope.pipeline.cluster_papers",
                  return_value=MOCK_CLUSTERS),
            patch("fieldscope.pipeline.label_clusters",
                  return_value=MOCK_CLUSTERS),
            patch("fieldscope.pipeline.analyze_evolution",
                  return_value=MOCK_EVENTS),
            patch("fieldscope.pipeline.generate_reports",
                  return_value=[tmp_path / "report.md"]),
        ):
            state = await run_pipeline(
                query="test query",
                config=config,
                output_dir=tmp_path,
                auto_accept=True,
            )

        # Find the run directory
        run_dirs = [d for d in tmp_path.iterdir() if d.is_dir()]
        assert len(run_dirs) == 1
        state_file = run_dirs[0] / "state.json"
        assert state_file.exists()

    @pytest.mark.asyncio
    async def test_auto_accept_passed(self, tmp_path):
        config = FieldscopeConfig()
        mock_validate = MagicMock(return_value=MOCK_SEEDS)
        mock_confirm = MagicMock(return_value=MOCK_MATURITY)

        with (
            patch("fieldscope.pipeline.expand_keywords", new_callable=AsyncMock,
                  return_value=["kw1"]),
            patch("fieldscope.pipeline.retrieve_papers", new_callable=AsyncMock,
                  return_value=MOCK_PAPERS),
            patch("fieldscope.pipeline.detect_seed_candidates",
                  return_value=MOCK_SEEDS),
            patch("fieldscope.pipeline.validate_seeds", mock_validate),
            patch("fieldscope.pipeline.expand_citations", new_callable=AsyncMock,
                  return_value=MOCK_PAPERS),
            patch("fieldscope.pipeline.filter_dataset",
                  return_value=MOCK_PAPERS),
            patch("fieldscope.pipeline.assess_maturity",
                  return_value=MOCK_MATURITY),
            patch("fieldscope.pipeline.confirm_maturity", mock_confirm),
            patch("fieldscope.pipeline.cluster_papers",
                  return_value=MOCK_CLUSTERS),
            patch("fieldscope.pipeline.label_clusters",
                  return_value=MOCK_CLUSTERS),
            patch("fieldscope.pipeline.analyze_evolution",
                  return_value=MOCK_EVENTS),
            patch("fieldscope.pipeline.generate_reports",
                  return_value=[]),
        ):
            await run_pipeline(
                query="test",
                config=config,
                output_dir=tmp_path,
                auto_accept=True,
            )

        # validate_seeds should be called with auto_accept=True
        mock_validate.assert_called_once()
        _, kwargs = mock_validate.call_args
        assert kwargs.get("auto_accept") is True or mock_validate.call_args[0][-1] is True

        # confirm_maturity should be called with auto_accept=True
        mock_confirm.assert_called_once()
