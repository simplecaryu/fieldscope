"""Pipeline orchestrator for fieldscope."""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from fieldscope.config import FieldscopeConfig
from fieldscope.embeddings.base import create_embedding_provider, prepare_text
from fieldscope.models import PipelineState
from fieldscope.stages.citation_expansion import expand_citations
from fieldscope.stages.clustering import cluster_papers
from fieldscope.stages.evolution import analyze_evolution
from fieldscope.stages.filtering import filter_dataset
from fieldscope.stages.keyword_expansion import expand_keywords
from fieldscope.stages.labeling import label_clusters
from fieldscope.stages.maturity import assess_maturity, confirm_maturity
from fieldscope.stages.reporting import generate_reports
from fieldscope.stages.retrieval import retrieve_papers
from fieldscope.stages.seed_validation import validate_seeds
from fieldscope.stages.seeds import detect_seed_candidates

logger = logging.getLogger(__name__)

STAGE_ORDER: list[str] = [
    "keyword_expansion",
    "initial_retrieval",
    "seed_candidate_detection",
    "seed_user_validation",
    "citation_expansion",
    "dataset_filtering",
    "field_maturity_assessment",
    "field_maturity_confirmation",
    "adaptive_clustering",
    "topic_labeling",
    "field_evolution_analysis",
    "report_generation",
]

STAGE_DIR_MAP: dict[str, str] = {
    "keyword_expansion": "01_keywords",
    "initial_retrieval": "02_retrieval",
    "seed_candidate_detection": "03_seeds",
    "seed_user_validation": "04_seed_validation",
    "citation_expansion": "05_citation_expansion",
    "dataset_filtering": "06_filtering",
    "field_maturity_assessment": "07_maturity",
    "field_maturity_confirmation": "08_maturity_confirmation",
    "adaptive_clustering": "09_clustering",
    "topic_labeling": "10_labeling",
    "field_evolution_analysis": "11_evolution",
    "report_generation": "12_reports",
}


def generate_run_id(query: str) -> str:
    """Generate a run ID from query: YYYYMMDD_HHMMSS_slug."""
    now = datetime.now(tz=timezone.utc)
    date_str = now.strftime("%Y%m%d")
    time_str = now.strftime("%H%M%S")
    slug = re.sub(r"\s+", "_", query.lower().strip())
    slug = re.sub(r"[^a-z0-9_]", "", slug)[:40]
    return f"{date_str}_{time_str}_{slug}"


class Pipeline:
    """Orchestrates the fieldscope analysis pipeline."""

    def __init__(
        self,
        query: str,
        config: FieldscopeConfig,
        output_dir: Path,
        run_id: str | None = None,
    ) -> None:
        self.query = query
        self.config = config
        self.output_dir = output_dir

        rid = run_id or generate_run_id(query)
        self.run_dir = output_dir / rid
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.state = PipelineState(
            run_id=rid,
            query=query,
            config_snapshot=config.model_dump(mode="json"),
            completed_stages=[],
            stage_outputs={},
        )

        self._save_config_snapshot()
        self._save_state()

    @classmethod
    def resume(cls, run_dir: Path, config: FieldscopeConfig) -> Pipeline:
        """Resume a pipeline from an existing run directory."""
        state_path = run_dir / "state.json"
        state = PipelineState.model_validate_json(state_path.read_text())

        obj = object.__new__(cls)
        obj.query = state.query
        obj.config = config
        obj.output_dir = run_dir.parent
        obj.run_dir = run_dir
        obj.state = state
        return obj

    def get_stages_to_run(self, from_stage: str | None = None) -> list[str]:
        """Return the list of stages to execute."""
        if from_stage is not None:
            if from_stage not in STAGE_ORDER:
                raise ValueError(f"Unknown stage: {from_stage}")
            idx = STAGE_ORDER.index(from_stage)
            return STAGE_ORDER[idx:]

        if self.state.completed_stages:
            last = self.state.completed_stages[-1]
            idx = STAGE_ORDER.index(last)
            return STAGE_ORDER[idx + 1 :]

        return list(STAGE_ORDER)

    def get_stage_dir(self, stage_name: str) -> Path:
        """Get or create the output subdirectory for a stage."""
        dirname = STAGE_DIR_MAP[stage_name]
        stage_dir = self.run_dir / dirname
        stage_dir.mkdir(parents=True, exist_ok=True)
        return stage_dir

    def mark_stage_started(self, stage_name: str) -> None:
        """Mark a stage as currently running."""
        self.state.current_stage = stage_name
        self._save_state()

    def mark_stage_completed(self, stage_name: str, output_path: str) -> None:
        """Mark a stage as completed and record its output path."""
        self.state.completed_stages.append(stage_name)
        self.state.stage_outputs[stage_name] = output_path
        self.state.current_stage = None
        self._save_state()

    def _save_state(self) -> None:
        state_path = self.run_dir / "state.json"
        state_path.write_text(self.state.model_dump_json(indent=2))

    def _save_config_snapshot(self) -> None:
        snapshot_path = self.run_dir / "config_snapshot.toml"
        # Simple TOML-ish dump of config for reference
        snapshot_path.write_text(json.dumps(self.state.config_snapshot, indent=2))


async def run_pipeline(
    query: str,
    config: FieldscopeConfig,
    output_dir: Path,
    auto_accept: bool = False,
    manual_keywords: list[str] | None = None,
    from_stage: str | None = None,
) -> PipelineState:
    """Execute the full fieldscope pipeline.

    Sequences all stages in order, managing checkpointing between stages.
    """
    pipeline = Pipeline(query=query, config=config, output_dir=output_dir)
    stages = pipeline.get_stages_to_run(from_stage)

    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
    llm_config = config.llm

    # Shared state across stages
    keywords: list[str] = manual_keywords or []
    papers = []
    seeds = []
    maturity = None
    clusters = []
    events = []

    for stage in stages:
        logger.info("Starting stage: %s", stage)
        pipeline.mark_stage_started(stage)
        stage_dir = pipeline.get_stage_dir(stage)

        if stage == "keyword_expansion":
            if not keywords:
                if llm_config:
                    keywords = await expand_keywords(query, llm_config, api_key)
                else:
                    keywords = [query]
            _save_json(stage_dir / "keywords.json", keywords)

        elif stage == "initial_retrieval":
            papers = await retrieve_papers(keywords, config.retrieval)
            _save_json(stage_dir / "papers.json",
                       [p.model_dump(mode="json") for p in papers])

        elif stage == "seed_candidate_detection":
            # Embed papers if centroid_proximity is in methods
            if "centroid_proximity" in config.seeds.methods:
                papers = _embed_papers(papers, config)
            seeds = detect_seed_candidates(papers, config.seeds)
            _save_json(stage_dir / "seeds.json",
                       [s.model_dump(mode="json") for s in seeds])

        elif stage == "seed_user_validation":
            seeds = validate_seeds(seeds, papers, auto_accept=auto_accept)
            _save_json(stage_dir / "validated_seeds.json",
                       [s.model_dump(mode="json") for s in seeds])

        elif stage == "citation_expansion":
            papers = await expand_citations(papers, seeds, config.citation_expansion)
            _save_json(stage_dir / "papers.json",
                       [p.model_dump(mode="json") for p in papers])

        elif stage == "dataset_filtering":
            # Compute seed centroid if papers have embeddings
            seed_centroid = _compute_seed_centroid(papers, seeds)
            papers = filter_dataset(papers, keywords, config.filtering,
                                    seed_centroid=seed_centroid)
            _save_json(stage_dir / "papers.json",
                       [p.model_dump(mode="json") for p in papers])

        elif stage == "field_maturity_assessment":
            maturity = assess_maturity(papers)
            _save_json(stage_dir / "maturity.json",
                       maturity.model_dump(mode="json"))

        elif stage == "field_maturity_confirmation":
            maturity = confirm_maturity(maturity, auto_accept=auto_accept)
            _save_json(stage_dir / "maturity_confirmed.json",
                       maturity.model_dump(mode="json"))

        elif stage == "adaptive_clustering":
            papers = _embed_papers(papers, config)
            clusters = cluster_papers(papers, maturity, config.clustering)
            _save_json(stage_dir / "clusters.json",
                       [c.model_dump(mode="json") for c in clusters])

        elif stage == "topic_labeling":
            clusters = label_clusters(clusters, papers, llm_config=llm_config)
            _save_json(stage_dir / "clusters_labeled.json",
                       [c.model_dump(mode="json") for c in clusters])

        elif stage == "field_evolution_analysis":
            events = analyze_evolution(papers, clusters, config.evolution)
            _save_json(stage_dir / "events.json",
                       [e.model_dump(mode="json") for e in events])

        elif stage == "report_generation":
            generate_reports(
                query=query,
                keywords=keywords,
                papers=papers,
                seeds=seeds,
                maturity=maturity,
                clusters=clusters,
                events=events,
                config=config.reporting,
                llm_config=llm_config,
                output_dir=stage_dir,
            )

        pipeline.mark_stage_completed(stage, str(stage_dir))
        logger.info("Completed stage: %s", stage)

    return pipeline.state


def _save_json(path: Path, data) -> None:
    """Save data as JSON."""
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str))


def _embed_papers(papers, config):
    """Embed papers that don't already have embeddings."""
    needs_embedding = [p for p in papers if p.embedding is None]
    if not needs_embedding:
        return papers

    try:
        provider = create_embedding_provider(config.embedding)
        texts = [prepare_text(p, config.embedding.text_fields) for p in needs_embedding]
        embeddings = provider.embed(texts)
        for i, p in enumerate(needs_embedding):
            p.embedding = embeddings[i].tolist()
    except Exception as e:
        logger.warning("Embedding failed: %s — continuing without embeddings", e)

    return papers


def _compute_seed_centroid(papers, seeds):
    """Compute centroid of seed paper embeddings."""
    seed_ids = {s.paper_id for s in seeds if s.validated}
    seed_embeddings = []
    for p in papers:
        if p.paper_id in seed_ids and p.embedding is not None:
            seed_embeddings.append(p.embedding)

    if not seed_embeddings:
        return None
    return np.mean(seed_embeddings, axis=0)
