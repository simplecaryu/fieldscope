"""Pipeline orchestrator for fieldscope."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from fieldscope.config import FieldscopeConfig
from fieldscope.models import PipelineState

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
