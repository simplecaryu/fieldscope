"""Tests for fieldscope pipeline orchestrator."""

import json

import pytest

from fieldscope.config import FieldscopeConfig
from fieldscope.models import PipelineState
from fieldscope.pipeline import (
    STAGE_ORDER,
    Pipeline,
    generate_run_id,
)


# ---------------------------------------------------------------------------
# Run ID generation
# ---------------------------------------------------------------------------


class TestGenerateRunId:
    def test_format(self):
        run_id = generate_run_id("AI in quantum magnetism")
        # Format: YYYYMMDD_HHMMSS_slug
        parts = run_id.split("_", 2)
        assert len(parts) == 3
        assert len(parts[0]) == 8  # date
        assert len(parts[1]) == 6  # time
        assert parts[2] == "ai_in_quantum_magnetism"

    def test_slug_lowercase_and_underscores(self):
        run_id = generate_run_id("Spintronics Field Evolution")
        slug = run_id.split("_", 2)[2]
        assert slug == "spintronics_field_evolution"

    def test_slug_truncated_to_40_chars(self):
        long_query = "a" * 60
        run_id = generate_run_id(long_query)
        slug = run_id.split("_", 2)[2]
        assert len(slug) <= 40


# ---------------------------------------------------------------------------
# Stage order
# ---------------------------------------------------------------------------


class TestStageOrder:
    def test_twelve_stages(self):
        assert len(STAGE_ORDER) == 12

    def test_first_stage(self):
        assert STAGE_ORDER[0] == "keyword_expansion"

    def test_last_stage(self):
        assert STAGE_ORDER[-1] == "report_generation"

    def test_expected_order(self):
        expected = [
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
        assert STAGE_ORDER == expected


# ---------------------------------------------------------------------------
# Pipeline initialization
# ---------------------------------------------------------------------------


class TestPipeline:
    def test_create_pipeline(self, tmp_path):
        cfg = FieldscopeConfig()
        p = Pipeline(query="spintronics", config=cfg, output_dir=tmp_path)
        assert p.query == "spintronics"
        assert p.config is cfg

    def test_pipeline_creates_run_directory(self, tmp_path):
        cfg = FieldscopeConfig()
        p = Pipeline(query="spintronics", config=cfg, output_dir=tmp_path)
        assert p.run_dir.exists()
        assert p.run_dir.parent == tmp_path

    def test_pipeline_initializes_state(self, tmp_path):
        cfg = FieldscopeConfig()
        p = Pipeline(query="spintronics", config=cfg, output_dir=tmp_path)
        assert isinstance(p.state, PipelineState)
        assert p.state.query == "spintronics"
        assert p.state.completed_stages == []
        assert p.state.current_stage is None

    def test_pipeline_saves_config_snapshot(self, tmp_path):
        cfg = FieldscopeConfig()
        p = Pipeline(query="test", config=cfg, output_dir=tmp_path)
        snapshot_path = p.run_dir / "config_snapshot.toml"
        assert snapshot_path.exists()

    def test_pipeline_saves_state_json(self, tmp_path):
        cfg = FieldscopeConfig()
        p = Pipeline(query="test", config=cfg, output_dir=tmp_path)
        state_path = p.run_dir / "state.json"
        assert state_path.exists()
        data = json.loads(state_path.read_text())
        assert data["query"] == "test"
        assert data["completed_stages"] == []

    def test_get_stages_to_run_all(self, tmp_path):
        cfg = FieldscopeConfig()
        p = Pipeline(query="test", config=cfg, output_dir=tmp_path)
        stages = p.get_stages_to_run()
        assert stages == STAGE_ORDER

    def test_get_stages_to_run_from_stage(self, tmp_path):
        cfg = FieldscopeConfig()
        p = Pipeline(query="test", config=cfg, output_dir=tmp_path)
        stages = p.get_stages_to_run(from_stage="adaptive_clustering")
        assert stages[0] == "adaptive_clustering"
        assert "keyword_expansion" not in stages

    def test_get_stages_to_run_invalid_stage(self, tmp_path):
        cfg = FieldscopeConfig()
        p = Pipeline(query="test", config=cfg, output_dir=tmp_path)
        with pytest.raises(ValueError, match="Unknown stage"):
            p.get_stages_to_run(from_stage="nonexistent_stage")

    def test_mark_stage_completed(self, tmp_path):
        cfg = FieldscopeConfig()
        p = Pipeline(query="test", config=cfg, output_dir=tmp_path)
        p.mark_stage_started("keyword_expansion")
        assert p.state.current_stage == "keyword_expansion"
        p.mark_stage_completed("keyword_expansion", "01_keywords/keywords.json")
        assert "keyword_expansion" in p.state.completed_stages
        assert p.state.current_stage is None
        # Verify persisted to disk
        data = json.loads((p.run_dir / "state.json").read_text())
        assert "keyword_expansion" in data["completed_stages"]

    def test_resume_from_state(self, tmp_path):
        cfg = FieldscopeConfig()
        p = Pipeline(query="test", config=cfg, output_dir=tmp_path)
        p.mark_stage_started("keyword_expansion")
        p.mark_stage_completed("keyword_expansion", "01_keywords/keywords.json")

        # Create a new pipeline that resumes from existing run dir
        p2 = Pipeline.resume(run_dir=p.run_dir, config=cfg)
        assert p2.state.completed_stages == ["keyword_expansion"]
        stages = p2.get_stages_to_run()
        assert stages[0] == "initial_retrieval"
        assert "keyword_expansion" not in stages

    def test_stage_subdirectory_creation(self, tmp_path):
        cfg = FieldscopeConfig()
        p = Pipeline(query="test", config=cfg, output_dir=tmp_path)
        subdir = p.get_stage_dir("keyword_expansion")
        assert subdir.exists()
        assert subdir.name == "01_keywords"
