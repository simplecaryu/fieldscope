"""Tests for fieldscope configuration loading."""

import pytest
from pydantic import ValidationError

from fieldscope.config import (
    CitationExpansionConfig,
    ClusteringConfig,
    EmbeddingConfig,
    EvolutionConfig,
    FieldscopeConfig,
    FilteringConfig,
    LLMConfig,
    LLMStageOverride,
    MaturityConfig,
    ReportingConfig,
    RetrievalConfig,
    SeedsConfig,
    load_config,
)


# ---------------------------------------------------------------------------
# Default config (no TOML file)
# ---------------------------------------------------------------------------


class TestFieldscopeConfigDefaults:
    def test_default_config_is_llm_free(self):
        cfg = FieldscopeConfig()
        assert cfg.llm is None

    def test_default_embedding(self):
        cfg = FieldscopeConfig()
        assert cfg.embedding.provider == "sentence-transformers"
        assert cfg.embedding.model == "all-MiniLM-L6-v2"
        assert cfg.embedding.dimensions == 384

    def test_default_retrieval(self):
        cfg = FieldscopeConfig()
        assert cfg.retrieval.primary_source == "openalex"
        assert cfg.retrieval.max_results_per_query == 1000

    def test_default_seeds(self):
        cfg = FieldscopeConfig()
        assert cfg.seeds.top_k == 15
        assert cfg.seeds.auto_accept is False

    def test_default_citation_expansion(self):
        cfg = FieldscopeConfig()
        assert cfg.citation_expansion.max_depth == 2

    def test_default_filtering(self):
        cfg = FieldscopeConfig()
        assert cfg.filtering.semantic_threshold == 0.3

    def test_default_maturity(self):
        cfg = FieldscopeConfig()
        assert cfg.maturity.auto_accept is False

    def test_default_clustering(self):
        cfg = FieldscopeConfig()
        assert cfg.clustering.leiden_resolution == 1.0

    def test_default_evolution(self):
        cfg = FieldscopeConfig()
        assert cfg.evolution.window_size_years == 3

    def test_default_reporting(self):
        cfg = FieldscopeConfig()
        assert cfg.reporting.formats == ["markdown", "json"]
        assert cfg.reporting.llm_narrative_enabled is False


# ---------------------------------------------------------------------------
# LLM config
# ---------------------------------------------------------------------------


class TestLLMConfig:
    def test_create_llm_config(self):
        llm = LLMConfig(
            base_url="https://api.openai.com/v1",
            model="gpt-4o-mini",
        )
        assert llm.temperature == 0.3
        assert llm.max_tokens == 2048

    def test_stage_overrides(self):
        llm = LLMConfig(
            base_url="https://api.openai.com/v1",
            model="gpt-4o-mini",
            keyword_expansion=LLMStageOverride(model="gpt-4o", temperature=0.5),
        )
        assert llm.keyword_expansion.model == "gpt-4o"
        assert llm.keyword_expansion.temperature == 0.5


# ---------------------------------------------------------------------------
# Load from TOML
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def test_load_minimal_toml(self, tmp_path):
        toml_file = tmp_path / "fieldscope.toml"
        toml_file.write_text(
            '[embedding]\nprovider = "sentence-transformers"\nmodel = "all-MiniLM-L6-v2"\ndimensions = 384\n'
        )
        cfg = load_config(toml_file)
        assert cfg.embedding.provider == "sentence-transformers"
        assert cfg.llm is None

    def test_load_with_llm(self, tmp_path):
        toml_file = tmp_path / "fieldscope.toml"
        toml_file.write_text(
            '[llm]\nbase_url = "http://localhost:11434/v1"\nmodel = "llama3.1:8b"\n'
        )
        cfg = load_config(toml_file)
        assert cfg.llm is not None
        assert cfg.llm.model == "llama3.1:8b"

    def test_load_with_stage_overrides(self, tmp_path):
        toml_file = tmp_path / "fieldscope.toml"
        toml_file.write_text(
            '[llm]\nbase_url = "http://localhost:11434/v1"\nmodel = "llama3.1:8b"\n\n'
            '[llm.reporting]\nmodel = "llama3.1:70b"\ntemperature = 0.4\nmax_tokens = 4096\n'
        )
        cfg = load_config(toml_file)
        assert cfg.llm.reporting.model == "llama3.1:70b"
        assert cfg.llm.reporting.max_tokens == 4096

    def test_load_nonexistent_returns_defaults(self, tmp_path):
        cfg = load_config(tmp_path / "nonexistent.toml")
        assert cfg.llm is None
        assert cfg.embedding.provider == "sentence-transformers"

    def test_load_overrides_specific_values(self, tmp_path):
        toml_file = tmp_path / "fieldscope.toml"
        toml_file.write_text(
            '[retrieval]\nprimary_source = "crossref"\nmax_results_per_query = 2000\n\n'
            '[seeds]\ntop_k = 20\nauto_accept = true\n'
        )
        cfg = load_config(toml_file)
        assert cfg.retrieval.primary_source == "crossref"
        assert cfg.retrieval.max_results_per_query == 2000
        assert cfg.seeds.top_k == 20
        assert cfg.seeds.auto_accept is True

    def test_load_full_config(self, tmp_path):
        toml_file = tmp_path / "fieldscope.toml"
        toml_file.write_text(
            '[llm]\nbase_url = "https://api.openai.com/v1"\nmodel = "gpt-4o-mini"\n\n'
            '[embedding]\nprovider = "openai-compatible"\n'
            'base_url = "https://api.openai.com/v1"\n'
            'model = "text-embedding-3-small"\ndimensions = 1536\n\n'
            '[retrieval]\nprimary_source = "openalex"\n\n'
            '[seeds]\ntop_k = 10\n\n'
            '[filtering]\nsemantic_threshold = 0.4\n\n'
            '[clustering]\nleiden_resolution = 0.8\n\n'
            '[reporting]\nformats = ["markdown", "json", "csv"]\nllm_narrative_enabled = true\n'
        )
        cfg = load_config(toml_file)
        assert cfg.llm.model == "gpt-4o-mini"
        assert cfg.embedding.provider == "openai-compatible"
        assert cfg.embedding.dimensions == 1536
        assert cfg.seeds.top_k == 10
        assert cfg.filtering.semantic_threshold == 0.4
        assert cfg.clustering.leiden_resolution == 0.8
        assert "csv" in cfg.reporting.formats
        assert cfg.reporting.llm_narrative_enabled is True
