"""Configuration models and TOML loading for fieldscope."""

from __future__ import annotations

import sys
from pathlib import Path

from pydantic import BaseModel

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


class LLMStageOverride(BaseModel):
    base_url: str | None = None
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None


class LLMConfig(BaseModel):
    base_url: str = ""
    model: str = ""
    temperature: float = 0.3
    max_tokens: int = 2048
    keyword_expansion: LLMStageOverride | None = None
    topic_labeling: LLMStageOverride | None = None
    reporting: LLMStageOverride | None = None


class EmbeddingConfig(BaseModel):
    provider: str = "sentence-transformers"
    model: str = "all-MiniLM-L6-v2"
    base_url: str = ""
    dimensions: int = 384
    batch_size: int = 64
    cache_dir: str = ".fieldscope/embeddings"
    normalize: bool = True
    text_fields: list[str] = ["title", "abstract"]


class RetrievalConfig(BaseModel):
    primary_source: str = "openalex"
    max_results_per_query: int = 1000
    rate_limit_rps: int = 10
    retry_max_attempts: int = 3
    retry_backoff_base: float = 2.0


class SeedsConfig(BaseModel):
    methods: list[str] = ["citation_count", "pagerank", "centroid_proximity"]
    top_k: int = 15
    auto_accept: bool = False


class CitationExpansionConfig(BaseModel):
    max_depth: int = 2
    max_papers_per_seed: int = 500
    directions: list[str] = ["references", "cited_by"]


class FilteringConfig(BaseModel):
    semantic_threshold: float = 0.3
    keyword_min_overlap: int = 1
    require_abstract: bool = False
    require_year: bool = True


class MaturityConfig(BaseModel):
    auto_accept: bool = False


class ClusteringConfig(BaseModel):
    leiden_resolution: float = 1.0
    hdbscan_min_cluster_size: int = 5
    embedding_similarity_threshold: float = 0.7


class EvolutionConfig(BaseModel):
    window_size_years: int = 3
    window_step_years: int = 1
    overlap_threshold: float = 0.3
    similarity_threshold: float = 0.5


class ReportingConfig(BaseModel):
    formats: list[str] = ["markdown", "json"]
    llm_narrative_enabled: bool = False


class FieldscopeConfig(BaseModel):
    llm: LLMConfig | None = None
    embedding: EmbeddingConfig = EmbeddingConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    seeds: SeedsConfig = SeedsConfig()
    citation_expansion: CitationExpansionConfig = CitationExpansionConfig()
    filtering: FilteringConfig = FilteringConfig()
    maturity: MaturityConfig = MaturityConfig()
    clustering: ClusteringConfig = ClusteringConfig()
    evolution: EvolutionConfig = EvolutionConfig()
    reporting: ReportingConfig = ReportingConfig()


def load_config(path: Path | None = None) -> FieldscopeConfig:
    """Load configuration from a TOML file.

    If the file does not exist, returns default configuration (LLM-free mode).
    """
    if path is None or not path.exists():
        return FieldscopeConfig()

    with open(path, "rb") as f:
        raw = tomllib.load(f)

    return FieldscopeConfig(**raw)
