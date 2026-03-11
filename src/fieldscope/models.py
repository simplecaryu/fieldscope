"""Core data models for fieldscope.

All models are pydantic BaseModel subclasses.
All inter-stage data exchange uses these models.
"""

from __future__ import annotations

from pydantic import BaseModel, field_validator


class Author(BaseModel):
    name: str
    orcid: str | None = None


class Provenance(BaseModel):
    method: str
    depth: int = 0
    seed_paper_id: str | None = None
    query: str | None = None


class Paper(BaseModel):
    doi: str | None = None
    openalex_id: str | None = None
    title: str
    abstract: str | None = None
    authors: list[Author] = []
    year: int | None = None
    venue: str | None = None
    citation_count: int = 0
    references: list[str] = []
    cited_by_count: int = 0
    source: str
    provenance: Provenance
    embedding: list[float] | None = None

    @field_validator("doi", mode="before")
    @classmethod
    def _normalize_doi(cls, v: str | None) -> str | None:
        if v is not None:
            return v.lower()
        return v

    @property
    def paper_id(self) -> str:
        if self.doi is not None:
            return self.doi
        if self.openalex_id is not None:
            return self.openalex_id
        raise ValueError("Paper must have at least one identifier (doi or openalex_id)")


class SeedCandidate(BaseModel):
    paper_id: str
    score: float
    methods: dict[str, float]
    rationale: str
    validated: bool | None = None


class Cluster(BaseModel):
    cluster_id: int
    member_paper_ids: list[str]
    label_extractive: str
    label_refined: str | None = None
    centroid: list[float] | None = None
    size: int
    top_keywords: list[str]


class EvolutionEvent(BaseModel):
    event_type: str
    time_window: tuple[int, int]
    source_cluster_ids: list[int]
    target_cluster_ids: list[int]
    evidence: dict[str, float]


class FieldMaturity(BaseModel):
    classification: str
    metrics: dict[str, float]
    user_override: bool


class PipelineState(BaseModel):
    run_id: str
    query: str
    config_snapshot: dict
    completed_stages: list[str]
    current_stage: str | None = None
    stage_outputs: dict[str, str]
