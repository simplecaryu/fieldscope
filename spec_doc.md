# fieldscope specification

This document contains the **initial technical specification** for the `fieldscope` repository.

`fieldscope` is a **bibliometric analysis toolkit** designed to help researchers define, construct, and analyze research fields starting from a user-provided keyword query.

The repository itself **is NOT an AI agent system**. AI agents may be used during development, but the resulting tool is a **deterministic research pipeline with bounded LLM-assisted stages**.

---

# Project Overview

`fieldscope` is a reproducible pipeline that:

1. accepts a research field keyword
2. expands the keyword using controlled LLM assistance
3. retrieves candidate literature
4. detects representative seed papers
5. requires user validation
6. performs citation expansion
7. constructs a filtered literature dataset
8. analyzes the structure of the field
9. analyzes the temporal evolution of the field

Primary goal:

> Make interdisciplinary research fields analyzable using reproducible bibliometric workflows.

Example target domains:

* AI in quantum magnetism
* machine learning for topological materials
* altermagnetism research landscape
* spintronics field evolution
* emerging condensed matter subfields

---

# Core Design Principles

* reproducibility
* modular pipeline design
* human‑reviewable intermediate steps
* retrieval‑first architecture
* bounded and explicit LLM usage
* domain‑adaptable analysis

The pipeline should remain **scientifically interpretable** and avoid opaque decision making.

---

# Scope

## Included

* keyword‑driven research field definition
* literature retrieval from scholarly metadata providers
* seed paper suggestion
* user‑validated seed selection
* citation expansion
* dataset filtering
* field maturity detection
* adaptive clustering analysis
* topic labeling
* field evolution analysis
* report generation

## Excluded

* autonomous AI research agents
* open‑ended planning agents
* conversational chat systems
* LLM‑dependent core bibliometric logic

---

# LLM and Text Model Scope

LLM and text-model usage should be **explicitly limited to a few well-defined stages**.

Primary LLM-assisted stages:

* keyword expansion
* human-readable cluster label refinement

Supporting text-model-assisted stages:

* text embedding, depending on the selected embedding backend

Default behavior:

* an LLM provider is configured and used for keyword expansion
* topic labeling uses extractive labeling first and, if enabled, LLM-based refinement

Fallback behavior:

* if no LLM provider is configured, keyword expansion is skipped unless the user explicitly supplies manual keyword sets
* topic labeling falls back to extractive methods only

Disallowed uses:

* pipeline-wide autonomous decision making
* uncontrolled literature search planning
* automatic field definition without user validation

The bibliometric logic must remain **graph- and data-driven**, and model-assisted stages must be transparent and reviewable.

---

# LLM and Embedding Backend Strategy

The system must support:

* BYOK (Bring Your Own Key)
* Local LLMs
* Local embedding models

The pipeline must **not require cloud‑only LLM services**.

### BYOK examples

* OpenAI‑compatible endpoints
* Anthropic‑compatible endpoints
* self‑hosted OpenAI‑compatible servers

### Local model examples

* Ollama
* llama.cpp
* vLLM

LLM access must be implemented through a **provider abstraction layer**.

---

# Embedding Policy

Text embeddings are used for:

* semantic similarity
* cluster centroid computation
* semantic filtering
* cluster labeling support

Embedding backends must be configurable.

Supported options:

* sentence‑transformers
* local embedding models
* BYOK embedding APIs

Embeddings are an infrastructure layer, not necessarily an LLM-specific feature.

Embedding infrastructure must **not be tied to a single vendor**.

---

# Pipeline Stages

The pipeline contains the following stages:

1. keyword_expansion_or_manual_keywords
2. initial_retrieval
3. seed_candidate_detection
4. seed_user_validation
5. citation_expansion
6. dataset_filtering
7. field_maturity_assessment
8. field_maturity_confirmation
9. adaptive_clustering
10. topic_labeling
11. field_evolution_analysis
12. report_generation

---

# Stage Descriptions

## Keyword Expansion or Manual Keywords

Goal:

Produce the keyword set used for initial literature retrieval.

Two supported modes:

### Mode A — LLM-assisted keyword expansion (default)

A configured LLM provider expands the user field query into search-ready keywords and related phrases.

Recommended target size:

* 8 to 20 expanded keywords or short keyword phrases

Practical policy:

* generate enough terms to improve recall
* avoid uncontrolled explosion of near-duplicate phrases
* keep the final keyword set reviewable by a human user

Example outputs:

* machine learning magnetism
* AI spintronics
* deep learning magnetic materials
* materials informatics magnetism

### Mode B — Manual keyword mode

If the user explicitly invokes manual mode, the system skips keyword expansion and uses only the keywords supplied by the user.

This mode is intended for:

* users who do not want LLM assistance
* offline or restricted environments
* highly controlled query construction

Manual keyword guidance:

* ask the user for 5 to 15 keywords or short phrases
* encourage a mix of canonical field names, adjacent terms, and important synonyms
* warn against overly generic terms that will cause topic drift

All generated or manually supplied keyword sets must be stored for reproducibility.

---

## Initial Literature Retrieval

Goal:

Build a **high-recall initial literature pool**.

Primary data sources:

* OpenAlex
* Crossref
* Semantic Scholar

Default implementation:

* OpenAlex as the primary source
* Crossref for metadata completion and DOI normalization where needed

Optional extended implementation:

* Semantic Scholar support when available and practical
* additional source merging only when the implementation cost is justified

Rules:

* prioritize recall over precision
* normalize metadata
* deduplicate records

---

## Seed Candidate Detection

Goal:

Suggest representative papers for field definition.

Default methods:

* citation count
* PageRank
* embedding centroid proximity

Optional methods:

* betweenness centrality
* additional structural ranking metrics

Output:

* seed candidates
* seed scores
* selection rationale

Implementation note:

* default methods should be sufficient for the first usable version
* optional methods may be added later if they clearly improve seed quality

---

## Seed User Validation

A mandatory human checkpoint unless auto-accept mode is enabled.

User may:

* accept seed
* reject seed
* add seed manually
* accept all suggested seeds

Citation expansion must not proceed without validated seeds unless auto-accept mode is explicitly enabled.

---

## Citation Expansion

Dataset expansion strategy:

* references
* citing papers

Maximum expansion depth:

citation distance ≤ 2

All records must track provenance.

---

## Dataset Filtering

Filtering layers:

* semantic similarity filter
* keyword alignment filter
* venue sanity filter
* metadata completeness filter

Embeddings may be used for semantic filtering.

---

## Field Maturity Assessment

Goal:

Estimate the likely development stage of the field.

Metrics:

* publication growth rate
* citation density
* keyword burst score
* age distribution

Classes:

* emerging
* growing
* mature

This stage should produce metrics and a suggested maturity class, but should not immediately force the analysis path.

---

## Field Maturity Confirmation

Before adaptive clustering, the system should present the computed maturity metrics and the suggested class to the user.

User may:

* accept the suggested maturity class
* override the maturity class manually

If auto-accept mode is enabled, the suggested class is accepted automatically.

---

## Adaptive Clustering

Clustering strategy depends on field maturity.

### Emerging

Graphs:

* bibliographic coupling
* embedding similarity

Algorithms:

* Leiden
* HDBSCAN

Purpose:

Detect emerging research fronts.

### Growing

Graphs:

* bibliographic coupling
* direct citation

Algorithm:

* Leiden

### Mature

Graphs:

* direct citation
* co‑citation

Purpose:

Map stable research communities.

---

## Topic Labeling

Cluster labeling methods:

* TF-IDF keywords
* KeyBERT
* centroid-nearest representative terms
* LLM-assisted label refinement

Default policy:

* always generate an extractive baseline label first
* by default, refine the final human-readable label with a configured LLM provider

Fallback policy:

* if no LLM provider is configured, use extractive labels only

Without a configured LLM provider, cluster labeling still works through extractive methods, but the final labels may be less natural and less concise.

---

## Field Evolution Analysis

Temporal analysis using sliding windows.

Recommended window size:

2–3 years

Detected events:

* birth
* growth
* split
* merge
* decline

Cluster linking signals:

* membership overlap
* embedding similarity
* citation flow

---

## Report Generation

Outputs:

* summary report
* methods report
* final markdown report
* optional LLM-refined narrative report

The report must include:

* field definition
* dataset construction method
* seed validation summary
* cluster structure
* emerging topics
* evolution analysis

Default behavior:

* generate structured non-LLM reports from pipeline outputs

Optional enhanced behavior:

* if explicitly enabled, generate an LLM-refined narrative version of the report
* this report-writing stage should be separately configurable from keyword expansion and topic-label refinement

Implementation note:

* keyword expansion, embedding, topic-label refinement, and report writing may use different models or providers in practice, so backend configuration should support stage-specific provider selection

---

# Repository Structure (Proposed)

```
src/

retrieval/
seeds/
citations/
filtering/
maturity/
clustering/
labeling/
evolution/
reporting/

llm/
embeddings/
```

LLM and embedding modules must be **provider-agnostic abstractions**.

The `llm/` module should handle:

* provider configuration
* keyword expansion calls
* optional topic label refinement calls
* optional report-writing calls
* local vs BYOK backend routing

---

# Non‑Goals

The project does NOT aim to build:

* a conversational AI product
* a fully autonomous scientific discovery agent
* a cloud‑dependent LLM system

---

# Success Criteria

Functional success:

* users can define a field from a keyword
* users can choose either LLM-assisted keyword expansion or manual keyword mode
* seed papers can be automatically suggested
* users can validate seeds before expansion
* an auto-accept mode can run the full pipeline without user prompts
* the system builds a reproducible literature dataset
* clustering adapts to field maturity

Engineering success:

* decisions are inspectable
* model usage is explicit, bounded, and configurable
* BYOK and local models are supported
* pipeline runs are reproducible

---

# Technology Stack

## Python

Version constraint: >= 3.10, < 3.13

Type hints are mandatory on all public interfaces.

## Package Manager

`uv` with `pyproject.toml` as the standard metadata file.

## Core Libraries

| Library | Purpose | Rationale |
|---------|---------|-----------|
| `pydantic` >= 2.0 | Data model validation, config schema | Type-safe schemas, JSON serialization, settings management |
| `networkx` | Graph construction, PageRank, centrality metrics | Standard Python graph library |
| `igraph` + `leidenalg` | Leiden community detection | networkx lacks Leiden; igraph+leidenalg is the reference implementation |
| `hdbscan` | Density-based clustering | Required for emerging-field clustering strategy |
| `scikit-learn` | TF-IDF, cosine similarity, general ML utilities | Standard ML toolkit |
| `sentence-transformers` | Local embedding models | Default local embedding backend |
| `keybert` | Keyword extraction from text | Used in extractive topic labeling |
| `httpx` | HTTP client for scholarly APIs | Async support, timeout control, connection pooling |
| `click` | CLI framework | Mature, composable subcommands |
| `rich` | Terminal UI (progress bars, tables, prompts) | Interactive validation checkpoints, progress display |
| `matplotlib` | Visualization | Citation network plots, cluster maps, evolution timelines |

Data handling uses pydantic models and standard Python collections (`list[Paper]`, etc.) rather than pandas, keeping the dependency footprint light and the data flow type-safe.

## Code Quality

* Linter and formatter: `ruff`
* Type checker: `mypy` (strict mode)

## Test Framework

`pytest` with `pytest-httpx` for HTTP API mocking.

---

# Data Models

All models are pydantic `BaseModel` subclasses. All inter-stage data exchange uses these models.

## Paper

The core record representing a scholarly work.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `doi` | `str \| None` | no | `None` | Digital Object Identifier (normalized, lowercase) |
| `openalex_id` | `str \| None` | no | `None` | OpenAlex work ID (e.g., `W2123456789`) |
| `title` | `str` | **yes** | — | Paper title |
| `abstract` | `str \| None` | no | `None` | Abstract text |
| `authors` | `list[Author]` | **yes** | `[]` | List of authors |
| `year` | `int \| None` | no | `None` | Publication year |
| `venue` | `str \| None` | no | `None` | Journal or conference name |
| `citation_count` | `int` | **yes** | `0` | Total citation count |
| `references` | `list[str]` | **yes** | `[]` | Referenced work IDs (DOIs or OpenAlex IDs) |
| `cited_by_count` | `int` | **yes** | `0` | Number of citing works |
| `source` | `str` | **yes** | — | Data source: `"openalex"`, `"crossref"`, `"semantic_scholar"` |
| `provenance` | `Provenance` | **yes** | — | How this paper entered the dataset |
| `embedding` | `list[float] \| None` | no | `None` | Embedding vector (populated after embedding stage) |

### Computed property: `paper_id`

`paper_id` is a read-only computed property, not a stored field.

Resolution order:

1. `doi` if present (normalized, lowercase)
2. `openalex_id` if present
3. raises an error — every paper must have at least one identifier

All cross-references between models (e.g., `SeedCandidate.paper_id`, `Cluster.member_paper_ids`) use this computed `paper_id`.

### Deduplication

Two papers are considered duplicates if they share the same `paper_id`. When merging records from different sources, prefer the record with more complete metadata (non-null fields).

## Author

| Field | Type | Required | Default |
|-------|------|----------|---------|
| `name` | `str` | **yes** | — |
| `orcid` | `str \| None` | no | `None` |

## Provenance

Tracks how a paper entered the dataset.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `method` | `str` | **yes** | — | `"initial_retrieval"` or `"citation_expansion"` |
| `depth` | `int` | **yes** | `0` | Citation distance from seed (0 = seed itself, 1 = direct ref/citation, 2 = max) |
| `seed_paper_id` | `str \| None` | no | `None` | Which seed paper led to this expansion |
| `query` | `str \| None` | no | `None` | The keyword query that retrieved this paper (initial retrieval only) |

## SeedCandidate

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `paper_id` | `str` | **yes** | — | Reference to Paper via computed `paper_id` |
| `score` | `float` | **yes** | — | Composite seed score |
| `methods` | `dict[str, float]` | **yes** | — | Per-method scores (e.g., `{"pagerank": 0.85, "citation_count": 0.72, "centroid_proximity": 0.65}`) |
| `rationale` | `str` | **yes** | — | Human-readable selection rationale |
| `validated` | `bool \| None` | no | `None` | User validation result (`None` = not yet reviewed) |

## Cluster

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `cluster_id` | `int` | **yes** | — | Unique cluster identifier |
| `member_paper_ids` | `list[str]` | **yes** | — | Paper IDs in this cluster |
| `label_extractive` | `str` | **yes** | — | TF-IDF / KeyBERT generated label |
| `label_refined` | `str \| None` | no | `None` | LLM-refined label (if enabled) |
| `centroid` | `list[float] \| None` | no | `None` | Cluster centroid in embedding space |
| `size` | `int` | **yes** | — | Number of papers |
| `top_keywords` | `list[str]` | **yes** | — | Top representative keywords |

## EvolutionEvent

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `event_type` | `str` | **yes** | `"birth"`, `"growth"`, `"split"`, `"merge"`, or `"decline"` |
| `time_window` | `tuple[int, int]` | **yes** | (start_year, end_year) |
| `source_cluster_ids` | `list[int]` | **yes** | Cluster(s) before the event |
| `target_cluster_ids` | `list[int]` | **yes** | Cluster(s) after the event |
| `evidence` | `dict[str, float]` | **yes** | Supporting metrics (overlap, similarity, citation flow) |

## FieldMaturity

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `classification` | `str` | **yes** | `"emerging"`, `"growing"`, or `"mature"` |
| `metrics` | `dict[str, float]` | **yes** | Keys: `growth_rate`, `citation_density`, `keyword_burst`, `median_age` |
| `user_override` | `bool` | **yes** | Whether the user overrode the automatic classification |

## PipelineState

Tracks the state of a single pipeline run for checkpointing and resumption.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `run_id` | `str` | **yes** | Unique run identifier |
| `query` | `str` | **yes** | Original user query |
| `config_snapshot` | `dict` | **yes** | Frozen configuration at run start |
| `completed_stages` | `list[str]` | **yes** | Stage names that have finished |
| `current_stage` | `str \| None` | no | Currently executing stage |
| `stage_outputs` | `dict[str, str]` | **yes** | Stage name → output file path |

## Inter-Stage Data Contracts

Each stage consumes and produces specific types:

| Stage | Input | Output |
|-------|-------|--------|
| keyword_expansion | `str` (user query) | `list[str]` (expanded keywords) |
| initial_retrieval | `list[str]` (keywords) | `list[Paper]` |
| seed_candidate_detection | `list[Paper]` | `list[SeedCandidate]` |
| seed_user_validation | `list[SeedCandidate]` | `list[SeedCandidate]` (with `validated` set) |
| citation_expansion | `list[Paper]`, `list[SeedCandidate]` (validated) | `list[Paper]` (expanded dataset) |
| dataset_filtering | `list[Paper]` | `list[Paper]` (filtered) |
| field_maturity_assessment | `list[Paper]` | `FieldMaturity` |
| field_maturity_confirmation | `FieldMaturity` | `FieldMaturity` (with possible user override) |
| adaptive_clustering | `list[Paper]`, `FieldMaturity` | `list[Cluster]` |
| topic_labeling | `list[Cluster]`, `list[Paper]` | `list[Cluster]` (with labels populated) |
| field_evolution_analysis | `list[Paper]`, `list[Cluster]` | `list[EvolutionEvent]` |
| report_generation | all above outputs | report files |

---

# Configuration Reference

## File Format and Location

Configuration uses TOML format.

Resolution order (first found wins):

1. path passed via `--config` CLI flag
2. `fieldscope.toml` in the current working directory
3. built-in defaults (LLM-free mode)

## Secrets and API Keys

API keys and secrets must be provided via environment variables, never in config files.

| Environment Variable | Purpose |
|---------------------|---------|
| `FIELDSCOPE_LLM_API_KEY` | API key for the default LLM provider |
| `FIELDSCOPE_LLM_KW_API_KEY` | API key override for keyword expansion LLM (falls back to default) |
| `FIELDSCOPE_LLM_LABEL_API_KEY` | API key override for topic labeling LLM (falls back to default) |
| `FIELDSCOPE_LLM_REPORT_API_KEY` | API key override for report writing LLM (falls back to default) |
| `FIELDSCOPE_EMBEDDING_API_KEY` | API key for BYOK embedding provider (not needed for local models) |
| `OPENALEX_EMAIL` | Polite pool email for OpenAlex (increases rate limit) |
| `CROSSREF_MAILTO` | Mailto header for Crossref API etiquette |
| `SEMANTIC_SCHOLAR_API_KEY` | Semantic Scholar API key (optional) |

## LLM Configuration

The default `[llm]` section is empty. When no LLM is configured, the pipeline operates in LLM-free mode:

* keyword expansion is skipped (user must supply manual keywords)
* topic labeling uses extractive methods only
* narrative report generation is disabled

When the user attempts an LLM-assisted feature without a configured provider, the pipeline must print a clear message explaining what to configure and exit gracefully — not crash or silently skip.

### Provider abstraction

All LLM access goes through an OpenAI-compatible chat completions interface. This covers:

* OpenAI directly
* Anthropic via OpenAI-compatible proxy or adapter
* Self-hosted servers (Ollama, vLLM, llama.cpp) that expose OpenAI-compatible endpoints

The config specifies `base_url` and `model` — no vendor-specific client libraries.

### Stage-specific LLM overrides

Each LLM-assisted stage can optionally override the default `[llm]` settings. If a stage-specific section is absent, it inherits from `[llm]`.

## Configuration Keys

```toml
# =============================================================================
# LLM Configuration
# =============================================================================

[llm]
# Default LLM provider. Leave empty for LLM-free mode.
# All LLM access uses the OpenAI-compatible chat completions interface.
# provider = "openai"           # Label only, for user reference
# base_url = "https://api.openai.com/v1"
# model = "gpt-4o-mini"
# temperature = 0.3
# max_tokens = 2048

# Stage-specific overrides (optional, inherits from [llm] if absent)

[llm.keyword_expansion]
# base_url = "https://api.openai.com/v1"
# model = "gpt-4o-mini"
# temperature = 0.5              # slightly higher for creative expansion

[llm.topic_labeling]
# base_url = "https://api.openai.com/v1"
# model = "gpt-4o-mini"
# temperature = 0.2              # lower for consistent labeling

[llm.reporting]
# base_url = "https://api.openai.com/v1"
# model = "gpt-4o"               # more capable model for narrative writing
# temperature = 0.4
# max_tokens = 4096

# =============================================================================
# Embedding Configuration
# =============================================================================

[embedding]
provider = "sentence-transformers"   # "sentence-transformers" or "openai-compatible"
model = "all-MiniLM-L6-v2"          # default local model
# base_url = ""                      # only for openai-compatible provider
dimensions = 384                     # must match the chosen model
batch_size = 64
cache_dir = ".fieldscope/embeddings" # relative to run output directory
normalize = true                     # L2-normalize vectors before similarity computation
text_fields = ["title", "abstract"]  # concatenated for embedding; falls back to title-only if abstract is missing

# =============================================================================
# Retrieval Configuration
# =============================================================================

[retrieval]
primary_source = "openalex"          # "openalex", "crossref", "semantic_scholar"
max_results_per_query = 1000
rate_limit_rps = 10                  # requests per second (OpenAlex: 10 unauth, 100 polite)
retry_max_attempts = 3
retry_backoff_base = 2.0             # exponential backoff base in seconds

# =============================================================================
# Seed Detection Configuration
# =============================================================================

[seeds]
methods = ["citation_count", "pagerank", "centroid_proximity"]
top_k = 15                           # number of seed candidates to suggest
auto_accept = false                  # skip user validation if true

# =============================================================================
# Citation Expansion Configuration
# =============================================================================

[citation_expansion]
max_depth = 2                        # maximum citation distance from seed
max_papers_per_seed = 500            # limit per seed to avoid explosion
directions = ["references", "cited_by"]

# =============================================================================
# Dataset Filtering Configuration
# =============================================================================

[filtering]
semantic_threshold = 0.3             # minimum cosine similarity to seed centroid
keyword_min_overlap = 1              # minimum keyword matches
require_abstract = false             # if true, drop papers without abstracts
require_year = true                  # if true, drop papers without publication year

# =============================================================================
# Field Maturity Configuration
# =============================================================================

[maturity]
auto_accept = false                  # skip user confirmation if true

# =============================================================================
# Clustering Configuration
# =============================================================================

[clustering]
leiden_resolution = 1.0              # resolution parameter for Leiden algorithm
hdbscan_min_cluster_size = 5         # minimum cluster size for HDBSCAN
embedding_similarity_threshold = 0.7 # for embedding similarity graph edges

# =============================================================================
# Field Evolution Configuration
# =============================================================================

[evolution]
window_size_years = 3                # sliding window size
window_step_years = 1                # sliding window step
overlap_threshold = 0.3              # Jaccard overlap for cluster linking
similarity_threshold = 0.5           # embedding similarity for cluster linking

# =============================================================================
# Report Generation Configuration
# =============================================================================

[reporting]
formats = ["markdown", "json"]       # output formats: "markdown", "json", "csv"
llm_narrative_enabled = false        # generate LLM-refined narrative report
```

## Example: Minimal LLM-Free Configuration

```toml
# fieldscope.toml — LLM-free mode
# No [llm] section needed. Pipeline uses manual keywords and extractive labels.

[embedding]
provider = "sentence-transformers"
model = "all-MiniLM-L6-v2"
dimensions = 384

[seeds]
auto_accept = true

[maturity]
auto_accept = true
```

## Example: Full Configuration with Ollama

```toml
# fieldscope.toml — local Ollama setup

[llm]
base_url = "http://localhost:11434/v1"
model = "llama3.1:8b"
temperature = 0.3
max_tokens = 2048

[llm.reporting]
model = "llama3.1:70b"
temperature = 0.4
max_tokens = 4096

[embedding]
provider = "sentence-transformers"
model = "all-MiniLM-L6-v2"
dimensions = 384

[retrieval]
primary_source = "openalex"
max_results_per_query = 2000
```

## Example: BYOK with OpenAI

```toml
# fieldscope.toml — OpenAI BYOK
# Set FIELDSCOPE_LLM_API_KEY in your environment

[llm]
base_url = "https://api.openai.com/v1"
model = "gpt-4o-mini"
temperature = 0.3

[llm.reporting]
model = "gpt-4o"
max_tokens = 4096

[embedding]
provider = "openai-compatible"
base_url = "https://api.openai.com/v1"
model = "text-embedding-3-small"
dimensions = 1536
```

---

# CLI Interface

## Entry Point

`fieldscope` (installed via `pyproject.toml` `[project.scripts]`) or `python -m fieldscope`.

## Commands

### `fieldscope run <query>`

Run the full pipeline for a research field query.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--config` | PATH | `./fieldscope.toml` | Path to config file |
| `--auto-accept` | flag | off | Skip all user validation checkpoints |
| `--manual-keywords` | flag | off | Enter manual keyword mode (skips LLM expansion) |
| `--output-dir` | PATH | `./fieldscope_output/` | Output directory |
| `--from-stage` | TEXT | — | Start from a specific stage (requires prior checkpoint data in output dir) |

Example usage:

```bash
# LLM-assisted run with user validation
fieldscope run "AI in quantum magnetism"

# Fully automated run, no prompts
fieldscope run "spintronics" --auto-accept

# Manual keywords, custom output
fieldscope run "altermagnetism" --manual-keywords --output-dir ./altermag_analysis/

# Resume from a specific stage (reuses prior stage outputs)
fieldscope run "AI in quantum magnetism" --from-stage adaptive_clustering --output-dir ./fieldscope_output/20260312_143022_ai_quantum/
```

### `fieldscope resume <run_id>`

Resume an interrupted or failed run from the last completed stage.

The `run_id` corresponds to a subdirectory under the output directory.

```bash
fieldscope resume 20260312_143022_ai_quantum
fieldscope resume 20260312_143022_ai_quantum --output-dir ./custom_output/
```

### `fieldscope status <run_id>`

Display the pipeline state for a run: completed stages, current stage, paper counts, and output paths.

```bash
fieldscope status 20260312_143022_ai_quantum
```

### `fieldscope export <run_id>`

Re-export results from a completed run in different formats without re-running the pipeline.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--format` | TEXT | `markdown` | Output format: `markdown`, `json`, `csv` |

```bash
fieldscope export 20260312_143022_ai_quantum --format json
```

### `fieldscope init`

Generate a default `fieldscope.toml` with all keys commented out and documented in the current directory. Intended as a starting point for configuration.

```bash
fieldscope init
```

## Interactive Validation Checkpoints

When `--auto-accept` is not set, the pipeline pauses at two stages:

### Seed User Validation

The pipeline displays a `rich` table of seed candidates with columns: rank, title, year, citation count, composite score, and per-method scores. The user is prompted with:

* `[a]ccept` — accept the suggested seed
* `[r]eject` — reject the seed
* `[A]ccept all` — accept all remaining suggestions
* `[m]anual add` — enter a DOI or title to add as a seed manually
* `[d]one` — finish seed selection and proceed

The pipeline iterates through candidates one by one. The user must accept at least one seed to proceed.

### Field Maturity Confirmation

The pipeline displays a `rich` panel with computed maturity metrics (growth rate, citation density, keyword burst score, median age) and the suggested classification (`emerging` / `growing` / `mature`). The user is prompted with:

* `[a]ccept` — accept the suggested classification
* `[o]verride` — manually select a different classification

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Configuration error (missing config, invalid keys) |
| 2 | LLM required but not configured |
| 3 | Retrieval failure (API errors after retries exhausted) |
| 4 | No results (zero papers retrieved for all keywords) |
| 5 | User abort (Ctrl+C or explicit quit at validation) |
| 10 | Internal error |

---

# Stage API Contracts

Each pipeline stage is implemented as a module under `src/fieldscope/stages/`. Every stage exposes a primary function with the signature documented below.

All stage functions receive a typed config dataclass (extracted from the relevant TOML section) rather than raw dicts.

## Stage 1: Keyword Expansion

```
Module: fieldscope.stages.keyword_expansion
```

```python
def expand_keywords(
    query: str,
    config: LlmConfig,
) -> list[str]
```

* **Pure**: no (LLM API call)
* **Side effects**: none (no file I/O)
* **Fallback**: if LLM is not configured, this function is not called; the pipeline uses manual keywords instead

```python
def prompt_manual_keywords() -> list[str]
```

* Interactive terminal prompt via `rich`; asks user for 5–15 keywords

## Stage 2: Initial Retrieval

```
Module: fieldscope.stages.retrieval
```

```python
async def retrieve_papers(
    keywords: list[str],
    config: RetrievalConfig,
) -> list[Paper]
```

* **Pure**: no (external API calls)
* **Side effects**: HTTP requests to OpenAlex/Crossref; writes checkpoint
* **Notes**: handles pagination, deduplication, metadata normalization internally

## Stage 3: Seed Candidate Detection

```
Module: fieldscope.stages.seeds
```

```python
def detect_seed_candidates(
    papers: list[Paper],
    config: SeedsConfig,
) -> list[SeedCandidate]
```

* **Pure**: yes (computed from paper data and embeddings)
* **Side effects**: none
* **Notes**: computes citation graph internally for PageRank; uses pre-computed embeddings for centroid proximity

## Stage 4: Seed User Validation

```
Module: fieldscope.stages.seed_validation
```

```python
def validate_seeds(
    candidates: list[SeedCandidate],
    papers: list[Paper],
    auto_accept: bool,
) -> list[SeedCandidate]
```

* **Pure**: no (user interaction)
* **Side effects**: terminal I/O via `rich`
* **Notes**: if `auto_accept=True`, marks all candidates as validated and returns immediately

## Stage 5: Citation Expansion

```
Module: fieldscope.stages.citation_expansion
```

```python
async def expand_citations(
    papers: list[Paper],
    validated_seeds: list[SeedCandidate],
    config: CitationExpansionConfig,
) -> list[Paper]
```

* **Pure**: no (external API calls)
* **Side effects**: HTTP requests; writes checkpoint
* **Notes**: returns the full expanded dataset (original + newly retrieved papers, deduplicated)

## Stage 6: Dataset Filtering

```
Module: fieldscope.stages.filtering
```

```python
def filter_dataset(
    papers: list[Paper],
    keywords: list[str],
    config: FilteringConfig,
) -> list[Paper]
```

* **Pure**: yes
* **Side effects**: none
* **Notes**: applies semantic, keyword, venue, and metadata filters in sequence

## Stage 7: Field Maturity Assessment

```
Module: fieldscope.stages.maturity
```

```python
def assess_maturity(
    papers: list[Paper],
) -> FieldMaturity
```

* **Pure**: yes
* **Side effects**: none

## Stage 8: Field Maturity Confirmation

```
Module: fieldscope.stages.maturity
```

```python
def confirm_maturity(
    maturity: FieldMaturity,
    auto_accept: bool,
) -> FieldMaturity
```

* **Pure**: no (user interaction)
* **Side effects**: terminal I/O via `rich`

## Stage 9: Adaptive Clustering

```
Module: fieldscope.stages.clustering
```

```python
def cluster_papers(
    papers: list[Paper],
    maturity: FieldMaturity,
    config: ClusteringConfig,
) -> list[Cluster]
```

* **Pure**: yes
* **Side effects**: none
* **Notes**: selects graph type and algorithm based on `maturity.classification`

## Stage 10: Topic Labeling

```
Module: fieldscope.stages.labeling
```

```python
def label_clusters(
    clusters: list[Cluster],
    papers: list[Paper],
    config: LlmConfig | None,
) -> list[Cluster]
```

* **Pure**: no (optional LLM call)
* **Side effects**: LLM API call if config is provided
* **Notes**: always generates extractive labels first; LLM refinement is applied only if `config` is not None

## Stage 11: Field Evolution Analysis

```
Module: fieldscope.stages.evolution
```

```python
def analyze_evolution(
    papers: list[Paper],
    clusters: list[Cluster],
    config: EvolutionConfig,
) -> list[EvolutionEvent]
```

* **Pure**: yes
* **Side effects**: none
* **Notes**: performs temporal sliding-window clustering internally and links clusters across windows

## Stage 12: Report Generation

```
Module: fieldscope.stages.reporting
```

```python
def generate_reports(
    query: str,
    keywords: list[str],
    papers: list[Paper],
    seeds: list[SeedCandidate],
    maturity: FieldMaturity,
    clusters: list[Cluster],
    events: list[EvolutionEvent],
    config: ReportingConfig,
    llm_config: LlmConfig | None,
    output_dir: Path,
) -> list[Path]
```

* **Pure**: no (file I/O, optional LLM call)
* **Side effects**: writes report files to `output_dir`; optional LLM API call for narrative report
* **Returns**: list of paths to generated report files

## Pipeline Orchestrator

```
Module: fieldscope.pipeline
```

```python
async def run_pipeline(
    query: str,
    config: FieldscopeConfig,
    output_dir: Path,
    auto_accept: bool = False,
    manual_keywords: bool = False,
    from_stage: str | None = None,
) -> PipelineState
```

* Sequences all stages in order
* Manages checkpointing between stages
* Handles `from_stage` resume by loading prior checkpoint data
* Updates `PipelineState` after each stage completes

---

# Graph Construction Details

The pipeline constructs several types of graphs. All graphs are built using `networkx` for general operations and converted to `igraph` when Leiden clustering is needed.

## Direct Citation Graph

* **Nodes**: papers (keyed by `paper_id`)
* **Edges**: directed, from citing paper to cited paper
* **Weight**: unweighted (binary edges)
* **Construction**: iterate over each paper's `references` list; add edge if the referenced paper exists in the dataset
* **Used by**: seed detection (PageRank), mature-field clustering, evolution analysis (citation flow)

Papers with missing or empty `references` lists are included as isolated nodes.

## Bibliographic Coupling Graph

Two papers are coupled if they share references.

* **Nodes**: papers
* **Edges**: undirected
* **Weight**: number of shared references, normalized by the minimum reference list length of the two papers: `w(A, B) = |refs(A) ∩ refs(B)| / min(|refs(A)|, |refs(B)|)`
* **Threshold**: edges with weight < 0.05 are dropped
* **Used by**: emerging-field clustering, growing-field clustering

Papers with no references are excluded from this graph (they cannot be coupled).

## Co-Citation Graph

Two papers are co-cited if they appear together in the reference list of a third paper.

* **Nodes**: papers
* **Edges**: undirected
* **Weight**: number of papers that co-cite the pair: `w(A, B) = |{P : A ∈ refs(P) ∧ B ∈ refs(P)}|`
* **Threshold**: edges with weight < 2 are dropped (require at least 2 co-citations)
* **Used by**: mature-field clustering

## Embedding Similarity Graph

* **Nodes**: papers (only those with non-null embeddings)
* **Edges**: undirected
* **Weight**: cosine similarity between embedding vectors
* **Construction**: k-nearest-neighbors with k = 15, then symmetrize; alternatively, threshold-based with `clustering.embedding_similarity_threshold` from config
* **Default method**: k-NN (more robust than threshold for varying embedding distributions)
* **Used by**: emerging-field clustering, seed detection (centroid proximity)

Papers without embeddings (missing abstracts and title-only embedding disabled) are excluded.

## Graph Selection by Maturity

| Maturity | Primary Graph | Secondary Graph | Algorithm |
|----------|--------------|-----------------|-----------|
| emerging | bibliographic coupling | embedding similarity | Leiden + HDBSCAN (ensemble) |
| growing | bibliographic coupling | direct citation | Leiden |
| mature | direct citation | co-citation | Leiden |

For the **emerging** ensemble strategy:

1. Run Leiden on the bibliographic coupling graph
2. Run HDBSCAN on the embedding space
3. Use the Leiden clusters as the primary partition
4. Flag papers where HDBSCAN and Leiden disagree for review in the cluster output

For **growing** and **mature**:

1. Build the primary graph
2. Add edges from the secondary graph with weight scaled by 0.5 to create a combined graph
3. Run Leiden on the combined graph

---

# Checkpointing and Persistence

## Run Directory Structure

Each pipeline run produces a self-contained output directory:

```
fieldscope_output/
└── <run_id>/
    ├── state.json                          # PipelineState (updated after each stage)
    ├── config_snapshot.toml                # Frozen config at run start
    ├── 01_keywords/
    │   └── keywords.json                   # list[str]
    ├── 02_retrieval/
    │   └── papers.jsonl                    # list[Paper], one JSON object per line
    ├── 03_seeds/
    │   └── seed_candidates.json            # list[SeedCandidate]
    ├── 04_seed_validation/
    │   └── validated_seeds.json            # list[SeedCandidate] (with validated set)
    ├── 05_citation_expansion/
    │   └── papers_expanded.jsonl           # list[Paper] (full expanded dataset)
    ├── 06_filtering/
    │   └── papers_filtered.jsonl           # list[Paper] (filtered dataset)
    ├── 07_maturity/
    │   └── maturity.json                   # FieldMaturity
    ├── 08_maturity_confirmation/
    │   └── maturity_confirmed.json         # FieldMaturity (with possible override)
    ├── 09_clustering/
    │   └── clusters.json                   # list[Cluster]
    ├── 10_labeling/
    │   └── clusters_labeled.json           # list[Cluster] (with labels)
    ├── 11_evolution/
    │   └── events.json                     # list[EvolutionEvent]
    └── 12_reports/
        ├── summary_report.md
        ├── methods_report.md
        ├── full_report.md
        ├── data_export.json
        └── narrative_report.md             # only if LLM narrative enabled
```

## Run ID Format

`<YYYYMMDD>_<HHMMSS>_<query_slug>`

Where `query_slug` is the user query lowercased, whitespace replaced with underscores, truncated to 40 characters. Example: `20260312_143022_ai_in_quantum_magnetism`.

## File Formats

* **JSONL** (`.jsonl`): for lists of Paper objects — one JSON object per line. Chosen over plain JSON for large datasets (streaming reads, append-friendly).
* **JSON** (`.json`): for single objects or small lists (config, seeds, clusters, maturity, events).
* **TOML** (`.toml`): for the config snapshot only.
* **Markdown** (`.md`): for report files.

All JSON/JSONL output uses pydantic's `.model_dump(mode="json")` for serialization.

## Checkpointing Behavior

* After each stage completes successfully, its output is written to the stage subdirectory and `state.json` is updated.
* If the pipeline is interrupted (crash, Ctrl+C, API failure), the last completed stage is recorded in `state.json`.
* `fieldscope resume` reads `state.json` and restarts from the next incomplete stage, loading prior stage outputs from disk.

## Stage Invalidation

When using `--from-stage` to re-run from a specific point:

* The specified stage and all downstream stages are re-executed.
* Prior outputs for re-executed stages are overwritten.
* Upstream stage outputs are loaded from disk and assumed valid.

No automatic dependency tracking or hash-based invalidation — the user explicitly chooses where to restart.

---

# Embedding Configuration

Configuration keys are defined in the Configuration Reference (`[embedding]` section). This section specifies behavioral details.

## When Embeddings Are Computed

Embeddings are computed once during the pipeline, after initial retrieval (stage 2), and reused throughout. If citation expansion adds new papers, embeddings are computed for the new papers only.

Embedding computation is not a separate pipeline stage — it is an infrastructure operation called by stages that need it (seed detection, filtering, clustering, evolution).

## Text Input

Text for embedding is constructed by concatenating the configured `text_fields` (default: `["title", "abstract"]`) separated by a period and space.

Fallback: if a paper has no abstract, embed title only. If title-only embedding produces poor results, the paper is still included — downstream filtering can remove low-quality records.

## Caching

Embeddings are cached in the run's `cache_dir` (default: `.fieldscope/embeddings/` relative to the run output directory).

Cache key: `paper_id` + embedding model name + text fields hash.

Cache format: a single `.npz` file (NumPy compressed archive) mapping `paper_id` → embedding vector. This avoids per-paper file overhead.

If the embedding model or text fields change between runs, the cache is invalidated (model name and text fields are part of the cache metadata).

## Provider Behavior

### `sentence-transformers` (default)

* Loads the model locally via the `sentence-transformers` library
* Batches papers according to `batch_size`
* Runs on CPU by default; uses GPU if available via PyTorch device detection

### `openai-compatible`

* Sends batched requests to the configured `base_url`
* Uses `FIELDSCOPE_EMBEDDING_API_KEY` for authentication
* Respects rate limits via the same retry/backoff logic as retrieval

---

# Error Handling and API Resilience

## Retry Strategy

All external API calls (scholarly APIs, LLM providers, embedding APIs) use the same retry pattern:

* Maximum attempts: `retrieval.retry_max_attempts` (default: 3)
* Backoff: exponential with base `retrieval.retry_backoff_base` (default: 2.0 seconds)
* Jitter: random 0–1 second added to each backoff interval
* Retryable errors: HTTP 429 (rate limit), 500, 502, 503, 504, connection timeouts

Non-retryable errors (400, 401, 403, 404) fail immediately with a descriptive error message.

## Rate Limiting

| API | Default Rate | With Auth/Polite Pool |
|-----|--------------|-----------------------|
| OpenAlex | 10 req/s | 100 req/s (set `OPENALEX_EMAIL`) |
| Crossref | 50 req/s | 50 req/s (set `CROSSREF_MAILTO` for priority) |
| Semantic Scholar | 1 req/s | 10 req/s (set `SEMANTIC_SCHOLAR_API_KEY`) |

The pipeline enforces rate limits using a token-bucket or simple sleep-based throttle before each request. The rate is configurable via `retrieval.rate_limit_rps`.

## Graceful Degradation

| Scenario | Behavior |
|----------|----------|
| LLM provider unreachable | Print error, suggest checking config; exit with code 2 |
| Zero papers retrieved for a keyword | Log warning, continue with remaining keywords |
| Zero papers retrieved for all keywords | Exit with code 4 and a message listing the attempted queries |
| Citation expansion hits rate limit | Retry with backoff; if exhausted, save partial results and checkpoint |
| Embedding model fails to load | Exit with descriptive error (missing model, CUDA issue, etc.) |
| Paper has no abstract | Embed title only; log info-level message |
| Paper has no DOI and no OpenAlex ID | Skip paper with warning (cannot assign `paper_id`) |

## Partial Progress

If the pipeline fails mid-stage, completed stages are preserved in checkpoints. The failing stage's partial output is discarded (not written to checkpoint). The user can fix the issue and `resume`.

---

# Output Specification

All outputs are written to the `12_reports/` subdirectory of the run directory.

## Summary Report (`summary_report.md`)

A concise overview intended for quick reading.

Sections:

1. **Field Definition** — query, expanded keywords, number of papers in final dataset
2. **Dataset Construction** — retrieval sources, paper counts at each stage (retrieved → expanded → filtered)
3. **Seed Papers** — table of validated seeds (title, year, citation count)
4. **Field Maturity** — classification and key metrics
5. **Cluster Overview** — table of clusters (label, size, top keywords)
6. **Key Findings** — top 3–5 clusters by size, notable evolution events

## Methods Report (`methods_report.md`)

Documents the pipeline configuration and methods for reproducibility.

Sections:

1. **Configuration** — config snapshot (all non-secret keys)
2. **Keyword Expansion** — method used (LLM or manual), full keyword list
3. **Retrieval** — sources queried, result counts per source
4. **Seed Selection** — methods used, scoring weights, number of candidates presented vs accepted
5. **Citation Expansion** — depth, directions, papers added
6. **Filtering** — thresholds applied, papers removed at each filter layer
7. **Clustering** — maturity class, graph types used, algorithm parameters, number of clusters
8. **Topic Labeling** — method (extractive / LLM-refined), model used if LLM
9. **Evolution Analysis** — window size, linking thresholds

## Full Report (`full_report.md`)

Combines summary and methods into a single comprehensive document with additional detail:

* Per-cluster detailed analysis (member papers, citation statistics, temporal range)
* Evolution timeline with event descriptions
* Cross-cluster citation flow summary

## Narrative Report (`narrative_report.md`, optional)

Generated only if `reporting.llm_narrative_enabled = true` and an LLM is configured for reporting.

An LLM-written prose version of the full report, structured as a research field review. The LLM receives the structured data from the full report as context and produces a readable narrative.

## Data Export (`data_export.json`)

Structured machine-readable export of all pipeline outputs:

```json
{
  "run_id": "...",
  "query": "...",
  "keywords": ["..."],
  "papers": [...],
  "seeds": [...],
  "maturity": {...},
  "clusters": [...],
  "evolution_events": [...],
  "statistics": {
    "total_retrieved": 0,
    "total_expanded": 0,
    "total_filtered": 0,
    "num_clusters": 0,
    "num_evolution_events": 0
  }
}
```

Note: the `papers` array in the data export excludes embedding vectors to keep file size manageable. Embeddings are available in the cached `.npz` file.

## CSV Export (optional, via `fieldscope export --format csv`)

When CSV format is requested:

* `papers.csv` — one row per paper (doi, title, year, venue, citation_count, cluster_id, cluster_label)
* `clusters.csv` — one row per cluster (cluster_id, label, size, top_keywords)
* `evolution_events.csv` — one row per event

## Visualizations

The report generation stage produces the following plots saved as PNG files in `12_reports/`:

* `cluster_sizes.png` — bar chart of cluster sizes
* `publication_timeline.png` — publication count per year, colored by cluster
* `evolution_timeline.png` — Sankey-style or alluvial diagram showing cluster evolution across time windows
* `citation_network.png` — force-directed layout of the citation graph, nodes colored by cluster (limited to top 200 papers by citation count if dataset exceeds 500 papers)

All plots use `matplotlib` with a clean, publication-ready style.

---

# Testing Strategy

## Test Structure

```
tests/
├── conftest.py                  # shared fixtures
├── fixtures/
│   ├── sample_papers.json       # ~50 papers with known structure
│   ├── sample_config.toml       # test config
│   └── api_responses/           # recorded API responses for mocking
│       ├── openalex_search.json
│       ├── openalex_cited_by.json
│       └── crossref_works.json
├── unit/
│   ├── test_models.py           # data model validation, paper_id computation
│   ├── test_keyword_expansion.py
│   ├── test_retrieval.py
│   ├── test_seeds.py
│   ├── test_filtering.py
│   ├── test_maturity.py
│   ├── test_clustering.py
│   ├── test_labeling.py
│   ├── test_evolution.py
│   └── test_reporting.py
├── integration/
│   └── test_pipeline.py         # end-to-end with mocked APIs
└── test_cli.py                  # CLI invocation tests via click.testing.CliRunner
```

## Test Fixtures

A fixture corpus of ~50 papers with:

* Known citation relationships (a small directed graph)
* Pre-computed embeddings (384-dimensional, deterministic)
* A mix of fields/years to test maturity detection
* At least 3 natural clusters

This fixture enables deterministic testing of seed detection, clustering, and evolution without API calls.

## Unit Tests

Each stage module has unit tests that:

* Test the primary function with fixture data
* Verify input/output types match the inter-stage contracts
* Test edge cases (empty input, single paper, papers with missing fields)
* For API-calling stages: use `pytest-httpx` to mock HTTP responses with recorded fixtures

## Integration Test

A single end-to-end test (`test_pipeline.py`) that:

1. Mocks all external APIs with fixture responses
2. Runs `run_pipeline()` with `auto_accept=True`
3. Verifies all 12 stages complete
4. Checks that output files exist in the expected directory structure
5. Validates that `state.json` records all stages as completed

## CLI Tests

Use `click.testing.CliRunner` to test:

* `fieldscope run` with `--auto-accept` and mocked APIs
* `fieldscope init` generates a valid TOML file
* `fieldscope status` with a pre-built run directory
* Invalid arguments produce correct exit codes

## Test Configuration

`pyproject.toml` test settings:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
```

---

# Logging and Progress Reporting

## Logging Library

Python standard library `logging` with `rich.logging.RichHandler` for formatted terminal output.

## Log Levels

| Level | Usage |
|-------|-------|
| `DEBUG` | Detailed internal state (paper counts per API page, individual filter decisions) |
| `INFO` | Stage transitions, summary statistics (papers retrieved, filtered, clustered) |
| `WARNING` | Non-fatal issues (paper missing abstract, zero results for one keyword, rate limit hit) |
| `ERROR` | Stage failures, API errors after retries exhausted |

Default log level: `INFO`.

Configurable via `--verbose` (sets `DEBUG`) and `--quiet` (sets `WARNING`) CLI flags.

## Progress Reporting

Long-running stages display progress using `rich.progress`:

| Stage | Progress Display |
|-------|-----------------|
| initial_retrieval | Progress bar: keywords processed / total keywords |
| citation_expansion | Progress bar: seeds expanded / total seeds |
| embedding computation | Progress bar: papers embedded / total papers |
| dataset_filtering | Spinner with paper count updates |

## Stage Summary

After each stage completes, a brief summary line is logged at `INFO` level:

```
[02/12] initial_retrieval: 2,847 papers retrieved from OpenAlex (12 keywords, 38s)
[06/12] dataset_filtering: 2,847 → 1,203 papers (semantic: -891, keyword: -412, metadata: -341)
[09/12] adaptive_clustering: 1,203 papers → 8 clusters (Leiden, growing maturity)
```

---

# Dependency Manifest

The `pyproject.toml` file serves as the single source of truth for project metadata and dependencies.

```toml
[project]
name = "fieldscope"
version = "0.1.0"
description = "Bibliometric analysis toolkit for research field definition and evolution"
requires-python = ">=3.10,<3.13"
license = {text = "MIT"}

dependencies = [
    "pydantic>=2.0",
    "networkx>=3.0",
    "python-igraph>=0.11",
    "leidenalg>=0.10",
    "hdbscan>=0.8.33",
    "scikit-learn>=1.3",
    "sentence-transformers>=2.2",
    "keybert>=0.8",
    "httpx>=0.25",
    "click>=8.1",
    "rich>=13.0",
    "matplotlib>=3.7",
    "numpy>=1.24",
    "tomli>=2.0; python_version < '3.11'",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4",
    "pytest-asyncio>=0.21",
    "pytest-httpx>=0.22",
    "ruff>=0.1",
    "mypy>=1.5",
]

[project.scripts]
fieldscope = "fieldscope.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.ruff]
target-version = "py310"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]

[tool.mypy]
python_version = "3.10"
strict = true

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
```

## Source Layout

```
fieldscope/
├── pyproject.toml
├── fieldscope.toml.example          # documented example config (generated by `fieldscope init`)
├── src/
│   └── fieldscope/
│       ├── __init__.py
│       ├── cli.py                   # click CLI entry point
│       ├── config.py                # pydantic config models, TOML loading
│       ├── models.py                # data models (Paper, Cluster, etc.)
│       ├── pipeline.py              # pipeline orchestrator
│       ├── llm/
│       │   ├── __init__.py
│       │   └── client.py            # OpenAI-compatible LLM client
│       ├── embeddings/
│       │   ├── __init__.py
│       │   ├── base.py              # embedding provider abstraction
│       │   ├── sentence_transformers.py
│       │   └── openai_compatible.py
│       └── stages/
│           ├── __init__.py
│           ├── keyword_expansion.py
│           ├── retrieval.py
│           ├── seeds.py
│           ├── seed_validation.py
│           ├── citation_expansion.py
│           ├── filtering.py
│           ├── maturity.py
│           ├── clustering.py
│           ├── labeling.py
│           ├── evolution.py
│           └── reporting.py
└── tests/
    ├── conftest.py
    ├── fixtures/
    ├── unit/
    ├── integration/
    └── test_cli.py
```
