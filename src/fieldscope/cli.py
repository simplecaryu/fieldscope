"""Click CLI entry point for fieldscope."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from fieldscope.config import load_config
from fieldscope.pipeline import STAGE_ORDER, Pipeline


@click.group()
@click.version_option(package_name="fieldscope")
def main() -> None:
    """fieldscope - Bibliometric analysis toolkit for research field definition and evolution."""


@main.command()
@click.argument("query")
@click.option("--config", "config_path", type=click.Path(exists=False), default=None)
@click.option("--auto-accept", is_flag=True, default=False)
@click.option("--manual-keywords", is_flag=True, default=False)
@click.option("--output-dir", type=click.Path(), default="./fieldscope_output/")
@click.option("--from-stage", default=None)
@click.option("--dry-run", is_flag=True, default=False, help="Show planned stages without executing.")
def run(
    query: str,
    config_path: str | None,
    auto_accept: bool,
    manual_keywords: bool,
    output_dir: str,
    from_stage: str | None,
    dry_run: bool,
) -> None:
    """Run the full pipeline for a research field query."""
    cfg_path = Path(config_path) if config_path else Path("fieldscope.toml")
    config = load_config(cfg_path)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    try:
        pipeline = Pipeline(query=query, config=config, output_dir=out)
    except Exception as e:
        click.echo(f"Error creating pipeline: {e}", err=True)
        sys.exit(1)

    try:
        stages = pipeline.get_stages_to_run(from_stage=from_stage)
    except ValueError as e:
        click.echo(str(e), err=True)
        sys.exit(1)

    if dry_run:
        click.echo(f"Run ID: {pipeline.state.run_id}")
        click.echo(f"Query: {query}")
        click.echo(f"Stages to run ({len(stages)}):")
        for i, stage in enumerate(stages, 1):
            click.echo(f"  {i}. {stage}")
        return

    # Actual pipeline execution will be implemented in later phases
    click.echo(f"Pipeline execution not yet implemented. Run ID: {pipeline.state.run_id}")


@main.command()
def init() -> None:
    """Generate a default fieldscope.toml in the current directory."""
    toml_path = Path("fieldscope.toml")
    if toml_path.exists():
        click.echo("fieldscope.toml already exists in current directory.", err=True)
        sys.exit(1)

    template = """\
# fieldscope configuration
# See documentation for all available options.

# [llm]
# base_url = "https://api.openai.com/v1"
# model = "gpt-4o-mini"
# temperature = 0.3
# max_tokens = 2048

[embedding]
provider = "sentence-transformers"
model = "all-MiniLM-L6-v2"
dimensions = 384

[retrieval]
primary_source = "openalex"
max_results_per_query = 1000

[seeds]
top_k = 15
auto_accept = false

[citation_expansion]
max_depth = 2

[filtering]
semantic_threshold = 0.3

[maturity]
auto_accept = false

[clustering]
leiden_resolution = 1.0

[evolution]
window_size_years = 3

[reporting]
formats = ["markdown", "json"]
llm_narrative_enabled = false
"""
    toml_path.write_text(template)
    click.echo("Created fieldscope.toml")


@main.command()
@click.argument("run_id")
@click.option("--output-dir", type=click.Path(), default="./fieldscope_output/")
def status(run_id: str, output_dir: str) -> None:
    """Display pipeline state for a run."""
    run_dir = Path(output_dir) / run_id
    state_path = run_dir / "state.json"

    if not state_path.exists():
        click.echo(f"Run not found: {run_id}", err=True)
        sys.exit(1)

    data = json.loads(state_path.read_text())
    click.echo(f"Run ID: {data['run_id']}")
    click.echo(f"Query: {data['query']}")
    click.echo(f"Completed stages ({len(data['completed_stages'])}/{len(STAGE_ORDER)}):")
    for stage in data["completed_stages"]:
        click.echo(f"  [done] {stage}")

    if data.get("current_stage"):
        click.echo(f"  [running] {data['current_stage']}")

    remaining = [s for s in STAGE_ORDER if s not in data["completed_stages"] and s != data.get("current_stage")]
    for stage in remaining:
        click.echo(f"  [pending] {stage}")


@main.command()
@click.argument("run_id")
@click.option("--output-dir", type=click.Path(), default="./fieldscope_output/")
def resume(run_id: str, output_dir: str) -> None:
    """Resume an interrupted pipeline run."""
    run_dir = Path(output_dir) / run_id
    state_path = run_dir / "state.json"

    if not state_path.exists():
        click.echo(f"Run not found: {run_id}", err=True)
        sys.exit(1)

    config = load_config(run_dir / "config_snapshot.toml")
    pipeline = Pipeline.resume(run_dir=run_dir, config=config)

    stages = pipeline.get_stages_to_run()
    click.echo(f"Resuming run {run_id} from stage: {stages[0] if stages else 'all completed'}")


@main.command()
@click.argument("run_id")
@click.option("--output-dir", type=click.Path(), default="./fieldscope_output/")
@click.option("--format", "fmt", default="markdown")
def export(run_id: str, output_dir: str, fmt: str) -> None:
    """Re-export results from a completed run."""
    run_dir = Path(output_dir) / run_id
    if not run_dir.exists():
        click.echo(f"Run not found: {run_id}", err=True)
        sys.exit(1)

    click.echo(f"Export not yet implemented for run {run_id} (format: {fmt})")
