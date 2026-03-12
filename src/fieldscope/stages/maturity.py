"""Stages 7-8: Field maturity assessment and confirmation."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fieldscope.models import FieldMaturity, Paper

logger = logging.getLogger(__name__)

CURRENT_YEAR = datetime.now(tz=timezone.utc).year


def assess_maturity(papers: list[Paper]) -> FieldMaturity:
    """Assess the maturity of the research field based on paper statistics."""
    if not papers:
        return FieldMaturity(
            classification="emerging",
            metrics={"growth_rate": 0.0, "citation_density": 0.0, "keyword_burst": 0.0, "median_age": 0.0},
            user_override=False,
        )

    # Papers with year
    dated = [p for p in papers if p.year is not None]
    if not dated:
        return FieldMaturity(
            classification="emerging",
            metrics={"growth_rate": 0.0, "citation_density": 0.0, "keyword_burst": 0.0, "median_age": 0.0},
            user_override=False,
        )

    # Compute metrics
    years = sorted(p.year for p in dated)
    ages = [CURRENT_YEAR - y for y in years]
    median_age = float(sorted(ages)[len(ages) // 2])

    # Growth rate: ratio of papers in recent 3 years vs total
    recent_count = sum(1 for y in years if CURRENT_YEAR - y <= 3)
    growth_rate = recent_count / len(dated)

    # Citation density: average citations per paper
    total_citations = sum(p.citation_count for p in papers)
    citation_density = total_citations / len(papers)
    # Normalize to rough [0, 1] scale (100+ avg citations = saturated)
    citation_density_norm = min(citation_density / 100.0, 1.0)

    # Keyword burst: approximated by concentration of recent publications
    # Higher burst = more recent papers relative to history
    if len(years) > 1:
        year_range = max(years) - min(years) + 1
        recent_3yr = sum(1 for y in years if CURRENT_YEAR - y <= 3)
        keyword_burst = (recent_3yr / len(years)) / (3 / max(year_range, 1))
        keyword_burst = min(keyword_burst, 1.0)
    else:
        keyword_burst = 1.0

    metrics = {
        "growth_rate": round(growth_rate, 3),
        "citation_density": round(citation_density_norm, 3),
        "keyword_burst": round(keyword_burst, 3),
        "median_age": round(median_age, 1),
    }

    # Classification logic
    if median_age <= 5 and growth_rate >= 0.5:
        classification = "emerging"
    elif median_age >= 15 and growth_rate <= 0.3:
        classification = "mature"
    else:
        classification = "growing"

    logger.info(
        "Field maturity: %s (median_age=%.1f, growth_rate=%.2f, citation_density=%.2f)",
        classification, median_age, growth_rate, citation_density_norm,
    )

    return FieldMaturity(
        classification=classification,
        metrics=metrics,
        user_override=False,
    )


def confirm_maturity(
    maturity: FieldMaturity,
    auto_accept: bool,
) -> FieldMaturity:
    """Confirm or override the maturity classification."""
    if auto_accept:
        return maturity.model_copy()

    # Interactive confirmation
    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    lines = [
        f"Classification: [bold]{maturity.classification}[/bold]",
        "",
    ]
    for key, val in maturity.metrics.items():
        lines.append(f"  {key}: {val}")

    console.print(Panel("\n".join(lines), title="Field Maturity Assessment"))

    while True:
        choice = console.input("[a]ccept / [o]verride > ").strip().lower()
        if choice == "a" or choice == "accept":
            return maturity.model_copy()
        elif choice == "o" or choice == "override":
            console.print("Options: emerging, growing, mature")
            new_class = console.input("Classification > ").strip().lower()
            if new_class in ("emerging", "growing", "mature"):
                return FieldMaturity(
                    classification=new_class,
                    metrics=maturity.metrics,
                    user_override=True,
                )
            console.print("[red]Invalid classification.[/red]")
        else:
            console.print("[red]Invalid choice.[/red]")
