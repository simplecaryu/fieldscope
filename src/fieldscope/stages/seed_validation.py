"""Stage 4: Seed user validation."""

from __future__ import annotations

import logging

from fieldscope.models import Paper, SeedCandidate

logger = logging.getLogger(__name__)


def validate_seeds(
    candidates: list[SeedCandidate],
    papers: list[Paper],
    auto_accept: bool,
) -> list[SeedCandidate]:
    """Validate seed candidates.

    If auto_accept is True, marks all candidates as validated.
    Otherwise, presents an interactive prompt (to be implemented in CLI layer).
    """
    if auto_accept:
        validated = []
        for c in candidates:
            validated.append(
                SeedCandidate(
                    paper_id=c.paper_id,
                    score=c.score,
                    methods=c.methods,
                    rationale=c.rationale,
                    validated=True,
                )
            )
        logger.info("Auto-accepted %d seed candidates", len(validated))
        return validated

    # Interactive validation - delegates to rich UI
    return _interactive_validation(candidates, papers)


def _interactive_validation(
    candidates: list[SeedCandidate],
    papers: list[Paper],
) -> list[SeedCandidate]:
    """Interactive seed validation using rich prompts."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    paper_map = {p.paper_id: p for p in papers}

    validated: list[SeedCandidate] = []

    console.print("\n[bold]Seed Candidate Validation[/bold]")
    console.print(f"Reviewing {len(candidates)} candidates. Commands: [a]ccept, [r]eject, [A]ccept all, [d]one\n")

    accept_all = False
    for i, candidate in enumerate(candidates, 1):
        paper = paper_map.get(candidate.paper_id)

        table = Table(title=f"Candidate {i}/{len(candidates)}")
        table.add_column("Field", style="bold")
        table.add_column("Value")
        table.add_row("Paper ID", candidate.paper_id)
        if paper:
            table.add_row("Title", paper.title)
            table.add_row("Year", str(paper.year or "N/A"))
            table.add_row("Citations", str(paper.citation_count))
            table.add_row("Venue", paper.venue or "N/A")
        table.add_row("Score", f"{candidate.score:.3f}")
        for method, score in candidate.methods.items():
            table.add_row(f"  {method}", f"{score:.3f}")
        table.add_row("Rationale", candidate.rationale)
        console.print(table)

        if accept_all:
            console.print("[green]Auto-accepted[/green]")
            validated.append(
                SeedCandidate(
                    paper_id=candidate.paper_id,
                    score=candidate.score,
                    methods=candidate.methods,
                    rationale=candidate.rationale,
                    validated=True,
                )
            )
            continue

        while True:
            choice = console.input("[a]ccept / [r]eject / [A]ccept all / [d]one > ").strip().lower()
            if choice == "a":
                validated.append(
                    SeedCandidate(
                        paper_id=candidate.paper_id,
                        score=candidate.score,
                        methods=candidate.methods,
                        rationale=candidate.rationale,
                        validated=True,
                    )
                )
                break
            elif choice == "r":
                validated.append(
                    SeedCandidate(
                        paper_id=candidate.paper_id,
                        score=candidate.score,
                        methods=candidate.methods,
                        rationale=candidate.rationale,
                        validated=False,
                    )
                )
                break
            elif choice == "accept all" or choice == "a!" or choice.startswith("A"):
                accept_all = True
                validated.append(
                    SeedCandidate(
                        paper_id=candidate.paper_id,
                        score=candidate.score,
                        methods=candidate.methods,
                        rationale=candidate.rationale,
                        validated=True,
                    )
                )
                break
            elif choice == "d" or choice == "done":
                # Mark remaining as not reviewed
                for remaining in candidates[i:]:
                    validated.append(
                        SeedCandidate(
                            paper_id=remaining.paper_id,
                            score=remaining.score,
                            methods=remaining.methods,
                            rationale=remaining.rationale,
                            validated=False,
                        )
                    )
                accepted = sum(1 for v in validated if v.validated)
                console.print(f"\n[bold]Done.[/bold] {accepted} seeds accepted.")
                return validated
            else:
                console.print("[red]Invalid choice.[/red]")

    accepted = sum(1 for v in validated if v.validated)
    console.print(f"\n[bold]Validation complete.[/bold] {accepted}/{len(validated)} seeds accepted.")
    return validated
