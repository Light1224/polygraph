"""Ingest markets, events, tags from Polymarket public APIs."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from polygraph.client import ComboClient, GammaClient
from polygraph.store import Store

console = Console()

Scope = Literal["active", "all"]


def _gamma_filters(scope: Scope) -> dict[str, str]:
    if scope == "active":
        return {"active": "true", "closed": "false"}
    return {}


def _count_stream(label: str, stream, store_fn, store: Store, *, commit_every: int = 200) -> int:
    """Ingest an API stream with a live counter."""
    count = 0
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(label, total=None)
        for item in stream:
            store_fn(item)
            count += 1
            if count % commit_every == 0:
                store.commit()
                progress.update(task, completed=count, description=f"{label} ({count})")
        store.commit()
        progress.update(task, completed=count, description=f"{label} done ({count})")
    return count


def fetch_all(
    db_path: Path,
    *,
    scope: Scope = "active",
    include_combo: bool = True,
    fresh: bool = False,
) -> Store:
    """Download Polymarket catalog into SQLite."""
    if fresh and db_path.exists():
        db_path.unlink()
        journal = db_path.with_suffix(".db-journal")
        if journal.exists():
            journal.unlink()

    store = Store(db_path)
    store.set_meta("fetch_scope", scope)
    filters = _gamma_filters(scope)

    with GammaClient() as gamma:
        console.print(
            f"[bold]Fetching markets[/bold] "
            f"({'active only' if scope == 'active' else 'all'})…"
        )
        market_count = _count_stream(
            "markets",
            gamma.iter_markets(**filters),
            store.upsert_market,
            store,
        )

        console.print(
            f"[bold]Fetching events[/bold] "
            f"({'active only' if scope == 'active' else 'all'})…"
        )
        event_count = _count_stream(
            "events",
            gamma.iter_events(**filters),
            store.upsert_event,
            store,
        )

        console.print("[bold]Fetching tags[/bold]…")
        tag_count = _count_stream("tags", gamma.iter_tags(), store.upsert_tag, store)

    combo_count = 0
    if include_combo:
        console.print("[bold]Fetching combo-eligible markets[/bold]…")
        with ComboClient() as combo:
            combo_count = _count_stream(
                "combo",
                combo.iter_combo_markets(),
                store.upsert_combo,
                store,
            )

    console.print(
        f"[green]Done.[/green] scope={scope} | "
        f"{market_count} markets, {event_count} events, {tag_count} tags, "
        f"{combo_count} combo → [cyan]{db_path}[/cyan]"
    )
    return store
