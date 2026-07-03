"""Fetch and store CLOB price history for belief dynamics."""

from __future__ import annotations

import time
from pathlib import Path

from rich.console import Console
from rich.progress import Progress

from polygraph.client import ClobClient
from polygraph.config import (
    PRICE_FETCH_DEFAULT_LIMIT,
    PRICE_FETCH_MIN_VOLUME,
    PRICE_FETCH_SLEEP_SEC,
    PRICE_HISTORY_FIDELITY,
    PRICE_HISTORY_INTERVAL,
)
from polygraph.store import Store

console = Console()


def fetch_prices(
    db_path: Path,
    *,
    min_volume: float = PRICE_FETCH_MIN_VOLUME,
    limit: int | None = PRICE_FETCH_DEFAULT_LIMIT,
    skip_fetched: bool = True,
    interval: str = PRICE_HISTORY_INTERVAL,
    fidelity: int = PRICE_HISTORY_FIDELITY,
    sleep_sec: float = PRICE_FETCH_SLEEP_SEC,
) -> dict[str, int]:
    """Download daily price series for liquid markets into SQLite."""
    stats = {"markets": 0, "points": 0, "errors": 0, "skipped": 0}

    with Store(db_path) as store:
        targets = list(
            store.iter_markets_for_prices(
                min_volume=min_volume,
                limit=limit,
                skip_fetched=skip_fetched,
            )
        )
        if not targets:
            console.print("[dim]No markets need price history (already fetched or none qualify).[/dim]")
            return stats

        console.print(
            f"[bold]Prices[/bold] — fetching {len(targets):,} markets "
            f"(vol ≥ {min_volume:,.0f}, interval={interval})…"
        )

        with ClobClient() as clob:
            with Progress() as progress:
                task = progress.add_task("tokens", total=len(targets))
                for m in targets:
                    token = m["token_id_yes"]
                    try:
                        points = clob.get_prices_history(
                            token,
                            interval=interval,
                            fidelity=fidelity,
                        )
                        if points:
                            n = store.upsert_price_points(token, points)
                            stats["points"] += n
                            stats["markets"] += 1
                        else:
                            stats["skipped"] += 1
                    except Exception as exc:
                        stats["errors"] += 1
                        if stats["errors"] <= 3:
                            console.print(f"[yellow]skip {m['id']}:[/yellow] {exc}")
                    progress.advance(task)
                    if sleep_sec:
                        time.sleep(sleep_sec)

        store.set_meta("price_history_interval", interval)
        store.set_meta("price_history_fidelity", str(fidelity))
        store.set_meta("price_history_fetched_at", str(int(time.time())))
        store.commit()

    console.print(
        f"[green]Done[/green] — {stats['markets']:,} tokens, "
        f"{stats['points']:,} points, {stats['errors']} errors"
    )
    return stats
