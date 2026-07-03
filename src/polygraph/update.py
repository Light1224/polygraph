"""Daily refresh — re-fetch catalog, prices, re-infer, rebuild graph."""

from __future__ import annotations

import json
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from rich.console import Console

from polygraph.config import (
    DEFAULT_DB_PATH,
    DEFAULT_GRAPH_PATH,
    PRICE_FETCH_DEFAULT_LIMIT,
    PRICE_FETCH_MIN_VOLUME,
)
from polygraph.fetch import fetch_all
from polygraph.graph import build_graph, graph_stats
from polygraph.store import Store
from polygraph.vector.index import EmbeddingIndex

console = Console()

Scope = Literal["active", "all"]


@dataclass
class UpdateResult:
    started_at: str
    finished_at: str = ""
    duration_sec: float = 0.0
    steps: dict[str, Any] = field(default_factory=dict)
    ok: bool = True
    error: str | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _fetch_step(db_path: Path, scope: Scope, include_combo: bool) -> dict[str, Any]:
    fetch_all(db_path, scope=scope, include_combo=include_combo, fresh=False)
    with Store(db_path) as store:
        return {
            "scope": scope,
            "markets": store.market_count(),
        }


def _step(name: str, fn) -> dict[str, Any]:
    t0 = time.perf_counter()
    try:
        out = fn()
        return {
            "ok": True,
            "duration_sec": round(time.perf_counter() - t0, 2),
            "detail": out if out is not None else {},
        }
    except Exception as exc:
        return {
            "ok": False,
            "duration_sec": round(time.perf_counter() - t0, 2),
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def _save_result(store: Store, result: UpdateResult) -> None:
    store.set_meta("last_update_at", result.finished_at or result.started_at)
    store.set_meta("last_update_ok", "1" if result.ok else "0")
    store.set_meta("last_update_duration_sec", str(round(result.duration_sec, 2)))
    store.set_meta("last_update_result", json.dumps(asdict(result), indent=2))


def run_update(
    db_path: Path = DEFAULT_DB_PATH,
    graph_path: Path = DEFAULT_GRAPH_PATH,
    *,
    scope: Scope = "active",
    include_combo: bool = True,
    refresh_embeddings: bool = True,
    refresh_prices: bool = True,
    price_min_volume: float = PRICE_FETCH_MIN_VOLUME,
    price_limit: int = PRICE_FETCH_DEFAULT_LIMIT,
    price_refetch: bool = True,
    empirical: bool = True,
    use_embeddings: bool = True,
) -> UpdateResult:
    """
    One full refresh cycle:
      fetch → embed (if stale) → prices → infer → build graph
    """
    if not db_path.exists():
        raise FileNotFoundError(
            f"No database at {db_path}. Run `polygraph fetch` once before `polygraph update`."
        )

    started = _utc_now()
    t0 = time.perf_counter()
    result = UpdateResult(started_at=started)

    console.print(f"[bold]Polygraph update[/bold] — started {started}")

    # 1. Re-fetch active catalog (upsert — keeps closed markets already in DB)
    step = _step(
        "fetch",
        lambda: _fetch_step(db_path, scope, include_combo),
    )
    result.steps["fetch"] = step
    if not step["ok"]:
        result.ok = False
        result.error = step.get("error", "fetch failed")
        _finalize(result, db_path, t0)
        return result

    # 2. Embeddings — skipped automatically when market count unchanged
    if refresh_embeddings:
        def _embed():
            from polygraph.infer.context import attach_event_context
            from polygraph.infer.pipeline import _events_as_dicts, _markets_as_dicts

            with Store(db_path) as store:
                markets = _markets_as_dicts(store)
                events = _events_as_dicts(store)
                attach_event_context(markets, events)
                before = store.market_count()
            index = EmbeddingIndex()
            n = index.build(markets, force=False)
            with Store(db_path) as store:
                index.sync_meta(store.path)
            return {"markets": before, "vectors": n}

        step = _step("embed", _embed)
        result.steps["embed"] = step
        if not step["ok"]:
            result.ok = False
            result.error = step.get("error", "embed failed")
            _finalize(result, db_path, t0)
            return result

    # 3. Price history for liquid markets
    if refresh_prices:
        def _prices():
            from polygraph.prices import fetch_prices

            limit = None if price_limit == 0 else price_limit
            stats = fetch_prices(
                db_path,
                min_volume=price_min_volume,
                limit=limit,
                skip_fetched=not price_refetch,
            )
            return stats

        step = _step("prices", _prices)
        result.steps["prices"] = step
        if not step["ok"]:
            result.ok = False
            result.error = step.get("error", "prices failed")
            _finalize(result, db_path, t0)
            return result

    # 4. Re-infer edges
    def _infer():
        from polygraph.infer.pipeline import run_inference_file

        run_inference_file(db_path, empirical=empirical, use_embeddings=use_embeddings)
        with Store(db_path) as store:
            return {"edge_count": store.edge_count()}

    step = _step("infer", _infer)
    result.steps["infer"] = step
    if not step["ok"]:
        result.ok = False
        result.error = step.get("error", "infer failed")
        _finalize(result, db_path, t0)
        return result

    # 5. Rebuild GraphML from SQLite (infer already ran)
    def _build():
        g = build_graph(
            db_path,
            graph_path,
            run_infer=False,
            empirical=empirical,
        )
        return graph_stats(g)

    step = _step("build", _build)
    result.steps["build"] = step
    if not step["ok"]:
        result.ok = False
        result.error = step.get("error", "build failed")
        _finalize(result, db_path, t0)
        return result

    _finalize(result, db_path, t0)
    console.print(
        f"[green]Update complete[/green] in {result.duration_sec:.1f}s "
        f"→ {graph_path}"
    )
    return result


def _finalize(result: UpdateResult, db_path: Path, t0: float) -> None:
    result.finished_at = _utc_now()
    result.duration_sec = round(time.perf_counter() - t0, 2)
    with Store(db_path) as store:
        _save_result(store, result)
    if not result.ok:
        console.print(f"[red]Update failed:[/red] {result.error}")


def run_update_loop(
    db_path: Path = DEFAULT_DB_PATH,
    graph_path: Path = DEFAULT_GRAPH_PATH,
    *,
    interval_hours: float = 24.0,
    max_runs: int = 0,
    **kwargs: Any,
) -> None:
    """Run `run_update` repeatedly. max_runs=0 means run until interrupted."""
    if interval_hours <= 0:
        raise ValueError("interval_hours must be positive")

    run = 0
    console.print(
        f"[bold]Polygraph update loop[/bold] — every {interval_hours:g}h "
        f"(Ctrl+C to stop)"
    )
    while True:
        run += 1
        console.print(f"\n[bold]── Run {run} ──[/bold]")
        try:
            run_update(db_path, graph_path, **kwargs)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            console.print(f"[red]Update run {run} crashed:[/red] {exc}")

        if max_runs and run >= max_runs:
            console.print(f"[dim]Reached max_runs={max_runs}, exiting.[/dim]")
            break

        sleep_sec = interval_hours * 3600
        next_at = datetime.now(timezone.utc).timestamp() + sleep_sec
        next_str = datetime.fromtimestamp(next_at, tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M UTC"
        )
        console.print(f"[dim]Sleeping until {next_str}…[/dim]")
        try:
            time.sleep(sleep_sec)
        except KeyboardInterrupt:
            console.print("\n[dim]Loop stopped.[/dim]")
            break
