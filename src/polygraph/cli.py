"""CLI for polygraph."""

from __future__ import annotations

from pathlib import Path

import click
import networkx as nx
from rich.console import Console
from rich.table import Table

from polygraph.config import DEFAULT_DB_PATH, DEFAULT_GRAPH_PATH
from polygraph.fetch import fetch_all
from polygraph.graph import EDGE_TYPES, build_graph, graph_stats
from polygraph.store import Store
from polygraph.vector.index import EmbeddingIndex

console = Console()


@click.group()
@click.version_option()
def main() -> None:
    """Polygraph — map Polymarket markets as a read-only interaction graph."""


@main.command()
@click.option("--db", "db_path", type=click.Path(path_type=Path), default=DEFAULT_DB_PATH)
@click.option(
    "--scope",
    type=click.Choice(["active", "all"]),
    default="active",
    help="active = open markets/events only; all = full historical crawl",
)
@click.option("--combo/--no-combo", default=True, help="Fetch combo-eligible catalog")
@click.option("--fresh/--no-fresh", default=False, help="Delete existing DB before fetch")
def fetch(db_path: Path, scope: str, combo: bool, fresh: bool) -> None:
    """Download markets, events, and tags from public Polymarket APIs."""
    fetch_all(db_path, scope=scope, include_combo=combo, fresh=fresh)


@main.command()
@click.option("--db", "db_path", type=click.Path(path_type=Path), default=DEFAULT_DB_PATH)
@click.option(
    "--output",
    "output_path",
    type=click.Path(path_type=Path),
    default=DEFAULT_GRAPH_PATH,
)
@click.option(
    "--semantic/--no-semantic",
    default=False,
    help="(legacy) ignored — use --empirical",
)
@click.option("--empirical/--no-empirical", default=True, help="Tier B co-movement edges")
@click.option("--skip-infer/--infer", default=False, help="Skip inference, use cached edges")
@click.option("--active-only/--all", default=True)
@click.option("--model", default="all-MiniLM-L6-v2", help="(legacy) unused")
def build(
    db_path: Path,
    output_path: Path,
    semantic: bool,
    empirical: bool,
    skip_infer: bool,
    active_only: bool,
    model: str,
) -> None:
    """Build belief graph (runs inference pipeline unless --skip-infer)."""
    if not db_path.exists():
        raise click.ClickException(f"No database at {db_path}. Run `polygraph fetch` first.")
    build_graph(
        db_path,
        output_path,
        run_infer=not skip_infer,
        empirical=empirical,
    )


@main.command()
@click.option("--db", "db_path", type=click.Path(path_type=Path), default=DEFAULT_DB_PATH)
@click.option("--force/--no-force", default=False, help="Rebuild even if index is current")
def embed(db_path: Path, force: bool) -> None:
    """Build local embedding index (all-MiniLM-L6-v2) for search + TEMPORAL anchors."""
    if not db_path.exists():
        raise click.ClickException(f"No database at {db_path}. Run `polygraph fetch` first.")
    from polygraph.infer.context import attach_event_context
    from polygraph.infer.pipeline import _events_as_dicts, _markets_as_dicts

    with Store(db_path) as store:
        markets = _markets_as_dicts(store)
        events = _events_as_dicts(store)
        attach_event_context(markets, events)
    index = EmbeddingIndex()
    index.build(markets, force=force)
    with Store(db_path) as store:
        index.sync_meta(store.path)
    console.print(f"[green]Ready[/green] — {index.size:,} vectors indexed")


@main.command()
@click.option("--db", "db_path", type=click.Path(path_type=Path), default=DEFAULT_DB_PATH)
@click.option("--min-volume", default=10_000, type=float, show_default=True)
@click.option("--limit", default=3000, type=int, help="Max markets to fetch (0 = no limit)")
@click.option("--refetch/--skip-fetched", default=False, help="Re-download even if token has data")
def prices(db_path: Path, min_volume: float, limit: int, refetch: bool) -> None:
    """Fetch CLOB daily price history for liquid markets."""
    if not db_path.exists():
        raise click.ClickException(f"No database at {db_path}. Run `polygraph fetch` first.")
    from polygraph.prices import fetch_prices

    fetch_prices(
        db_path,
        min_volume=min_volume,
        limit=None if limit == 0 else limit,
        skip_fetched=not refetch,
    )


@main.command()
@click.option("--db", "db_path", type=click.Path(path_type=Path), default=DEFAULT_DB_PATH)
@click.option("--empirical/--no-empirical", default=True)
@click.option("--embeddings/--no-embeddings", default=True, help="Use local embedding index")
def infer(db_path: Path, empirical: bool, embeddings: bool) -> None:
    """Infer directed edges (GROUND + EMPIRICAL) into SQLite."""
    if not db_path.exists():
        raise click.ClickException(f"No database at {db_path}. Run `polygraph fetch` first.")
    from polygraph.infer.pipeline import run_inference_file

    run_inference_file(db_path, empirical=empirical, use_embeddings=embeddings)


@main.command()
@click.option("--db", "db_path", type=click.Path(path_type=Path), default=DEFAULT_DB_PATH)
def stats(db_path: Path) -> None:
    """Show counts from the local cache."""
    if not db_path.exists():
        raise click.ClickException(f"No database at {db_path}. Run `polygraph fetch` first.")
    with Store(db_path) as store:
        n_markets = store.market_count()
        n_events = sum(1 for _ in store.iter_events())
        n_combo = len(store.combo_condition_ids())
        scope = store.get_meta("fetch_scope") or "unknown"
        ph = store.price_history_count()

    table = Table(title="Polygraph cache")
    table.add_column("Entity")
    table.add_column("Count", justify="right")
    table.add_row("Scope", scope)
    table.add_row("Markets", str(n_markets))
    table.add_row("Events", str(n_events))
    table.add_row("Combo-eligible", str(n_combo))
    if ph:
        table.add_row("Price points", f"{ph:,}")
    console.print(table)


@main.command("edge-types")
def edge_types() -> None:
    """Explain graph edge types."""
    table = Table(title="Edge types")
    table.add_column("Type")
    table.add_column("Meaning")
    for t, desc in EDGE_TYPES.items():
        table.add_row(t, desc)
    console.print(table)


@main.command()
@click.argument("market_id")
@click.option(
    "--graph",
    "graph_path",
    type=click.Path(exists=True, path_type=Path),
    default=DEFAULT_GRAPH_PATH,
)
@click.option("--depth", default=1, help="Hops from seed market")
def neighbors(market_id: str, graph_path: Path, depth: int) -> None:
    """Show connected markets for a given market ID."""
    g = nx.read_graphml(graph_path)
    if market_id not in g:
        raise click.ClickException(f"Market {market_id} not in graph")

    seen = {market_id}
    frontier = {market_id}
    for _ in range(depth):
        nxt: set[str] = set()
        for node in frontier:
            nxt.update(g.neighbors(node))
        nxt -= seen
        seen |= nxt
        frontier = nxt

    seed = g.nodes[market_id]
    console.print(f"[bold]{seed.get('question', market_id)}[/bold] (id={market_id})\n")

    table = Table()
    table.add_column("ID")
    table.add_column("Question")
    table.add_column("Edge types")
    table.add_column("Weight", justify="right")

    for nid in sorted(seen - {market_id}):
        if nid not in g:
            continue
        data = g[market_id][nid] if g.has_edge(market_id, nid) else {}
        # For depth>1, edge to seed may not exist — show node anyway
        if not data and depth == 1:
            continue
        types = ", ".join(data.get("types", [])) if data else "(via hop)"
        node = g.nodes[nid]
        table.add_row(
            nid,
            (node.get("question") or "")[:60],
            types,
            str(round(data.get("weight", 0), 2)) if data else "",
        )
    console.print(table)


@main.command()
@click.option(
    "--graph",
    "graph_path",
    type=click.Path(exists=True, path_type=Path),
    default=DEFAULT_GRAPH_PATH,
)
@click.option("--output", type=click.Path(path_type=Path), default=Path("data/graph.html"))
@click.option("--max-nodes", default=300, help="Limit nodes for browser viz")
def viz(graph_path: Path, output: Path, max_nodes: int) -> None:
    """Export interactive HTML visualization (needs [viz] extra)."""
    try:
        from pyvis.network import Network
    except ImportError as exc:
        raise click.ClickException(
            "Install viz extras: pip install -e '.[viz]'"
        ) from exc

    g = nx.read_graphml(graph_path)
    if g.number_of_nodes() > max_nodes:
        # Keep highest-degree nodes
        top = sorted(g.degree, key=lambda x: x[1], reverse=True)[:max_nodes]
        g = g.subgraph(n[0] for n in top).copy()

    net = Network(height="800px", width="100%", bgcolor="#111", font_color="white")
    net.from_nx(g)
    output.parent.mkdir(parents=True, exist_ok=True)
    net.save_graph(str(output))
    console.print(f"Wrote [cyan]{output}[/cyan] ({g.number_of_nodes()} nodes)")


@main.command()
@click.option("--db", "db_path", type=click.Path(path_type=Path), default=DEFAULT_DB_PATH)
@click.option(
    "--graph",
    "graph_path",
    type=click.Path(path_type=Path),
    default=DEFAULT_GRAPH_PATH,
)
@click.option(
    "--scope",
    type=click.Choice(["active", "all"]),
    default="active",
    show_default=True,
    help="Catalog scope for fetch step",
)
@click.option("--combo/--no-combo", default=True, help="Refresh combo-eligible catalog")
@click.option("--embed/--no-embed", default=True, help="Rebuild embedding index if stale")
@click.option("--prices/--no-prices", default=True, help="Refresh CLOB price history")
@click.option("--price-min-volume", default=10_000, type=float, show_default=True)
@click.option("--price-limit", default=3000, type=int, show_default=True, help="0 = no limit")
@click.option(
    "--price-refetch/--new-prices-only",
    default=True,
    help="Re-download history for tracked tokens (daily refresh)",
)
@click.option("--empirical/--no-empirical", default=True)
@click.option("--embeddings/--no-embeddings", default=True, help="Use embedding index during infer")
@click.option(
    "--loop",
    "run_loop",
    is_flag=True,
    help="Run on a schedule until Ctrl+C",
)
@click.option(
    "--interval-hours",
    default=24.0,
    type=float,
    show_default=True,
    help="Hours between runs (with --loop)",
)
@click.option(
    "--max-runs",
    default=0,
    type=int,
    help="Stop after N runs (0 = infinite, with --loop)",
)
def update(
    db_path: Path,
    graph_path: Path,
    scope: str,
    combo: bool,
    embed: bool,
    prices: bool,
    price_min_volume: float,
    price_limit: int,
    price_refetch: bool,
    empirical: bool,
    embeddings: bool,
    run_loop: bool,
    interval_hours: float,
    max_runs: int,
) -> None:
    """Refresh catalog, prices, edges, and graph (daily maintenance)."""
    from polygraph.update import run_update, run_update_loop

    kwargs = dict(
        scope=scope,
        include_combo=combo,
        refresh_embeddings=embed,
        refresh_prices=prices,
        price_min_volume=price_min_volume,
        price_limit=price_limit,
        price_refetch=price_refetch,
        empirical=empirical,
        use_embeddings=embeddings,
    )
    if run_loop:
        run_update_loop(
            db_path,
            graph_path,
            interval_hours=interval_hours,
            max_runs=max_runs,
            **kwargs,
        )
        return

    result = run_update(db_path, graph_path, **kwargs)
    if not result.ok:
        raise click.ClickException(result.error or "update failed")


@main.command()
@click.option("--db", "db_path", type=click.Path(path_type=Path), default=DEFAULT_DB_PATH)
@click.option(
    "--graph",
    "graph_path",
    type=click.Path(path_type=Path),
    default=DEFAULT_GRAPH_PATH,
)
@click.option("--host", default="127.0.0.1")
@click.option("--port", default=8080, type=int)
def serve(db_path: Path, graph_path: Path, host: str, port: int) -> None:
    """Launch the web UI to explore the market graph."""
    try:
        import uvicorn
    except ImportError as exc:
        raise click.ClickException(
            "Install web extras: pip install -e '.[web]'"
        ) from exc

    if not db_path.exists():
        raise click.ClickException(f"No database at {db_path}. Run `polygraph fetch` first.")
    if not graph_path.exists():
        raise click.ClickException(f"No graph at {graph_path}. Run `polygraph build` first.")

    from polygraph.web.server import create_app

    app = create_app(db_path=db_path, graph_path=graph_path)
    console.print(f"[green]Polygraph UI[/green] → http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
