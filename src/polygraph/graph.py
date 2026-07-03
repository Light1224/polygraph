"""Build directed belief graph from inferred edges."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import networkx as nx
from rich.console import Console

from polygraph.config import DEFAULT_GRAPH_PATH
from polygraph.infer.models import EDGE_TYPES
from polygraph.store import Store

console = Console()


def build_belief_graph(store: Store) -> nx.MultiDiGraph:
    """Assemble nodes from markets + directed edges from inference."""
    g = nx.MultiDiGraph()

    for m in store.iter_markets():
        mid = str(m["id"])
        g.add_node(
            mid,
            node_type="market",
            question=m["question"],
            slug=m["slug"],
            condition_id=m["condition_id"],
            active=bool(m["active"]),
            closed=bool(m["closed"]),
            neg_risk=bool(m["neg_risk"]),
            volume=m["volume"],
            prob_yes=m["prob_yes"],
            prob_no=m["prob_no"],
            delta_1d=m["delta_1d"],
            delta_7d=m["delta_7d"],
            liquidity=m["liquidity"],
        )

    for e in store.iter_edges(active_only=True):
        if e["relation"] == "EXCLUDES":
            # Stored for logic checks; omitted from visual graph (use event grouping)
            continue
        src, tgt = e["source_id"], e["target_id"]
        if src not in g or tgt not in g:
            continue
        relation = e["relation"]
        direction = e["direction"]
        key = relation
        attrs = {
            "relation": relation,
            "tier": e["tier"],
            "direction": direction,
            "confidence": e["confidence"],
            "mechanism": e["mechanism"] or "",
            "evidence_quote": e["evidence_quote"] or "",
            "evidence_json": e["evidence_json"] or "{}",
            "weight": e["confidence"],
        }
        if direction == "undirected":
            g.add_edge(src, tgt, key=key, **attrs)
            g.add_edge(tgt, src, key=f"{relation}_rev", **attrs)
        else:
            g.add_edge(src, tgt, key=key, **attrs)

    return g


def graph_stats(g: nx.Graph) -> dict[str, Any]:
    if g.number_of_nodes() == 0:
        return {"nodes": 0, "edges": 0}

    relation_counts: dict[str, int] = defaultdict(int)
    if isinstance(g, nx.MultiDiGraph):
        for _, _, data in g.edges(data=True):
            rel = data.get("relation", "unknown")
            if not str(rel).endswith("_rev"):
                relation_counts[rel] += 1
        n_edges = sum(1 for _, _, d in g.edges(data=True) if not str(d.get("relation", "")).endswith("_rev"))
    else:
        for _, _, data in g.edges(data=True):
            for t in data.get("types", []):
                relation_counts[t] += 1
        n_edges = g.number_of_edges()

    degrees = [d for _, d in g.degree()]
    return {
        "nodes": g.number_of_nodes(),
        "edges": n_edges,
        "relation_counts": dict(sorted(relation_counts.items())),
        "avg_degree": round(sum(degrees) / max(len(degrees), 1), 2),
    }


def _sanitize_for_graphml(g: nx.Graph) -> nx.Graph:
    export = nx.DiGraph() if isinstance(g, nx.MultiDiGraph) else nx.Graph()
    for n, data in g.nodes(data=True):
        clean = {k: "" if v is None else str(v) for k, v in data.items()}
        export.add_node(n, **clean)

    seen_undir: set[tuple[str, str]] = set()
    for u, v, data in g.edges(data=True):
        rel = data.get("relation", "")
        if str(rel).endswith("_rev"):
            continue
        if data.get("direction") == "undirected":
            key = tuple(sorted((u, v)))
            if key in seen_undir:
                continue
            seen_undir.add(key)
        clean = {k: "" if v is None else str(v) for k, v in data.items()}
        export.add_edge(u, v, **clean)
    return export


def build_graph(
    db_path: Path,
    output_path: Path = DEFAULT_GRAPH_PATH,
    *,
    run_infer: bool = True,
    empirical: bool = True,
    **_,
) -> nx.MultiDiGraph:
    with Store(db_path) as store:
        if run_infer or store.edge_count() == 0:
            from polygraph.infer.pipeline import run_inference

            console.print("[bold]Inferring edges[/bold] (math pipeline)…")
            run_inference(store, empirical=empirical)
        g = build_belief_graph(store)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(_sanitize_for_graphml(g), output_path)
    stats = graph_stats(g)
    console.print(f"[green]Belief graph written[/green] to [cyan]{output_path}[/cyan]")
    for k, v in stats.items():
        console.print(f"  {k}: {v}")
    return g
