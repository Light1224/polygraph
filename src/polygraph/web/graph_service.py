"""Load and query the belief graph for the web UI."""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import networkx as nx

from polygraph.infer.models import EDGE_TYPES

EDGE_COLORS = {
    "EXCLUDES": "#ef4444",
    "CO_EVENT": "#f97316",
    "SHARED_TAG": "#eab308",
    "SUBEVENT": "#a855f7",
    "TEMPORAL": "#6366f1",
    "RESOLVES_IF": "#3b82f6",
    "RELATED": "#06b6d4",
    "LEADS": "#f59e0b",
    "COMOVES": "#22c55e",
    "IMPLIES": "#64748b",
}

# Relations omitted from exploration (too dense)
SKIP_RELATIONS = frozenset({"EXCLUDES"})


@lru_cache(maxsize=2)
def load_graph(path: str) -> nx.DiGraph:
    return nx.read_graphml(path)


@lru_cache(maxsize=2)
def load_adjacency(db_path: str) -> tuple[dict[str, set[str]], list[dict]]:
    """Build adjacency for BFS + edge list (cached)."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    adj: dict[str, set[str]] = defaultdict(set)
    all_edges: list[dict] = []
    for row in conn.execute(
        "SELECT * FROM edges WHERE active=1"
    ):
        e = dict(row)
        if e["relation"] in SKIP_RELATIONS:
            continue
        s, t = e["source_id"], e["target_id"]
        adj[s].add(t)
        if e["direction"] == "undirected":
            adj[t].add(s)
        else:
            adj[t].add(s)  # traverse both ways for neighborhood discovery
        all_edges.append(e)
    conn.close()
    return dict(adj), all_edges


def _edges_for_nodes(all_edges: list[dict], node_ids: set[str]) -> list[dict]:
    out = []
    for e in all_edges:
        if e["source_id"] in node_ids and e["target_id"] in node_ids:
            rel = e["relation"]
            out.append(
                {
                    "from": e["source_id"],
                    "to": e["target_id"],
                    "relation": rel,
                    "tier": e["tier"],
                    "direction": e["direction"],
                    "color": EDGE_COLORS.get(rel, "#94a3b8"),
                    "confidence": e["confidence"],
                    "mechanism": e["mechanism"] or "",
                    "evidence_quote": e["evidence_quote"] or "",
                    "title": f"{rel}: {e['mechanism'] or ''}",
                    "arrows": "to" if e["direction"] == "forward" else "",
                }
            )
    return out


def bfs_neighborhood(
    adj: dict[str, set[str]],
    seed: str,
    *,
    depth: int,
    max_nodes: int,
) -> set[str]:
    seen = {seed}
    frontier = {seed}
    for _ in range(depth):
        if len(seen) >= max_nodes:
            break
        nxt: set[str] = set()
        for node in frontier:
            for nb in adj.get(node, ()):
                if nb not in seen:
                    nxt.add(nb)
        seen |= nxt
        frontier = nxt
        if not frontier:
            break
    if len(seen) > max_nodes:
        # BFS layers: keep nodes closest to seed
        layers: list[set[str]] = [{seed}]
        visited = {seed}
        cur = {seed}
        for _ in range(depth):
            nxt = set()
            for n in cur:
                for nb in adj.get(n, ()):
                    if nb not in visited:
                        nxt.add(nb)
                        visited.add(nb)
            if not nxt:
                break
            layers.append(nxt)
            cur = nxt
        kept = {seed}
        for layer in layers[1:]:
            for n in layer:
                if len(kept) >= max_nodes:
                    break
                kept.add(n)
            if len(kept) >= max_nodes:
                break
        seen = kept
    return seen


def ego_subgraph(
    graph: nx.DiGraph,
    seed: str,
    *,
    db_path: Path | None = None,
    depth: int = 1,
    max_nodes: int = 500,
    include_excludes: bool = False,
    mode: str = "explore",
) -> dict:
    if seed not in graph:
        return {"seed": seed, "nodes": [], "edges": [], "error": "Market not in graph"}

    skip = set() if include_excludes else SKIP_RELATIONS

    if db_path and db_path.exists():
        adj, all_edges = load_adjacency(str(db_path.resolve()))
        if skip:
            all_edges = [e for e in all_edges if e["relation"] not in skip]
            adj = defaultdict(set)
            for e in all_edges:
                s, t = e["source_id"], e["target_id"]
                adj[s].add(t)
                adj[t].add(s)
            adj = dict(adj)

        if mode == "focus":
            seen = {seed} | adj.get(seed, set())
        else:
            seen = bfs_neighborhood(adj, seed, depth=depth, max_nodes=max_nodes)
        edges = _edges_for_nodes(all_edges, seen)
    else:
        seen = bfs_neighborhood(
            {n: set(graph.successors(n)) | set(graph.predecessors(n)) for n in graph},
            seed,
            depth=depth,
            max_nodes=max_nodes,
        )
        edges = []
        for u, v, data in graph.edges(data=True):
            if u in seen and v in seen:
                rel = data.get("relation", "link")
                edges.append(
                    {
                        "from": u,
                        "to": v,
                        "relation": rel,
                        "color": EDGE_COLORS.get(rel, "#94a3b8"),
                        "mechanism": data.get("mechanism", ""),
                        "title": data.get("mechanism", rel),
                        "arrows": "to",
                    }
                )

    nodes = []
    for nid in seen:
        if nid not in graph:
            continue
        data = graph.nodes[nid]
        py = _float(data.get("prob_yes"))
        nodes.append(
            {
                "id": nid,
                "label": _short_label(data.get("question", nid), max_len=36 if len(seen) > 200 else 42),
                "question": data.get("question", ""),
                "slug": data.get("slug", ""),
                "volume": _float(data.get("volume")),
                "prob_yes": py,
                "neg_risk": str(data.get("neg_risk", "")).lower() in ("true", "1"),
                "degree": graph.degree(nid),
                "is_seed": nid == seed,
            }
        )

    return {
        "seed": seed,
        "depth": 1 if mode == "focus" else depth,
        "max_nodes": max_nodes,
        "mode": mode,
        "nodes": nodes,
        "edges": edges,
        "node_count": len(nodes),
        "edge_count": len(edges),
    }


def graph_summary(graph: nx.DiGraph, db_path: Path | None = None) -> dict:
    edge_count = graph.number_of_edges()
    if db_path and db_path.exists():
        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT COUNT(*) FROM edges WHERE active=1 AND relation NOT IN ('EXCLUDES')"
        ).fetchone()
        conn.close()
        edge_count = row[0]
    return {
        "nodes": graph.number_of_nodes(),
        "edges": edge_count,
        "edge_types": EDGE_TYPES,
        "edge_colors": EDGE_COLORS,
    }


def _short_label(question: str, max_len: int = 42) -> str:
    q = question or ""
    return q if len(q) <= max_len else q[: max_len - 1] + "…"


def _float(val, default: float = 0.0) -> float:
    try:
        return float(val) if val not in (None, "") else default
    except (TypeError, ValueError):
        return default
