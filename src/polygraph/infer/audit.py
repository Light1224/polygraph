"""Graph quality audit — connectivity and domain sanity checks."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from polygraph.infer.domain import domains_compatible, market_domain
from polygraph.infer.models import EdgeRecord


def audit_edges(
    edges: list[EdgeRecord],
    markets_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Post-inference audit stats for validation_report.json."""
    active = [e for e in edges if e.active]
    visual = [e for e in active if e.relation != "EXCLUDES"]

    adj: dict[str, set[str]] = defaultdict(set)
    for e in visual:
        adj[e.source_id].add(e.target_id)
        adj[e.target_id].add(e.source_id)

    all_ids = set(markets_by_id.keys())
    connected = set(adj.keys())
    isolated = len(all_ids - connected)

    cross_domain = 0
    for e in visual:
        if e.relation not in ("RELATED", "TEMPORAL", "COMOVES"):
            continue
        a = markets_by_id.get(e.source_id)
        b = markets_by_id.get(e.target_id)
        if a and b and not domains_compatible(a, b):
            cross_domain += 1

    by_rel = Counter(e.relation for e in active)
    temporal_implied = sum(
        1 for e in active
        if e.relation == "TEMPORAL" and e.evidence.get("anchor_source") == "implied"
    )

    degrees = [len(adj.get(mid, ())) for mid in connected]
    avg_degree = sum(degrees) / len(degrees) if degrees else 0.0

    return {
        "visual_edges": len(visual),
        "connected_markets": len(connected),
        "isolated_markets": isolated,
        "isolation_rate": round(isolated / max(1, len(all_ids)), 4),
        "avg_visual_degree": round(avg_degree, 2),
        "cross_domain_violations": cross_domain,
        "temporal_implied_count": temporal_implied,
        "active_by_relation": dict(by_rel),
    }
