"""Structural edges — connect markets via Polymarket's own grouping (fast, high precision)."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from polygraph.infer.models import EdgeRecord

MAX_CO_EVENT_DEGREE = 12
MAX_TAG_MARKET_FRAC = 0.12


def infer_co_event(
    markets: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> list[EdgeRecord]:
    """Same multi-outcome event — star to hub (highest volume) to limit clutter."""
    markets_by_id = {str(m["id"]): m for m in markets}
    edges: list[EdgeRecord] = []
    seen: set[tuple[str, str]] = set()

    for event in events:
        mids = [m for m in json.loads(event.get("market_ids") or "[]") if m in markets_by_id]
        if len(mids) < 2:
            continue
        title = (event.get("title") or event.get("id") or "")[:80]
        hub = max(mids, key=lambda mid: float(markets_by_id[mid].get("volume") or 0))
        others = sorted(
            [m for m in mids if m != hub],
            key=lambda mid: float(markets_by_id[mid].get("volume") or 0),
            reverse=True,
        )[:MAX_CO_EVENT_DEGREE]
        for other in others:
            key = tuple(sorted((hub, other)))
            if key in seen:
                continue
            seen.add(key)
            edges.append(
                EdgeRecord(
                    source_id=hub,
                    target_id=other,
                    relation="CO_EVENT",
                    tier="GROUND",
                    direction="undirected",
                    confidence=1.0,
                    mechanism=f"Same event: «{title}»",
                    evidence={"event_id": event.get("id")},
                )
            )
    return edges


def infer_shared_tag(markets: list[dict[str, Any]]) -> list[EdgeRecord]:
    """Link markets sharing a rare tag (star topology per tag)."""
    n = len(markets)
    if n == 0:
        return []

    markets_by_id = {str(m["id"]): m for m in markets}
    tag_to_markets: dict[str, list[str]] = defaultdict(list)
    for m in markets:
        mid = str(m["id"])
        tags = m.get("event_tags") or m.get("tag_slugs") or []
        if isinstance(tags, str):
            try:
                tags = json.loads(tags)
            except json.JSONDecodeError:
                tags = []
        for t in tags:
            if t:
                tag_to_markets[str(t)].append(mid)

    max_tag_size = max(int(n * MAX_TAG_MARKET_FRAC), 50)
    edges: list[EdgeRecord] = []
    seen: set[tuple[str, str]] = set()

    for tag, mids in tag_to_markets.items():
        if len(mids) < 2 or len(mids) > max_tag_size:
            continue
        by_vol = sorted(
            mids,
            key=lambda mid: float(markets_by_id[mid].get("volume") or 0),
            reverse=True,
        )
        hub = by_vol[0]
        for other in by_vol[1:6]:
            key = tuple(sorted((hub, other)))
            if key in seen:
                continue
            seen.add(key)
            edges.append(
                EdgeRecord(
                    source_id=hub,
                    target_id=other,
                    relation="SHARED_TAG",
                    tier="RELATED",
                    direction="undirected",
                    confidence=0.75,
                    mechanism=f"Shared topic tag: {tag}",
                    evidence={"tag": tag},
                )
            )
    return edges
