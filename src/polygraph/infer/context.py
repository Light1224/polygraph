"""Attach event-level metadata so embeddings and inference see full market context."""

from __future__ import annotations

import json
from typing import Any


def attach_event_context(
    markets: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Join event title, tags, description onto each market dict (in-place)."""
    from polygraph.infer.domain import market_domain

    events_by_id = {str(e["id"]): e for e in events}
    for m in markets:
        eids = m.get("event_ids")
        if isinstance(eids, str):
            try:
                eids = json.loads(eids)
            except json.JSONDecodeError:
                eids = []
        titles: list[str] = []
        tags: list[str] = []
        for eid in eids or []:
            ev = events_by_id.get(str(eid))
            if not ev:
                continue
            if ev.get("title"):
                titles.append(str(ev["title"]))
            raw_tags = ev.get("tag_slugs")
            if isinstance(raw_tags, str):
                try:
                    raw_tags = json.loads(raw_tags)
                except json.JSONDecodeError:
                    raw_tags = []
            if raw_tags:
                tags.extend(str(t) for t in raw_tags)
        m["event_title"] = " | ".join(dict.fromkeys(titles))
        m["event_description"] = ""
        m["event_tags"] = list(dict.fromkeys(tags))
        if not m.get("tag_slugs") or m.get("tag_slugs") == "[]":
            m["tag_slugs"] = json.dumps(m["event_tags"])
        m["context_text"] = build_context_text(m)
        m["_event_id_set"] = {str(x) for x in (eids or [])}
        m["domain"] = market_domain(m)
    return markets


def build_context_text(market: dict[str, Any]) -> str:
    """Single narrative block for entity extraction and display."""
    parts = [market.get("question") or ""]
    if market.get("event_title"):
        parts.append(f"Event: {market['event_title']}")
    if market.get("group_item_title"):
        parts.append(f"Outcome: {market['group_item_title']}")
    if market.get("event_description"):
        parts.append(market["event_description"][:300])
    elif market.get("description"):
        parts.append(str(market["description"])[:300])
    tags = market.get("event_tags") or []
    if isinstance(tags, str):
        try:
            tags = json.loads(tags)
        except json.JSONDecodeError:
            tags = []
    if tags:
        parts.append("Topics: " + ", ".join(tags[:10]))
    return " ".join(p for p in parts if p)
