"""Canonical text for context-aware embeddings."""

from __future__ import annotations

import json
from typing import Any

from polygraph.infer.entities import extract_proper_spans


def market_document(market: dict[str, Any], *, max_desc: int = 360) -> str:
    """
    Rich context document: market question is NOT enough — include event framing,
    outcome label, topics, and named entities for disambiguation.
    """
    q = market.get("question") or ""
    parts = [f"[MARKET] {q}"]

    if market.get("event_title"):
        parts.append(f"[EVENT] {market['event_title']}")

    if market.get("group_item_title"):
        parts.append(f"[OUTCOME] {market['group_item_title']}")

    entities = extract_proper_spans(q)
    if market.get("event_title"):
        entities = list(dict.fromkeys(entities + extract_proper_spans(market["event_title"])))
    if entities:
        parts.append("[ENTITIES] " + ", ".join(entities[:12]))

    desc = market.get("event_description") or market.get("description") or ""
    if desc:
        parts.append(f"[RESOLUTION] {str(desc)[:max_desc]}")

    tags = market.get("event_tags") or market.get("tag_slugs")
    if isinstance(tags, str):
        try:
            tags = json.loads(tags)
        except json.JSONDecodeError:
            tags = []
    if tags:
        parts.append("[TOPICS] " + ", ".join(str(t) for t in tags[:10]))

    try:
        eids = market.get("event_ids")
        if isinstance(eids, str):
            eids = json.loads(eids)
        if eids:
            parts.append(f"[EVENT_ID] {eids[0]}")
    except (json.JSONDecodeError, TypeError):
        pass

    return " ".join(p.strip() for p in parts if p and p.strip())


def anchor_query(phrase: str, *, source_question: str = "") -> str:
    entities = extract_proper_spans(phrase) or extract_proper_spans(source_question)
    ent = ", ".join(entities[:8]) if entities else phrase
    return (
        f"[ANCHOR] When does {phrase} happen? "
        f"[ENTITIES] {ent}. "
        f"[TYPE] Official release, election, IPO, or announced resolution deadline market."
    )
