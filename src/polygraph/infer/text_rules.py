"""Tier A (GROUND) edges from resolution text and market structure."""

from __future__ import annotations

import itertools
import json
import re
from collections import defaultdict
from datetime import datetime
from typing import Any

from polygraph.infer.models import EdgeRecord
from polygraph.infer.structural import infer_co_event, infer_shared_tag
from polygraph.infer.temporal import infer_temporal

# --- regex patterns for resolution text ---
BEFORE_QUESTION = re.compile(r"before\s+(.+?)\s*\?", re.IGNORECASE)


def is_dependent_market(question: str) -> bool:
    """Markets that resolve *before* another event (not anchor deadlines)."""
    q = question or ""
    if re.search(r"\breleased?\s+before\b", q, re.IGNORECASE):
        return False
    return bool(BEFORE_QUESTION.search(q))


def _parse_date(val: str | None) -> datetime | None:
    if not val:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d"):
        try:
            return datetime.strptime(val[:26], fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(val.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


def infer_excludes(
    events: list[dict[str, Any]],
    markets_by_id: dict[str, dict[str, Any]],
) -> list[EdgeRecord]:
    edges: list[EdgeRecord] = []
    for event in events:
        if not (event.get("neg_risk") or event.get("enable_neg_risk")):
            continue
        market_ids = json.loads(event["market_ids"] or "[]")
        valid = [m for m in market_ids if m in markets_by_id]
        if len(valid) < 2:
            continue
        title = event.get("title") or event["id"]
        for a, b in itertools.combinations(valid, 2):
            edges.append(
                EdgeRecord(
                    source_id=a,
                    target_id=b,
                    relation="EXCLUDES",
                    tier="GROUND",
                    direction="undirected",
                    confidence=1.0,
                    mechanism=f"Neg-risk: only one outcome wins in «{title}»",
                    evidence_quote="",
                    evidence={"event_id": event["id"]},
                )
            )
    return edges


MAX_SUBEVENT_GROUP = 24
MAX_SUBEVENT_CHAIN = 8


def infer_subevent(
    markets: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> list[EdgeRecord]:
    """Narrow deadline → broader deadline within the same event only."""
    edges: list[EdgeRecord] = []
    markets_by_id = {str(m["id"]): m for m in markets}
    groups: list[list[tuple[datetime, dict[str, Any]]]] = []

    for event in events:
        mids = json.loads(event["market_ids"] or "[]")
        dated: list[tuple[datetime, dict[str, Any]]] = []
        for mid in mids:
            m = markets_by_id.get(mid)
            if not m or not m.get("end_date"):
                continue
            d = _parse_date(m.get("end_date"))
            if d is None:
                continue
            dated.append((d, m))
        if len(dated) >= 2:
            groups.append(sorted(dated, key=lambda x: x[0]))

    seen: set[tuple[str, str]] = set()
    for group in groups:
        if len(group) > MAX_SUBEVENT_GROUP:
            # Large events: chain nearest broader deadlines only (avoid O(n²)).
            for i in range(len(group) - 1):
                narrow, broad = group[i][1], group[i + 1][1]
                a, b = str(narrow["id"]), str(broad["id"])
                key = (a, b)
                if key in seen:
                    continue
                seen.add(key)
                edges.append(
                    EdgeRecord(
                        source_id=a,
                        target_id=b,
                        relation="SUBEVENT",
                        tier="GROUND",
                        direction="forward",
                        confidence=0.95,
                        mechanism=(
                            f"Earlier deadline ({narrow.get('end_date', '')[:10]}) "
                            f"⊆ later ({broad.get('end_date', '')[:10]})"
                        ),
                        evidence_quote="",
                        evidence={
                            "end_narrow": narrow.get("end_date"),
                            "end_broad": broad.get("end_date"),
                        },
                    )
                )
            continue

        for i, (d_n, narrow) in enumerate(group):
            limit = min(len(group), i + 1 + MAX_SUBEVENT_CHAIN)
            for j in range(i + 1, limit):
                d_b, broad = group[j]
                if d_n >= d_b:
                    continue
                a, b = str(narrow["id"]), str(broad["id"])
                key = (a, b)
                if key in seen:
                    continue
                seen.add(key)
                edges.append(
                    EdgeRecord(
                        source_id=a,
                        target_id=b,
                        relation="SUBEVENT",
                        tier="GROUND",
                        direction="forward",
                        confidence=0.95,
                        mechanism=(
                            f"Earlier deadline ({narrow.get('end_date', '')[:10]}) "
                            f"⊆ later ({broad.get('end_date', '')[:10]})"
                        ),
                        evidence_quote="",
                        evidence={
                            "end_narrow": narrow.get("end_date"),
                            "end_broad": broad.get("end_date"),
                        },
                    )
                )
    return edges


def infer_resolves_if(markets: list[dict[str, Any]]) -> list[EdgeRecord]:
    return []


def infer_ground_edges(
    markets: list[dict[str, Any]],
    events: list[dict[str, Any]],
    *,
    embedding_index=None,
    corpus=None,
    entity_index=None,
) -> list[EdgeRecord]:
    from polygraph.infer.corpus import CorpusIndex
    from polygraph.infer.entities import EntityIndex

    markets_by_id = {str(m["id"]): m for m in markets}
    event_rows = [dict(e) for e in events]
    edges: list[EdgeRecord] = []
    edges.extend(infer_excludes(event_rows, markets_by_id))
    edges.extend(infer_co_event(markets, event_rows))
    edges.extend(infer_shared_tag(markets))
    edges.extend(infer_subevent(markets, event_rows))
    if embedding_index is not None and (corpus is None or entity_index is None):
        corpus = corpus or CorpusIndex(markets)
        entity_index = entity_index or EntityIndex(markets)
    edges.extend(
        infer_temporal(
            markets,
            markets_by_id,
            embedding_index=embedding_index,
            corpus=corpus,
            entity_index=entity_index,
        )
    )
    edges.extend(infer_resolves_if(markets))
    return edges
