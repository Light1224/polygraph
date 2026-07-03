"""RELATED edges — fast entity links for isolated markets (no query re-encode)."""

from __future__ import annotations

from typing import Any

from polygraph.infer.domain import domains_compatible, market_domain, same_event
from polygraph.infer.entities import EntityIndex
from polygraph.infer.models import EdgeRecord
from polygraph.vector.index import EmbeddingIndex

MIN_SIM = 0.52
MIN_SIM_ENTITY = 0.45
MAX_PER_MARKET = 2
MAX_ISOLATED = 2500
MIN_VOLUME = 10_000


def _connected_ids(edges: list[EdgeRecord]) -> set[str]:
    ids: set[str] = set()
    for e in edges:
        if not e.active or e.relation == "EXCLUDES":
            continue
        ids.add(e.source_id)
        ids.add(e.target_id)
    return ids


def infer_related(
    markets: list[dict[str, Any]],
    markets_by_id: dict[str, dict[str, Any]],
    embedding_index: EmbeddingIndex,
    *,
    existing_pairs: set[tuple[str, str]] | None = None,
    ground_edges: list[EdgeRecord] | None = None,
    entity_index: EntityIndex | None = None,
) -> list[EdgeRecord]:
    existing = existing_pairs or set()
    connected = _connected_ids(ground_edges or [])

    candidates = [
        m
        for m in markets
        if str(m["id"]) not in connected
        and float(m.get("volume") or 0) >= MIN_VOLUME
    ]
    candidates.sort(key=lambda m: float(m.get("volume") or 0), reverse=True)
    candidates = candidates[:MAX_ISOLATED]
    if len(candidates) < 2:
        return []

    entity_index = entity_index or EntityIndex(markets)
    embedding_index._ensure_loaded()
    edges: list[EdgeRecord] = []
    seen: set[tuple[str, str]] = set()
    per_market: dict[str, int] = {}

    for m in candidates:
        mid_a = str(m["id"])
        if per_market.get(mid_a, 0) >= MAX_PER_MARKET:
            continue
        vec_a = embedding_index.vector_for(mid_a)
        if vec_a is None:
            continue

        ctx_a = m.get("context_text") or m.get("question") or ""
        phrase = m.get("question") or ""
        pool = entity_index.candidates_for(phrase, ctx_a, exclude_id=mid_a, limit=25)
        if not pool:
            continue

        sims = dict(embedding_index.score_ids_from_vector(vec_a, pool))

        for mid_b in pool:
            mb = markets_by_id.get(mid_b)
            if not mb:
                continue
            if not domains_compatible(m, mb):
                continue

            ctx_b = mb.get("context_text") or mb.get("question") or ""
            ents = entity_index.market_tokens.get(mid_a, set()) & entity_index.market_tokens.get(mid_b, set())
            se = same_event(m, mb)
            dom = market_domain(m)
            sim = sims.get(mid_b, 0.0)

            if dom in ("sports", "esports"):
                if not se or sim < 0.4:
                    continue
                reason = "same event (isolated)"
            elif ents:
                if sim < MIN_SIM_ENTITY:
                    continue
                reason = f"entities: {', '.join(sorted(ents)[:3])}"
            elif se and sim >= 0.42:
                reason = "same event"
            elif sim < MIN_SIM:
                continue
            else:
                reason = "similar context"

            key = tuple(sorted((mid_a, mid_b)))
            if key in seen or key in existing:
                continue
            if per_market.get(mid_a, 0) >= MAX_PER_MARKET:
                break
            if per_market.get(mid_b, 0) >= MAX_PER_MARKET:
                continue

            seen.add(key)
            per_market[mid_a] = per_market.get(mid_a, 0) + 1
            per_market[mid_b] = per_market.get(mid_b, 0) + 1
            edges.append(
                EdgeRecord(
                    source_id=mid_a,
                    target_id=mid_b,
                    relation="RELATED",
                    tier="RELATED",
                    direction="undirected",
                    confidence=min(0.88, 0.5 + sim * 0.45),
                    mechanism=f"{reason} (sim={sim:.2f})",
                    evidence={"sim": round(sim, 4), "entities": list(ents)[:6], "isolated": True},
                )
            )
    return edges
