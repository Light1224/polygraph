"""TEMPORAL edges — explicit + implied entity anchors with domain gating."""

from __future__ import annotations

import re
from typing import Any

from polygraph.infer.anchors import resolve_anchor_phrase
from polygraph.infer.corpus import CorpusIndex, is_dependent_question, tokenize
from polygraph.infer.domain import domains_compatible, market_domain
from polygraph.infer.entities import EntityIndex, anchor_entities
from polygraph.infer.models import EdgeRecord
from polygraph.vector.index import EmbeddingIndex

_DEADLINE = re.compile(
    r"\b(by|before|on|in)\s+.{3,40}?\b(20\d{2}|january|february|march|april|may|june|july|august|september|october|november|december)\b",
    re.IGNORECASE,
)
_RELEASE = re.compile(
    r"\b(release[sd]?|launch(?:ed)?|ipo|out|announced|available|election|elected|inaugurat)\b",
    re.IGNORECASE,
)

MIN_COMBINED = 9.0
MIN_ENTITY_OVERLAP = 1
MIN_EMBED_SIM = 0.48
MAX_TEMPORAL_SOURCES = 1800


def _compatible_pair(source: dict[str, Any], target: dict[str, Any] | None) -> bool:
    if not target:
        return False
    return domains_compatible(source, target)


def _structural_anchor_score(question: str) -> float:
    q = question or ""
    score = 0.0
    if _DEADLINE.search(q):
        score += 4.0
    if _RELEASE.search(q):
        score += 3.0
    if is_dependent_question(q):
        score -= 8.0
    return score


def _eligible_dependent(market: dict[str, Any]) -> bool:
    q = market.get("question") or ""
    if is_dependent_question(q):
        return True
    phrase, src = resolve_anchor_phrase(market)
    return phrase is not None and src == "implied"


def infer_temporal(
    markets: list[dict[str, Any]],
    markets_by_id: dict[str, dict[str, Any]],
    *,
    embedding_index: EmbeddingIndex | None = None,
    corpus: CorpusIndex | None = None,
    entity_index: EntityIndex | None = None,
) -> list[EdgeRecord]:
    corpus = corpus or CorpusIndex(markets)
    entity_index = entity_index or EntityIndex(markets)
    edges: list[EdgeRecord] = []
    seen: set[tuple[str, str]] = set()
    dependents = [m for m in markets if _eligible_dependent(m)]
    dependents.sort(key=lambda m: float(m.get("volume") or 0), reverse=True)
    dependents = dependents[:MAX_TEMPORAL_SOURCES]

    for m in dependents:

        q = m.get("question") or ""
        phrase, anchor_src = resolve_anchor_phrase(m)
        if not phrase:
            continue

        phrase_tokens = set(tokenize(phrase))
        ctx = m.get("context_text") or q
        source_tokens = set(tokenize(ctx)) - phrase_tokens
        source_id = str(m["id"])

        entity_cands = set(entity_index.candidates_for(phrase, ctx, exclude_id=source_id))
        corpus_cands = set(
            corpus.retrieve(phrase_tokens, source_tokens, exclude_id=source_id, limit=60)
        )
        candidate_ids = [
            cid
            for cid in list(entity_cands | corpus_cands)[:100]
            if _compatible_pair(m, markets_by_id.get(cid))
        ][:60]
        if not candidate_ids:
            continue

        src_tokens = entity_index.market_tokens.get(source_id, set())

        embed_sims: dict[str, float] = {}
        if embedding_index is not None:
            vec_src = embedding_index.vector_for(source_id)
            if vec_src is not None:
                embed_sims = dict(
                    embedding_index.score_ids_from_vector(vec_src, candidate_ids)
                )

        best_id: str | None = None
        best_score = 0.0
        best_sim = 0.0

        for cid in candidate_ids:
            cand = markets_by_id.get(cid)
            if not cand:
                continue
            ent_overlap = len(src_tokens & entity_index.market_tokens.get(cid, set()))
            if ent_overlap < MIN_ENTITY_OVERLAP:
                continue

            sim = embed_sims.get(cid, 0.0)
            token_score = corpus.token_overlap_score(phrase_tokens, source_tokens, cid)
            struct = _structural_anchor_score(cand.get("question") or "")
            vol = float(cand.get("volume") or 0)
            vol_boost = min(2.0, (vol / 500_000) ** 0.4)
            entity_boost = ent_overlap * 3.5
            implied_boost = 1.5 if anchor_src == "implied" else 0.0
            combined = token_score + struct + sim * 7.0 + vol_boost + entity_boost + implied_boost

            if embedding_index is not None and sim < MIN_EMBED_SIM and ent_overlap < 2:
                continue
            if combined > best_score:
                best_score = combined
                best_id = cid
                best_sim = sim

        if not best_id or best_score < MIN_COMBINED:
            continue

        key = (source_id, best_id)
        if key in seen:
            continue
        seen.add(key)

        ents = anchor_entities(ctx, phrase)
        target_q = markets_by_id[best_id].get("question", "")[:60]
        src_label = "implied anchor" if anchor_src == "implied" else "explicit anchor"
        confidence = min(0.97, 0.7 + best_score * 0.018 + best_sim * 0.08)
        edges.append(
            EdgeRecord(
                source_id=source_id,
                target_id=best_id,
                relation="TEMPORAL",
                tier="GROUND",
                direction="forward",
                confidence=confidence,
                mechanism=(
                    f"{src_label}: «{target_q}» "
                    f"(entities={','.join(ents[:3])}, sim={best_sim:.2f})"
                ),
                evidence_quote=phrase[:200],
                evidence={
                    "anchor_phrase": phrase,
                    "anchor_source": anchor_src,
                    "entities": ents,
                    "resolver": "entity+corpus+embed",
                    "embed_sim": round(best_sim, 3),
                    "combined_score": round(best_score, 3),
                    "domain": market_domain(m),
                },
            )
        )
    return edges
