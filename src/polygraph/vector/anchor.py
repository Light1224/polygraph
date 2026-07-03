"""Hybrid embedding + rules temporal anchor resolution."""

from __future__ import annotations

import re
from typing import Any

from polygraph.vector.index import EmbeddingIndex
from polygraph.vector.text import anchor_query

# Re-use rule helpers without circular import at module level
_RELEASE = re.compile(
    r"\b(release[sd]?|launch(?:ed)?|ipo|announced|officially\s+available)\b",
    re.IGNORECASE,
)
_BEFORE_Q = re.compile(r"before\s+(.+?)\s*\?", re.IGNORECASE)
_DEPENDENT_SKIP = re.compile(
    r"\b(postponed|trailer|launch price|cost \$|another .+ before)\b",
    re.IGNORECASE,
)


def is_dependent_market(question: str) -> bool:
    q = question or ""
    if re.search(r"\breleased?\s+before\b", q, re.IGNORECASE):
        return False
    return bool(_BEFORE_Q.search(q))


def _rule_score(phrase: str, candidate: dict[str, Any]) -> float:
    q = candidate.get("question") or ""
    text = q.lower()
    score = 0.0
    pl = phrase.lower()
    if pl in text:
        score += 5.0
    if _RELEASE.search(q):
        score += 8.0
    if re.search(r"\breleased?\s+before\b", q, re.IGNORECASE):
        score += 15.0
    if _DEPENDENT_SKIP.search(q):
        score -= 15.0
    if is_dependent_market(q):
        score -= 20.0
    vol = candidate.get("volume") or 0
    score += min(3.0, (float(vol) / 1_000_000) ** 0.5)
    return score


def resolve_temporal_anchor(
    source: dict[str, Any],
    markets_by_id: dict[str, dict[str, Any]],
    index: EmbeddingIndex,
    *,
    phrase: str,
    min_combined: float = 12.0,
) -> tuple[str | None, float, str]:
    """
    Returns (anchor_market_id, confidence, mechanism_detail).
    Hybrid: embedding retrieval → rule re-rank.
    """
    source_id = str(source["id"])
    query = anchor_query(phrase)
    hits = index.search(query, k=30, exclude_ids={source_id})

    best_id: str | None = None
    best_combined = 0.0
    best_sim = 0.0

    for mid, sim in hits:
        cand = markets_by_id.get(mid)
        if not cand:
            continue
        if is_dependent_market(cand.get("question", "")):
            continue
        rule = _rule_score(phrase, cand)
        combined = rule + sim * 12.0
        if combined > best_combined:
            best_combined = combined
            best_id = mid
            best_sim = sim

    if best_id and best_combined >= min_combined:
        target_q = markets_by_id[best_id].get("question", "")[:60]
        detail = f"embedding+rules sim={best_sim:.2f} → «{target_q}»"
        confidence = min(0.98, 0.75 + best_sim * 0.2)
        return best_id, confidence, detail
    return None, 0.0, ""
