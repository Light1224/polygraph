"""Corpus-level token statistics for fast, data-driven entity matching."""

from __future__ import annotations

import heapq
import math
import re
from collections import defaultdict
from typing import Any

# Standard English stopwords — not entity-specific hardcoding.
_STOP = frozenset(
    """
    a an the and or but if in on at to for of is are was were be been being
    will would could should may might must do does did has have had this that
    these those it its they them their we our you your he she his her not no yes
    by with from as into than then so such any all each both other another
    about over under after before when where who whom which what how up out
    """.split()
)

_YEAR = re.compile(r"^\d{4}$")
_BEFORE_Q = re.compile(r"\b(?:before|until)\s+(.+?)\s*\?", re.IGNORECASE)
_TOKEN = re.compile(r"[a-z0-9]{2,}", re.IGNORECASE)


def tokenize(text: str, *, include_years: bool = False) -> list[str]:
    raw = _TOKEN.findall((text or "").lower())
    out: list[str] = []
    for t in raw:
        if t in _STOP:
            continue
        if _YEAR.match(t) and not include_years:
            continue
        out.append(t)
    return out


def is_dependent_question(question: str) -> bool:
    """Market resolves relative to another named event (question text only)."""
    q = question or ""
    if re.search(r"\breleased?\s+before\b", q, re.IGNORECASE):
        return False
    return bool(_BEFORE_Q.search(q))


def extract_anchor_phrase(question: str) -> str | None:
    m = _BEFORE_Q.search(question or "")
    return m.group(1).strip() if m else None


class CorpusIndex:
    """Inverted index + IDF over all market questions (built once per infer run)."""

    def __init__(self, markets: list[dict[str, Any]]):
        self.n_docs = len(markets)
        self.doc_tokens: dict[str, set[str]] = {}
        self.postings: dict[str, set[str]] = defaultdict(set)
        df: dict[str, int] = defaultdict(int)

        for m in markets:
            mid = str(m["id"])
            text = m.get("context_text") or " ".join(
                filter(
                    None,
                    [m.get("question"), m.get("group_item_title"), m.get("slug")],
                )
            )
            toks = set(tokenize(text))
            self.doc_tokens[mid] = toks
            for t in toks:
                df[t] += 1

        self.idf: dict[str, float] = {
            t: math.log((self.n_docs + 1) / (c + 1)) + 1.0 for t, c in df.items()
        }
        for mid, toks in self.doc_tokens.items():
            for t in toks:
                self.postings[t].add(mid)

        idf_vals = sorted(self.idf.values())
        self.idf_p25 = idf_vals[len(idf_vals) // 4] if idf_vals else 1.0
        self.idf_p50 = idf_vals[len(idf_vals) // 2] if idf_vals else 1.5
        self.idf_p75 = idf_vals[(3 * len(idf_vals)) // 4] if idf_vals else 2.0

    def idf_score(self, token: str) -> float:
        return self.idf.get(token, 0.0)

    def is_entity_phrase(self, phrase: str) -> bool:
        """Reject bare deadlines and ultra-common boilerplate (IDF-driven)."""
        p = (phrase or "").strip()
        if len(p) < 3:
            return False
        if _YEAR.match(p):
            return False
        toks = tokenize(p)
        if not toks:
            return False
        scores = [self.idf_score(t) for t in toks]
        if max(scores) < self.idf_p50:
            return False
        rare = sum(1 for s in scores if s >= self.idf_p50)
        return rare >= 1 and (len(toks) == 1 or rare >= min(2, len(toks)))

    def retrieve(
        self,
        phrase_tokens: set[str],
        source_tokens: set[str],
        *,
        exclude_id: str,
        limit: int = 150,
    ) -> list[str]:
        """Fast candidate retrieval via rare-token postings."""
        if not phrase_tokens:
            return []

        ranked_tokens = sorted(
            phrase_tokens,
            key=lambda t: self.idf_score(t),
            reverse=True,
        )
        pool: dict[str, float] = defaultdict(float)
        for t in ranked_tokens[:6]:
            hits = 0
            for mid in self.postings.get(t, ()):
                if mid == exclude_id:
                    continue
                pool[mid] += self.idf_score(t)
                hits += 1
                if hits >= 200:
                    break

        for mid in list(pool.keys()):
            ctoks = self.doc_tokens.get(mid, set())
            overlap = phrase_tokens & ctoks
            if len(overlap) < min(2, len(phrase_tokens)):
                src_overlap = source_tokens & ctoks
                if not overlap and len(src_overlap) < 2:
                    del pool[mid]
                    continue
            pool[mid] += 2.0 * len(overlap)
            pool[mid] += 1.0 * len(source_tokens & ctoks)

        return [
            mid
            for mid, _ in heapq.nlargest(limit, pool.items(), key=lambda x: x[1])
        ]

    def token_overlap_score(
        self, phrase_tokens: set[str], source_tokens: set[str], candidate_id: str
    ) -> float:
        ctoks = self.doc_tokens.get(candidate_id, set())
        if not ctoks:
            return 0.0
        po = len(phrase_tokens & ctoks) / max(1, len(phrase_tokens))
        so = len(source_tokens & ctoks) / max(1, len(source_tokens))
        rare = sum(
            1
            for t in phrase_tokens & ctoks
            if self.idf_score(t) >= self.idf_p50
        )
        return 4.0 * po + 2.0 * so + 1.5 * rare
