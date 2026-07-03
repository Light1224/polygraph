"""Entity (noun / proper-noun) extraction from market questions — data-driven, no alias tables."""

from __future__ import annotations

import heapq
import re
from collections import defaultdict
from typing import Any

# Title-case or ALL-CAPS spans in the original question text.
_PROPER_SPAN = re.compile(
    r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,4}|[A-Z]{2,}(?:\s+[A-Z]{2,}){0,3})\b"
)
_ROMAN = re.compile(r"\b[IVXLC]{1,4}\b")
_YEAR = re.compile(r"^\d{4}$")


def extract_proper_spans(text: str) -> list[str]:
    """Named spans from capitalization (Trump, GTA VI, New York)."""
    if not text:
        return []
    spans: list[str] = []
    for m in _PROPER_SPAN.finditer(text):
        s = m.group(0).strip()
        if s.lower() in {"will", "the", "new", "who", "what", "how"}:
            continue
        spans.append(s)
    tokens = text.split()
    for i, tok in enumerate(tokens):
        if _ROMAN.match(tok) and i > 0 and tokens[i - 1][0].isupper():
            spans.append(f"{tokens[i - 1]} {tok}")
    return list(dict.fromkeys(spans))


def extract_entities(question: str) -> list[str]:
    """Normalized entity keys for matching (lowercase slug form)."""
    keys: list[str] = []
    for s in extract_proper_spans(question or ""):
        k = re.sub(r"\s+", " ", s.lower().strip())
        if len(k) >= 2 and k not in keys:
            keys.append(k)
    return keys


def entity_key_tokens(key: str) -> set[str]:
    out: set[str] = set()
    for t in re.findall(r"[a-z0-9]{2,}", (key or "").lower()):
        if not _YEAR.match(t):
            out.add(t)
    return out


def entity_tokens(question: str) -> set[str]:
    out: set[str] = set()
    for e in extract_entities(question):
        for t in re.findall(r"[a-z0-9]{2,}", e):
            if not _YEAR.match(t):
                out.add(t)
    return out


def shared_entities(a: str, b: str) -> set[str]:
    return entity_tokens(a) & entity_tokens(b)


def anchor_entities(source_question: str, phrase: str | None = None) -> list[str]:
    if phrase:
        ents = extract_entities(phrase)
        if ents:
            return ents
    return extract_entities(source_question)


def is_entity_anchor(phrase: str, source_question: str) -> bool:
    p = (phrase or "").strip()
    if not p or _YEAR.fullmatch(p):
        return False
    if anchor_entities(source_question, phrase):
        return True
    toks = re.findall(r"[a-zA-Z]{2,}", p)
    return len(toks) >= 1 and any(len(t) >= 3 or t.isupper() for t in toks)


class EntityIndex:
    """entity_key → market ids for fast anchor retrieval."""

    def __init__(self, markets: list[dict[str, Any]]):
        self.entity_to_markets: dict[str, set[str]] = defaultdict(set)
        self.market_entities: dict[str, set[str]] = {}
        self.market_tokens: dict[str, set[str]] = {}
        for m in markets:
            mid = str(m["id"])
            keys: set[str] = set()
            for field in (m.get("question"), m.get("event_title")):
                for e in extract_entities(str(field or "")):
                    keys.add(e)
            token_union: set[str] = set()
            for k in keys:
                token_union |= entity_key_tokens(k)
            self.market_entities[mid] = keys
            self.market_tokens[mid] = token_union
            for k in keys:
                self.entity_to_markets[k].add(mid)

    def shared_token_count(self, mid_a: str, mid_b: str) -> int:
        return len(self.market_tokens.get(mid_a, set()) & self.market_tokens.get(mid_b, set()))

    def candidates_for(
        self, phrase: str, source_question: str, *, exclude_id: str, limit: int = 120
    ) -> list[str]:
        scores: dict[str, float] = defaultdict(float)
        keys = anchor_entities(source_question, phrase)
        phrase_ents = set(keys)
        for key in keys:
            hits = 0
            for mid in self.entity_to_markets.get(key, ()):
                if mid != exclude_id:
                    scores[mid] += 3.0
                    hits += 1
                    if hits >= 150:
                        break
        for mid in list(scores.keys()):
            overlap = len(phrase_ents & self.market_entities.get(mid, set()))
            scores[mid] += overlap * 2.0
        return [mid for mid, _ in heapq.nlargest(limit, scores.items(), key=lambda x: x[1])]
