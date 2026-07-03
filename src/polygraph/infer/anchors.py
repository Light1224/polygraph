"""Explicit and implied temporal anchor extraction."""

from __future__ import annotations

import re
from typing import Any

from polygraph.infer.corpus import extract_anchor_phrase, is_dependent_question
from polygraph.infer.domain import is_boilerplate_anchor, market_domain
from polygraph.infer.entities import is_entity_anchor

_BEFORE_DESC = re.compile(
    r"\b(?:before|prior to|contingent on|gated on|until)\s+"
    r"([^.\n]{5,90}?)(?:\s+(?:is|are|has|have|occurs|happens|released|announced)|[.,;])",
    re.IGNORECASE,
)
_RESOLVES_IF = re.compile(
    r"\bresolves?\s+(?:to\s+)?(?:Yes|No)\s+if\s+([^.\n]{8,100}?)[.,]",
    re.IGNORECASE,
)


def extract_implied_anchor(market: dict[str, Any]) -> str | None:
    """Anchor implied by resolution criteria (description), not question title."""
    if market_domain(market) in ("sports", "esports"):
        return None
    q = market.get("question") or ""
    for text in (market.get("description"), market.get("event_description")):
        if not text:
            continue
        for pat in (_BEFORE_DESC, _RESOLVES_IF):
            m = pat.search(str(text))
            if not m:
                continue
            phrase = m.group(1).strip()
            if is_boilerplate_anchor(phrase):
                continue
            if is_entity_anchor(phrase, q):
                return phrase
    return None


def resolve_anchor_phrase(market: dict[str, Any]) -> tuple[str | None, str]:
    """
    Returns (phrase, source) where source is 'question' | 'implied' | ''.
    """
    q = market.get("question") or ""
    explicit = extract_anchor_phrase(q)
    if explicit and is_entity_anchor(explicit, q) and not is_boilerplate_anchor(explicit):
        return explicit, "question"
    implied = extract_implied_anchor(market)
    if implied:
        return implied, "implied"
    if explicit and is_entity_anchor(explicit, q):
        return explicit, "question"
    return None, ""
