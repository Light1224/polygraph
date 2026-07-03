"""Market domain classification — keeps sports/esports from cross-linking to politics/crypto."""

from __future__ import annotations

import json
import re
from typing import Any

_SPORTS_MARKET = re.compile(
    r"\b("
    r"vs\.?|o/u|over/under|spread|moneyline|touchdowns?|rebounds?|assists?|"
    r"set\s+\d|map\s+\d|game\s+\d|round\s+\d|quarter|inning|halftime|"
    r"both teams to score|btts|match winner|total kills|first blood|"
    r"draw no bet|handicap|series winner"
    r")\b",
    re.IGNORECASE,
)
_SPORTS_TOPIC = re.compile(
    r"\b(nba|nfl|mlb|nhl|mls|ufc|mma|boxing|cs2|counter-strike|dota|"
    r"league of legends|valorant|premier league|champions league|"
    r"world cup|super bowl|stanley cup|march madness|f1|formula 1|tennis|atp|wta)\b",
    re.IGNORECASE,
)
_ESPORTS_TOPIC = re.compile(
    r"\b(esports|cs2|dota|valorant|league of legends|lck|lec|vct)\b",
    re.IGNORECASE,
)
_CRYPTO = re.compile(
    r"\b(bitcoin|ethereum|crypto|btc|eth|solana|defi|token|blockchain|nft)\b",
    re.IGNORECASE,
)
_POLITICS = re.compile(
    r"\b(trump|biden|election|president|congress|senate|governor|parliament|vote)\b",
    re.IGNORECASE,
)

_SPORTS_TAGS = frozenset({
    "sports", "nba", "nfl", "mlb", "nhl", "soccer", "tennis", "mma", "ufc",
    "esports", "cs2", "dota-2", "league-of-legends", "valorant", "cricket",
    "golf", "f1", "formula-1", "ncaa", "college-football", "college-basketball",
})


def _tags_list(market: dict[str, Any]) -> list[str]:
    raw = market.get("event_tags") or market.get("tag_slugs") or []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raw = []
    return [str(t).lower() for t in raw]


def market_domain(market: dict[str, Any]) -> str:
    """Coarse domain for compatibility gating."""
    cached = market.get("domain")
    if isinstance(cached, str) and cached:
        return cached
    text = " ".join(
        filter(
            None,
            [
                market.get("question"),
                market.get("event_title"),
                market.get("context_text"),
            ],
        )
    )
    tags = _tags_list(market)
    if any(t in _SPORTS_TAGS for t in tags):
        return "esports" if _ESPORTS_TOPIC.search(text) else "sports"
    if _SPORTS_MARKET.search(text) or _SPORTS_TOPIC.search(text):
        return "esports" if _ESPORTS_TOPIC.search(text) else "sports"
    if _CRYPTO.search(text) or any("crypto" in t for t in tags):
        return "crypto"
    if _POLITICS.search(text) or any(t in tags for t in ("politics", "elections", "trump")):
        return "politics"
    return "general"


def same_event(a: dict[str, Any], b: dict[str, Any]) -> bool:
    ea = a.get("_event_id_set")
    eb = b.get("_event_id_set")
    if isinstance(ea, set) and isinstance(eb, set):
        return bool(ea and eb and ea & eb)

    def eids(m: dict) -> set[str]:
        raw = m.get("event_ids") or []
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                raw = []
        return {str(x) for x in raw}

    ea, eb = eids(a), eids(b)
    return bool(ea and eb and ea & eb)


def domains_compatible(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Whether two markets may receive semantic/temporal cross-links."""
    if same_event(a, b):
        return True
    da = a.get("domain") or market_domain(a)
    db = b.get("domain") or market_domain(b)
    if da in ("sports", "esports") or db in ("sports", "esports"):
        return da == db and same_event(a, b)
    if da == "general" or db == "general":
        return True
    return da == db


def is_boilerplate_anchor(phrase: str) -> bool:
    """Resolution-rule fragments that are not real-world anchors."""
    p = (phrase or "").lower()
    boiler = (
        "play begins",
        "set ",
        "game ",
        "map ",
        "the start",
        "the end",
        "specified period",
        "seasonal adjustment",
        "official result",
        "market close",
    )
    return any(b in p for b in boiler)
