from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EdgeRecord:
    source_id: str
    target_id: str
    relation: str
    tier: str  # GROUND, EMPIRICAL, RELATED, INFERRED
    direction: str  # forward, undirected
    confidence: float
    mechanism: str = ""
    evidence_quote: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)
    active: int = 1

    def key(self) -> tuple[str, str, str]:
        return (self.source_id, self.target_id, self.relation)


RELATION_TIERS = {
    "EXCLUDES": "GROUND",
    "CO_EVENT": "GROUND",
    "SUBEVENT": "GROUND",
    "TEMPORAL": "GROUND",
    "RESOLVES_IF": "GROUND",
    "SHARED_TAG": "RELATED",
    "RELATED": "RELATED",
    "LEADS": "EMPIRICAL",
    "COMOVES": "EMPIRICAL",
    "SAME_TOPIC": "RELATED",
    "SEMANTIC": "RELATED",
    "IMPLIES": "INFERRED",
}

EDGE_TYPES = {
    "EXCLUDES": "Mutually exclusive outcomes (neg-risk) — at most one Yes",
    "CO_EVENT": "Same multi-outcome event — sibling markets",
    "SHARED_TAG": "Shared topic tag",
    "SUBEVENT": "Narrower deadline ⊆ broader (P(early) ≤ P(late))",
    "TEMPORAL": "Resolution gated on anchor event (before / until)",
    "RESOLVES_IF": "Explicit conditional in resolution criteria",
    "RELATED": "Same entities or strong semantic similarity",
    "LEADS": "Price of source leads target (Granger / lead-lag)",
    "COMOVES": "Prices move together — correlated belief",
    "SAME_TOPIC": "Shared event or tag (layout only)",
    "SEMANTIC": "Similar question/description text",
    "IMPLIES": "Inferred implication (NLI / LLM)",
}
