"""Validation gates — logic consistency and edge quality."""

from __future__ import annotations

import json
from typing import Any

from polygraph.infer.domain import domains_compatible
from polygraph.infer.models import EdgeRecord

EPSILON = 0.08  # allow spread / noise on probability constraints


def _prob(m: dict[str, Any]) -> float | None:
    p = m.get("prob_yes")
    if p is None:
        return None
    try:
        return float(p)
    except (TypeError, ValueError):
        return None


def validate_edges(
    edges: list[EdgeRecord],
    markets_by_id: dict[str, dict[str, Any]],
) -> tuple[list[EdgeRecord], list[dict[str, Any]]]:
    """
    V1: SUBEVENT / TEMPORAL probability constraints.
    Returns (updated_edges, violations_list).
    """
    violations: list[dict[str, Any]] = []

    for edge in edges:
        if edge.relation not in ("SUBEVENT", "TEMPORAL"):
            continue
        src = markets_by_id.get(edge.source_id)
        tgt = markets_by_id.get(edge.target_id)
        if not src or not tgt:
            continue
        p_src, p_tgt = _prob(src), _prob(tgt)
        if p_src is None or p_tgt is None:
            continue

        if edge.relation == "SUBEVENT" and p_src > p_tgt + EPSILON:
            violations.append(
                {
                    "type": "CONSTRAINT_VIOLATION",
                    "edge": edge.key(),
                    "relation": edge.relation,
                    "message": f"P(narrow)={p_src:.2f} > P(broad)={p_tgt:.2f} — possible mispricing",
                    "source_question": src.get("question", "")[:80],
                    "target_question": tgt.get("question", "")[:80],
                }
            )
            # Keep edge active — violation IS the insight
            edge.evidence["violation"] = True
            edge.mechanism += " [⚠ prob constraint violated]"

    # Deactivate COMOVES on same-event pairs already linked by GROUND
    ground_pairs = {
        tuple(sorted((e.source_id, e.target_id)))
        for e in edges
        if e.tier == "GROUND" and e.relation != "EXCLUDES"
    }
    for edge in edges:
        if edge.relation != "COMOVES":
            continue
        key = tuple(sorted((edge.source_id, edge.target_id)))
        if key in ground_pairs:
            edge.active = 0
            edge.evidence["deactivated"] = "redundant_with_ground"

    # Drop cross-domain semantic/temporal links
    for edge in edges:
        if edge.relation not in ("RELATED", "TEMPORAL", "COMOVES"):
            continue
        a = markets_by_id.get(edge.source_id)
        b = markets_by_id.get(edge.target_id)
        if a and b and not domains_compatible(a, b):
            edge.active = 0
            edge.evidence["deactivated"] = "cross_domain"

    # Drop low-quality TEMPORAL links (post-inference safety net)
    for edge in edges:
        if edge.relation != "TEMPORAL":
            continue
        combined = edge.evidence.get("combined_score", 0)
        token_score = edge.evidence.get("token_score", 0)
        if combined and combined < 9.0:
            edge.active = 0
            edge.evidence["deactivated"] = "low_combined_score"
        elif token_score and token_score < 4.0:
            edge.active = 0
            edge.evidence["deactivated"] = "low_token_overlap"

    return edges, violations


def neg_risk_probability_check(
    events: list[dict[str, Any]],
    markets_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Sum of prob_yes in neg-risk event should not greatly exceed 1."""
    violations: list[dict[str, Any]] = []
    for event in events:
        if not (event.get("neg_risk") or event.get("enable_neg_risk")):
            continue
        mids = json.loads(event.get("market_ids") or "[]")
        probs = []
        for mid in mids:
            m = markets_by_id.get(mid)
            if m and m.get("prob_yes") is not None:
                probs.append(float(m["prob_yes"]))
        if len(probs) < 2:
            continue
        total = sum(probs)
        if total > 1.0 + EPSILON * len(probs):
            violations.append(
                {
                    "type": "NEG_RISK_SUM",
                    "event_id": event.get("id"),
                    "title": event.get("title", ""),
                    "sum_prob_yes": round(total, 3),
                    "message": f"Outcome probs sum to {total:.2f} (>1) — mutually exclusive set",
                }
            )
    return violations
