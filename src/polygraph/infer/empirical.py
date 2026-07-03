"""Tier B (EMPIRICAL) edges from price co-movement."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from itertools import combinations
from typing import Any

import numpy as np

from polygraph.infer.domain import domains_compatible, market_domain, same_event
from polygraph.infer.models import EdgeRecord

MIN_VOLUME = 50_000
MIN_HISTORY_POINTS = 12
MIN_ABS_R_HISTORY = 0.38
MIN_ABS_R_SNAPSHOT = 0.45


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v) if v is not None else default
    except (TypeError, ValueError):
        return default


def _returns_from_series(series: list[tuple[int, float]]) -> dict[int, float]:
    """Map timestamp → daily return (Δp)."""
    if len(series) < 2:
        return {}
    out: dict[int, float] = {}
    prev_ts, prev_p = series[0]
    for ts, p in series[1:]:
        out[ts] = p - prev_p
        prev_ts, prev_p = ts, p
    return out


def _pearson_on_overlap(
    series_a: list[tuple[int, float]],
    series_b: list[tuple[int, float]],
) -> float | None:
    ra = _returns_from_series(series_a)
    rb = _returns_from_series(series_b)
    keys = sorted(set(ra) & set(rb))
    if len(keys) < MIN_HISTORY_POINTS:
        return None
    va = np.array([ra[k] for k in keys], dtype=np.float64)
    vb = np.array([rb[k] for k in keys], dtype=np.float64)
    if np.std(va) < 1e-9 or np.std(vb) < 1e-9:
        return None
    r = float(np.corrcoef(va, vb)[0, 1])
    return None if math.isnan(r) else r


def _snapshot_r(a: dict[str, Any], b: dict[str, Any]) -> float | None:
    va = np.array([_safe_float(a.get("delta_1d")), _safe_float(a.get("delta_7d"))])
    vb = np.array([_safe_float(b.get("delta_1d")), _safe_float(b.get("delta_7d"))])
    if np.std(va) < 1e-9 or np.std(vb) < 1e-9:
        return None
    r = float(np.corrcoef(va, vb)[0, 1])
    return None if math.isnan(r) else r


def infer_comoves(
    markets: list[dict[str, Any]],
    *,
    price_series: dict[str, list[tuple[int, float]]] | None = None,
    min_abs_r: float | None = None,
    max_pairs_per_event: int = 20,
) -> list[EdgeRecord]:
    """
    COMOVES from synchronized price changes within the same event.

    Uses daily return correlation when price_history is available;
    falls back to [delta_1d, delta_7d] snapshot Pearson r.
    """
    price_series = price_series or {}
    use_history = bool(price_series)
    threshold = min_abs_r or (MIN_ABS_R_HISTORY if use_history else MIN_ABS_R_SNAPSHOT)

    by_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for m in markets:
        if _safe_float(m.get("volume")) < MIN_VOLUME:
            continue
        for eid in json.loads(m.get("event_ids") or "[]"):
            by_event[eid].append(m)

    edges: list[EdgeRecord] = []
    seen: set[tuple[str, str]] = set()
    markets_by_id = {str(m["id"]): m for m in markets}

    for group in by_event.values():
        if len(group) < 2:
            continue
        pairs_scored: list[tuple[float, str, str, float, str, int]] = []
        for a, b in combinations(group, 2):
            aid, bid = str(a["id"]), str(b["id"])
            sa = price_series.get(aid)
            sb = price_series.get(bid)
            method = "delta_snapshot"
            n_points = 0
            if sa and sb and len(sa) >= MIN_HISTORY_POINTS and len(sb) >= MIN_HISTORY_POINTS:
                r = _pearson_on_overlap(sa, sb)
                method = "daily_returns"
                n_points = len(set(_returns_from_series(sa)) & set(_returns_from_series(sb)))
            else:
                r = _snapshot_r(a, b)
            if r is None or abs(r) < threshold:
                continue
            pairs_scored.append((abs(r), aid, bid, r, method, n_points))

        pairs_scored.sort(reverse=True)
        for _, aid, bid, r, method, n_points in pairs_scored[:max_pairs_per_event]:
            ma, mb = markets_by_id.get(aid), markets_by_id.get(bid)
            if ma and mb and not domains_compatible(ma, mb):
                continue
            if ma and mb and not same_event(ma, mb) and market_domain(ma) in ("sports", "esports"):
                continue
            key = tuple(sorted((aid, bid)))
            if key in seen:
                continue
            seen.add(key)
            mech = (
                f"Daily returns correlated (r={r:.2f}, n={n_points})"
                if method == "daily_returns"
                else f"Recent moves correlated (r={r:.2f})"
            )
            edges.append(
                EdgeRecord(
                    source_id=aid,
                    target_id=bid,
                    relation="COMOVES",
                    tier="EMPIRICAL",
                    direction="undirected",
                    confidence=min(0.88, 0.5 + abs(r) * 0.45),
                    mechanism=mech,
                    evidence={
                        "r": round(r, 4),
                        "method": method,
                        "n_points": n_points,
                    },
                )
            )
    return edges
