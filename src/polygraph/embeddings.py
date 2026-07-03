"""Semantic similarity edges via local sentence embeddings."""

from __future__ import annotations

import json
from typing import Iterator

import numpy as np
from sklearn.neighbors import NearestNeighbors

from polygraph.config import SEMANTIC_MIN_SIMILARITY, SEMANTIC_TOP_K


def market_text(row: dict) -> str:
    """Text blob for embedding: question + event context + description snippet."""
    parts = [row.get("question") or ""]
    if row.get("group_item_title"):
        parts.append(str(row["group_item_title"]))
    if row.get("description"):
        parts.append(str(row["description"])[:500])
    try:
        event_ids = json.loads(row.get("event_ids") or "[]")
        if event_ids:
            parts.append(f"event:{event_ids[0]}")
    except json.JSONDecodeError:
        pass
    return " | ".join(p for p in parts if p)


def load_encoder(model_name: str = "all-MiniLM-L6-v2"):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "Semantic edges need sentence-transformers. "
            "Install with: pip install -e '.[embeddings]'"
        ) from exc
    return SentenceTransformer(model_name)


def semantic_edges(
    markets: list[dict],
    *,
    model_name: str = "all-MiniLM-L6-v2",
    top_k: int = SEMANTIC_TOP_K,
    min_similarity: float = SEMANTIC_MIN_SIMILARITY,
) -> Iterator[tuple[str, str, float]]:
    """
    Yield (market_id_a, market_id_b, cosine_similarity) for semantically
    similar markets. Uses approximate k-NN — O(n·k) not O(n²).
    """
    if len(markets) < 2:
        return

    ids = [str(m["id"]) for m in markets]
    texts = [market_text(m) for m in markets]

    encoder = load_encoder(model_name)
    embeddings = encoder.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.asarray(embeddings, dtype=np.float32)

    k = min(top_k + 1, len(markets))
    nn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=k)
    nn.fit(embeddings)
    distances, indices = nn.kneighbors(embeddings)

    seen: set[tuple[str, str]] = set()
    for i, (dists, nbrs) in enumerate(zip(distances, indices)):
        for dist, j in zip(dists, nbrs):
            if i == j:
                continue
            sim = 1.0 - float(dist)
            if sim < min_similarity:
                continue
            a, b = ids[i], ids[j]
            key = (min(a, b), max(a, b))
            if key in seen:
                continue
            seen.add(key)
            yield a, b, sim
