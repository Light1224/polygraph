"""Persistent embedding index — numpy archive + metadata."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console
from rich.progress import track

from polygraph.config import DEFAULT_DATA_DIR, EMBEDDING_BATCH_SIZE, EMBEDDING_MODEL
from polygraph.vector.model import get_encoder
from polygraph.vector.text import market_document

console = Console()

INDEX_DIR = DEFAULT_DATA_DIR / "vectors"
DOC_VERSION = 3
VECTORS_PATH = INDEX_DIR / "embeddings.npy"
IDS_PATH = INDEX_DIR / "market_ids.json"
META_PATH = INDEX_DIR / "meta.json"


class EmbeddingIndex:
    """Memory-mapped cosine search over market embeddings."""

    def __init__(self, root: Path = INDEX_DIR):
        self.root = root
        self.vectors_path = root / "embeddings.npy"
        self.ids_path = root / "market_ids.json"
        self.meta_path = root / "meta.json"
        self._vectors: np.ndarray | None = None
        self._ids: list[str] | None = None
        self._id_to_idx: dict[str, int] | None = None

    def exists(self) -> bool:
        return self.vectors_path.exists() and self.ids_path.exists()

    def load(self) -> None:
        if not self.exists():
            raise FileNotFoundError(f"No embedding index at {self.root}")
        self._vectors = np.load(self.vectors_path, mmap_mode="r")
        self._ids = json.loads(self.ids_path.read_text())
        self._id_to_idx = {mid: i for i, mid in enumerate(self._ids)}

    def _ensure_loaded(self) -> None:
        if self._vectors is None:
            self.load()

    @property
    def size(self) -> int:
        self._ensure_loaded()
        return len(self._ids or [])

    def build(
        self,
        markets: list[dict[str, Any]],
        *,
        model_name: str = EMBEDDING_MODEL,
        batch_size: int = EMBEDDING_BATCH_SIZE,
        force: bool = False,
    ) -> int:
        if self.exists() and not force:
            meta = json.loads(self.meta_path.read_text()) if self.meta_path.exists() else {}
            if (
                meta.get("market_count") == len(markets)
                and meta.get("model") == model_name
                and meta.get("doc_version", 1) >= 1
            ):
                console.print(f"[dim]Embedding index up to date ({len(markets)} markets)[/dim]")
                self.load()
                return len(markets)

        self.root.mkdir(parents=True, exist_ok=True)
        ids = [str(m["id"]) for m in markets]
        docs = [market_document(m) for m in markets]

        encoder = get_encoder(model_name)
        console.print(f"[bold]Encoding[/bold] {len(docs)} markets with {model_name}…")

        vectors: list[np.ndarray] = []
        for i in track(range(0, len(docs), batch_size), description="embed"):
            batch = docs[i : i + batch_size]
            emb = encoder.encode(
                batch,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            vectors.append(np.asarray(emb, dtype=np.float32))

        matrix = np.vstack(vectors)
        np.save(self.vectors_path, matrix)
        self.ids_path.write_text(json.dumps(ids))
        self.meta_path.write_text(
            json.dumps(
                {
                    "model": model_name,
                    "dim": int(matrix.shape[1]),
                    "market_count": len(ids),
                    "doc_version": DOC_VERSION,
                },
                indent=2,
            )
        )
        self._vectors = matrix
        self._ids = ids
        self._id_to_idx = {mid: i for i, mid in enumerate(ids)}
        console.print(f"[green]Index saved[/green] → {self.root} ({matrix.shape})")
        return len(ids)

    def encode_query(self, text: str, *, model_name: str = EMBEDDING_MODEL) -> np.ndarray:
        encoder = get_encoder(model_name)
        vec = encoder.encode([text], normalize_embeddings=True)
        return np.asarray(vec[0], dtype=np.float32)

    def search(
        self,
        query: str,
        *,
        k: int = 25,
        exclude_ids: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Top-k markets by cosine similarity (vectors are L2-normalized)."""
        self._ensure_loaded()
        assert self._vectors is not None and self._ids is not None

        q = self.encode_query(query)
        scores = self._vectors @ q  # cosine sim

        exclude = exclude_ids or set()
        ranked = np.argsort(-scores)
        hits: list[tuple[str, float]] = []
        for idx in ranked:
            mid = self._ids[int(idx)]
            if mid in exclude:
                continue
            sim = float(scores[int(idx)])
            hits.append((mid, sim))
            if len(hits) >= k:
                break
        return hits

    def vector_for(self, market_id: str) -> np.ndarray | None:
        self._ensure_loaded()
        assert self._vectors is not None and self._id_to_idx is not None
        idx = self._id_to_idx.get(market_id)
        if idx is None:
            return None
        return np.asarray(self._vectors[idx], dtype=np.float32)

    def score_ids_from_vector(
        self, query_vec: np.ndarray, market_ids: list[str]
    ) -> list[tuple[str, float]]:
        """Score candidates given a precomputed unit query vector."""
        if not market_ids:
            return []
        self._ensure_loaded()
        assert self._vectors is not None and self._id_to_idx is not None
        pairs = [(mid, self._id_to_idx[mid]) for mid in market_ids if mid in self._id_to_idx]
        if not pairs:
            return []
        idxs = [p[1] for p in pairs]
        sims = np.asarray(self._vectors[idxs], dtype=np.float32) @ query_vec
        hits = [(pairs[i][0], float(sims[i])) for i in range(len(pairs))]
        hits.sort(key=lambda x: -x[1])
        return hits

    def score_ids(self, query: str, market_ids: list[str]) -> list[tuple[str, float]]:
        """Cosine similarity for a restricted set of markets (vectorized)."""
        if not market_ids:
            return []
        self._ensure_loaded()
        assert self._vectors is not None and self._id_to_idx is not None

        pairs = [(mid, self._id_to_idx[mid]) for mid in market_ids if mid in self._id_to_idx]
        if not pairs:
            return []
        q = self.encode_query(query)
        idxs = [p[1] for p in pairs]
        sims = np.asarray(self._vectors[idxs], dtype=np.float32) @ q
        hits = [(pairs[i][0], float(sims[i])) for i in range(len(pairs))]
        hits.sort(key=lambda x: -x[1])
        return hits

    def sync_meta(self, db_path: Path) -> None:
        """Record index build in SQLite meta."""
        conn = sqlite3.connect(db_path)
        conn.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            ("embedding_model", EMBEDDING_MODEL),
        )
        if self.meta_path.exists():
            conn.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                ("embedding_index", self.meta_path.read_text()),
            )
        conn.commit()
        conn.close()


def try_load(root: Path = INDEX_DIR) -> EmbeddingIndex | None:
    """Load a persisted index if present; None on missing or error."""
    index = EmbeddingIndex(root)
    if not index.exists():
        return None
    try:
        index.load()
        return index
    except Exception:
        return None
