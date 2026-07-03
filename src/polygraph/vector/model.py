"""Lazy-loaded sentence-transformer encoder."""

from __future__ import annotations

from functools import lru_cache

from polygraph.config import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def get_encoder(model_name: str = EMBEDDING_MODEL):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "Embeddings require sentence-transformers. "
            "Install with: pip install -e '.[embeddings]'"
        ) from exc
    return SentenceTransformer(model_name)
