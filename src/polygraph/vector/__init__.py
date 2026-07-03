"""Local embedding index for semantic retrieval (anchor resolution, search)."""

from polygraph.vector.anchor import resolve_temporal_anchor
from polygraph.vector.index import EmbeddingIndex

__all__ = ["EmbeddingIndex", "resolve_temporal_anchor", "try_load"]
