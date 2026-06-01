"""In-memory state store for local testing and unit tests.

Hardened versus the prototype: it enforces embedding dimensions, tracks per-node
timestamps and embedding model so TTL/erasure work identically to Postgres, and
guards against NaN/zero-norm vectors.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np

from .base import BaseStateStore, CachedHit
from ..exceptions import DimensionMismatchError


class InMemoryStateStore(BaseStateStore):
    """A lightweight, dependency-free, thread-safe vector cache for local use."""

    def __init__(self, expected_dim: Optional[int] = None):
        self._expected_dim = expected_dim
        self._lock = threading.RLock()
        # session_id -> list of node dicts {doc, embedding, turn_added, created_at, model}
        self._store: Dict[str, List[Dict[str, Any]]] = {}

    def _check_dim(self, vec: List[float]) -> None:
        if self._expected_dim is not None and len(vec) != self._expected_dim:
            raise DimensionMismatchError(
                f"Embedding has dim {len(vec)}, expected {self._expected_dim}"
            )

    def save_session_context(
        self, session_id, turn, documents, embeddings, *,
        embedding_model="unknown", owner=None,
    ) -> None:
        if not documents or not embeddings:
            return
        if len(documents) != len(embeddings):
            raise ValueError("documents and embeddings must be the same length")
        now = time.time()
        with self._lock:
            bucket = self._store.setdefault(session_id, [])
            for doc, emb in zip(documents, embeddings):
                self._check_dim(emb)
                bucket.append({
                    "doc": dict(doc),
                    "embedding": np.asarray(emb, dtype=np.float64),
                    "turn_added": turn,
                    "created_at": now,
                    "embedding_model": embedding_model,
                })

    def search_cache(
        self, session_id, query_embedding, top_k=3, *, ttl_seconds=0,
    ) -> List[CachedHit]:
        self._check_dim(query_embedding)
        now = time.time()
        with self._lock:
            nodes = self._store.get(session_id, [])
            if not nodes:
                return []
            live = [
                n for n in nodes
                if ttl_seconds <= 0 or (now - n["created_at"]) <= ttl_seconds
            ]
            if not live:
                return []

            query_vec = np.asarray(query_embedding, dtype=np.float64)
            norm_query = np.linalg.norm(query_vec)
            if norm_query == 0 or not np.isfinite(norm_query):
                return []

            cache_vecs = np.vstack([n["embedding"] for n in live])
            norms = np.linalg.norm(cache_vecs, axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                sims = (cache_vecs @ query_vec) / (norms * norm_query)
            sims = np.nan_to_num(sims, nan=-1.0, posinf=-1.0, neginf=-1.0)

            order = np.argsort(sims)[::-1][:top_k]
            results: List[CachedHit] = []
            for idx in order:
                node = live[int(idx)]
                doc = dict(node["doc"])
                doc["turn_added"] = node["turn_added"]
                results.append(CachedHit(
                    doc=doc,
                    score=float(sims[int(idx)]),
                    turn_added=node["turn_added"],
                    age_seconds=now - node["created_at"],
                ))
            return results

    def next_turn(self, session_id: str) -> int:
        with self._lock:
            nodes = self._store.get(session_id, [])
            if not nodes:
                return 1
            return max(n["turn_added"] for n in nodes) + 1

    def delete_session(self, session_id: str) -> int:
        with self._lock:
            removed = len(self._store.get(session_id, []))
            self._store.pop(session_id, None)
            return removed

    def purge_expired(self, ttl_seconds: int) -> int:
        if ttl_seconds <= 0:
            return 0
        now = time.time()
        removed = 0
        with self._lock:
            for sid in list(self._store.keys()):
                kept = [n for n in self._store[sid] if (now - n["created_at"]) <= ttl_seconds]
                removed += len(self._store[sid]) - len(kept)
                if kept:
                    self._store[sid] = kept
                else:
                    del self._store[sid]
        return removed

    def count(self, session_id: str) -> int:
        with self._lock:
            return len(self._store.get(session_id, []))
