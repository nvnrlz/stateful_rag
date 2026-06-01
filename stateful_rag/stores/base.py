"""Abstract state-store interface.

Beyond the original save/search pair, every store must now support the
operations a compliant clinical backend needs: deriving the next turn from
persisted state (not fragile in-memory counters), right-to-erasure deletion, and
TTL-based purging of stale PHI.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class CachedHit:
    """A single cached document returned from a similarity search."""

    doc: Dict[str, Any]
    score: float          # cosine similarity in [-1, 1]
    turn_added: int
    age_seconds: float    # how long ago this node was cached


class BaseStateStore(ABC):
    """Interface every StatefulRAG cache backend must implement."""

    @abstractmethod
    def save_session_context(
        self,
        session_id: str,
        turn: int,
        documents: List[Dict[str, Any]],
        embeddings: List[List[float]],
        *,
        embedding_model: str = "unknown",
        owner: Optional[str] = None,
    ) -> None:
        ...

    @abstractmethod
    def search_cache(
        self,
        session_id: str,
        query_embedding: List[float],
        top_k: int = 3,
        *,
        ttl_seconds: int = 0,
    ) -> List[CachedHit]:
        ...

    @abstractmethod
    def next_turn(self, session_id: str) -> int:
        """Return the next turn number for a session, derived from stored state."""

    @abstractmethod
    def delete_session(self, session_id: str) -> int:
        """Hard-delete all data for a session (GDPR/HIPAA erasure). Returns rows removed."""

    @abstractmethod
    def purge_expired(self, ttl_seconds: int) -> int:
        """Remove cached nodes older than ``ttl_seconds``. Returns rows removed."""

    @abstractmethod
    def count(self, session_id: str) -> int:
        """Return the number of cached nodes for a session."""
