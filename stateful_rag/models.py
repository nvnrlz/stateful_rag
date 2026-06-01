"""SQLAlchemy models for the persistent cache and audit trail.

Changes from the prototype:

- The vector dimension is no longer hardcoded to 1536; it is read from
  ``STATEFUL_RAG_EMBED_DIM`` so the schema matches whatever embedding model the
  deployment actually uses (e.g. 384 for MiniLM, 1536 for OpenAI).
- ``CachedNode`` now records ``created_at`` (for TTL/retention) and the
  ``embedding_model`` used, so cached vectors from incompatible models are never
  silently compared.
- An ``AuditLog`` table provides a queryable, append-only compliance trail.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, String, DateTime, ForeignKey, Integer, Text, Float, Index
from sqlalchemy.orm import declarative_base, relationship
from pgvector.sqlalchemy import Vector

Base = declarative_base()

# Dimension is fixed at import time so the column type and any ANN index agree.
# Override with STATEFUL_RAG_EMBED_DIM before importing this module.
EMBEDDING_DIM = int(os.environ.get("STATEFUL_RAG_EMBED_DIM", "1536"))


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class SessionModel(Base):
    __tablename__ = "sessions"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at = Column(DateTime(timezone=True), default=_utcnow)
    # Owner principal lets the DB enforce tenancy (and supports row-level security).
    owner = Column(String, nullable=True, index=True)
    cached_nodes = relationship(
        "CachedNode", back_populates="session", cascade="all, delete-orphan"
    )


class CachedNode(Base):
    __tablename__ = "cached_nodes"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String, ForeignKey("sessions.id", ondelete="CASCADE"), index=True)
    turn_added = Column(Integer, nullable=False)
    created_at = Column(DateTime(timezone=True), default=_utcnow, index=True)
    # Encrypted (or plaintext-tagged) document JSON. See security.ContentCipher.
    content = Column(Text, nullable=False)
    embedding = Column(Vector(EMBEDDING_DIM), nullable=False)
    # Which model produced the embedding; guards against cross-model comparison.
    embedding_model = Column(String, nullable=False, default="unknown")
    session = relationship("SessionModel", back_populates="cached_nodes")


class AuditLog(Base):
    """Append-only audit trail of retrieval decisions (no raw PHI stored)."""

    __tablename__ = "audit_log"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    timestamp = Column(DateTime(timezone=True), default=_utcnow, index=True)
    correlation_id = Column(String, index=True)
    session_id = Column(String, index=True)
    principal = Column(String, nullable=True)
    turn = Column(Integer)
    route = Column(String, index=True)
    query_hash = Column(String)
    best_similarity = Column(Float, nullable=True)
    num_results = Column(Integer, default=0)
    embedding_model = Column(String, default="unknown")
    divergence = Column(Float, nullable=True)
    latency_ms = Column(Float, nullable=True)
    error = Column(Text, nullable=True)


# Composite index to make per-session, recency-ordered scans fast.
Index("ix_cached_nodes_session_created", CachedNode.session_id, CachedNode.created_at)
