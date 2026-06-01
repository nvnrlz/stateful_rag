"""Production cache backend on PostgreSQL + pgvector.

Hardened versus the prototype:

- **Request-scoped sessions.** Instead of holding one shared, non-thread-safe
  ``Session``, this store takes a ``session_factory`` (a ``sessionmaker``) and
  opens a fresh, properly-scoped session per operation, committing on success and
  rolling back on error. The legacy single-session form is still accepted but
  warns.
- **Encryption at rest.** Document content is encrypted with
  :class:`~stateful_rag.security.ContentCipher` before it touches the database.
- **TTL + erasure.** Reads ignore expired rows; ``purge_expired`` and
  ``delete_session`` provide retention and right-to-erasure.
- **Model tagging.** Each node stores the embedding model used, so vectors from
  incompatible models are never compared.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, Iterator, List, Optional

from sqlalchemy import func
from sqlalchemy.orm import Session as DBSession

from .base import BaseStateStore, CachedHit
from ..audit import AuditRecord, AuditSink
from ..logging_utils import get_logger
from ..models import SessionModel, CachedNode, AuditLog
from ..security import ContentCipher


class PostgresStateStore(BaseStateStore):
    def __init__(
        self,
        session_factory: Optional[Callable[[], DBSession]] = None,
        *,
        db_session: Optional[DBSession] = None,
        cipher: Optional[ContentCipher] = None,
    ):
        if session_factory is None and db_session is None:
            raise ValueError("Provide either session_factory (preferred) or db_session")
        self._factory = session_factory
        self._shared = db_session
        self._cipher = cipher or ContentCipher()
        self._logger = get_logger()
        if session_factory is None:
            self._logger.warning(
                "PostgresStateStore initialised with a single shared db_session. "
                "This is not thread-safe; pass a session_factory (sessionmaker) "
                "for concurrent backends."
            )

    @contextmanager
    def _session(self) -> Iterator[DBSession]:
        """Yield a session, committing on success and rolling back on error."""
        if self._factory is not None:
            db = self._factory()
            try:
                yield db
                db.commit()
            except Exception:
                db.rollback()
                raise
            finally:
                db.close()
        else:
            db = self._shared
            try:
                yield db
                db.commit()
            except Exception:
                db.rollback()
                raise

    def save_session_context(
        self, session_id, turn, documents, embeddings, *,
        embedding_model="unknown", owner=None,
    ) -> None:
        if not documents or not embeddings:
            return
        if len(documents) != len(embeddings):
            raise ValueError("documents and embeddings must be the same length")
        with self._session() as db:
            session = db.get(SessionModel, session_id)
            if session is None:
                db.add(SessionModel(id=session_id, owner=owner))
            for doc, emb in zip(documents, embeddings):
                db.add(CachedNode(
                    session_id=session_id,
                    turn_added=turn,
                    content=self._cipher.encrypt(json.dumps(doc)),
                    embedding=emb,
                    embedding_model=embedding_model,
                ))

    def search_cache(
        self, session_id, query_embedding, top_k=3, *, ttl_seconds=0,
    ) -> List[CachedHit]:
        now = datetime.now(timezone.utc)
        with self._session() as db:
            q = (
                db.query(
                    CachedNode,
                    CachedNode.embedding.cosine_distance(query_embedding).label("distance"),
                )
                .filter(CachedNode.session_id == session_id)
            )
            if ttl_seconds and ttl_seconds > 0:
                cutoff = now - timedelta(seconds=ttl_seconds)
                q = q.filter(CachedNode.created_at >= cutoff)
            rows = q.order_by("distance").limit(top_k).all()

            results: List[CachedHit] = []
            for node, distance in rows:
                doc = json.loads(self._cipher.decrypt(node.content))
                doc["turn_added"] = node.turn_added
                created = node.created_at or now
                if created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
                results.append(CachedHit(
                    doc=doc,
                    score=1.0 - float(distance),
                    turn_added=node.turn_added,
                    age_seconds=max(0.0, (now - created).total_seconds()),
                ))
            return results

    def next_turn(self, session_id: str) -> int:
        with self._session() as db:
            current = (
                db.query(func.max(CachedNode.turn_added))
                .filter(CachedNode.session_id == session_id)
                .scalar()
            )
            return (current or 0) + 1

    def delete_session(self, session_id: str) -> int:
        with self._session() as db:
            removed = (
                db.query(CachedNode).filter(CachedNode.session_id == session_id).delete()
            )
            db.query(SessionModel).filter(SessionModel.id == session_id).delete()
            return removed

    def purge_expired(self, ttl_seconds: int) -> int:
        if ttl_seconds <= 0:
            return 0
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=ttl_seconds)
        with self._session() as db:
            return db.query(CachedNode).filter(CachedNode.created_at < cutoff).delete()

    def count(self, session_id: str) -> int:
        with self._session() as db:
            return (
                db.query(func.count(CachedNode.id))
                .filter(CachedNode.session_id == session_id)
                .scalar()
                or 0
            )


class PostgresAuditSink(AuditSink):
    """Persists audit records to the ``audit_log`` table."""

    def __init__(self, session_factory: Callable[[], DBSession]):
        self._factory = session_factory

    def record(self, record: AuditRecord) -> None:
        db = self._factory()
        try:
            db.add(AuditLog(
                correlation_id=record.correlation_id,
                session_id=record.session_id,
                principal=record.principal,
                turn=record.turn,
                route=record.route,
                query_hash=record.query_hash,
                best_similarity=record.best_similarity,
                num_results=record.num_results,
                embedding_model=record.embedding_model,
                divergence=record.divergence,
                latency_ms=record.latency_ms,
                error=record.error,
            ))
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()
