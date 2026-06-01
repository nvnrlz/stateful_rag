"""Audit trail for every retrieval decision.

The README claimed the system was "strictly auditable" while the code only had
``print()`` calls. This module makes that claim real: every retrieval produces an
immutable :class:`AuditRecord` describing exactly what happened — which route was
taken, the similarity scores, how many documents were returned, the embedding
model, latency, and any divergence detected — and writes it to one or more
:class:`AuditSink` backends.

PHI hygiene: the raw query text is **never** stored in the audit log. Only a
salted SHA-256 hash is recorded, which is enough to correlate turns within a
session without persisting clinical free-text in the trail.
"""

from __future__ import annotations

import hashlib
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Optional

from .logging_utils import get_correlation_id, log_event


def hash_query(query: str) -> str:
    """Return a salted, non-reversible hash of a query for safe auditing."""
    salt = os.environ.get("STATEFUL_RAG_AUDIT_SALT", "stateful-rag")
    return hashlib.sha256((salt + "::" + query).encode("utf-8")).hexdigest()


@dataclass
class AuditRecord:
    """One immutable record of a single retrieval decision."""

    session_id: str
    turn: int
    route: str  # cache_hit | cache_hit_verified | drift_break | main_db | fail_open | error
    query_hash: str
    correlation_id: str = field(default_factory=get_correlation_id)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    principal: Optional[str] = None
    best_similarity: Optional[float] = None
    num_results: int = 0
    result_scores: list[float] = field(default_factory=list)
    embedding_model: str = "unknown"
    divergence: Optional[float] = None
    latency_ms: Optional[float] = None
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AuditSink(ABC):
    """Destination for audit records (DB table, log pipeline, SIEM, ...)."""

    @abstractmethod
    def record(self, record: AuditRecord) -> None:  # pragma: no cover - interface
        ...


class LoggingAuditSink(AuditSink):
    """Writes audit records to the structured logger. Always-on, cheap default."""

    def record(self, record: AuditRecord) -> None:
        log_event(logging.INFO, "audit", **record.to_dict())


class InMemoryAuditSink(AuditSink):
    """Collects audit records in memory. Intended for tests and local inspection."""

    def __init__(self) -> None:
        self.records: list[AuditRecord] = []

    def record(self, record: AuditRecord) -> None:
        self.records.append(record)


class CompositeAuditSink(AuditSink):
    """Fan a record out to several sinks; a failing sink must not break retrieval."""

    def __init__(self, *sinks: AuditSink) -> None:
        self.sinks = list(sinks)

    def record(self, record: AuditRecord) -> None:
        for sink in self.sinks:
            try:
                sink.record(record)
            except Exception as exc:  # auditing must never crash the request path
                log_event(logging.ERROR, "audit_sink_failed", error=str(exc))
