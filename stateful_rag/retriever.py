"""The core routing engine.

This is a substantial rewrite of the prototype. The key behavioural changes that
make it fit for a clinical backend:

1. **The cache is a hint, not a replacement.** Safety mode governs how much we
   trust it. In ``STRICT`` (the default) every cache hit is *verified* against a
   fresh retrieval and divergence is recorded; if the cache disagrees with the
   authoritative database beyond ``max_divergence`` we **fail open** to fresh
   results. A cache hit can therefore never silently hide a finding the main DB
   would have surfaced.
2. **Real scores and provenance** are attached to every returned document
   (``_score``, ``_route``, ``_source``, ``_turn_added``) instead of fabricated
   ``1.0`` confidences.
3. **Resilience**: embedding and main-retriever calls go through retry +
   circuit-breaker guards and the engine fails *open* (to the authoritative DB)
   rather than crashing a turn.
4. **Auditing**: every decision emits an :class:`~stateful_rag.audit.AuditRecord`.
5. **Authorization & turn derivation** come from the host authorizer and the
   persisted store, not from client-trusted or in-memory state.

The engine keeps no per-request mutable state on ``self`` (only immutable config
and thread-safe guards), so a single instance is safe to share across concurrent
requests.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .audit import AuditRecord, AuditSink, LoggingAuditSink, hash_query
from .config import SafetyMode, StatefulRAGConfig
from .eval.divergence import compare_results
from .exceptions import EmbeddingError, DimensionMismatchError
from .logging_utils import get_correlation_id, log_event, set_correlation_id
from .resilience import GuardedCall
from .security import Authorizer, enforce_authorization
from .stores.base import BaseStateStore, CachedHit


@dataclass
class _Decision:
    """Per-request working state, kept on the stack (never on ``self``)."""

    session_id: str
    turn: int
    query: str
    principal: Optional[str]
    t0: float
    route: str = "error"
    source: str = "main_db"
    best_sim: Optional[float] = None
    divergence: Optional[float] = None
    served: List[Dict[str, Any]] = field(default_factory=list)
    scores: Optional[List[float]] = None
    error: Optional[str] = None


class StatefulRetriever:
    """Routes between the authoritative main vector DB and the session cache."""

    def __init__(
        self,
        state_store: BaseStateStore,
        main_retriever_fn: Callable[[str], List[Dict[str, Any]]],
        embed_fn: Callable[[str], List[float]],
        *,
        config: Optional[StatefulRAGConfig] = None,
        audit_sink: Optional[AuditSink] = None,
        authorizer: Optional[Authorizer] = None,
        drift_threshold: Optional[float] = None,  # backward-compatible override
        rng: Optional[random.Random] = None,
    ):
        self.store = state_store
        self.main_retriever_fn = main_retriever_fn
        self.embed_fn = embed_fn
        self.config = config or StatefulRAGConfig()
        if drift_threshold is not None:  # honour the legacy positional knob
            self.config.drift_threshold = drift_threshold
        self.audit_sink = audit_sink or LoggingAuditSink()
        self.authorizer = authorizer
        self._rng = rng or random.Random()

        self._guard_embed = GuardedCall(
            name="embed_fn",
            max_retries=self.config.max_retries,
            base_delay=self.config.retry_base_delay,
            fail_threshold=self.config.circuit_fail_threshold,
            reset_seconds=self.config.circuit_reset_seconds,
        )
        self._guard_main = GuardedCall(
            name="main_retriever_fn",
            max_retries=self.config.max_retries,
            base_delay=self.config.retry_base_delay,
            fail_threshold=self.config.circuit_fail_threshold,
            reset_seconds=self.config.circuit_reset_seconds,
        )

    # ------------------------------------------------------------------ embedding
    def _embed(self, text: str) -> List[float]:
        vec = self._guard_embed(lambda: self.embed_fn(text))
        if vec is None or len(vec) == 0:
            raise EmbeddingError("embed_fn returned an empty vector")
        if self.config.validate_dimensions and len(vec) != self.config.embedding_dim:
            raise DimensionMismatchError(
                f"embed_fn returned dim {len(vec)}, expected {self.config.embedding_dim}"
            )
        return vec

    # ------------------------------------------------------------------ main DB
    def fresh_retrieve(self, query: str) -> List[Dict[str, Any]]:
        """Authoritative, cache-bypassing retrieval. Does not mutate the cache."""
        docs = self._guard_main(lambda: self.main_retriever_fn(query))
        return [dict(d) if isinstance(d, dict) else {"content": str(d)} for d in docs]

    def _fresh_and_cache(
        self, query: str, session_id: str, turn: int, principal: Optional[str]
    ) -> List[Dict[str, Any]]:
        docs = self.fresh_retrieve(query)
        if docs:
            embeddings = [self._embed(d.get("content", "")) for d in docs]
            self.store.save_session_context(
                session_id, turn, docs, embeddings,
                embedding_model=self.config.embedding_model, owner=principal,
            )
        return docs

    # ------------------------------------------------------------------ cache eval
    def _rank_hits(self, hits: List[CachedHit], turn: int) -> List[CachedHit]:
        """Filter by the per-doc floor and re-rank with optional recency decay."""
        decay = self.config.recency_decay
        floored = [h for h in hits if h.score >= self.config.per_doc_floor]

        def adjusted(h: CachedHit) -> float:
            if decay <= 0:
                return h.score
            age_turns = max(0, turn - h.turn_added)
            return h.score * ((1.0 - decay) ** age_turns)

        return sorted(floored, key=adjusted, reverse=True)

    # ------------------------------------------------------------------ public API
    def retrieve(
        self,
        query: str,
        session_id: str,
        current_turn: Optional[int] = None,
        *,
        principal: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if get_correlation_id() == "-":
            set_correlation_id()
        enforce_authorization(self.authorizer, session_id, principal)
        turn = current_turn if current_turn is not None else self.store.next_turn(session_id)
        d = _Decision(session_id=session_id, turn=turn, query=query,
                      principal=principal, t0=time.perf_counter())
        try:
            self._route(d)
        except Exception as exc:
            d.error = d.error or str(exc)
            d.route = "error"
            self._audit(d)
            raise
        annotated = self._annotate(d)
        self._audit(d)
        return annotated

    def _route(self, d: _Decision) -> None:
        """Populate the decision with a route and a served document set."""
        try:
            query_vec = self._embed(d.query)
        except DimensionMismatchError:
            # A dimension mismatch is a configuration error, not a transient
            # fault. Failing open would corrupt every cache write, so surface it.
            raise
        except Exception as exc:
            # Embedding failed: fail OPEN to the authoritative DB (no caching).
            d.error = f"embed_failed: {exc}"
            log_event(logging.ERROR, "embed_failed_fail_open", error=str(exc))
            d.served = self.fresh_retrieve(d.query)
            d.route, d.source = "fail_open", "main_db"
            return

        hits = self.store.search_cache(
            d.session_id, query_vec, top_k=self.config.top_k,
            ttl_seconds=self.config.cache_ttl_seconds,
        )
        d.best_sim = hits[0].score if hits else None
        ranked = self._rank_hits(hits, d.turn)
        is_hit = bool(ranked) and d.best_sim is not None and d.best_sim >= self.config.drift_threshold

        if not is_hit:
            d.route = "drift_break" if hits else "main_db"
            d.source = "main_db"
            d.served = self._fresh_and_cache(d.query, d.session_id, d.turn, d.principal)
            return

        cached_docs = [h.doc for h in ranked]
        cached_scores = [h.score for h in ranked]

        if self.config.safety_mode is SafetyMode.FAST:
            d.route, d.source = "cache_hit", "cache"
            d.served, d.scores = cached_docs, cached_scores
            return

        if self.config.safety_mode is SafetyMode.STRICT:
            # Verify against the authoritative DB on every hit; fail open on disagreement.
            fresh = self._fresh_and_cache(d.query, d.session_id, d.turn, d.principal)
            d.divergence = compare_results(fresh, cached_docs).divergence
            if d.divergence <= self.config.max_divergence:
                d.route, d.source = "cache_hit_verified", "cache"
                d.served, d.scores = cached_docs, cached_scores
            else:
                log_event(logging.WARNING, "cache_divergence_exceeded_fail_open",
                          divergence=d.divergence, budget=self.config.max_divergence)
                d.route, d.source = "fail_open", "main_db"
                d.served = fresh
            return

        # BALANCED: serve cache, shadow-sample a fraction for divergence telemetry.
        d.route, d.source = "cache_hit", "cache"
        d.served, d.scores = cached_docs, cached_scores
        if self._rng.random() < self.config.shadow_sample_rate:
            shadow = self.fresh_retrieve(d.query)
            d.divergence = compare_results(shadow, cached_docs).divergence
            if d.divergence > self.config.max_divergence:
                log_event(logging.WARNING, "shadow_divergence_exceeded",
                          divergence=d.divergence, budget=self.config.max_divergence)

    # ------------------------------------------------------------------ helpers
    def _annotate(self, d: _Decision) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for i, doc in enumerate(d.served):
            item = dict(doc) if isinstance(doc, dict) else {"content": str(doc)}
            item.setdefault("_turn_added", item.get("turn_added"))
            item["_route"] = d.route
            item["_source"] = d.source
            if d.scores is not None and i < len(d.scores):
                item["_score"] = float(d.scores[i])
            else:
                item.setdefault("_score", None)
            out.append(item)
        return out

    def _audit(self, d: _Decision) -> None:
        record = AuditRecord(
            session_id=d.session_id,
            turn=d.turn,
            route=d.route,
            query_hash=hash_query(d.query),
            principal=d.principal,
            best_similarity=d.best_sim,
            num_results=len(d.served),
            result_scores=[float(s) for s in (d.scores or [])],
            embedding_model=self.config.embedding_model,
            divergence=d.divergence,
            latency_ms=(time.perf_counter() - d.t0) * 1000.0,
            error=d.error,
        )
        try:
            self.audit_sink.record(record)
        except Exception as exc:  # audit must never break retrieval
            log_event(logging.ERROR, "audit_record_failed", error=str(exc))
