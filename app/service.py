"""Assembles the full backend: embedder → main store → StatefulRAG → triage engine.

This is the single composition root the API (and tests) use. It wires the
hardened ``stateful_rag`` caching layer in front of the authoritative ENT
knowledge store and the deterministic triage engine.
"""

from __future__ import annotations

import threading
from typing import Dict, Optional

from stateful_rag import (
    InMemoryStateStore,
    SafetyMode,
    StatefulRAGConfig,
    StatefulRetriever,
)

from .config import AppConfig, load_config
from .feedback import Feedback, FeedbackLog
from .providers.embeddings import build_embedder
from .providers.llm import build_llm
from .retrieval.knowledge_retriever import KnowledgeRetriever
from .retrieval.main_store import build_main_store
from .triage.agent import TriageEngine, TurnResult
from .triage.extractor import build_extractor
from .triage.responder import build_responder
from .triage.state import InMemorySessionStore, SessionState


class SessionBusy(Exception):
    """Raised when a message arrives for a session that is still processing one.

    This prevents a duplicate / impatient resend from spawning a second long
    provider call on the same session — the root cause of the Batch-2 'server
    became unresponsive after a duplicate message' hang.
    """


class ReceptionistService:
    def __init__(self, cfg: Optional[AppConfig] = None):
        self.cfg = cfg or load_config()

        # 1) Embedder + authoritative knowledge store.
        self.embedder = build_embedder(
            self.cfg.embedding_provider, dim=self.cfg.embedding_dim, model=self.cfg.embedding_model
        )
        self.main_store = build_main_store(self.cfg)
        self.knowledge = KnowledgeRetriever(self.embedder, self.main_store, top_k=self.cfg.main_top_k)
        self._verify_index_compatibility()

        # 2) StatefulRAG caching/safety layer in front of the main store.
        rag_config = StatefulRAGConfig(
            embedding_dim=self.embedder.dim,
            embedding_model=self.embedder.name,
            # Calibrated against the configured embedder; see app.eval.calibrate.
            drift_threshold=self.cfg.drift_threshold,
            per_doc_floor=self.cfg.per_doc_floor,
            safety_mode=SafetyMode.STRICT,  # verify every cache hit against the main store
            cache_ttl_seconds=0,
            max_retries=1,
        )
        cache_store = InMemoryStateStore(expected_dim=self.embedder.dim)
        self.retriever = StatefulRetriever(
            state_store=cache_store,
            main_retriever_fn=self.knowledge.as_fn(),
            embed_fn=self.embedder.embed,
            config=rag_config,
        )

        # 3) Dialogue: deterministic engine + (optional) LLM phrasing.
        self.llm = build_llm(self.cfg.llm_provider, model=self.cfg.llm_model)
        self.responder = build_responder(self.llm)
        # Intake extraction is LLM-only (no heuristic fallback). When no LLM is
        # configured the engine reports "unavailable" rather than guessing.
        self.extractor = build_extractor(self.llm)
        self.engine = TriageEngine(self.retriever, self.responder,
                                   extractor=self.extractor, max_turns=self.cfg.max_turns)

        # 4) Session + feedback stores.
        self.sessions = InMemorySessionStore()
        self.feedback = FeedbackLog(path=f"{self.cfg.data_dir}/feedback.jsonl")

        # Per-session in-flight locks so one session processes one message at a time.
        self._locks_guard = threading.Lock()
        self._session_locks: Dict[str, threading.Lock] = {}

    def _session_lock(self, session_id: str) -> threading.Lock:
        with self._locks_guard:
            lock = self._session_locks.get(session_id)
            if lock is None:
                lock = threading.Lock()
                self._session_locks[session_id] = lock
            return lock

    def _verify_index_compatibility(self) -> None:
        """Guard against serving an index built with a different embedder."""
        import logging
        from stateful_rag.logging_utils import get_logger

        store = self.main_store
        name = getattr(store, "embedder_name", None)
        dim = getattr(store, "dim", None)
        if store.count() == 0:
            return
        if dim is not None and dim != self.embedder.dim:
            raise ValueError(
                f"Index dim {dim} != embedder '{self.embedder.name}' dim {self.embedder.dim}. "
                f"Re-ingest with the configured embedder."
            )
        if name and name not in ("unknown", self.embedder.name):
            get_logger().log(
                logging.WARNING,
                f"Knowledge index was built with embedder '{name}' but the service is "
                f"configured for '{self.embedder.name}'. Re-ingest to avoid degraded retrieval.",
            )

    # -- API surface ---------------------------------------------------------
    def start_session(self, principal: Optional[str] = None) -> tuple[SessionState, TurnResult]:
        state = self.sessions.create(principal=principal)
        result = self.engine.start(state)
        self.sessions.save(state)
        return state, result

    def message(self, session_id: str, text: str) -> TurnResult:
        if self.sessions.get(session_id) is None:
            raise KeyError(session_id)
        lock = self._session_lock(session_id)
        if not lock.acquire(blocking=False):
            # A previous message for this session is still being processed.
            raise SessionBusy(session_id)
        try:
            state = self.sessions.get(session_id)
            if state is None:
                raise KeyError(session_id)
            if state.concluded:
                # Allow a fresh follow-up complaint after conclusion.
                state.concluded = False
            result = self.engine.handle(state, text)
            self.sessions.save(state)
            return result
        finally:
            lock.release()

    def record_feedback(self, fb: Feedback) -> None:
        self.feedback.record(fb)

    def record_session_feedback(self, sf) -> None:
        self.feedback.record_session(sf)

    def knowledge_count(self) -> int:
        return self.main_store.count()
