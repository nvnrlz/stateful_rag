"""Central configuration for StatefulRAG.

All tunables live here in one validated, env-overridable place instead of being
sprinkled as magic numbers across the codebase. Safety-relevant defaults are
chosen conservatively: the cache is treated as a *hint*, not a replacement, and
the system fails **open** to the authoritative main database.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum

from .exceptions import ConfigurationError


class SafetyMode(str, Enum):
    """How aggressively the cache is allowed to short-circuit the main database.

    - ``STRICT``  : On a cache hit, still run a constrained fresh retrieval and
                    merge/verify. Highest safety, lowest latency win. Recommended
                    default for clinical use.
    - ``BALANCED``: Serve from cache, but shadow-sample a fraction of hits against
                    a fresh retrieval to continuously measure divergence.
    - ``FAST``    : Serve directly from cache. Lowest latency, no live safety net.
                    Only appropriate for non-clinical workloads.
    """

    STRICT = "strict"
    BALANCED = "balanced"
    FAST = "fast"


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a float, got {raw!r}") from exc


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an int, got {raw!r}") from exc


@dataclass
class StatefulRAGConfig:
    """Immutable-ish configuration object passed into the retriever and stores."""

    # --- Vector / embedding ---
    embedding_dim: int = field(default_factory=lambda: _env_int("STATEFUL_RAG_EMBED_DIM", 1536))
    embedding_model: str = field(
        default_factory=lambda: os.environ.get("STATEFUL_RAG_EMBED_MODEL", "unknown")
    )
    # If True, the retriever rejects embeddings whose length != embedding_dim.
    validate_dimensions: bool = True

    # --- Retrieval / drift ---
    top_k: int = field(default_factory=lambda: _env_int("STATEFUL_RAG_TOP_K", 3))
    # Minimum cosine similarity for the *best* cached doc to be considered a hit.
    drift_threshold: float = field(
        default_factory=lambda: _env_float("STATEFUL_RAG_DRIFT_THRESHOLD", 0.85)
    )
    # Per-document floor: cached docs scoring below this are dropped from the
    # returned set even on a hit (prevents off-topic docs riding along).
    per_doc_floor: float = field(
        default_factory=lambda: _env_float("STATEFUL_RAG_PER_DOC_FLOOR", 0.5)
    )
    # Recency weighting: cached docs decay by this factor per elapsed turn so that
    # stale context loses to fresher context. 0.0 disables decay.
    recency_decay: float = field(
        default_factory=lambda: _env_float("STATEFUL_RAG_RECENCY_DECAY", 0.0)
    )

    # --- Safety ---
    safety_mode: SafetyMode = field(
        default_factory=lambda: SafetyMode(
            os.environ.get("STATEFUL_RAG_SAFETY_MODE", SafetyMode.STRICT.value)
        )
    )
    # Fraction of cache hits to shadow-check against a fresh retrieval in BALANCED
    # mode. Used only to measure divergence; results are still served from cache.
    shadow_sample_rate: float = field(
        default_factory=lambda: _env_float("STATEFUL_RAG_SHADOW_SAMPLE_RATE", 0.1)
    )
    # If a STRICT-mode constrained fresh retrieval disagrees with the cache beyond
    # this Jaccard-distance budget, fail open and serve the fresh results.
    max_divergence: float = field(
        default_factory=lambda: _env_float("STATEFUL_RAG_MAX_DIVERGENCE", 0.5)
    )

    # --- Resilience ---
    max_retries: int = field(default_factory=lambda: _env_int("STATEFUL_RAG_MAX_RETRIES", 2))
    retry_base_delay: float = field(
        default_factory=lambda: _env_float("STATEFUL_RAG_RETRY_BASE_DELAY", 0.2)
    )
    circuit_fail_threshold: int = field(
        default_factory=lambda: _env_int("STATEFUL_RAG_CIRCUIT_FAILS", 5)
    )
    circuit_reset_seconds: float = field(
        default_factory=lambda: _env_float("STATEFUL_RAG_CIRCUIT_RESET", 30.0)
    )

    # --- Retention / compliance ---
    # Cached nodes older than this many seconds are ignored on read and eligible
    # for purge. 0 disables TTL. Default: 24h.
    cache_ttl_seconds: int = field(
        default_factory=lambda: _env_int("STATEFUL_RAG_CACHE_TTL", 86_400)
    )

    def __post_init__(self) -> None:
        if self.embedding_dim <= 0:
            raise ConfigurationError("embedding_dim must be positive")
        if self.top_k <= 0:
            raise ConfigurationError("top_k must be positive")
        if not 0.0 <= self.drift_threshold <= 1.0:
            raise ConfigurationError("drift_threshold must be in [0, 1]")
        if not 0.0 <= self.per_doc_floor <= 1.0:
            raise ConfigurationError("per_doc_floor must be in [0, 1]")
        if not 0.0 <= self.shadow_sample_rate <= 1.0:
            raise ConfigurationError("shadow_sample_rate must be in [0, 1]")
        if not 0.0 <= self.max_divergence <= 1.0:
            raise ConfigurationError("max_divergence must be in [0, 1]")
        if isinstance(self.safety_mode, str):
            self.safety_mode = SafetyMode(self.safety_mode)
