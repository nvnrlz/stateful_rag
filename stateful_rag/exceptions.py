"""Typed exceptions for StatefulRAG.

A diagnostic backend must never fail with an opaque ``KeyError`` halfway through a
retrieval. Every failure mode that callers may reasonably want to handle has a
dedicated, documented exception type here.
"""

from __future__ import annotations


class StatefulRAGError(Exception):
    """Base class for every error raised by StatefulRAG."""


class ConfigurationError(StatefulRAGError):
    """Raised when configuration is invalid or internally inconsistent."""


class AuthorizationError(StatefulRAGError):
    """Raised when a principal is not permitted to access a session's cache."""


class EmbeddingError(StatefulRAGError):
    """Raised when the embedding function fails or returns an invalid vector."""


class RetrievalError(StatefulRAGError):
    """Raised when the main retriever fails and no safe fallback is available."""


class DimensionMismatchError(StatefulRAGError):
    """Raised when an embedding does not match the configured vector dimension."""


class CircuitOpenError(StatefulRAGError):
    """Raised when a circuit breaker is open and the call is short-circuited."""
