"""StatefulRAG: a measured, auditable, fail-open caching layer for multi-turn RAG."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .config import StatefulRAGConfig, SafetyMode
from .retriever import StatefulRetriever
from .stores.memory import InMemoryStateStore
from .stores.postgres import PostgresStateStore, PostgresAuditSink
from .stores.base import BaseStateStore, CachedHit
from .security import ContentCipher
from .audit import (
    AuditRecord,
    AuditSink,
    LoggingAuditSink,
    InMemoryAuditSink,
    CompositeAuditSink,
)
from .exceptions import (
    StatefulRAGError,
    ConfigurationError,
    AuthorizationError,
    EmbeddingError,
    RetrievalError,
    DimensionMismatchError,
    CircuitOpenError,
)

__all__ = [
    "StatefulRetriever",
    "StatefulRAGConfig",
    "SafetyMode",
    "InMemoryStateStore",
    "PostgresStateStore",
    "PostgresAuditSink",
    "BaseStateStore",
    "CachedHit",
    "ContentCipher",
    "AuditRecord",
    "AuditSink",
    "LoggingAuditSink",
    "InMemoryAuditSink",
    "CompositeAuditSink",
    "StatefulRAGError",
    "ConfigurationError",
    "AuthorizationError",
    "EmbeddingError",
    "RetrievalError",
    "DimensionMismatchError",
    "CircuitOpenError",
    # Lazy framework wrappers (see __getattr__)
    "StatefulLangChainRetriever",
    "StatefulLlamaIndexRetriever",
]

if TYPE_CHECKING:
    from .wrappers.langchain_wrapper import StatefulLangChainRetriever
    from .wrappers.llamaindex_wrapper import StatefulLlamaIndexRetriever


def __getattr__(name: str) -> Any:
    if name in ("StatefulLangChainRetriever", "StatefulLlamaIndexRetriever"):
        from . import wrappers
        return getattr(wrappers, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
