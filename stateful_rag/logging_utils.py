"""Structured, JSON-friendly logging for StatefulRAG.

The original prototype used bare ``print()`` calls, which cannot be aggregated,
filtered, or correlated in production. This module provides a single logger that
emits structured records with a per-request correlation id, so every routing
decision can be traced through a log pipeline (CloudWatch, Loki, Datadog, ...).
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from contextvars import ContextVar
from typing import Any

# A correlation id that flows through a single retrieval request. Set it at the
# top of a request handler with ``set_correlation_id()`` and every log line plus
# audit record emitted downstream will carry it.
_correlation_id: ContextVar[str] = ContextVar("stateful_rag_correlation_id", default="-")

_LOGGER_NAME = "stateful_rag"
_CONFIGURED = False


class _JsonFormatter(logging.Formatter):
    """Render log records as single-line JSON for machine ingestion."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "correlation_id": getattr(record, "correlation_id", _correlation_id.get()),
        }
        # Attach any structured extras passed via ``logger.info(..., extra={"fields": {...}})``.
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            payload.update(fields)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def get_logger() -> logging.Logger:
    """Return the shared StatefulRAG logger, configuring it once on first use."""
    global _CONFIGURED
    logger = logging.getLogger(_LOGGER_NAME)
    if not _CONFIGURED:
        level = os.environ.get("STATEFUL_RAG_LOG_LEVEL", "INFO").upper()
        handler = logging.StreamHandler()
        # JSON in production; human-readable when explicitly requested for local dev.
        if os.environ.get("STATEFUL_RAG_LOG_FORMAT", "json").lower() == "json":
            handler.setFormatter(_JsonFormatter())
        else:
            handler.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
            )
        logger.addHandler(handler)
        logger.setLevel(getattr(logging, level, logging.INFO))
        logger.propagate = False
        _CONFIGURED = True
    return logger


def set_correlation_id(value: str | None = None) -> str:
    """Bind a correlation id to the current context, generating one if needed."""
    cid = value or uuid.uuid4().hex
    _correlation_id.set(cid)
    return cid


def get_correlation_id() -> str:
    """Return the correlation id bound to the current context."""
    return _correlation_id.get()


def log_event(level: int, msg: str, **fields: Any) -> None:
    """Emit a structured log event with arbitrary key/value fields."""
    logger = get_logger()
    logger.log(
        level,
        msg,
        extra={"fields": fields, "correlation_id": _correlation_id.get()},
    )
