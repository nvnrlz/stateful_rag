"""Resilience primitives: bounded retries with backoff and a circuit breaker.

In the prototype, a single transient failure of the embedding API or main
retriever would crash the entire retrieval turn. For a diagnostic backend that
is unacceptable. These helpers wrap the two external calls (embed / main
retrieve) so that transient faults are retried, and a persistently failing
dependency trips a circuit breaker instead of hammering it.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, TypeVar

from .exceptions import CircuitOpenError
from .logging_utils import get_logger, log_event
import logging

T = TypeVar("T")


def retry_call(
    fn: Callable[[], T],
    *,
    max_retries: int,
    base_delay: float,
    name: str,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call ``fn`` with exponential backoff. Re-raises the last error on exhaustion."""
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as exc:
            if attempt >= max_retries:
                log_event(
                    logging.ERROR,
                    "dependency_call_failed",
                    dependency=name,
                    attempts=attempt + 1,
                    error=str(exc),
                )
                raise
            delay = base_delay * (2 ** attempt)
            log_event(
                logging.WARNING,
                "dependency_call_retry",
                dependency=name,
                attempt=attempt + 1,
                delay=delay,
                error=str(exc),
            )
            sleep(delay)
            attempt += 1


class CircuitBreaker:
    """A minimal thread-safe circuit breaker.

    After ``fail_threshold`` consecutive failures the circuit *opens* and calls
    short-circuit with ``CircuitOpenError`` for ``reset_seconds``. The first call
    after that window is allowed through (half-open); success closes the circuit,
    failure re-opens it.
    """

    def __init__(self, *, fail_threshold: int, reset_seconds: float, name: str):
        self._fail_threshold = fail_threshold
        self._reset_seconds = reset_seconds
        self._name = name
        self._failures = 0
        self._opened_at: float | None = None
        self._lock = threading.Lock()

    @property
    def is_open(self) -> bool:
        with self._lock:
            return self._opened_at is not None and not self._half_open_ready()

    def _half_open_ready(self) -> bool:
        return (
            self._opened_at is not None
            and (time.monotonic() - self._opened_at) >= self._reset_seconds
        )

    def call(self, fn: Callable[[], T]) -> T:
        with self._lock:
            if self._opened_at is not None and not self._half_open_ready():
                raise CircuitOpenError(
                    f"Circuit '{self._name}' is open; dependency is unhealthy."
                )
        try:
            result = fn()
        except Exception:
            self._record_failure()
            raise
        self._record_success()
        return result

    def _record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._failures >= self._fail_threshold and self._opened_at is None:
                self._opened_at = time.monotonic()
                log_event(
                    logging.ERROR,
                    "circuit_opened",
                    circuit=self._name,
                    failures=self._failures,
                )

    def _record_success(self) -> None:
        with self._lock:
            if self._opened_at is not None:
                log_event(logging.INFO, "circuit_closed", circuit=self._name)
            self._failures = 0
            self._opened_at = None


class GuardedCall:
    """Combines retry + circuit breaker for a single external dependency."""

    def __init__(self, *, name: str, max_retries: int, base_delay: float,
                 fail_threshold: int, reset_seconds: float,
                 sleep: Callable[[float], None] = time.sleep):
        self._name = name
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._sleep = sleep
        self._breaker = CircuitBreaker(
            fail_threshold=fail_threshold, reset_seconds=reset_seconds, name=name
        )

    def __call__(self, fn: Callable[[], T]) -> T:
        return self._breaker.call(
            lambda: retry_call(
                fn,
                max_retries=self._max_retries,
                base_delay=self._base_delay,
                name=self._name,
                sleep=self._sleep,
            )
        )
