"""Shared HTTP helper for the Gemini REST providers: retry with backoff.

Free-tier Gemini enforces tight rate limits, so a burst of calls (extraction +
embeddings + phrasing within one turn) can transiently return 429. A single 429
must NOT surface to the patient as 'service unavailable' — we retry with
exponential backoff (honouring Retry-After) and only give up after several
attempts.
"""

from __future__ import annotations

import time

import httpx

# Status codes worth retrying (rate limit + transient server errors).
_RETRYABLE = {429, 500, 502, 503, 504}


def post_with_retry(client: httpx.Client, url: str, *, params: dict, json: dict,
                    max_retries: int = 2, base_delay: float = 1.0) -> httpx.Response:
    attempt = 0
    while True:
        try:
            r = client.post(url, params=params, json=json)
            if r.status_code in _RETRYABLE and attempt < max_retries:
                raise _Retry(_retry_after(r, base_delay, attempt))
            r.raise_for_status()
            return r
        except _Retry as rt:
            time.sleep(rt.delay)
            attempt += 1
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError):
            if attempt >= max_retries:
                raise
            time.sleep(base_delay * (2 ** attempt))
            attempt += 1


class _Retry(Exception):
    def __init__(self, delay: float):
        self.delay = delay


def _retry_after(resp: httpx.Response, base_delay: float, attempt: int) -> float:
    # Cap waits low so an interactive request fails fast (to a friendly retry
    # message) instead of blocking a worker thread for many seconds.
    hdr = resp.headers.get("Retry-After")
    if hdr:
        try:
            return min(float(hdr), 5.0)
        except ValueError:
            pass
    return min(base_delay * (2 ** attempt), 5.0)
