"""Retention job: purge cached PHI older than the configured TTL.

Run on a schedule (cron / k8s CronJob) to enforce data-retention limits:

    DATABASE_URL=... STATEFUL_RAG_CACHE_TTL=86400 python scripts/purge_expired.py
"""

from __future__ import annotations

import os
import sys

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from stateful_rag.config import StatefulRAGConfig
from stateful_rag.stores.postgres import PostgresStateStore


def main() -> int:
    url = os.environ.get("DATABASE_URL")
    if not url:
        sys.stderr.write("ERROR: DATABASE_URL is required\n")
        return 1
    cfg = StatefulRAGConfig()
    factory = sessionmaker(bind=create_engine(url))
    store = PostgresStateStore(factory)
    removed = store.purge_expired(cfg.cache_ttl_seconds)
    print(f"Purged {removed} cached nodes older than {cfg.cache_ttl_seconds}s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
