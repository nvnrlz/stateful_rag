"""Initialise the StatefulRAG PostgreSQL schema.

Versus the prototype this script:

- Reads the connection string and embedding dimension from the environment
  (never hardcoded credentials).
- Enables pgvector, creates the tables (sessions, cached_nodes, audit_log), and
  **creates an HNSW ANN index** on the embedding column. Without this index
  pgvector falls back to an exact sequential scan, which is what makes the
  prototype's "sub-10ms" claim impossible to reproduce at scale.
"""

from __future__ import annotations

import os
import sys

from sqlalchemy import create_engine, text

from stateful_rag.models import Base, EMBEDDING_DIM


def _db_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        sys.stderr.write(
            "ERROR: set DATABASE_URL, e.g.\n"
            "  export DATABASE_URL='postgresql+psycopg://USER:PASS@localhost:5433/rag_state'\n"
        )
        sys.exit(1)
    return url


def init_database() -> None:
    engine = create_engine(_db_url())
    print("Enabling pgvector extension...")
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        conn.commit()

    print("Creating tables (sessions, cached_nodes, audit_log)...")
    Base.metadata.create_all(engine)

    # HNSW index for fast approximate nearest-neighbour search. Cosine ops match
    # the cosine_distance used by the store. m / ef_construction are reasonable
    # defaults; tune for your corpus size.
    print(f"Creating HNSW index on cached_nodes.embedding (dim={EMBEDDING_DIM})...")
    with engine.connect() as conn:
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS ix_cached_nodes_embedding_hnsw "
            "ON cached_nodes USING hnsw (embedding vector_cosine_ops) "
            "WITH (m = 16, ef_construction = 64);"
        ))
        conn.commit()

    print("✅ Database initialised.")


if __name__ == "__main__":
    init_database()
