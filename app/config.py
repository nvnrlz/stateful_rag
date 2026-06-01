"""Application configuration (env-overridable).

Defaults are chosen so the backend runs fully offline for development and CI
(hashing embedder + on-disk vector store + rule-based dialogue). Production runs
opt into real providers (Sentence-Transformers / OpenAI embeddings, an LLM, and
pgvector) purely through environment variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader: sets vars that aren't already in the environment.

    Avoids a python-dotenv dependency. Existing environment variables win, so
    explicit overrides on the command line are respected.
    """
    if not os.path.exists(path):
        return
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)


# Load .env as early as possible so config + providers see the values.
load_dotenv()


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass
class AppConfig:
    # --- Corpus / data ---
    corpus_dir: str = field(default_factory=lambda: _env("RAG_CORPUS_DIR", "/Users/naveen/Downloads/RAG DB"))
    data_dir: str = field(default_factory=lambda: _env("RAG_DATA_DIR", "data"))
    # On-disk index path for the offline file-backed main store.
    index_path: str = field(default_factory=lambda: _env("RAG_INDEX_PATH", "data/ent_index.npz"))

    # --- Embeddings ---
    # provider: hashing | sentence_transformers | openai
    embedding_provider: str = field(default_factory=lambda: _env("RAG_EMBED_PROVIDER", "hashing"))
    embedding_model: str = field(default_factory=lambda: _env("RAG_EMBED_MODEL", "hashing-v1"))
    embedding_dim: int = field(default_factory=lambda: _env_int("RAG_EMBED_DIM", 384))

    # --- Vector store ---
    # store: file | pgvector
    main_store: str = field(default_factory=lambda: _env("RAG_MAIN_STORE", "file"))
    database_url: str = field(default_factory=lambda: _env("DATABASE_URL", ""))

    # --- Chunking ---
    chunk_chars: int = field(default_factory=lambda: _env_int("RAG_CHUNK_CHARS", 1200))
    chunk_overlap: int = field(default_factory=lambda: _env_int("RAG_CHUNK_OVERLAP", 200))

    # --- Retrieval ---
    main_top_k: int = field(default_factory=lambda: _env_int("RAG_MAIN_TOP_K", 5))

    # --- StatefulRAG drift thresholds (calibrate per embedder: app.eval.calibrate) ---
    drift_threshold: float = field(default_factory=lambda: _env_float("RAG_DRIFT_THRESHOLD", 0.6))
    per_doc_floor: float = field(default_factory=lambda: _env_float("RAG_PER_DOC_FLOOR", 0.2))

    # --- Dialogue / LLM ---
    # provider: rule_based | openai | anthropic
    llm_provider: str = field(default_factory=lambda: _env("RAG_LLM_PROVIDER", "rule_based"))
    llm_model: str = field(default_factory=lambda: _env("RAG_LLM_MODEL", ""))
    max_turns: int = field(default_factory=lambda: _env_int("RAG_MAX_TURNS", 8))

    # --- Stateful cache ---
    cache_store: str = field(default_factory=lambda: _env("RAG_CACHE_STORE", "memory"))  # memory | pgvector

    def is_offline(self) -> bool:
        return self.embedding_provider == "hashing" and self.llm_provider == "rule_based"


def load_config() -> AppConfig:
    return AppConfig()
