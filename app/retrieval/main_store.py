"""The authoritative knowledge store ("main DB") over the reference corpus.

Two backends behind one interface:

- ``FileMainStore``    : on-disk NumPy index (brute-force cosine). Dependency-free,
  ideal for offline development and beta-scale corpora.
- ``PgVectorMainStore``: PostgreSQL + pgvector with an HNSW index, for the full
  corpus at production scale.

Each stored unit is a :class:`Chunk` carrying its source citation (book + page
range), which the receptionist surfaces so reviewers can verify grounding.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class Chunk:
    id: str
    content: str
    source: str          # human-readable book/source name
    page_start: int
    page_end: int
    section: str = ""

    def citation(self) -> str:
        if self.page_start == self.page_end:
            return f"{self.source}, p.{self.page_start}"
        return f"{self.source}, pp.{self.page_start}-{self.page_end}"

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "section": self.section,
            "citation": self.citation(),
        }


class MainStore(ABC):
    @abstractmethod
    def add(self, chunks: List[Chunk], embeddings: List[List[float]]) -> None: ...

    @abstractmethod
    def search(self, query_embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        """Return docs as dicts: {content, _score, id, source, page_*, citation}."""

    @abstractmethod
    def count(self) -> int: ...


class FileMainStore(MainStore):
    """Brute-force cosine search backed by a .npz embedding matrix + JSON sidecar."""

    def __init__(self, index_path: str, dim: int, embedder_name: str = "unknown"):
        self.index_path = index_path
        self.dim = dim
        self.embedder_name = embedder_name
        self._embeddings = np.empty((0, dim), dtype=np.float32)
        self._chunks: List[Chunk] = []
        if os.path.exists(index_path):
            self.load()

    def add(self, chunks: List[Chunk], embeddings: List[List[float]]) -> None:
        if not chunks:
            return
        arr = np.asarray(embeddings, dtype=np.float32)
        if arr.shape[1] != self.dim:
            raise ValueError(f"embedding dim {arr.shape[1]} != store dim {self.dim}")
        self._embeddings = np.vstack([self._embeddings, arr]) if len(self._embeddings) else arr
        self._chunks.extend(chunks)

    def search(self, query_embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        if len(self._chunks) == 0:
            return []
        q = np.asarray(query_embedding, dtype=np.float32)
        nq = np.linalg.norm(q)
        if nq == 0:
            return []
        norms = np.linalg.norm(self._embeddings, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            sims = (self._embeddings @ q) / (norms * nq)
        sims = np.nan_to_num(sims, nan=-1.0)
        order = np.argsort(sims)[::-1][:top_k]
        out: List[Dict[str, Any]] = []
        for idx in order:
            chunk = self._chunks[int(idx)]
            doc = {"content": chunk.content, "_score": float(sims[int(idx)])}
            doc.update(chunk.to_metadata())
            out.append(doc)
        return out

    def count(self) -> int:
        return len(self._chunks)

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.index_path) or ".", exist_ok=True)
        np.savez_compressed(self.index_path, embeddings=self._embeddings)
        sidecar = self.index_path + ".chunks.json"
        with open(sidecar, "w") as f:
            json.dump([asdict(c) for c in self._chunks], f)
        with open(self.index_path + ".meta.json", "w") as f:
            json.dump({"dim": self.dim, "embedder_name": self.embedder_name,
                       "count": len(self._chunks)}, f)

    def load(self) -> None:
        data = np.load(self.index_path)
        self._embeddings = data["embeddings"].astype(np.float32)
        sidecar = self.index_path + ".chunks.json"
        if os.path.exists(sidecar):
            with open(sidecar) as f:
                self._chunks = [Chunk(**c) for c in json.load(f)]
        meta_path = self.index_path + ".meta.json"
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            self.dim = int(meta.get("dim", self.dim))
            self.embedder_name = meta.get("embedder_name", self.embedder_name)


class PgVectorMainStore(MainStore):
    """PostgreSQL + pgvector backend for the full corpus (production scale)."""

    def __init__(self, session_factory, dim: int):
        self._factory = session_factory
        self.dim = dim
        from .pg_models import KnowledgeChunk  # local import to avoid hard dep at import time
        self._model = KnowledgeChunk

    def add(self, chunks: List[Chunk], embeddings: List[List[float]]) -> None:
        db = self._factory()
        try:
            for chunk, emb in zip(chunks, embeddings):
                db.add(self._model(
                    id=chunk.id, content=chunk.content, source=chunk.source,
                    page_start=chunk.page_start, page_end=chunk.page_end,
                    section=chunk.section, embedding=emb,
                ))
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def search(self, query_embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        db = self._factory()
        try:
            rows = (
                db.query(self._model, self._model.embedding.cosine_distance(query_embedding).label("d"))
                .order_by("d").limit(top_k).all()
            )
            out = []
            for node, dist in rows:
                chunk = Chunk(node.id, node.content, node.source, node.page_start,
                              node.page_end, node.section or "")
                doc = {"content": chunk.content, "_score": 1.0 - float(dist)}
                doc.update(chunk.to_metadata())
                out.append(doc)
            return out
        finally:
            db.close()

    def count(self) -> int:
        db = self._factory()
        try:
            return db.query(self._model).count()
        finally:
            db.close()


def build_main_store(cfg) -> MainStore:
    if cfg.main_store == "file":
        return FileMainStore(cfg.index_path, cfg.embedding_dim)
    if cfg.main_store == "pgvector":
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        if not cfg.database_url:
            raise ValueError("DATABASE_URL required for pgvector main store")
        factory = sessionmaker(bind=create_engine(cfg.database_url))
        return PgVectorMainStore(factory, cfg.embedding_dim)
    raise ValueError(f"Unknown main_store: {cfg.main_store!r}")
