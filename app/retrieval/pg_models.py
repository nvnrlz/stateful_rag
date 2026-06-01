"""SQLAlchemy model for the production knowledge store (separate from the cache)."""

from __future__ import annotations

import os

from sqlalchemy import Column, String, Integer, Text
from sqlalchemy.orm import declarative_base
from pgvector.sqlalchemy import Vector

Base = declarative_base()

EMBEDDING_DIM = int(os.environ.get("RAG_EMBED_DIM", "384"))


class KnowledgeChunk(Base):
    __tablename__ = "knowledge_chunks"
    id = Column(String, primary_key=True)
    content = Column(Text, nullable=False)
    source = Column(String, nullable=False, index=True)
    page_start = Column(Integer, nullable=False)
    page_end = Column(Integer, nullable=False)
    section = Column(String, nullable=True)
    embedding = Column(Vector(EMBEDDING_DIM), nullable=False)
