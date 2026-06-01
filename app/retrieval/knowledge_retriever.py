"""Adapts the embedder + main store into the ``main_retriever_fn`` StatefulRAG expects."""

from __future__ import annotations

from typing import Any, Callable, Dict, List

from ..providers.embeddings import EmbeddingProvider
from .main_store import MainStore


class KnowledgeRetriever:
    """Embeds a query and fetches grounded chunks from the authoritative store."""

    def __init__(self, embedder: EmbeddingProvider, store: MainStore, top_k: int = 5):
        self.embedder = embedder
        self.store = store
        self.top_k = top_k

    def __call__(self, query: str) -> List[Dict[str, Any]]:
        vec = self.embedder.embed(query)
        return self.store.search(vec, self.top_k)

    def as_fn(self) -> Callable[[str], List[Dict[str, Any]]]:
        return self.__call__
