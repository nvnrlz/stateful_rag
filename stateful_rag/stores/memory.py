import numpy as np
from typing import List, Dict, Any, Tuple
from .base import BaseStateStore


class InMemoryStateStore(BaseStateStore):
    """A lightweight, dependency-free vector store for local testing."""

    def __init__(self):
        self.store: Dict[str, Dict[str, Any]] = {}

    def save_session_context(self, session_id: str, turn: int, documents: List[Dict[str, Any]], embeddings: List[List[float]]):
        if not documents or not embeddings:
            return
        if session_id not in self.store:
            self.store[session_id] = {"documents": [], "embeddings": np.empty((0, len(embeddings[0])))}

        for doc in documents:
            doc_copy = doc.copy()
            doc_copy["turn_added"] = turn
            self.store[session_id]["documents"].append(doc_copy)

        new_embeddings_array = np.array(embeddings)
        self.store[session_id]["embeddings"] = np.vstack([self.store[session_id]["embeddings"], new_embeddings_array])

    def search_cache(self, session_id: str, query_embedding: List[float], top_k: int = 3) -> List[Tuple[Dict[str, Any], float]]:
        if session_id not in self.store or len(self.store[session_id]["documents"]) == 0:
            return []

        query_vec = np.array(query_embedding)
        cache_vecs = self.store[session_id]["embeddings"]

        dot_products = np.dot(cache_vecs, query_vec)
        norms_cache = np.linalg.norm(cache_vecs, axis=1)
        norm_query = np.linalg.norm(query_vec)

        if norm_query == 0 or np.any(norms_cache == 0):
            return []

        similarities = dot_products / (norms_cache * norm_query)
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            doc = self.store[session_id]["documents"][idx]
            results.append((doc, score))
        return results
