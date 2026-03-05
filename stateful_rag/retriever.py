from typing import List, Callable, Any, Dict
from .stores.base import BaseStateStore


class StatefulRetriever:
    """The core engine that routes between the main Vector DB and the Stateful Cache."""

    def __init__(
        self,
        state_store: BaseStateStore,
        main_retriever_fn: Callable[[str], List[Dict[str, Any]]],
        embed_fn: Callable[[str], List[float]],
        drift_threshold: float = 0.85
    ):
        self.store = state_store
        self.main_retriever_fn = main_retriever_fn
        self.embed_fn = embed_fn
        self.drift_threshold = drift_threshold

    def retrieve(self, query: str, session_id: str, current_turn: int) -> List[Dict[str, Any]]:
        query_vector = self.embed_fn(query)
        cached_results = self.store.search_cache(session_id, query_vector, top_k=3)

        if cached_results:
            best_doc, best_similarity = cached_results[0]
            if best_similarity >= self.drift_threshold:
                print(f"[Turn {current_turn}] Cache hit! Executing Constrained Re-Analysis... (Sim: {best_similarity:.2f})")
                return [doc for doc, score in cached_results]
            else:
                print(f"[Turn {current_turn}] Context drift detected (Sim: {best_similarity:.2f} < {self.drift_threshold}). Breaking cache...")
        else:
            print(f"[Turn {current_turn}] Cache empty. Executing exhaustive Main DB search...")

        documents = self.main_retriever_fn(query)
        embeddings = [self.embed_fn(doc.get("content", "")) for doc in documents]
        self.store.save_session_context(session_id, current_turn, documents, embeddings)

        return documents
