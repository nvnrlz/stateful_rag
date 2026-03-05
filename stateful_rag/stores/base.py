from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple


class BaseStateStore(ABC):
    """The abstract interface that all StatefulRAG databases must implement."""

    @abstractmethod
    def save_session_context(self, session_id: str, turn: int, documents: List[Dict[str, Any]], embeddings: List[List[float]]):
        pass

    @abstractmethod
    def search_cache(self, session_id: str, query_embedding: List[float], top_k: int = 3) -> List[Tuple[Dict[str, Any], float]]:
        pass
