import json
from typing import List, Dict, Any, Tuple
from sqlalchemy.orm import Session as DBSession
from .base import BaseStateStore
from ..models import SessionModel, CachedNode


class PostgresStateStore(BaseStateStore):
    """Production-grade caching using PostgreSQL and pgvector."""

    def __init__(self, db_session: DBSession):
        self.db = db_session

    def save_session_context(self, session_id: str, turn: int, documents: List[Dict[str, Any]], embeddings: List[List[float]]):
        if not documents or not embeddings:
            return
        session = self.db.query(SessionModel).filter(SessionModel.id == session_id).first()
        if not session:
            session = SessionModel(id=session_id)
            self.db.add(session)

        for doc, emb in zip(documents, embeddings):
            node = CachedNode(session_id=session_id, turn_added=turn, content=json.dumps(doc), embedding=emb)
            self.db.add(node)
        self.db.commit()

    def search_cache(self, session_id: str, query_embedding: List[float], top_k: int = 3) -> List[Tuple[Dict[str, Any], float]]:
        count = self.db.query(CachedNode).filter(CachedNode.session_id == session_id).count()
        if count == 0:
            return []

        results = (
            self.db.query(CachedNode, CachedNode.embedding.cosine_distance(query_embedding).label("distance"))
            .filter(CachedNode.session_id == session_id)
            .order_by("distance")
            .limit(top_k)
            .all()
        )

        output = []
        for node, distance in results:
            similarity = 1.0 - distance
            doc = json.loads(node.content)
            doc["turn_added"] = node.turn_added
            output.append((doc, float(similarity)))
        return output
