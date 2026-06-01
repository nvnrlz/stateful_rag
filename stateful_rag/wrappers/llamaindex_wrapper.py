"""LlamaIndex-compatible retriever wrapper.

Fixes versus the prototype:

- Real similarity scores are propagated to ``NodeWithScore`` instead of a
  fabricated constant ``1.0`` (which falsely signalled perfect confidence for
  every result).
- The conversation turn is derived from the persisted store rather than an
  in-memory counter.
"""

from __future__ import annotations

from typing import Any, List, Optional

from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, TextNode, QueryBundle


class StatefulLlamaIndexRetriever(BaseRetriever):
    """A LlamaIndex-compatible wrapper for the StatefulRAG engine."""

    def __init__(
        self,
        stateful_retriever: Any,
        session_id: str,
        *,
        principal: Optional[str] = None,
        pinned_turn: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.stateful_retriever = stateful_retriever
        self.session_id = session_id
        self.principal = principal
        self.pinned_turn = pinned_turn

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        raw_docs = self.stateful_retriever.retrieve(
            query=query_bundle.query_str,
            session_id=self.session_id,
            current_turn=self.pinned_turn,
            principal=self.principal,
        )

        nodes: List[NodeWithScore] = []
        for doc in raw_docs:
            if isinstance(doc, dict):
                content = doc.pop("content", None)
                if content is None:
                    content = str(doc)
                score = doc.get("_score")
                metadata = doc
            else:
                content, score, metadata = str(doc), None, {}
            node = TextNode(text=content, metadata=metadata)
            nodes.append(NodeWithScore(node=node, score=score))
        return nodes
