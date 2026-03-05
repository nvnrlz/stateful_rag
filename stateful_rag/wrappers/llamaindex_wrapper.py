from typing import List, Any
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, TextNode, QueryBundle


class StatefulLlamaIndexRetriever(BaseRetriever):
    """A LlamaIndex-compatible wrapper for the StatefulRAG engine."""

    def __init__(
        self,
        stateful_retriever: Any,
        session_id: str,
        current_turn: int = 1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.stateful_retriever = stateful_retriever
        self.session_id = session_id
        self.current_turn = current_turn

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        raw_docs = self.stateful_retriever.retrieve(
            query=query_bundle.query_str,
            session_id=self.session_id,
            current_turn=self.current_turn
        )
        self.current_turn += 1

        nodes = []
        for doc in raw_docs:
            content = doc.pop("content", str(doc)) if isinstance(doc, dict) else str(doc)
            metadata = doc if isinstance(doc, dict) else {}

            # Wrap output in LlamaIndex TextNodes
            node = TextNode(text=content, metadata=metadata)
            nodes.append(NodeWithScore(node=node, score=1.0))

        return nodes
