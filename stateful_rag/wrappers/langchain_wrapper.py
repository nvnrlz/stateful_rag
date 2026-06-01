"""LangChain-compatible retriever wrapper.

Fixes versus the prototype:

- The conversation turn is derived from the persisted store (``current_turn=None``
  -> ``store.next_turn``), not from a fragile in-memory counter that breaks under
  concurrency and across restarts.
- Real similarity scores and provenance flow into LangChain ``Document`` metadata
  (``score``, ``route``, ``source``) instead of being discarded.
"""

from __future__ import annotations

from typing import Any, List, Optional

from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from pydantic import Field, ConfigDict


class StatefulLangChainRetriever(BaseRetriever):
    """A LangChain-compatible wrapper for the StatefulRAG engine."""

    stateful_retriever: Any = Field(description="The core StatefulRetriever instance")
    session_id: str = Field(description="The unique session ID for the patient/user")
    principal: Optional[str] = Field(default=None, description="Calling principal for authz")
    # Optional explicit turn pin; when None the turn is derived from the store.
    pinned_turn: Optional[int] = Field(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        raw_docs = self.stateful_retriever.retrieve(
            query=query,
            session_id=self.session_id,
            current_turn=self.pinned_turn,
            principal=self.principal,
        )

        lc_documents: List[Document] = []
        for doc in raw_docs:
            if isinstance(doc, dict):
                content = doc.pop("content", None)
                if content is None:
                    content = str(doc)
                metadata = doc
            else:
                content, metadata = str(doc), {}
            lc_documents.append(Document(page_content=content, metadata=metadata))
        return lc_documents
