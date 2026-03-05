from typing import List, Any
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from pydantic import Field, ConfigDict


class StatefulLangChainRetriever(BaseRetriever):
    """A LangChain-compatible wrapper for the StatefulRAG engine."""

    stateful_retriever: Any = Field(description="The core StatefulRetriever instance")
    session_id: str = Field(description="The unique session ID for the patient/user")
    current_turn: int = Field(default=1, description="The auto-incrementing conversation turn")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:

        raw_docs = self.stateful_retriever.retrieve(
            query=query,
            session_id=self.session_id,
            current_turn=self.current_turn
        )

        # Auto-increment the turn so the developer doesn't have to manage state!
        self.current_turn += 1

        lc_documents = []
        for doc in raw_docs:
            # Safely extract content and push the rest into LangChain metadata
            content = doc.pop("content", str(doc)) if isinstance(doc, dict) else str(doc)
            metadata = doc if isinstance(doc, dict) else {}
            lc_documents.append(Document(page_content=content, metadata=metadata))

        return lc_documents
