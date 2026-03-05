import time
import numpy as np
from stateful_rag.stores.memory import InMemoryStateStore
from stateful_rag.retriever import StatefulRetriever
from stateful_rag.wrappers.langchain_wrapper import StatefulLangChainRetriever


def mock_embed_fn(text: str) -> list[float]:
    return (np.ones(1536) / np.sqrt(1536)).tolist()


def mock_main_retriever(text: str) -> list[dict]:
    time.sleep(1.0)
    return [{"content": "LangChain integration verified.", "source": "mock_db"}]


if __name__ == "__main__":
    print("=== Testing LangChain Wrapper ===")
    core_retriever = StatefulRetriever(
        state_store=InMemoryStateStore(), main_retriever_fn=mock_main_retriever, embed_fn=mock_embed_fn
    )
    # Developer injects our wrapper into their LangChain pipeline
    lc_retriever = StatefulLangChainRetriever(
        stateful_retriever=core_retriever, session_id="lc_user_123"
    )

    print("\n--- Turn 1 (Invoking LangChain Retriever) ---")
    start = time.time()
    docs = lc_retriever.invoke("Test query")
    print(f"Latency: {(time.time() - start)*1000:.2f} ms | Turn Counter Internally: {lc_retriever.current_turn}")
    print(f"Type returned: {type(docs[0])} | Content: {docs[0].page_content} | Meta: {docs[0].metadata}")

    print("\n--- Turn 2 (Cache Hit via LangChain Invoke) ---")
    start = time.time()
    docs = lc_retriever.invoke("Follow up query")  # Notice the user doesn't pass 'current_turn' anymore!
    print(f"Latency: {(time.time() - start)*1000:.2f} ms | Turn Counter Internally: {lc_retriever.current_turn}")
