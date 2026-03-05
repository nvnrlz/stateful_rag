import time
import numpy as np
from llama_index.core.schema import QueryBundle
from stateful_rag.stores.memory import InMemoryStateStore
from stateful_rag.retriever import StatefulRetriever
from stateful_rag.wrappers.llamaindex_wrapper import StatefulLlamaIndexRetriever


def mock_embed_fn(text: str) -> list[float]:
    return (np.ones(1536) / np.sqrt(1536)).tolist()


def mock_main_retriever(text: str) -> list[dict]:
    time.sleep(1.0)
    return [{"content": "LlamaIndex integration verified.", "source": "mock_db"}]


if __name__ == "__main__":
    print("=== Testing LlamaIndex Wrapper ===")
    core_retriever = StatefulRetriever(
        state_store=InMemoryStateStore(),
        main_retriever_fn=mock_main_retriever,
        embed_fn=mock_embed_fn
    )

    li_retriever = StatefulLlamaIndexRetriever(
        stateful_retriever=core_retriever,
        session_id="li_user_777"
    )

    print("\n--- Turn 1 (Invoking LlamaIndex Retriever) ---")
    start = time.time()
    nodes = li_retriever.retrieve(QueryBundle("First query"))
    print(f"Latency: {(time.time() - start)*1000:.2f} ms | Turn Internally: {li_retriever.current_turn}")
    print(f"Type returned: {type(nodes[0])} | Content: {nodes[0].node.text}")

    print("\n--- Turn 2 (Cache Hit via LlamaIndex) ---")
    start = time.time()
    nodes = li_retriever.retrieve(QueryBundle("Follow up query"))
    print(f"Latency: {(time.time() - start)*1000:.2f} ms | Turn Internally: {li_retriever.current_turn}")
