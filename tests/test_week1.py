import time
import numpy as np
from stateful_rag.retriever import StatefulRetriever
from stateful_rag.stores.memory import InMemoryStateStore

# Hardcoded vectors to ensure deterministic cache hits/misses for testing
MATCHING_VECTOR = (np.ones(1536) / np.sqrt(1536)).tolist()
DRIFT_VECTOR = (np.ones(1536) * -1 / np.sqrt(1536)).tolist()


def mock_embed_fn(text: str) -> list[float]:
    if "chest" in text.lower() or "hours" in text.lower():
        return MATCHING_VECTOR
    return DRIFT_VECTOR


def mock_main_retriever(text: str) -> list[dict]:
    time.sleep(1.2)  # Simulate slow O(N) database search
    return [{"content": f"Mock medical knowledge for: {text}"}]


if __name__ == "__main__":
    store = InMemoryStateStore()
    retriever = StatefulRetriever(
        state_store=store,
        main_retriever_fn=mock_main_retriever,
        embed_fn=mock_embed_fn,
        drift_threshold=0.85
    )
    session_id = "patient_123"

    print("--- INTERACTION TURN 1 ---")
    start = time.time()
    retriever.retrieve("I have severe chest pain", session_id, current_turn=1)
    latency = (time.time() - start) * 1000
    print(f"Turn 1 Latency: {latency:.2f} ms\n")

    print("--- INTERACTION TURN 2 ---")
    start = time.time()
    retriever.retrieve("It started about 2 hours ago", session_id, current_turn=2)
    latency = (time.time() - start) * 1000
    print(f"Turn 2 Latency: {latency:.2f} ms\n")

    print("--- INTERACTION TURN 3 (Context Drift) ---")
    start = time.time()
    retriever.retrieve("Can you also check my foot?", session_id, current_turn=3)
    latency = (time.time() - start) * 1000
    print(f"Turn 3 Latency: {latency:.2f} ms\n")
