import time
import uuid
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from stateful_rag.retriever import StatefulRetriever
from stateful_rag.stores.postgres import PostgresStateStore

DB_URL = "postgresql+psycopg://rag_user:rag_password@localhost:5433/rag_state"
engine = create_engine(DB_URL)
SessionLocal = sessionmaker(bind=engine)

MATCHING_VECTOR = (np.ones(1536) / np.sqrt(1536)).tolist()


def mock_embed_fn(text: str) -> list[float]:
    return MATCHING_VECTOR


def mock_main_retriever(text: str) -> list[dict]:
    time.sleep(1.2)  # Simulate slow O(N) database search
    return [{"content": f"Postgres Mock Data: {text}", "source": "main_db"}]


if __name__ == "__main__":
    db_session = SessionLocal()
    store = PostgresStateStore(db_session=db_session)
    retriever = StatefulRetriever(
        state_store=store, main_retriever_fn=mock_main_retriever, embed_fn=mock_embed_fn
    )
    session_id = f"test_pg_{uuid.uuid4().hex[:6]}"
    print(f"=== Testing Postgres Backend (Session: {session_id}) ===")

    print("\n--- TURN 1 (Massive O(N) Search -> Writes to Postgres) ---")
    start = time.time()
    docs1 = retriever.retrieve("My stomach hurts", session_id, 1)
    print(f"Latency: {(time.time() - start)*1000:.2f} ms | Output: {docs1[0]['content']}")

    print("\n--- TURN 2 (O(1) Constrained Re-Analysis -> Reads from Postgres) ---")
    start = time.time()
    docs2 = retriever.retrieve("It started after eating", session_id, 2)
    print(f"Latency: {(time.time() - start)*1000:.2f} ms | Output: {docs2[0]['content']}")
