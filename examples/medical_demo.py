import time
import numpy as np
from sentence_transformers import SentenceTransformer
from stateful_rag.retriever import StatefulRetriever
from stateful_rag.stores.memory import InMemoryStateStore

print("Loading local embedding model (all-MiniLM-L6-v2)...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Note: all-MiniLM outputs 384-dimensional vectors. Cosine similarities for this model
# generally hover between 0.3 and 0.7 for loosely related text. We adjust the drift threshold to 0.35.
DRIFT_THRESHOLD = 0.35

# --- MOCK MEDICAL KNOWLEDGE GRAPH (Main DB) ---
MOCK_KB = [
    {"content": "Cardiology: Patient presents with acute chest pain radiating to the left arm. High risk of myocardial infarction. Heart rate may be elevated or irregular."},
    {"content": "Cardiology: EKG shows ST elevation. Cardiac troponin levels are elevated. Immediate intervention required."},
    {"content": "Dermatology: Patient complains of an itchy red rash on the leg or foot, consistent with contact dermatitis or eczema."},
    {"content": "Dermatology: Treatment for contact dermatitis includes topical hydrocortisone cream and avoiding allergens."}
]


def real_embed_fn(text: str) -> list[float]:
    return model.encode(text).tolist()


def mock_heavy_main_db_search(query: str) -> list[dict]:
    """Simulates an exhaustive, slow vector database search."""
    time.sleep(1.5)  # Simulate O(N) latency
    query_vec = model.encode(query)
    # Simple brute-force cosine similarity for the mock DB
    results = []
    for doc in MOCK_KB:
        doc_vec = model.encode(doc["content"])
        sim = np.dot(query_vec, doc_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(doc_vec))
        results.append({"doc": doc, "sim": float(sim)})

    # Return top 2 results
    results.sort(key=lambda x: x["sim"], reverse=True)
    return [{"content": res["doc"]["content"]} for res in results[:2]]


if __name__ == "__main__":
    store = InMemoryStateStore()
    retriever = StatefulRetriever(
        state_store=store,
        main_retriever_fn=mock_heavy_main_db_search,
        embed_fn=real_embed_fn,
        drift_threshold=DRIFT_THRESHOLD
    )
    session_id = "patient_777"
    print("\n" + "=" * 80)
    print("🩺 STATEFUL-RAG: SEMANTIC DRIFT & CLINICAL SAFETY DEMO")
    print("=" * 80)

    conversation = [
        ("I'm having severe chest pain and my left arm hurts.", "Turn 1: Initial Triage (Expected: Slow, Main DB)"),
        ("Is my heart rate normal? I feel dizzy.", "Turn 2: Follow-up on Cardiology (Expected: Fast, Cache Hit)"),
        ("By the way, I also have this really itchy red rash on my leg.", "Turn 3: Topic Shift (Expected: Context Drift! Breaks Cache -> Slow, Main DB)"),
        ("Should I put some cream on the rash?", "Turn 4: Follow-up on Dermatology (Expected: Fast, Cache Hit on NEW context)")
    ]

    for turn_idx, (user_input, description) in enumerate(conversation, 1):
        print(f"\n🗣️  Patient: '{user_input}'")
        print(f"   {description}")
        start_time = time.time()
        docs = retriever.retrieve(user_input, session_id, current_turn=turn_idx)
        latency = (time.time() - start_time) * 1000

        print(f"⏱️  Latency: {latency:.2f} ms")
        print(f"📄 Retrieved Top Context: {docs[0]['content'] if isinstance(docs[0], dict) else docs[0]}")
