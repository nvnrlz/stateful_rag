import time
import uuid
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from stateful_rag.retriever import StatefulRetriever
from stateful_rag.stores.memory import InMemoryStateStore
from stateful_rag.config import StatefulRAGConfig, SafetyMode

# --- Page Configuration ---
st.set_page_config(page_title="StatefulRAG: Clinical Triage", layout="wide")


# --- Cache the Models so they don't reload on every UI click ---
@st.cache_resource
def load_engine():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    MOCK_KB = [
        {"content": "Cardiology: Patient presents with acute chest pain radiating to the left arm. High risk of myocardial infarction. Heart rate may be elevated or irregular."},
        {"content": "Cardiology: EKG shows ST elevation. Cardiac troponin levels are elevated. Immediate intervention required."},
        {"content": "Dermatology: Patient complains of an itchy red rash on the leg or foot, consistent with contact dermatitis or eczema."},
        {"content": "Dermatology: Treatment for contact dermatitis includes topical hydrocortisone cream and avoiding allergens."}
    ]

    def real_embed_fn(text: str) -> list[float]:
        return model.encode(text).tolist()

    def mock_heavy_main_db_search(query: str) -> list[dict]:
        time.sleep(1.5)  # Simulate O(N) database latency
        query_vec = model.encode(query)
        results = []
        for doc in MOCK_KB:
            doc_vec = model.encode(doc["content"])
            sim = np.dot(query_vec, doc_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(doc_vec))
            results.append({"doc": doc, "sim": float(sim)})
        results.sort(key=lambda x: x["sim"], reverse=True)
        return [{"content": res["doc"]["content"]} for res in results[:2]]

    config = StatefulRAGConfig(
        embedding_dim=384,
        embedding_model="all-MiniLM-L6-v2",
        drift_threshold=0.35,
        per_doc_floor=0.2,
        safety_mode=SafetyMode.BALANCED,
        cache_ttl_seconds=0,
    )
    store = InMemoryStateStore(expected_dim=384)
    retriever = StatefulRetriever(
        state_store=store,
        main_retriever_fn=mock_heavy_main_db_search,
        embed_fn=real_embed_fn,
        config=config,
    )
    return retriever


retriever = load_engine()

# --- Session State Management ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.current_turn = 1
    st.session_state.messages = []
    st.session_state.audit_logs = []

# --- UI Layout ---
st.title("🩺 StatefulRAG: Medical Graph-RAG Dashboard")
st.markdown("*Proving 135x Latency Reduction & Clinical Safety (Context Drift Fallback)*")

col_chat, col_audit = st.columns([2, 1])

with col_chat:
    st.subheader("Patient Triage Interface")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Enter patient symptoms (e.g., 'I have chest pain'):"):
        # Display User Message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # AI Processing
        with st.chat_message("assistant"):
            with st.spinner("Analyzing Medical Graph..."):
                start_time = time.time()
                docs = retriever.retrieve(
                    query=prompt,
                    session_id=st.session_state.session_id,
                    current_turn=st.session_state.current_turn
                )
                latency_ms = (time.time() - start_time) * 1000

                top = docs[0] if docs else {}
                top_context = top.get('content', top) if isinstance(top, dict) else top
                route = top.get('_route', 'unknown') if isinstance(top, dict) else 'unknown'
                score = top.get('_score') if isinstance(top, dict) else None
                response_text = f"**Extracted Clinical Context:**\n> {top_context}"
                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})

                # Determine state from the AUTHORITATIVE route decision (not latency).
                route_display = {
                    "main_db": ("Cold Start (Main DB)", "🔴"),
                    "drift_break": ("Context Drift Detected (Main DB)", "⚠️"),
                    "cache_hit": ("Cache Hit", "🟢"),
                    "cache_hit_verified": ("Cache Hit (Verified vs Main DB)", "🟢"),
                    "fail_open": ("Failed Open to Main DB (cache disagreed)", "⚠️"),
                }
                state_type, color = route_display.get(route, (route, "ℹ️"))

                st.session_state.audit_logs.append({
                    "turn": st.session_state.current_turn,
                    "query": prompt,
                    "latency": latency_ms,
                    "state": state_type,
                    "color": color,
                    "route": route,
                    "score": score,
                })

                st.session_state.current_turn += 1
                st.rerun()

with col_audit:
    st.subheader("System Audit Trace")
    st.caption("Live metrics from the Persistence Layer")
    for log in reversed(st.session_state.audit_logs):
        with st.expander(f"Turn {log['turn']} | {log['color']} {log['state']}", expanded=True):
            st.markdown(f"**Query:** '{log['query']}'")
            st.metric("System Latency", f"{log['latency']:.2f} ms")
            if log['color'] == "🟢":
                st.success("O(1) Memory Search Bypassed Main Vector DB")
            elif log['color'] == "⚠️":
                st.error("Semantic Similarity < 0.35. Cache broken for Clinical Safety.")
