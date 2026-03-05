# 🩺 StatefulRAG

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![PostgreSQL + pgvector](https://img.shields.io/badge/PostgreSQL-pgvector-blue.svg)](https://github.com/pgvector/pgvector)

**Sustainable, $\mathcal{O}(1)$ memory caching for multi-turn Retrieval-Augmented Generation (RAG).**

**Cut your vector database costs and reduce system latency by up to 135x.**

---

## 📖 About

StatefulRAG is the official open-source implementation of the caching and routing architecture proposed in the research paper: **"Stateful, Multilingual Medical Graph-RAG Framework for Sustainable and Iterative Clinical Triage"** (Kamalakannan et al., 2026).

While originally designed to solve critical bottlenecks in clinical decision support systems (CDSS) specifically **extreme system latency**, **cross-lingual AI hallucinations**, and **unstructured generative guessing**, this library abstracts those solutions into a **domain-agnostic framework**. Whether you are building medical triage agents, legal research assistants, or customer support bots, StatefulRAG provides a strictly auditable, highly constrained, and environmentally sustainable ("Green AI") memory layer for multi-turn conversations.

**Authors & Researchers:**
- **Naveen Kamalakannan** (TechZilla Solutions)
- **Ankita Jogekar** (Independent Researcher)
- **Umaima Haider** (University of East London)

---

## 🚨 The Problem: Multi-Turn RAG is Unsustainable

Standard conversational RAG agents are **stateless**. For every multi-turn follow-up question a user asks, the system must re-embed the entire chat history and execute a massive $\mathcal{O}(N)$ similarity search across millions of vectors.

In high-traffic deployments, this continuous redundant querying results in **extreme latency**, **prohibitive cloud GPU costs**, and a **massive carbon footprint**.

---

## 💡 The Solution: Stateful Caching & Constrained Re-Analysis

StatefulRAG introduces a robust **memory middleware layer** backed by PostgreSQL + pgvector (or in-memory NumPy for local testing).

- **Turn 1 (Cold Start):** Queries your heavy, main Vector DB (Pinecone, Qdrant, etc.) and **caches** the retrieved semantic sub-graph to PostgreSQL.
- **Turn 2+ (Cache Hit):** **Bypasses the main DB entirely.** Executes lightning-fast "constrained re-analysis" directly on the PostgreSQL localized cache.
- **Context Drift Fallback:** If the user abruptly changes topics (e.g., from "chest pain" to "billing questions"), the system detects a drop in semantic similarity, **safely breaks the cache**, and re-queries the main database to prevent AI hallucinations.

---

## 📊 Validated Benchmarks (The "Green AI" Proof)

Derived from our published clinical triage research:

| Metric | Stateless RAG (Standard) | StatefulRAG (PostgreSQL) | Speedup |
|---|---|---|---|
| Turn 1 (Cold DB Search) | ~1270 ms | ~1270 ms | 1x |
| Turn 2 (Follow-up Query) | ~1450 ms | ~9.4 ms | **~135x** |

---

## 🚀 Quick Start

### Installation

**Core framework** (lightweight - SQLAlchemy, pgvector, NumPy, LangChain, LlamaIndex):

```bash
pip install stateful-rag
```

**With demo extras** (adds Sentence-Transformers + Streamlit for running examples):

```bash
pip install "stateful-rag[demo]"
```

### 1. Framework Integrations (LangChain & LlamaIndex)

StatefulRAG works natively with your existing pipelines. Just wrap your core retriever.

**Using LangChain:**

```python
from stateful_rag.wrappers import StatefulLangChainRetriever
from stateful_rag.stores.memory import InMemoryStateStore  # Switch to PostgresStateStore for production
from stateful_rag.retriever import StatefulRetriever

# 1. Setup your state store
store = InMemoryStateStore()

# 2. Initialize the core engine with your existing DB fetcher
core_retriever = StatefulRetriever(
    state_store=store,
    main_retriever_fn=my_pinecone_search,
    embed_fn=my_openai_embedder,
    drift_threshold=0.35  # Safety threshold
)

# 3. Wrap it for LangChain LCEL!
lc_retriever = StatefulLangChainRetriever(
    stateful_retriever=core_retriever,
    session_id="user_123"
)

docs = lc_retriever.invoke("I have chest pain")           # Turn 1: Slow DB Search
docs = lc_retriever.invoke("Is my heart rate normal?")    # Turn 2: Fast Cache Hit!
```

### 2. Using PostgreSQL for Production

Swap out the `InMemoryStateStore` for persistent, scalable caching across server restarts.

```python
from stateful_rag.stores import PostgresStateStore
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Note: Use os.environ.get("DATABASE_URL") in real production environments!
engine = create_engine("postgresql+psycopg://rag_user:rag_password@localhost:5433/rag_state")
db_session = sessionmaker(bind=engine)()

production_store = PostgresStateStore(db_session=db_session)
```

---

## 🖥️ Interactive Demo (The Doctor Dashboard)

Run the Streamlit UI locally to visualize the "Context Drift" safety mechanism in real-time:

```bash
git clone https://github.com/naveenkamalakannan/stateful-rag.git
cd stateful-rag
pip install -e ".[demo]"
streamlit run examples/demo_app.py
```

![Doctor Dashboard — Showing Cache Hits, Context Drift Detection, and Live Audit Trace](docs/doctor_dashboard_screenshot.png)

---

## 📚 Academic Citation

This library is the official implementation of the architecture proposed in:

> **"Stateful, Multilingual Medical Graph-RAG Framework for Sustainable and Iterative Clinical Triage"**
> N. Kamalakannan, A. Jogekar, U. Haider (2026).

```bibtex
@article{kamalakannan2026stateful,
  title={Stateful, Multilingual Medical Graph-RAG Framework for Sustainable and Iterative Clinical Triage},
  author={Kamalakannan, Naveen and Jogekar, Ankita and Haider, Umaima},
  year={2026}
}
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
