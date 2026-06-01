# 🩺 StatefulRAG

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![PostgreSQL + pgvector](https://img.shields.io/badge/PostgreSQL-pgvector-blue.svg)](https://github.com/pgvector/pgvector)

**A measured, auditable, fail-open caching layer for multi-turn Retrieval-Augmented Generation (RAG).**

StatefulRAG reduces redundant vector-database traffic on multi-turn conversations by caching a session's retrieved context — **without silently trading away retrieval quality.** Every cache decision is scored, audited, and (in the default safety mode) verified against the authoritative database, so the cache can never quietly hide a result the main database would have returned.

---

## ⚠️ Clinical-use disclaimer

This library is **infrastructure**, not a medical device. It does not diagnose, triage, or make clinical decisions. If you deploy it in software that influences patient care, that software is very likely a regulated **Software as a Medical Device** (FDA SaMD / EU MDR) and requires clinical validation, a quality-management system, and regulatory clearance. The engineering controls here (encryption, audit, fail-open routing, divergence measurement) are necessary groundwork — **not** a substitute for that process. See [`docs/SAFETY.md`](docs/SAFETY.md).

---

## 📖 About

StatefulRAG accompanies the research paper **"Stateful, Multilingual Medical Graph-RAG Framework for Sustainable and Iterative Clinical Triage"** (Kamalakannan et al., 2026). The original idea — caching a session's retrieved sub-graph to avoid repeated exhaustive searches — is preserved here, but the implementation has been hardened for backend use: the cache is treated as a *hint that must be verified*, not a replacement for retrieval.

**Authors & Researchers:**
- **Naveen Kamalakannan** (TechZilla Solutions)
- **Ankita Jogekar** (Independent Researcher)
- **Umaima Haider** (University of East London)

---

## 🧠 How it works

For each turn the engine embeds the query and searches the session cache:

- **Cold start / cache miss** → query the authoritative main vector DB (Pinecone, Qdrant, pgvector, …), then cache the result with its embedding, model tag, and timestamp.
- **Context drift** → if the best cached similarity is below `drift_threshold`, the cache is *broken* and the main DB is queried again. Off-topic cached docs are also dropped via a `per_doc_floor`.
- **Cache hit** → behaviour depends on the **safety mode** (below).

Every returned document carries provenance: `_route`, `_source`, `_score`, `_turn_added`. Every decision emits a structured **audit record** (no raw query text is stored — only a salted hash).

### Safety modes

| Mode | On a cache hit | Latency | When to use |
|---|---|---|---|
| `STRICT` *(default)* | Runs a fresh retrieval, measures **divergence** vs the cache, serves cache only if within `max_divergence`; otherwise **fails open** to fresh results. | No win on hits, full safety | Clinical / high-stakes |
| `BALANCED` | Serves cache, **shadow-samples** a fraction of hits to monitor divergence. | Win on most hits, live safety net | After STRICT data shows low divergence |
| `FAST` | Serves cache directly. | Max win, no safety net | Non-clinical workloads |

The recommended path is to **start in `STRICT`**, watch the divergence metric on real traffic, and only relax to `BALANCED`/`FAST` once the data justifies it.

---

## 📊 Benchmarks — read the methodology

Caching can reduce follow-up latency substantially when a hit avoids an exhaustive main-DB search, but the magnitude is entirely workload-dependent (corpus size, index type, network, embedding cost). **Run the included harness against your own stack and index before quoting any number.** The PostgreSQL path requires the HNSW index created by [`init_db.py`](init_db.py); without it, pgvector does an exact sequential scan and the latency advantage disappears. We deliberately do not headline a fixed "Nx" figure here, because it is not reproducible without the original paper's specific corpus and hardware.

---

## 🚀 Quick Start

### Installation

```bash
pip install stateful-rag                 # core (in-memory + Postgres)
pip install "stateful-rag[frameworks]"   # + LangChain & LlamaIndex wrappers
pip install "stateful-rag[demo]"         # + Sentence-Transformers & Streamlit
```

### Core usage

```python
import os
from stateful_rag import (
    StatefulRetriever, StatefulRAGConfig, SafetyMode, InMemoryStateStore,
)

config = StatefulRAGConfig(
    embedding_dim=1536,                 # MUST match your embedder's output
    embedding_model="text-embedding-3-small",
    drift_threshold=0.85,
    safety_mode=SafetyMode.STRICT,      # verify every cache hit
)

retriever = StatefulRetriever(
    state_store=InMemoryStateStore(expected_dim=1536),
    main_retriever_fn=my_pinecone_search,   # Callable[[str], list[dict]]
    embed_fn=my_openai_embedder,            # Callable[[str], list[float]]
    config=config,
    authorizer=lambda session_id, principal: principal == owner_of(session_id),
)

docs = retriever.retrieve("I have chest pain", session_id="user_123",
                          current_turn=1, principal="user_123")
print(docs[0]["_route"], docs[0]["_score"])   # e.g. "main_db", None
```

### PostgreSQL for production

Use a `sessionmaker` (request-scoped sessions, **not** a single shared session) and field-level encryption:

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from stateful_rag import PostgresStateStore, PostgresAuditSink, ContentCipher

factory = sessionmaker(bind=create_engine(os.environ["DATABASE_URL"]))
store = PostgresStateStore(factory, cipher=ContentCipher())   # key from env
audit = PostgresAuditSink(factory)

retriever = StatefulRetriever(store, my_search, my_embedder,
                              config=config, audit_sink=audit, authorizer=my_authz)
```

Initialise the schema + HNSW index once: `DATABASE_URL=... python init_db.py`.

---

## 🔐 Security & compliance

- **Encryption at rest** — cached content is encrypted with `ContentCipher` (Fernet/AES). Set `STATEFUL_RAG_ENCRYPTION_KEY`; without it the store warns and runs in plaintext. *Residual risk:* embeddings stay unencrypted to remain searchable — mitigate with DB-level encryption and strict access control.
- **Session isolation** — `session_id` is client-supplied and is **not** trusted alone. Provide an `authorizer` to enforce that the calling principal owns the session (closes the IDOR hole).
- **Retention & erasure** — `cache_ttl_seconds` ignores/excludes stale rows on read; [`scripts/purge_expired.py`](scripts/purge_expired.py) deletes them; `store.delete_session(id)` implements right-to-erasure.
- **Audit trail** — every decision is recorded (route, scores, divergence, latency, model) with a **salted hash** of the query, never the raw text.

See [`.env.example`](.env.example) for all configuration variables.

---

## 📐 Measuring retrieval quality (divergence harness)

Trust in a cache is empirical. The eval harness replays conversations and compares cache-served results against a fresh authoritative search, reporting recall, precision, and Jaccard divergence:

```python
from stateful_rag.eval import EvaluationHarness, EvalScenario

report = EvaluationHarness(retriever).run([
    EvalScenario("patient-1", ["chest pain", "is my heart rate normal", "I feel dizzy"]),
])
print(report.summary())   # mean_recall / mean_divergence / worst_divergence on cache hits
```

Wire this into CI against golden clinical scenarios so retrieval-quality regressions fail the build.

---

## 🖥️ Interactive demo

```bash
pip install -e ".[demo]"
streamlit run examples/demo_app.py
```

![Doctor Dashboard](docs/doctor_dashboard_screenshot.png)

---

## ✅ Tests

```bash
pip install -e ".[test]"
pytest                                   # unit tests (no DB needed)
# Postgres integration tests (optional):
export STATEFUL_RAG_TEST_DATABASE_URL=postgresql+psycopg://USER:PASS@localhost:5433/rag_state
export STATEFUL_RAG_EMBED_DIM=8
pytest tests/test_postgres.py
```

---

## 📚 Academic Citation

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

MIT — see [LICENSE](LICENSE).
