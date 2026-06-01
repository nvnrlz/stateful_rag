# ENT AI-Receptionist Backend — Runbook

A multi-turn, retrieval-grounded ENT triage/receptionist chat service built on the
hardened `stateful_rag` layer. It asks the patient questions to localise an ENT
complaint, grounds statements in the licensed reference corpus (with citations),
escalates red-flag emergencies, and captures reviewer verdicts for the beta.

> **Test-only.** This is an experimental tool for a supervised study with
> student-doctor reviewers. It does not diagnose and must not be used for real
> care. See [`SAFETY.md`](SAFETY.md).

## Architecture

```
patient ──chat──> FastAPI (app/api)
                     │
                     ▼
            TriageEngine (app/triage)   ← deterministic question plan + red-flag safety
                     │ grounding query
                     ▼
            StatefulRetriever (stateful_rag)  ← multi-turn cache, STRICT verify, fail-open
                     │ cache miss / verify
                     ▼
            KnowledgeRetriever → MainStore (app/retrieval)  ← FileMainStore (offline) | pgvector
                     ▲
        ingest (app/ingestion): PDF → clean → chunk(+page cites) → embed
```

Clinical control flow (what to ask, when to escalate, when to stop) is
**deterministic and auditable**. An optional LLM only rephrases questions/summaries.

## 1. Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[app]"          # offline-capable backend
# optional, for the real test:
pip install -e ".[app-local]"    # local sentence-transformers embeddings
```

## 2. Ingest the corpus

Offline smoke (hashing embedder, MBBS-tier texts, capped pages):

```bash
python -m app.ingestion.ingest --corpus "/path/to/RAG DB" \
    --out data/ent_index.npz --provider hashing --dim 384 \
    --books dhingra bansal --max-pages 150
```

For the real test, use semantic embeddings (and remove the page cap):

```bash
python -m app.ingestion.ingest --corpus "/path/to/RAG DB" \
    --out data/ent_index.npz --provider sentence_transformers \
    --books dhingra bansal
# then set RAG_EMBED_PROVIDER=sentence_transformers and RAG_EMBED_DIM to match the model.
```

The ingester **skips byte-identical duplicates** automatically (the corpus has one)
and flags any book with no extractable text (would need OCR).

## 3. Run the chat API

```bash
export RAG_INDEX_PATH=data/ent_index.npz
uvicorn app.api.server:create_app --factory --host 0.0.0.0 --port 8000
```

### Endpoints
| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | corpus size, providers, offline flag |
| POST | `/session/start` | **consent-gated** session start → greeting + disclaimer |
| POST | `/chat` | one conversational turn |
| POST | `/feedback` | reviewer verdict (`correct`/`partially_correct`/`wrong`/`unsafe`) |
| GET | `/feedback/summary` | running accuracy summary |

### Example
```bash
SID=$(curl -s localhost:8000/session/start -H 'content-type: application/json' \
  -d '{"consent_acknowledged":true,"principal":"mbbs-1"}' | jq -r .session_id)

curl -s localhost:8000/chat -H 'content-type: application/json' \
  -d "{\"session_id\":\"$SID\",\"text\":\"my right ear hurts with discharge\"}" | jq
```

## Chat UI (test console)

A minimal single-page chat console is served by the same app (no separate build
or port). Launch it with the auto-port helper — it scans 8000–8100 and falls back
to an OS-assigned port, so it won't clash with your other running apps:

```bash
python scripts/serve.py                 # 127.0.0.1, first free port; prints the URL
python scripts/serve.py --host 0.0.0.0  # reachable from test devices on the LAN
python scripts/serve.py --port 8123     # force a port
```

The page provides:
- a **consent gate** (reviewer role: MBBS / ENT-PG + optional reviewer id) that
  must be accepted before any chat,
- the **back-and-forth chat**, with each AI reply showing its grounding citations
  (source + page), complaint area, and StatefulRAG route,
- a **✓ Correct / ✗ Wrong** button on every AI reply → `POST /feedback`,
- a **final feedback panel** (overall comment, requested enhancements, 1–5 rating)
  → `POST /session/feedback`.

All verdicts and feedback are appended to `data/feedback.jsonl`; `GET /feedback/summary`
gives a live accuracy + feedback count. Endpoints are unchanged, so the same
backend serves the future IVR channel.

## 4. Reviewer (student-doctor) eval loop

For each assistant turn the reviewer judges, POST `/feedback` with the verdict and
an optional comment. `/feedback/summary` gives live accuracy. Verdicts are stored
append-only in `data/feedback.jsonl` for later analysis (batch 1 = MBBS,
batch 2 = ENT PG via the `reviewer_role` field).

## 5. Providers (Gemini / Google AI Studio)

The beta uses Google AI Studio for both embeddings and dialogue phrasing:

```bash
# .env
GOOGLE_API_KEY=...                         # from Google AI Studio
RAG_EMBED_PROVIDER=gemini
RAG_EMBED_MODEL=gemini-embedding-001        # 768-dim
RAG_EMBED_DIM=768
RAG_LLM_PROVIDER=gemini
RAG_LLM_MODEL=gemini-2.5-flash              # there is no "3.5 flash"
RAG_DRIFT_THRESHOLD=0.78                     # calibrated (app.eval)
RAG_PER_DOC_FLOOR=0.476
```

Ingest with the same embedder before serving:

```bash
python scripts/build_and_eval.py   # re-ingest + re-run calibration/validation
```

Offline fallback (no key / CI): `RAG_EMBED_PROVIDER=hashing`, `RAG_LLM_PROVIDER=rule_based`.

## 6. Retrieval validation & calibration (measured)

Run any time the embedding model changes:

```bash
python scripts/build_and_eval.py          # writes data/eval_report.json
# or, against an existing index:
python -m app.eval.run --index data/ent_index.npz --provider gemini --model gemini-embedding-001
```

**Latest results — `gemini-embedding-001`, 1063 chunks (Dhingra + Bansal), 28-query ENT gold set:**

| Metric | Value |
|---|---|
| recall@5 | **0.93** |
| MRR@5 | **0.76** |
| precision@5 | 0.62 |
| recall by area | ear 1.00 · neck 1.00 · nose 0.88 · throat 0.86 |

**Drift calibration:** on-topic similarity ≈ 0.80 vs cross-topic ≈ 0.77 — **separation ≈ 0.03**.
Interpretation: in this tightly-clustered ENT embedding space, query-vs-cached-doc
similarity is a *weak* drift signal. **Keep `SafetyMode.STRICT`** (verify every cache
hit against the authoritative store); treat the cache as a verified latency
optimisation, not a safety mechanism.

## Switching to a database backend

- `RAG_MAIN_STORE=pgvector` + `DATABASE_URL=...` (run `init_db.py` for the cache
  schema; create a `knowledge_chunks` table + HNSW index for the main store).
- `RAG_CACHE_STORE=pgvector` to persist the StatefulRAG session cache.
- Set `STATEFUL_RAG_ENCRYPTION_KEY`, `STATEFUL_RAG_AUDIT_SALT`, and an authorizer.

## Known limitations (read before testing)

- **recall@5 ≈ 0.93, not 1.0.** ~7% of gold queries missed the top-5 (e.g. a
  generic "sore throat + fever" query). Expand the corpus beyond the two MBBS
  texts and add clinician-labelled relevance before relying on the numbers.
- **Cache drift detection is weak in ENT** (separation ≈ 0.03) — STRICT mode
  compensates by always verifying, so expect frequent `fail_open`/`drift_break`
  routes. The cache helps latency only when a follow-up is very close to a cached
  passage.
- **The gold set uses a concept-keyword proxy** for relevance, not clinician
  judgement. Good for regression tracking; not a substitute for expert evaluation.
- Red-flag rules are keyword-based (high sensitivity, will over-escalate); review
  and extend `app/safety/guardrails.py` with your clinicians before the test.
