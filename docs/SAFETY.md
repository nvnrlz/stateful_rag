# StatefulRAG — Safety, Security & Compliance Notes

This document records the safety properties StatefulRAG *does* and *does not*
provide, the residual risks an operator must own, and the controls available.
It is engineering documentation, not regulatory or legal advice.

## 1. What StatefulRAG is and is not

StatefulRAG is a **caching/routing layer** for retrieval. It:

- does **not** diagnose, triage, or make any clinical decision;
- does **not** generate text — it only returns retrieved documents with scores;
- **must not** be the sole safeguard in a patient-facing system.

Any application that uses retrieved context to influence patient care is very
likely a regulated **Software as a Medical Device** (FDA SaMD / EU MDR). That
brings obligations — clinical validation, risk management (ISO 14971), a quality
management system (ISO 13485), post-market surveillance — that are out of scope
for this library and remain the deploying organisation's responsibility.

## 2. The core safety property

The central risk of any stateful cache is **silent recall loss**: serving stale
cached context that omits a finding a fresh search would have surfaced. The
prototype this replaced had exactly that failure mode — a "cache hit" returned
turn-1 documents and never re-queried.

StatefulRAG addresses it with **safety modes** (`StatefulRAGConfig.safety_mode`):

| Mode | Guarantee on a cache hit |
|---|---|
| `STRICT` *(default)* | Fresh retrieval runs every hit; if Jaccard divergence vs the cache exceeds `max_divergence`, the engine **fails open** and serves the fresh result. A cache hit cannot hide a fresh finding. |
| `BALANCED` | Cache served; a sampled fraction (`shadow_sample_rate`) is checked against fresh and divergence is logged. Bounded, *measured* risk. |
| `FAST` | Cache served unverified. No safety net — non-clinical only. |

**Fail-open everywhere:** embedding/main-DB failures route to the authoritative
database rather than raising or returning empty. A dimension mismatch is treated
as a configuration error and raised (failing open would corrupt every write).

## 3. Measure before you trust

Use `stateful_rag.eval.EvaluationHarness` against golden clinical scenarios in CI.
It reports recall, precision and divergence on cache hits. Promotion path:
`STRICT` in production → observe divergence → relax to `BALANCED`/`FAST` only when
the data shows divergence stays within budget. Treat a divergence regression like
a failing test.

## 4. Security controls

| Concern | Control | Operator action |
|---|---|---|
| PHI at rest | Field-level encryption (`ContentCipher`, Fernet/AES) | Set `STATEFUL_RAG_ENCRYPTION_KEY` from a secret manager |
| Tenant isolation / IDOR | `authorizer(session_id, principal)` callback | Always provide one; verify ownership |
| Retention | `cache_ttl_seconds` + `scripts/purge_expired.py` | Schedule the purge job |
| Right to erasure | `store.delete_session(id)` | Wire to your data-subject-request flow |
| Auditability | Append-only `audit_log`, salted query hashes | Ship to immutable/WORM storage; retain per policy |
| Secrets | Env-only; nothing hardcoded | Use a real secret manager, rotate keys |

## 5. Residual risks (operator-owned)

- **Embedding inversion.** Embeddings are stored unencrypted so they remain
  searchable; they can leak information about the source text. Mitigate with
  database-level encryption, network isolation, and least-privilege DB access.
- **Threshold calibration.** `drift_threshold` / `per_doc_floor` are
  model-specific. Calibrate them per embedding model on representative data;
  uncalibrated thresholds degrade either recall or the latency benefit.
  *Measured (gemini-embedding-001, ENT corpus, `app.eval`):* drift_threshold=0.78,
  per_doc_floor=0.476, with **on-topic vs cross-topic similarity separation of only
  ~0.03**. In this tightly-clustered ENT embedding space the cache's drift signal
  is weak, so **`SafetyMode.STRICT` is mandatory** — the cache is a *verified*
  latency optimisation, never a standalone safety mechanism. Re-run `app.eval`
  whenever the embedding model changes.
- **Retrieval recall is not perfect.** Measured recall@5 ≈ 0.93 on the ENT gold
  set (concept-keyword proxy, two MBBS texts). ~7% of queries missed the top-5;
  broaden the corpus and add clinician-labelled relevance before relying on it.
- **Key management.** Losing `STATEFUL_RAG_ENCRYPTION_KEY` makes cached content
  unrecoverable; leaking it defeats encryption. Manage and rotate via KMS.
- **Audit completeness.** `LoggingAuditSink` is best-effort; for compliance use
  `PostgresAuditSink` (or a SIEM) with durable, tamper-evident storage.
- **Not validated clinically.** No claim is made about retrieval *quality* for
  any clinical task. That must be established by the deploying organisation.

## 6. Recommended production checklist

- [ ] `safety_mode=STRICT` (until divergence data justifies relaxing)
- [ ] `authorizer` enforcing per-principal session ownership
- [ ] `STATEFUL_RAG_ENCRYPTION_KEY` and `STATEFUL_RAG_AUDIT_SALT` set from secrets
- [ ] `PostgresStateStore` built with a `sessionmaker` (request-scoped sessions)
- [ ] HNSW index created (`init_db.py`)
- [ ] Retention purge job scheduled
- [ ] `PostgresAuditSink` shipping to durable storage
- [ ] Divergence eval wired into CI against golden scenarios
- [ ] Embedding `embedding_dim` matches the model and the DB schema
