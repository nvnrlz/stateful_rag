"""Swap to a semantic embedder, re-ingest the ENT corpus, then calibrate + validate.

Tries a biomedical sentence model first (best for ENT), falls back to a general
small model if it can't be downloaded. Re-ingests the MBBS-tier texts, runs the
ENT retrieval evaluation + drift calibration, and writes data/eval_report.json.
"""

from __future__ import annotations

import glob
import json
import os
import sys

from app.eval.calibrate import calibrate_thresholds
from app.eval.retrieval_eval import evaluate_retrieval
from app.ingestion.ingest import ingest
from app.providers.embeddings import build_embedder
from app.retrieval.main_store import FileMainStore

CORPUS = os.environ.get("RAG_CORPUS_DIR", "/Users/naveen/Downloads/RAG DB")
OUT = os.environ.get("RAG_INDEX_PATH", "data/ent_index.npz")
BOOKS = ["dhingra", "bansal"]
MAX_PAGES = int(os.environ.get("SWAP_MAX_PAGES", "150"))

# (display name, HuggingFace repo id). Biomedical first (best for ENT), then a
# reliable general fallback. Each is tried via the local HF cache first (offline),
# then over the network if available.
CANDIDATE_MODELS = [
    ("pubmedbert-base-embeddings", "NeuML/pubmedbert-base-embeddings"),
    ("all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L6-v2"),
]


def _local_snapshot(repo_id: str) -> str | None:
    """Return a local HF cache snapshot dir for repo_id, if fully present."""
    cache = os.path.expanduser(os.environ.get("HF_HOME", "~/.cache/huggingface"))
    folder = "models--" + repo_id.replace("/", "--")
    pattern = os.path.join(cache, "hub", folder, "snapshots", "*")
    for snap in sorted(glob.glob(pattern)):
        if os.path.exists(os.path.join(snap, "config.json")):
            return snap
    return None


def pick_model() -> tuple[str, str]:
    """Return (display_name, model_path_or_id). Prefers offline cache."""
    from sentence_transformers import SentenceTransformer

    for display, repo in CANDIDATE_MODELS:
        snap = _local_snapshot(repo)
        if snap:
            try:
                print(f"Loading {display} from local cache (offline) …")
                SentenceTransformer(snap)
                print(f"  ✓ loaded from cache: {display}")
                return display, snap
            except Exception as exc:
                print(f"  ✗ cache load failed ({display}): {exc}")
        try:
            print(f"Downloading {display} ({repo}) …")
            SentenceTransformer(repo)
            print(f"  ✓ downloaded: {display}")
            return display, repo
        except Exception as exc:
            print(f"  ✗ unavailable ({display}): {exc}")
    raise SystemExit("No sentence-transformer model could be loaded (no cache, no network).")


def main() -> int:
    display, path = pick_model()
    print(f"\n=== Re-ingesting with semantic embedder: {display} ===")
    embedder = build_embedder("sentence_transformers", model=path, name=display)
    # Ingest reusing the already-loaded embedder for the clean recorded name.
    from app.ingestion.chunker import chunk_pages
    from app.ingestion.pdf_loader import load_pdf_pages, pdf_page_count
    from app.ingestion.ingest import discover_pdfs, source_for, _slug, _file_md5
    from app.retrieval.main_store import Chunk

    store = FileMainStore(OUT, dim=embedder.dim, embedder_name=display)
    store._embeddings = store._embeddings[:0]  # fresh build
    store._chunks = []
    seen: set[str] = set()
    for p in discover_pdfs(CORPUS, BOOKS):
        fn = os.path.basename(p)
        d = _file_md5(p)
        if d in seen:
            continue
        seen.add(d)
        meta = source_for(fn)
        slug = _slug(fn).replace(" ", "_")
        specs = chunk_pages(load_pdf_pages(p, max_pages=MAX_PAGES), chunk_chars=1200, overlap=200)
        embs = embedder.embed_batch([s.content for s in specs])
        chunks = [Chunk(id=f"{slug}:{i}:{s.page_start}-{s.page_end}", content=s.content,
                        source=str(meta["name"]), page_start=s.page_start, page_end=s.page_end,
                        section=str(meta.get("tier", ""))) for i, s in enumerate(specs)]
        store.add(chunks, embs)
        print(f"  • {meta['name']}: {len(chunks)} chunks")
    store.save()

    print(f"\n=== Validating retrieval ({store.count()} chunks, dim={embedder.dim}) ===")
    retrieval = evaluate_retrieval(embedder, store, k=5)
    calibration = calibrate_thresholds(embedder, store, top_k=5)

    report = {
        "model": display,
        "retrieval": retrieval.summary(),
        "calibration": calibration.summary(),
        "weak_queries": [
            {"query": q.query, "area": q.area} for q in retrieval.per_query if not q.hit
        ],
    }
    os.makedirs("data", exist_ok=True)
    with open("data/eval_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n================ RESULTS ================")
    print(json.dumps(report, indent=2))
    c = report["calibration"]
    print("\nRecommended settings:")
    print(f"  RAG_EMBED_PROVIDER=sentence_transformers")
    print(f"  RAG_EMBED_MODEL={display}")
    print(f"  RAG_EMBED_DIM={embedder.dim}")
    print(f"  RAG_DRIFT_THRESHOLD={c['recommended_drift_threshold']}")
    print(f"  RAG_PER_DOC_FLOOR={c['recommended_per_doc_floor']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
