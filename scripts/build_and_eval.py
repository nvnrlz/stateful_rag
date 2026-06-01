"""Build the ENT index with the configured embedder, then calibrate + validate.

Provider-agnostic: reads AppConfig (which loads .env), so it works with the
Gemini embedder, a local sentence-transformer, or the hashing fallback. Writes
data/eval_report.json and prints recommended drift thresholds.

    python scripts/build_and_eval.py            # uses .env (RAG_EMBED_PROVIDER=…)
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import load_config
from app.eval.calibrate import calibrate_thresholds
from app.eval.retrieval_eval import evaluate_retrieval
from app.ingestion.chunker import chunk_pages
from app.ingestion.ingest import _file_md5, _slug, discover_pdfs, source_for
from app.ingestion.pdf_loader import load_pdf_pages
from app.providers.embeddings import build_embedder
from app.retrieval.main_store import Chunk, FileMainStore

BOOKS = ["dhingra", "bansal"]
MAX_PAGES = int(os.environ.get("SWAP_MAX_PAGES", "150"))


def main() -> int:
    cfg = load_config()
    embedder = build_embedder(cfg.embedding_provider, dim=cfg.embedding_dim, model=cfg.embedding_model)
    print(f"Embedder: {embedder.name} (provider={cfg.embedding_provider}, dim={embedder.dim})")

    store = FileMainStore(cfg.index_path, dim=embedder.dim, embedder_name=embedder.name)
    store._embeddings = store._embeddings[:0]
    store._chunks = []
    seen: set[str] = set()
    for p in discover_pdfs(cfg.corpus_dir, BOOKS):
        digest = _file_md5(p)
        if digest in seen:
            continue
        seen.add(digest)
        meta = source_for(os.path.basename(p))
        slug = _slug(os.path.basename(p)).replace(" ", "_")
        specs = chunk_pages(load_pdf_pages(p, max_pages=MAX_PAGES),
                            chunk_chars=cfg.chunk_chars, overlap=cfg.chunk_overlap)
        embs = embedder.embed_batch([s.content for s in specs])
        chunks = [Chunk(id=f"{slug}:{i}:{s.page_start}-{s.page_end}", content=s.content,
                        source=str(meta["name"]), page_start=s.page_start,
                        page_end=s.page_end, section=str(meta.get("tier", "")))
                  for i, s in enumerate(specs)]
        store.add(chunks, embs)
        print(f"  • {meta['name']}: {len(chunks)} chunks")
    store.save()
    print(f"Indexed {store.count()} chunks → {cfg.index_path}")

    print("\nEvaluating retrieval + calibrating drift thresholds …")
    retrieval = evaluate_retrieval(embedder, store, k=cfg.main_top_k)
    calibration = calibrate_thresholds(embedder, store, top_k=cfg.main_top_k)
    report = {
        "embedder": embedder.name,
        "retrieval": retrieval.summary(),
        "calibration": calibration.summary(),
        "weak_queries": [{"query": q.query, "area": q.area}
                         for q in retrieval.per_query if not q.hit],
    }
    os.makedirs(os.path.dirname(cfg.index_path) or ".", exist_ok=True)
    with open("data/eval_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n================ RESULTS ================")
    print(json.dumps(report, indent=2))
    c = report["calibration"]
    print("\nRecommended (copy into .env):")
    print(f"  RAG_DRIFT_THRESHOLD={c['recommended_drift_threshold']}")
    print(f"  RAG_PER_DOC_FLOOR={c['recommended_per_doc_floor']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
