"""Run ENT retrieval validation + threshold calibration and write a report.

    python -m app.eval.run --index data/ent_index.npz --provider sentence_transformers \
        --model NeuML/pubmedbert-base-embeddings --k 5 --out data/eval_report.json

Uses the same embedder/store the service would use, so the numbers reflect the
deployed configuration. Prints recommended drift_threshold / per_doc_floor to
copy into the environment (RAG_DRIFT_THRESHOLD / RAG_PER_DOC_FLOOR).
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from ..providers.embeddings import build_embedder
from ..retrieval.main_store import FileMainStore
from .calibrate import calibrate_thresholds
from .retrieval_eval import evaluate_retrieval


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="ENT retrieval validation + calibration")
    p.add_argument("--index", default="data/ent_index.npz")
    p.add_argument("--provider", default="hashing",
                   choices=["hashing", "sentence_transformers", "openai"])
    p.add_argument("--model", default="")
    p.add_argument("--dim", type=int, default=384, help="only used for hashing")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--out", default="data/eval_report.json")
    args = p.parse_args(argv)

    embedder = build_embedder(args.provider, dim=args.dim, model=args.model)
    store = FileMainStore(args.index, dim=embedder.dim)
    if store.count() == 0:
        sys.stderr.write(f"ERROR: index {args.index!r} is empty. Ingest first.\n")
        return 1
    if store.embedder_name not in ("unknown", embedder.name):
        sys.stderr.write(
            f"WARNING: index built with '{store.embedder_name}' but evaluating with "
            f"'{embedder.name}'. Re-ingest for valid numbers.\n"
        )

    print(f"Evaluating retrieval over {store.count()} chunks with '{embedder.name}' (dim={embedder.dim})…")
    retrieval = evaluate_retrieval(embedder, store, k=args.k)
    calibration = calibrate_thresholds(embedder, store, top_k=args.k)

    report = {
        "retrieval": retrieval.summary(),
        "calibration": calibration.summary(),
        "weak_queries": [
            {"query": q.query, "area": q.area, "first_rank": q.first_rank}
            for q in retrieval.per_query if not q.hit
        ],
    }

    print("\n=== RETRIEVAL QUALITY ===")
    print(json.dumps(report["retrieval"], indent=2))
    print("\n=== DRIFT CALIBRATION ===")
    print(json.dumps(report["calibration"], indent=2))
    if report["weak_queries"]:
        print(f"\n⚠ {len(report['weak_queries'])} queries had no relevant chunk in top-{args.k}:")
        for w in report["weak_queries"]:
            print(f"   - [{w['area']}] {w['query']}")

    c = report["calibration"]
    print("\n=== RECOMMENDED SETTINGS ===")
    print(f"  export RAG_DRIFT_THRESHOLD={c['recommended_drift_threshold']}")
    print(f"  export RAG_PER_DOC_FLOOR={c['recommended_per_doc_floor']}")

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n✅ Report written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
