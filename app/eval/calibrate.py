"""Calibrate StatefulRAG drift thresholds for the configured embedder.

The drift decision compares a follow-up query's embedding against the cached
*document* embeddings from the previous turn. The right threshold is therefore
embedder-specific and must be measured, not guessed.

Method:
1. For each gold conversation pair, embed the first complaint, retrieve its top-k
   chunks (the "cache"), then measure the best cosine similarity between the
   follow-up query and those cached chunk embeddings.
2. Same-area follow-ups are positives (should stay cached); cross-area follow-ups
   are negatives (should break the cache).
3. Sweep candidate thresholds and pick the one maximising Youden's J
   (sensitivity + specificity − 1) — the value that best separates the two.
4. Recommend a per-document floor from the distribution of genuinely-relevant
   top-1 retrieval scores, so off-topic chunks are dropped on a hit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean, pstdev
from typing import Any, Dict, List, Tuple

import numpy as np

from ..providers.embeddings import EmbeddingProvider
from ..retrieval.main_store import MainStore
from .ent_gold import GOLD_PAIRS, GOLD_QUERIES


def _best_sim_to_cache(q_vec: np.ndarray, cache: np.ndarray) -> float:
    if cache.size == 0:
        return -1.0
    nq = np.linalg.norm(q_vec)
    norms = np.linalg.norm(cache, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sims = (cache @ q_vec) / (norms * nq)
    sims = np.nan_to_num(sims, nan=-1.0)
    return float(np.max(sims))


@dataclass
class CalibrationReport:
    embedder: str
    positives: List[float] = field(default_factory=list)
    negatives: List[float] = field(default_factory=list)
    recommended_drift_threshold: float = 0.0
    recommended_per_doc_floor: float = 0.0
    youden_j: float = 0.0
    accuracy_at_threshold: float = 0.0
    separation: float = 0.0   # mean(pos) - mean(neg)

    def summary(self) -> Dict[str, Any]:
        def stats(xs):
            return {"n": len(xs), "mean": round(mean(xs), 4) if xs else None,
                    "min": round(min(xs), 4) if xs else None,
                    "max": round(max(xs), 4) if xs else None,
                    "std": round(pstdev(xs), 4) if len(xs) > 1 else 0.0}
        return {
            "embedder": self.embedder,
            "on_topic_similarity": stats(self.positives),
            "cross_topic_similarity": stats(self.negatives),
            "separation": round(self.separation, 4),
            "recommended_drift_threshold": round(self.recommended_drift_threshold, 3),
            "recommended_per_doc_floor": round(self.recommended_per_doc_floor, 3),
            "youden_j": round(self.youden_j, 4),
            "accuracy_at_threshold": round(self.accuracy_at_threshold, 4),
        }


def calibrate_thresholds(
    embedder: EmbeddingProvider,
    store: MainStore,
    *,
    top_k: int = 5,
) -> CalibrationReport:
    positives: List[float] = []
    negatives: List[float] = []

    for turn1, turn2, on_topic in GOLD_PAIRS:
        docs = store.search(embedder.embed(turn1), top_k)
        if not docs:
            continue
        cache = np.asarray(embedder.embed_batch([d["content"] for d in docs]), dtype=np.float64)
        best = _best_sim_to_cache(np.asarray(embedder.embed(turn2), dtype=np.float64), cache)
        (positives if on_topic else negatives).append(best)

    # Sweep thresholds for the best Youden's J separating positives/negatives.
    all_vals = sorted(set(positives + negatives))
    candidates = all_vals or [0.5]
    best_j, best_thr, best_acc = -1.0, candidates[0], 0.0
    P, N = len(positives), len(negatives)
    for thr in candidates:
        tp = sum(1 for v in positives if v >= thr)
        fp = sum(1 for v in negatives if v >= thr)
        tpr = tp / P if P else 0.0
        fpr = fp / N if N else 0.0
        j = tpr - fpr
        acc = ((tp) + (N - fp)) / (P + N) if (P + N) else 0.0
        if j > best_j or (j == best_j and acc > best_acc):
            best_j, best_thr, best_acc = j, thr, acc

    # Nudge threshold to the midpoint between the chosen value and the next lower
    # negative, for a small safety margin.
    sep = (mean(positives) - mean(negatives)) if positives and negatives else 0.0

    # per_doc_floor from genuinely-relevant top-1 retrieval scores (10th percentile).
    top1_scores: List[float] = []
    for g in GOLD_QUERIES:
        docs = store.search(embedder.embed(g.query), 1)
        if docs:
            top1_scores.append(float(docs[0].get("_score", 0.0)))
    per_doc_floor = float(np.percentile(top1_scores, 10)) * 0.6 if top1_scores else 0.2
    per_doc_floor = max(0.0, min(per_doc_floor, best_thr))

    return CalibrationReport(
        embedder=embedder.name, positives=positives, negatives=negatives,
        recommended_drift_threshold=best_thr, recommended_per_doc_floor=per_doc_floor,
        youden_j=best_j, accuracy_at_threshold=best_acc, separation=sep,
    )
