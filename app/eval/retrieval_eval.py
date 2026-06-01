"""Retrieval-quality evaluation over the ENT gold set.

Metrics (concept-keyword proxy relevance):
- recall@k     : fraction of queries with >=1 relevant chunk in the top-k.
- MRR@k        : mean reciprocal rank of the first relevant chunk.
- precision@k  : mean fraction of top-k chunks that are relevant.
- area accuracy: fraction where the top chunk's source/area matches expectation.

Run against whatever embedder/store the AppConfig selects, so it measures the
*actual* deployed retrieval (e.g. the biomedical Sentence-Transformer).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Any, Dict, List

from ..providers.embeddings import EmbeddingProvider
from ..retrieval.main_store import MainStore
from .ent_gold import GOLD_QUERIES, GoldQuery


def _is_relevant(chunk_text: str, anchors: List[str]) -> bool:
    low = (chunk_text or "").lower()
    return any(a.lower() in low for a in anchors)


@dataclass
class PerQuery:
    query: str
    area: str
    hit: bool
    first_rank: int          # 1-based rank of first relevant chunk, 0 if none
    precision: float
    top_citation: str


@dataclass
class RetrievalReport:
    k: int
    embedder: str
    n: int
    recall_at_k: float
    mrr_at_k: float
    precision_at_k: float
    per_area: Dict[str, float] = field(default_factory=dict)
    per_query: List[PerQuery] = field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        return {
            "embedder": self.embedder,
            "queries": self.n,
            "k": self.k,
            "recall@k": round(self.recall_at_k, 4),
            "mrr@k": round(self.mrr_at_k, 4),
            "precision@k": round(self.precision_at_k, 4),
            "recall_by_area": {a: round(v, 4) for a, v in self.per_area.items()},
        }


def evaluate_retrieval(
    embedder: EmbeddingProvider,
    store: MainStore,
    *,
    k: int = 5,
    gold: List[GoldQuery] | None = None,
) -> RetrievalReport:
    gold = gold or GOLD_QUERIES
    per_query: List[PerQuery] = []
    area_hits: Dict[str, List[float]] = {}

    for g in gold:
        docs = store.search(embedder.embed(g.query), k)
        first_rank = 0
        rel_count = 0
        for rank, d in enumerate(docs, start=1):
            if _is_relevant(d.get("content", ""), g.anchors):
                rel_count += 1
                if first_rank == 0:
                    first_rank = rank
        hit = first_rank > 0
        precision = rel_count / len(docs) if docs else 0.0
        per_query.append(PerQuery(
            query=g.query, area=g.area, hit=hit, first_rank=first_rank,
            precision=precision, top_citation=docs[0].get("citation", "") if docs else "",
        ))
        area_hits.setdefault(g.area, []).append(1.0 if hit else 0.0)

    n = len(gold)
    recall = mean(1.0 if q.hit else 0.0 for q in per_query) if n else 0.0
    mrr = mean((1.0 / q.first_rank) if q.first_rank else 0.0 for q in per_query) if n else 0.0
    precision = mean(q.precision for q in per_query) if n else 0.0
    per_area = {a: mean(v) for a, v in area_hits.items()}

    return RetrievalReport(
        k=k, embedder=embedder.name, n=n, recall_at_k=recall, mrr_at_k=mrr,
        precision_at_k=precision, per_area=per_area, per_query=per_query,
    )
