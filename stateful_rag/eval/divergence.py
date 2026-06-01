"""Retrieval-quality / divergence evaluation.

This is the piece that turns "trust me, the cache is fine" into a measurement.
For any cache hit we can ask: *would a fresh retrieval from the authoritative
main database have returned the same clinical context?* If not, the cache is
silently degrading recall — the central safety risk of stateful caching.

The harness replays scenarios and reports, per cache hit:

- **recall@k**  : fraction of fresh (ground-truth) docs the cache also returned.
- **precision@k**: fraction of cached docs that a fresh search would have returned.
- **Jaccard divergence**: 1 - |intersection| / |union| of the two doc sets.

Use it offline (in CI, against golden scenarios) and online (via the retriever's
BALANCED shadow sampling) to keep the cache honest.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from statistics import mean
from typing import Any, Callable, Dict, List, Sequence


def document_key(doc: Any) -> str:
    """A stable identity for a document, used for set comparison.

    Prefers an explicit id/source, falling back to a hash of the content text.
    """
    if isinstance(doc, dict):
        for id_field in ("id", "doc_id", "node_id"):
            if doc.get(id_field) is not None:
                return f"id:{doc[id_field]}"
        content = doc.get("content")
        if content is not None:
            return "h:" + hashlib.sha256(str(content).encode("utf-8")).hexdigest()[:16]
    return "h:" + hashlib.sha256(str(doc).encode("utf-8")).hexdigest()[:16]


def _keyset(docs: Sequence[Any]) -> set[str]:
    return {document_key(d) for d in docs}


def recall_at_k(fresh: Sequence[Any], cached: Sequence[Any]) -> float:
    """Of the ground-truth (fresh) docs, how many did the cache return?"""
    fresh_keys = _keyset(fresh)
    if not fresh_keys:
        return 1.0
    return len(fresh_keys & _keyset(cached)) / len(fresh_keys)


def precision_at_k(fresh: Sequence[Any], cached: Sequence[Any]) -> float:
    """Of the cached docs returned, how many are in the fresh ground-truth set?"""
    cached_keys = _keyset(cached)
    if not cached_keys:
        return 1.0
    return len(cached_keys & _keyset(fresh)) / len(cached_keys)


def jaccard_divergence(fresh: Sequence[Any], cached: Sequence[Any]) -> float:
    """1 - Jaccard similarity between the two doc sets (0 = identical, 1 = disjoint)."""
    a, b = _keyset(fresh), _keyset(cached)
    union = a | b
    if not union:
        return 0.0
    return 1.0 - len(a & b) / len(union)


@dataclass
class DivergenceResult:
    recall: float
    precision: float
    divergence: float

    def is_safe(self, max_divergence: float) -> bool:
        return self.divergence <= max_divergence


def compare_results(fresh: Sequence[Any], cached: Sequence[Any]) -> DivergenceResult:
    return DivergenceResult(
        recall=recall_at_k(fresh, cached),
        precision=precision_at_k(fresh, cached),
        divergence=jaccard_divergence(fresh, cached),
    )


@dataclass
class EvalScenario:
    """A multi-turn conversation to replay against a retriever."""

    session_id: str
    turns: List[str]


@dataclass
class _TurnReport:
    turn: int
    query: str
    route: str
    result: DivergenceResult


@dataclass
class EvaluationReport:
    turns: List[_TurnReport] = field(default_factory=list)

    @property
    def mean_recall(self) -> float:
        cached_turns = [t for t in self.turns if t.route.startswith("cache")]
        return mean([t.result.recall for t in cached_turns]) if cached_turns else 1.0

    @property
    def mean_divergence(self) -> float:
        cached_turns = [t for t in self.turns if t.route.startswith("cache")]
        return mean([t.result.divergence for t in cached_turns]) if cached_turns else 0.0

    @property
    def worst_divergence(self) -> float:
        cached_turns = [t for t in self.turns if t.route.startswith("cache")]
        return max([t.result.divergence for t in cached_turns], default=0.0)

    def summary(self) -> Dict[str, Any]:
        return {
            "cache_turns": sum(1 for t in self.turns if t.route.startswith("cache")),
            "total_turns": len(self.turns),
            "mean_recall_on_cache_hits": round(self.mean_recall, 4),
            "mean_divergence_on_cache_hits": round(self.mean_divergence, 4),
            "worst_divergence_on_cache_hits": round(self.worst_divergence, 4),
        }


class EvaluationHarness:
    """Replays scenarios and compares cache-path results to fresh retrieval.

    ``retriever`` must expose ``retrieve(query, session_id, current_turn)`` and a
    ``fresh_retrieve(query)`` that bypasses the cache (the authoritative answer).
    """

    def __init__(self, retriever: Any):
        self._retriever = retriever

    def run(self, scenarios: Sequence[EvalScenario]) -> EvaluationReport:
        report = EvaluationReport()
        for scenario in scenarios:
            for turn_idx, query in enumerate(scenario.turns, start=1):
                # Ground truth: what an exhaustive fresh search would return now.
                fresh = self._retriever.fresh_retrieve(query)
                # Actual: what the stateful retriever serves (may be cache or fresh).
                served = self._retriever.retrieve(query, scenario.session_id, turn_idx)
                route = served[0].get("_route", "unknown") if served and isinstance(served[0], dict) else "unknown"
                report.turns.append(_TurnReport(
                    turn=turn_idx,
                    query=query,
                    route=route,
                    result=compare_results(fresh, served),
                ))
        return report
