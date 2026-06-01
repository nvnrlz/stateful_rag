"""Embedding providers.

Three implementations behind one interface:

- ``HashingEmbedder``  : dependency-free, deterministic hashed bag-of-features.
  Good enough to build/validate the pipeline and run CI offline; NOT semantically
  strong — use a real model for the actual clinical test.
- ``SentenceTransformerEmbedder`` : local transformer (e.g. all-MiniLM, or a
  biomedical model). Recommended default for the real beta.
- ``OpenAIEmbedder`` : hosted embeddings via env API key.

All return L2-normalised vectors of a fixed dimension so cosine similarity is
consistent across the main store and the stateful cache.
"""

from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod
from typing import List

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class EmbeddingProvider(ABC):
    dim: int
    name: str

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        ...

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed(t) for t in texts]


def _normalise(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0 or not np.isfinite(norm):
        # Avoid zero vectors (cosine undefined); nudge to a tiny uniform vector.
        vec = np.full_like(vec, 1.0 / math.sqrt(len(vec)))
        return vec
    return vec / norm


class HashingEmbedder(EmbeddingProvider):
    """Deterministic hashed n-gram embedding. No external dependencies."""

    def __init__(self, dim: int = 384, name: str = "hashing-v1"):
        self.dim = dim
        self.name = name

    def embed(self, text: str) -> List[float]:
        vec = np.zeros(self.dim, dtype=np.float64)
        tokens = _TOKEN_RE.findall((text or "").lower())
        if not tokens:
            return _normalise(vec).tolist()
        # Unigrams + bigrams for a little context sensitivity.
        features = list(tokens)
        features += [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
        for feat in features:
            h = int(hashlib.md5(feat.encode()).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if (h >> 1) & 1 else -1.0
            vec[idx] += sign
        return _normalise(vec).tolist()


class SentenceTransformerEmbedder(EmbeddingProvider):
    """Local transformer embeddings (optional dependency: sentence-transformers).

    ``model`` may be a HuggingFace repo id or a local snapshot path; ``name`` lets
    callers record a clean label (e.g. ``all-MiniLM-L6-v2``) when loading from a
    cache directory path.
    """

    def __init__(self, model: str = "all-MiniLM-L6-v2", name: str | None = None):
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Install with: pip install sentence-transformers"
            ) from exc
        self._model = SentenceTransformer(model)
        self.dim = int(self._model.get_sentence_embedding_dimension())
        self.name = name or model

    def embed(self, text: str) -> List[float]:
        vec = self._model.encode(text or "", normalize_embeddings=True)
        return np.asarray(vec, dtype=np.float64).tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        arr = self._model.encode(list(texts), normalize_embeddings=True, batch_size=32)
        return [np.asarray(v, dtype=np.float64).tolist() for v in arr]


class GeminiEmbedder(EmbeddingProvider):
    """Google AI Studio (Gemini) embeddings via REST. Needs GOOGLE_API_KEY.

    Returns L2-normalised vectors so cosine similarity is consistent with the rest
    of the system. ``text-embedding-004`` is 768-dimensional.
    """

    _BASE = "https://generativelanguage.googleapis.com/v1beta"
    # Requested output dimensionality (gemini-embedding-001 supports 768/1536/3072).
    _OUTPUT_DIM = 768

    def __init__(self, model: str = "gemini-embedding-001", api_key: str | None = None,
                 output_dim: int = 768, task_type: str = "SEMANTIC_SIMILARITY",
                 max_workers: int = 8):
        import os
        self.name = model
        self._model_path = f"models/{model}"
        self.dim = output_dim
        self._task_type = task_type
        self._max_workers = max_workers
        self._key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not self._key:
            raise RuntimeError("GOOGLE_API_KEY / GEMINI_API_KEY not set for GeminiEmbedder")
        import httpx
        try:
            import certifi
            self._client = httpx.Client(timeout=60, verify=certifi.where())
        except Exception:
            self._client = httpx.Client(timeout=60)

    def _normalise(self, values: List[float]) -> List[float]:
        return _normalise(np.asarray(values, dtype=np.float64)).tolist()

    def embed(self, text: str) -> List[float]:
        from ._http import post_with_retry
        url = f"{self._BASE}/{self._model_path}:embedContent"
        body = {
            "model": self._model_path,
            "content": {"parts": [{"text": text or ""}]},
            "taskType": self._task_type,
            "outputDimensionality": self.dim,
        }
        r = post_with_retry(self._client, url, params={"key": self._key}, json=body)
        return self._normalise(r.json()["embedding"]["values"])

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        # The synchronous batch endpoint isn't available for this model, so fan out
        # single embedContent calls with a small thread pool for throughput.
        from concurrent.futures import ThreadPoolExecutor
        if not texts:
            return []
        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            return list(pool.map(self.embed, texts))


class OpenAIEmbedder(EmbeddingProvider):
    """Hosted OpenAI embeddings (optional; needs OPENAI_API_KEY)."""

    _DIMS = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072}

    def __init__(self, model: str = "text-embedding-3-small"):
        try:
            from openai import OpenAI
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("openai is not installed. Install with: pip install openai") from exc
        self._client = OpenAI()
        self.name = model
        self.dim = self._DIMS.get(model, 1536)

    def embed(self, text: str) -> List[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        resp = self._client.embeddings.create(model=self.name, input=list(texts))
        return [d.embedding for d in resp.data]


def build_embedder(provider: str, *, dim: int = 384, model: str = "", name: str = "") -> EmbeddingProvider:
    provider = provider.lower()
    if provider == "hashing":
        return HashingEmbedder(dim=dim, name=model or "hashing-v1")
    if provider == "sentence_transformers":
        return SentenceTransformerEmbedder(model=model or "all-MiniLM-L6-v2", name=name or None)
    if provider in ("gemini", "google"):
        return GeminiEmbedder(model=model or "text-embedding-004")
    if provider == "openai":
        return OpenAIEmbedder(model=model or "text-embedding-3-small")
    raise ValueError(f"Unknown embedding provider: {provider!r}")
