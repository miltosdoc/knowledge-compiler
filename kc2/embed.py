"""Pluggable dense-embedding backend.

Deliberately zero hard dependencies. The first backend that is actually
available wins, and if none is, retrieval stays lexical-plus-graph rather than
failing. Nothing here needs installing to make the rest of the package work.

Backends, in priority order:
  1. sentence-transformers (local, CPU, offline once cached)
  2. an OpenAI-compatible /v1/embeddings endpoint - e.g. a local llama-server
  3. none -> dense scoring is skipped
"""
from __future__ import annotations

import math
from typing import Sequence

from . import config


class Embedder:
    name = "none"
    available = False

    def encode(self, texts: Sequence[str]) -> list[list[float]]:  # pragma: no cover
        raise NotImplementedError

    def encode_query(self, text: str) -> list[float]:
        return self.encode([text])[0]


class SentenceTransformerEmbedder(Embedder):
    name = "sentence-transformers"

    #: BGE-family models are trained with an asymmetric retrieval objective: the
    #: query carries an instruction prefix and the passage does not. Encoding both
    #: sides identically collapses the score range - measured on this corpus, a
    #: query for "refractory angina" then ranked a *vagal AFib* note top, because
    #: raw cosine could not separate 0.786 from 0.779. With the prefix applied the
    #: same query returns the vasospastic and recurrent-angina notes instead.
    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5"):
        try:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415

            self._m = SentenceTransformer(model)
            self.model_name = model
            self.available = True
        except Exception:
            self.available = False

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        vecs = self._m.encode(
            list(texts), normalize_embeddings=True, batch_size=32, show_progress_bar=False
        )
        return [list(map(float, v)) for v in vecs]

    def encode_query(self, text: str) -> list[float]:
        prefix = self.QUERY_PREFIX if "bge" in getattr(self, "model_name", "").lower() else ""
        return self.encode([prefix + text])[0]


class OpenAICompatEmbedder(Embedder):
    name = "openai-compatible"

    def __init__(self, base_url: str | None = None, model: str | None = None):
        self.base_url = (base_url or config.BASE_URL).rstrip("/")
        self.model = model or "text-embedding-3-small"
        self.available = self._probe()

    def _probe(self) -> bool:
        try:
            import urllib.request  # noqa: PLC0415

            req = urllib.request.Request(f"{self.base_url}/embeddings", method="OPTIONS")
            with urllib.request.urlopen(req, timeout=2):
                return True
        except Exception:
            return False

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        import json  # noqa: PLC0415
        import urllib.request  # noqa: PLC0415

        body = json.dumps({"model": self.model, "input": list(texts)}).encode()
        req = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {config.API_KEY}",
            },
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.loads(r.read())
        return [d["embedding"] for d in data["data"]]


_CACHED: Embedder | None = None


def get_embedder(refresh: bool = False) -> Embedder:
    """Return the first available backend.

    Model loading dominates startup, so the resolved backend is memoised for the
    life of the process - a Retriever built per request must not reload weights.
    """
    global _CACHED
    if _CACHED is not None and not refresh:
        return _CACHED
    for factory in (SentenceTransformerEmbedder, OpenAICompatEmbedder):
        try:
            e = factory()
            if e.available:
                _CACHED = e
                return e
        except Exception:
            continue
    _CACHED = Embedder()
    return _CACHED


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return num / (na * nb) if na and nb else 0.0


def rrf(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """Reciprocal rank fusion - combines lexical and dense orderings."""
    out: dict[str, float] = {}
    for ranking in rankings:
        for i, item in enumerate(ranking):
            out[item] = out.get(item, 0.0) + 1.0 / (k + i + 1)
    return out
