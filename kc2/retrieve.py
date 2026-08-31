"""Hybrid retrieval: lexical + graph expansion, fused and ranked.

v1 matched seeds by substring, so ``"refractory angina"`` returned 0 of 717
notes. Retrieval here scores every note lexically, then expands along the
canonicalised graph with distance decay.

Dense vectors slot in at ``_dense_scores`` once an embedding backend is
available; the fusion path is already reciprocal-rank based.
"""
from __future__ import annotations

from pathlib import Path

from . import config
from .embed import cosine, get_embedder, rrf
from .index import BM25, canonicalize, load_notes, normalize
from .schema import Note

GRAPH_DECAY = 0.45
DENSE_CACHE = ".kc2_embeddings.json"


class Retriever:
    def __init__(self, atomic_dir: Path | None = None, dense: bool = True):
        self.notes: dict[str, Note] = load_notes(atomic_dir)
        self.adjacency, self.alias, self.links_resolved, self.links_dangling = canonicalize(
            self.notes
        )
        docs = {
            t: normalize(t) * 3 + normalize(" ".join(n.tags)) * 2 + normalize(n.content)
            for t, n in self.notes.items()
        }
        self.bm25 = BM25(docs)
        self._dir = Path(atomic_dir or config.ATOMIC_DIR)
        self.embedder = get_embedder() if dense else None
        self._vectors: dict[str, list[float]] = {}
        if self.embedder is not None and self.embedder.available:
            self._load_or_build_vectors()

    # -- dense -------------------------------------------------------------
    def _load_or_build_vectors(self) -> None:
        """Embed each note once and cache to disk, keyed by content hash."""
        import hashlib
        import json

        payload = {t: f"{t}. {n.content}" for t, n in self.notes.items()}
        digest = hashlib.sha256(
            "\x00".join(f"{k}\x01{v}" for k, v in sorted(payload.items())).encode()
        ).hexdigest()

        cache = self._dir / DENSE_CACHE
        if cache.exists():
            try:
                blob = json.loads(cache.read_text())
                if blob.get("digest") == digest:
                    self._vectors = blob["vectors"]
                    return
            except Exception:
                pass

        titles = list(payload)
        vecs = self.embedder.encode([payload[t] for t in titles])
        self._vectors = dict(zip(titles, vecs))
        try:
            cache.write_text(json.dumps({"digest": digest, "vectors": self._vectors}))
        except OSError:
            pass

    # -- stats -------------------------------------------------------------
    def stats(self) -> dict:
        edges = sum(len(v) for v in self.adjacency.values()) // 2
        return {
            "notes": len(self.notes),
            "dense_backend": self.embedder.name if self.embedder else "none",
            "edges": edges,
            "links_resolved": self.links_resolved,
            "links_dangling": self.links_dangling,
            "resolution_rate": round(
                self.links_resolved / max(self.links_resolved + self.links_dangling, 1), 3
            ),
        }

    # -- retrieval ---------------------------------------------------------
    def _dense_scores(self, query: str) -> dict[str, float]:
        if not self._vectors or self.embedder is None:
            return {}
        qv = self.embedder.encode_query(query)
        return {t: cosine(qv, v) for t, v in self._vectors.items()}

    def retrieve(self, query: str, k: int = 12, hops: int = 1) -> list[tuple[str, float]]:
        lexical = self.bm25.score(normalize(query))
        dense = self._dense_scores(query)
        if not lexical and not dense:
            return []

        if dense:
            # Reciprocal rank fusion: cosine and BM25 live on incompatible scales,
            # so combine their ORDERINGS rather than their magnitudes.
            lex_rank = [t for t, _ in sorted(lexical.items(), key=lambda x: -x[1])]
            den_rank = [t for t, _ in sorted(dense.items(), key=lambda x: -x[1])]
            raw = rrf([lex_rank, den_rank])
        else:
            raw = dict(lexical)

        top = max(raw.values())
        fused: dict[str, float] = {t: s / top for t, s in raw.items()}

        seeds = [t for t, _ in sorted(fused.items(), key=lambda x: -x[1])[:k]]
        for depth in range(1, hops + 1):
            for seed in list(seeds):
                for neighbour in self.adjacency.get(seed, ()):
                    if neighbour in self.notes:
                        decayed = fused.get(seed, 0.0) * (GRAPH_DECAY**depth)
                        fused[neighbour] = max(fused.get(neighbour, 0.0), decayed)

        return sorted(fused.items(), key=lambda x: -x[1])[:k]

    def neighbors(self, concept: str, depth: int = 1) -> list[str]:
        seen = {concept}
        frontier = {concept}
        for _ in range(max(depth, 1)):
            nxt: set[str] = set()
            for c in frontier:
                nxt |= self.adjacency.get(c, set())
            nxt -= seen
            seen |= nxt
            frontier = nxt
        return sorted(seen - {concept})
