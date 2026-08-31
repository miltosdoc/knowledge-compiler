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
from .index import BM25, canonicalize, load_notes, normalize
from .schema import Note

GRAPH_DECAY = 0.45


class Retriever:
    def __init__(self, atomic_dir: Path | None = None):
        self.notes: dict[str, Note] = load_notes(atomic_dir)
        self.adjacency, self.alias, self.links_resolved, self.links_dangling = canonicalize(
            self.notes
        )
        docs = {
            t: normalize(t) * 3 + normalize(" ".join(n.tags)) * 2 + normalize(n.content)
            for t, n in self.notes.items()
        }
        self.bm25 = BM25(docs)

    # -- stats -------------------------------------------------------------
    def stats(self) -> dict:
        edges = sum(len(v) for v in self.adjacency.values()) // 2
        return {
            "notes": len(self.notes),
            "edges": edges,
            "links_resolved": self.links_resolved,
            "links_dangling": self.links_dangling,
            "resolution_rate": round(
                self.links_resolved / max(self.links_resolved + self.links_dangling, 1), 3
            ),
        }

    # -- retrieval ---------------------------------------------------------
    def _dense_scores(self, query: str) -> dict[str, float]:
        """Placeholder for the dense half of hybrid retrieval."""
        return {}

    def retrieve(self, query: str, k: int = 12, hops: int = 1) -> list[tuple[str, float]]:
        lexical = self.bm25.score(normalize(query))
        if not lexical:
            return []
        top = max(lexical.values())
        fused: dict[str, float] = {t: s / top for t, s in lexical.items()}

        for title, score in self._dense_scores(query).items():
            fused[title] = max(fused.get(title, 0.0), score)

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
