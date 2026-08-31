"""The knowledge graph, and traversal over it.

This is the primary structure. Retrieval exists only to find an entry point;
everything after that is determined by graph topology, not by query-document
similarity. Compilation returns a *connected neighbourhood* - a subgraph with
the edges that produced it - rather than a ranked list of independent notes.

The authored graph alone cannot carry that. Measured on the recovered corpus:
average degree 0.89, 43% of notes isolated, 63 components, largest 16 nodes.
So edges come from two sources:

  authored   an explicit [[link]] the clinician's distillation asserted.
             High weight - it is a claimed relationship.
  inferred   a k-nearest-neighbour edge from embedding similarity. Lower
             weight - it is a resemblance, not an assertion.

Embeddings densify the graph. They do not rank the answer.
"""
from __future__ import annotations

import heapq
from collections import defaultdict
from dataclasses import dataclass, field

AUTHORED_WEIGHT = 1.0
INFERRED_MAX_WEIGHT = 0.85


@dataclass
class Edge:
    target: str
    weight: float
    kind: str  # "authored" | "inferred"


@dataclass
class Graph:
    adjacency: dict[str, list[Edge]] = field(default_factory=lambda: defaultdict(list))

    def neighbours(self, node: str) -> list[Edge]:
        return self.adjacency.get(node, [])

    def degree(self, node: str) -> int:
        return len(self.adjacency.get(node, []))

    def stats(self) -> dict:
        nodes = list(self.adjacency)
        deg = [len(v) for v in self.adjacency.values()]
        authored = sum(1 for v in self.adjacency.values() for e in v if e.kind == "authored")
        inferred = sum(1 for v in self.adjacency.values() for e in v if e.kind == "inferred")
        return {
            "nodes": len(nodes),
            "edges": (authored + inferred) // 2,
            "authored_edges": authored // 2,
            "inferred_edges": inferred // 2,
            "avg_degree": round(sum(deg) / max(len(deg), 1), 2),
            "isolated": sum(1 for d in deg if d == 0),
        }


def _cos(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))  # vectors are pre-normalised


def build_graph(
    notes: dict,
    vectors: dict[str, list[float]] | None = None,
    k: int = 6,
    min_similarity: float = 0.55,
) -> Graph:
    """Authored links plus a kNN semantic layer, as one weighted graph."""
    g = Graph(defaultdict(list))
    seen: set[tuple[str, str]] = set()

    def add(a: str, b: str, w: float, kind: str) -> None:
        if a == b:
            return
        key = (a, b) if a < b else (b, a)
        if key in seen:
            return
        seen.add(key)
        g.adjacency[a].append(Edge(b, w, kind))
        g.adjacency[b].append(Edge(a, w, kind))

    for title in notes:
        g.adjacency.setdefault(title, [])

    for title, note in notes.items():
        for target in getattr(note, "links_resolved", []):
            if target in notes:
                add(title, target, AUTHORED_WEIGHT, "authored")

    if vectors:
        titles = [t for t in notes if t in vectors]
        for a in titles:
            va = vectors[a]
            sims = ((_cos(va, vectors[b]), b) for b in titles if b != a)
            for s, b in heapq.nlargest(k, sims):
                if s >= min_similarity:
                    add(a, b, min(s, INFERRED_MAX_WEIGHT), "inferred")
    return g


def spreading_activation(
    graph: Graph,
    seeds: dict[str, float],
    decay: float = 0.65,
    max_nodes: int = 20,
    min_activation: float = 0.05,
) -> tuple[dict[str, float], dict[str, tuple[str, str]]]:
    """Flow activation outward from seed nodes along weighted edges.

    Returns ``(activation, provenance)`` where provenance maps each reached node
    to the ``(parent, edge_kind)`` that carried the most activation into it - so
    the compiler can show *how* a concept was reached rather than asserting a
    bare relevance score.
    """
    activation: dict[str, float] = dict(seeds)
    provenance: dict[str, tuple[str, str]] = {}
    frontier = [(-v, n) for n, v in seeds.items()]
    heapq.heapify(frontier)
    settled: set[str] = set()

    while frontier and len(settled) < max_nodes:
        neg, node = heapq.heappop(frontier)
        if node in settled:
            continue
        settled.add(node)
        current = -neg
        for edge in graph.neighbours(node):
            if edge.target in settled:
                continue
            flowed = current * decay * edge.weight
            if flowed < min_activation:
                continue
            if flowed > activation.get(edge.target, 0.0):
                activation[edge.target] = flowed
                provenance[edge.target] = (node, edge.kind)
                heapq.heappush(frontier, (-flowed, edge.target))

    return {n: activation[n] for n in settled}, provenance
