"""MCP server for Knowledge Compiler v2.

Reinstates the five v1 tools (signatures recovered from Hermes request dumps
after ``knowledge_mcp_server.py`` was lost with the Mac mini) and adds the norm
tools that make read-time resolution possible.

``mcp`` is imported lazily so the rest of the package - and the test suite -
works without it installed.
"""
from __future__ import annotations

import json
from typing import Any

from .compile import Compiler
from .norms import NormStore

_compiler: Compiler | None = None
_norms: NormStore | None = None


def _c() -> Compiler:
    global _compiler
    if _compiler is None:
        _compiler = Compiler()
    return _compiler


def _n() -> NormStore:
    global _norms
    if _norms is None:
        _norms = NormStore()
    return _norms


def _dump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, default=str)


# --- tool implementations (plain functions, unit-testable without mcp) -------

def intuition_search(query: str, limit: int = 10) -> str:
    """Search the knowledge graph for clinical reasoning matching a query."""
    hits = _c().retriever.retrieve(query, k=limit)
    notes = _c().retriever.notes
    return _dump(
        {
            "query": query,
            "results": [
                {
                    "title": t,
                    "score": round(s, 3),
                    "tags": notes[t].tags,
                    "preview": notes[t].content[:240],
                    "links": getattr(notes[t], "links_resolved", []),
                }
                for t, s in hits
            ],
        }
    )


def intuition_get_note(title: str) -> str:
    """Get the full content of a specific clinical reasoning note by title."""
    note = _c().retriever.notes.get(title)
    if note is None:
        return _dump({"error": "not found", "title": title})
    return _dump(
        {
            "title": note.title,
            "kind": note.kind,
            "tags": note.tags,
            "content": note.content,
            "links": getattr(note, "links_resolved", []),
            "superseded_parameters": _n().audit_text(note.content),
        }
    )


def intuition_neighbors(concept: str, depth: int = 1) -> str:
    """Get neighbouring concepts in the knowledge graph."""
    return _dump(
        {"concept": concept, "depth": depth, "neighbors": _c().retriever.neighbors(concept, depth)}
    )


def intuition_compile(seed: str, max_tokens: int = 8000) -> str:
    """Compile a reasoning module for a seed concept or a whole case description."""
    r = _c().compile(seed, max_tokens=max_tokens)
    return _dump(
        {
            "seed": seed,
            "compiled_prompt": r.prompt,
            "concepts": len(r.concepts),
            "estimated_tokens": r.estimated_tokens,
            "corrections": r.corrections,
            "norm_refs": r.norm_refs,
            "unverified_values": r.unverified_values,
        }
    )


def intuition_graph_stats() -> str:
    """Statistics about the knowledge graph and the norms layer."""
    s = _c().retriever.stats()
    ns = _n()
    s.update(
        {
            "norms": len(ns.norms),
            "stale_norms": [n.id for n in ns.stale()],
        }
    )
    return _dump(s)


def norm_lookup(norm_id: str) -> str:
    """Authoritative current value for a clinical parameter. THE ONLY sanctioned
    source for a threshold, score definition, target or dose."""
    n = _n().get(norm_id)
    asked_retired = False
    if n is None:
        n, asked_retired = _n().resolve_name(norm_id)
    if n is None:
        return _dump({"error": "no such norm", "norm_id": norm_id,
                      "action": "state that the norm is missing; do NOT answer from memory"})
    return _dump(
        {
            "id": n.id,
            "current": n.current,
            "authority": n.authority,
            "valid_from": n.valid_from,
            "thresholds": n.thresholds,
            "supersedes": n.supersedes,
            "notes": n.notes,
            "last_verified": n.last_verified,
            "expires_on": n.expires_on,
            "stale": n.is_stale(),
            **(
                {
                    "warning": (
                        f"{norm_id!r} is a RETIRED name. The current parameter is "
                        f"{n.current!r} ({n.authority}). Answer with the current one."
                    )
                }
                if asked_retired
                else {}
            ),
        }
    )


def norm_search(query: str, limit: int = 5) -> str:
    """Find the norm records relevant to a question before answering it."""
    return _dump(
        {
            "query": query,
            "results": [
                {"id": n.id, "current": n.current, "authority": n.authority,
                 "stale": n.is_stale()}
                for n in _n().search(query, limit)
            ],
        }
    )


TOOLS = [
    intuition_search,
    intuition_get_note,
    intuition_neighbors,
    intuition_compile,
    intuition_graph_stats,
    norm_lookup,
    norm_search,
]


def create_server():
    """Build the FastMCP server. Requires the optional ``mcp`` dependency."""
    from mcp.server.fastmcp import FastMCP  # noqa: PLC0415

    server = FastMCP("knowledge-compiler")
    for fn in TOOLS:
        server.tool()(fn)
    return server


def main() -> None:
    create_server().run()


if __name__ == "__main__":
    main()


# --- concept memory: the vault as a compounding graph -----------------------

_store = None


def _s():
    global _store
    if _store is None:
        from .config import VAULT_DIR  # noqa: PLC0415
        from .ingest import ConceptStore  # noqa: PLC0415

        _store = ConceptStore(VAULT_DIR / "concepts")
    return _store


def concept_candidates(title: str, content: str = "", limit: int = 5) -> str:
    """Which existing concepts is this observation closest to?

    Call before adding anything, to see whether the vault already holds it."""
    return _dump(
        {
            "query": title,
            "candidates": [
                {"id": cid, "title": _s().concepts[cid].title,
                 "similarity": round(score, 3),
                 "strength": _s().concepts[cid].strength}
                for score, cid in _s().nearest(title, content, n=limit)
            ],
        }
    )


def concept_ingest(
    title: str, content: str, source: str, observed_on: str = "", detail: str = ""
) -> str:
    """Add an observation to the vault, merging into an existing concept when it
    already exists.

    Returns action "merge", "link", "create", or "ambiguous". On "ambiguous" the
    server deliberately does not guess: decide from the candidates and then call
    `concept_merge`, or `concept_ingest` again once you have chosen a title."""
    r = _s().ingest(title, content, source, observed_on or None, detail)
    return _dump(
        {
            "action": r.action,
            "concept_id": r.concept_id,
            "matched": r.matched,
            "similarity": round(r.score, 3),
            "candidates": r.candidates,
            "next": (
                "Same concept? call concept_merge. Different? call concept_ingest "
                "with a distinct title."
            )
            if r.action == "ambiguous"
            else None,
        }
    )


def concept_merge(keep_id: str, absorb_id: str) -> str:
    """Fold one concept into another, keeping both names findable as aliases."""
    ok = _s().merge(keep_id, absorb_id)
    return _dump(
        {"merged": ok, "keep": keep_id, "absorbed": absorb_id,
         "strength": _s().concepts[keep_id].strength if ok else None}
    )


def concept_compounding() -> str:
    """Is the vault compounding, or merely accumulating?

    Evidence per concept should rise as encounters accrue. If concept count grows
    in step with transcripts, knowledge is being appended rather than integrated."""
    return _dump(_s().compounding())


TOOLS += [concept_candidates, concept_ingest, concept_merge, concept_compounding]
