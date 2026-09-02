"""MCP server for Knowledge Compiler v2.

Reinstates the five v1 tools (signatures recovered from Hermes request dumps
after ``knowledge_mcp_server.py`` was lost with the Mac mini) and adds the norm
tools that make read-time resolution possible.

Protocol layer (2026-09-02): every tool result carries a ``protocol`` object
(see ``protocol.py``) so a fresh agent with no memory inherits the operating
contract from the output itself. ``session_brief`` makes the start-of-case
interview a tool, and thin-context calls get an ``ask_next`` block telling
the agent what to ask before relying on the answer.

``mcp`` is imported lazily so the rest of the package - and the test suite -
works without it installed.
"""
from __future__ import annotations

import json
from typing import Any

from .compile import Compiler
from .norms import NormStore
from . import protocol as _protocol
from .protocol import protocol_block
from .protocol import session_brief

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
            "protocol": protocol_block(checkpoint=True),
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
            "protocol": protocol_block(),
        }
    )


def intuition_neighbors(concept: str, depth: int = 1) -> str:
    """Get neighbouring concepts in the knowledge graph."""
    return _dump(
        {
            "concept": concept,
            "depth": depth,
            "neighbors": _c().retriever.neighbors(concept, depth),
            "protocol": protocol_block(),
        }
    )


def intuition_compile(
    seed: str, context: dict[str, Any] | None = None, max_tokens: int = 8000
) -> str:
    """Compile a reasoning module for a seed concept or a whole case description.

    `context` (optional): what is already known about the case — e.g.
    ``{"age": "68", "duration": "2 weeks, worsening", "meds": "metoprolol
    50 mg", "workup": "rest ECG normal"}``. When omitted or empty, the result
    carries an ``ask_next`` block: the case is being answered on thin
    context, so put the missing pieces to the patient/owner before relying
    on this output."""
    r = _c().compile(seed, max_tokens=max_tokens)
    missing = _protocol._missing_context(context)
    payload: dict[str, Any] = {
        "seed": seed,
        "compiled_prompt": r.prompt,
        "concepts": len(r.concepts),
        "estimated_tokens": r.estimated_tokens,
        "corrections": r.corrections,
        "norm_refs": r.norm_refs,
        "unverified_values": r.unverified_values,
    }
    if missing:
        payload["ask_next"] = _protocol._CONTEXTUAL_MSG
        payload["missing_context"] = missing
    payload["protocol"] = protocol_block()
    return _dump(payload)


def intuition_graph_stats() -> str:
    """Statistics about the knowledge graph and the norms layer."""
    s = _c().retriever.stats()
    ns = _n()
    s.update(
        {
            "norms": len(ns.norms),
            "stale_norms": [n.id for n in ns.stale()],
            "protocol": protocol_block(),
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
                      "action": "state that the norm is missing; do NOT answer from memory",
                      "protocol": protocol_block()})
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
            "protocol": protocol_block(),
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
            "protocol": protocol_block(),
        }
    )


TOOLS = [
    session_brief,
    intuition_search,
    intuition_get_note,
    intuition_neighbors,
    intuition_compile,
    intuition_graph_stats,
    norm_lookup,
    norm_search,
]


SERVER_INSTRUCTIONS = """\
kc2 is the distilled reasoning graph of one cardiologist (Dr Triantafyllou,
Hjartcentrum Halland / Pulsus Hem EKG, Varberg, Sweden): ~5,600 concepts
from 2,100+ encounters, plus a norms layer of authoritative parameter values.

HOW TO DRIVE THIS SERVER
1. New patient case -> call `session_brief` first. It returns the interview
   (age, duration/trend, meds, prior workup, goal) and a coverage score.
   Ask the missing items to the patient/owner, then `intuition_compile` with
   the full picture.
2. A patient CALLING with symptoms is a TRIAGE call: `intuition_compile` on
   the case, answer with red-flag questions (syncope, rest pain, dyspnoea,
   palpitations), suggest the clinic contact path. Admin messages (booking,
   result follow-up, complaints) are process questions - no compile needed.
3. Numbers: NEVER state a threshold, dose, target or score from memory.
   `norm_lookup` is the only sanctioned source. Missing norm -> say so.
4. Notes are historical evidence of past reasoning, not current guidance.

LANGUAGE & ROLE RULES (patient-facing Swedish text)
- Name the ROLE, never the physician: 'läkaren', 'sjuksköterskan',
  'personalen'. Exceptions: signature line, legal/insurance documents,
  prescriptions.
- Keep it short, warm, concrete. Swedish for patients; this protocol speaks
  English to the agent.
"""


def create_server():
    """Build the MCP server. Requires the optional ``mcp`` dependency.

    The SDK renamed FastMCP to MCPServer in 2.x, so both are accepted rather
    than pinning users to one major version.
    """
    try:
        from mcp.server.mcpserver import MCPServer as _Server  # noqa: PLC0415
    except ImportError:  # mcp 1.x
        from mcp.server.fastmcp import FastMCP as _Server  # noqa: PLC0415

    server = _Server(
        "knowledge-compiler",
        instructions=SERVER_INSTRUCTIONS,
    )
    # TOOLS is fully populated by the time create_server() is called: the
    # module appends the concept- and source-family tools at the end, so a
    # single loop covers all 14.
    for fn in TOOLS:
        server.tool()(fn)

    @server.prompt()
    def start_guidance() -> str:
        """Start-of-case briefing: how to interview before compiling."""
        return session_brief(None)

    @server.prompt()
    def role_rules() -> str:
        """Swedish role-naming rules for patient-facing text."""
        return (
            "Patient-facing Swedish: name the ROLE, never the physician.\n"
            "- decision/key fact -> 'läkaren'\n"
            "- nurse tasks (reminders, instructions, follow-up calls) -> "
            "'sjuksköterskan'\n"
            "- clinic facilities (scheduling, reception) -> 'personalen'\n"
            "Exception: signature line, legal/insurance documents, prescriptions.\n"
            "Short, warm, concrete. No 'Dr Triantafyllou' in the body."
        )

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
        from .config import CONCEPTS_DIR  # noqa: PLC0415
        from .ingest import ConceptStore  # noqa: PLC0415

        _store = ConceptStore(CONCEPTS_DIR)
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
            "protocol": protocol_block(),
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
            "protocol": protocol_block(),
        }
    )


def concept_merge(keep_id: str, absorb_id: str) -> str:
    """Fold one concept into another, keeping both names findable as aliases."""
    ok = _s().merge(keep_id, absorb_id)
    return _dump(
        {"merged": ok, "keep": keep_id, "absorbed": absorb_id,
         "strength": _s().concepts[keep_id].strength if ok else None,
         "protocol": protocol_block()}
    )


def concept_compounding() -> str:
    """Is the vault compounding, or merely accumulating?

    Evidence per concept should rise as encounters accrue. If concept count grows
    in step with transcripts, knowledge is being appended rather than integrated."""
    out = _s().compounding()
    out["protocol"] = protocol_block()
    return _dump(out)


TOOLS += [concept_candidates, concept_ingest, concept_merge, concept_compounding]


# --- raw sources: let the driving agent do the distillation -----------------

def source_discover(directory: str = "") -> str:
    """Find clinical databases (.db/.sqlite) and pg_dump files (.dump/.sql) and
    report which columns hold the transcript and the clinical note.

    Inspect this before reading anything, to confirm the column mapping."""
    from .config import DATA_DIR  # noqa: PLC0415
    from .sources import discover  # noqa: PLC0415

    directory = directory or str(DATA_DIR)
    return _dump(
        {
            "directory": directory,
            "sources": [
                {
                    "file": str(s.path), "kind": s.kind, "table": s.table,
                    "rows": s.rows, "id_column": s.id_col,
                    "transcript_column": s.transcript_col, "note_column": s.note_col,
                    "columns": s.columns, "usable": s.usable,
                }
                for s in discover(directory)
            ],
            "protocol": protocol_block(),
        }
    )


def source_read(directory: str = "", file: str = "", limit: int = 5,
                offset: int = 0) -> str:
    """Read raw encounters as {id, transcript, notes}.

    Distil each one yourself, then call `concept_ingest` per reasoning pattern
    found. Read in small batches - transcripts are long."""
    from .config import DATA_DIR  # noqa: PLC0415
    from .sources import discover, extract  # noqa: PLC0415

    directory = directory or str(DATA_DIR)
    sources = [s for s in discover(directory) if s.usable]
    if file:
        sources = [s for s in sources if s.path.name == file or str(s.path) == file]
    if not sources:
        return _dump({"error": "no usable source found", "directory": directory})
    src = sources[0]
    records = extract(src, limit=offset + limit)[offset:]
    return _dump(
        {
            "file": str(src.path), "table": src.table,
            "offset": offset, "returned": len(records), "records": records,
            "protocol": protocol_block(),
        }
    )


TOOLS += [source_discover, source_read]
