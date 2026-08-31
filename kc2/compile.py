"""Compile a holographic prompt that carries references, not values.

Three rules, all of them consequences of the v2 invariant:

1. Values embedded in historical notes are relabelled as *observations of what
   was used at the time*, never as current guidance.
2. Any retired parameter name found in retrieved notes raises a correction block
   at the very top of the prompt, where it cannot be missed.
3. The reader is forbidden from stating a threshold from memory and directed to
   ``norm_lookup`` instead.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from . import config
from .norms import NormStore
from .retrieve import Retriever
from .schema import detect_normative_claims

DIRECTIVE = """\
## BINDING RULES FOR USING THIS MODULE

1. DO NOT state any numeric threshold, score definition, target value, drug dose
   or guideline class from memory. Call `norm_lookup` (or `norm_search`) and quote
   what it returns.
2. Values that appear inside the notes below are HISTORICAL OBSERVATIONS of what
   was applied at the time of the encounter. They are evidence of past reasoning.
   They are NOT current guidance and must not be repeated as such.
3. The durable content here is the REASONING PATTERN - what the clinician attended
   to, what they ruled out, and why. Take the pattern; look the parameter up.
4. If a required norm is missing from the norms layer, say so explicitly rather
   than filling the gap from memory.
"""


@dataclass
class Compiled:
    query: str
    prompt: str
    concepts: list[tuple[str, float]] = field(default_factory=list)
    corrections: list[dict] = field(default_factory=list)
    norm_refs: list[str] = field(default_factory=list)
    unverified_values: list[str] = field(default_factory=list)
    stale_norms: list[str] = field(default_factory=list)
    estimated_tokens: int = 0


class Compiler:
    def __init__(self, atomic_dir: Path | None = None, norms_dir: Path | None = None):
        self.retriever = Retriever(atomic_dir)
        self.norms = NormStore(norms_dir)

    def compile(
        self,
        query: str,
        max_tokens: int | None = None,
        k: int = 14,
        today: date | None = None,
    ) -> Compiled:
        max_tokens = max_tokens or config.MAX_PROMPT_TOKENS
        hits = self.retriever.retrieve(query, k=k)

        corrections: list[dict] = []
        seen_corr: set[str] = set()
        unverified: set[str] = set()
        norm_refs: set[str] = set()

        blocks: list[str] = []
        for title, score in hits:
            note = self.retriever.notes[title]
            text = f"{title}\n{note.content}"

            for corr in self.norms.audit_text(text):
                if corr["found"] not in seen_corr:
                    seen_corr.add(corr["found"])
                    corrections.append(corr)
                    norm_refs.add(corr["norm_id"])

            claims = detect_normative_claims(note.content)
            unverified.update(claims)
            for c in claims:
                n, _ = self.norms.resolve_name(c)
                if n:
                    norm_refs.add(n.id)

            flag = " ⚠ contains a retired parameter - see corrections above" if any(
                c["found"] in text.lower() for c in corrections
            ) else ""
            connected = ", ".join(getattr(note, "links_resolved", [])) or "-"
            blocks.append(
                f"### {title}   [relevance {score:.2f}]{flag}\n"
                f"Tags: {' '.join('#' + t for t in note.tags)}\n"
                f"{note.content}\n"
                f"Connected: {connected}\n"
            )

        stale = [n.id for n in self.norms.stale(today)]

        header = [f"# Clinical Reasoning Module: {query}", ""]
        if corrections:
            header += [
                "## ⚠ SUPERSEDED PARAMETERS DETECTED IN THIS MODULE",
                "",
                "The notes below were distilled from encounters that predate current",
                "guidance. The following parameters have since changed:",
                "",
            ]
            for c in corrections:
                header.append(
                    f"- **{c['found']}** → **{c['superseded_by']}** ({c['authority']})\n"
                    f"  - {c['migration'].strip()}\n"
                    f"  - authoritative record: `{c['norm_id']}` (call `norm_lookup`)"
                )
            header.append("")
        if stale:
            header += [
                f"> Freshness warning: {len(stale)} norm record(s) past re-verification: "
                + ", ".join(f"`{s}`" for s in stale),
                "",
            ]
        header += [DIRECTIVE, ""]
        if norm_refs:
            header += [
                "## NORMS REFERENCED BY THIS MODULE (look each one up)",
                "",
                *[f"- `{r}`" for r in sorted(norm_refs)],
                "",
            ]
        if unverified:
            header += [
                "## VALUES APPEARING BELOW - TREAT AS HISTORICAL, VERIFY BEFORE USE",
                "",
                ", ".join(f"`{v}`" for v in sorted(unverified)),
                "",
            ]
        header.append("---\n")

        head = "\n".join(header)
        used = config.estimate_tokens(head)
        kept: list[str] = []
        for b in blocks:
            t = config.estimate_tokens(b)
            if used + t > max_tokens:
                break
            kept.append(b)
            used += t

        prompt = head + "\n".join(kept) + (
            f"\n---\n# END | {len(kept)} concepts | ~{used} tokens | "
            f"{len(corrections)} correction(s)\n"
        )
        return Compiled(
            query=query,
            prompt=prompt,
            concepts=hits[: len(kept)],
            corrections=corrections,
            norm_refs=sorted(norm_refs),
            unverified_values=sorted(unverified),
            stale_norms=stale,
            estimated_tokens=used,
        )
