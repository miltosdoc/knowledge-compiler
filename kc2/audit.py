"""Vault auditor: find every note whose content can go stale.

This is the operator-facing half of the norms layer. It walks the vault and
reports, per note, which parameters are already retired and which raw values
should be lifted out of prose into norm records.
"""
from __future__ import annotations

from pathlib import Path

from .index import load_notes
from .norms import NormStore
from .schema import detect_normative_claims


def audit_vault(atomic_dir: Path | None = None, norms_dir: Path | None = None) -> dict:
    notes = load_notes(atomic_dir)
    store = NormStore(norms_dir)

    retired, unmapped, clean = [], [], 0
    for title, note in notes.items():
        text = f"{title}\n{note.content}"
        corrections = store.audit_text(text)
        claims = detect_normative_claims(note.content)
        covered = set()
        for c in claims:
            n, _ = store.resolve_name(c)
            if n:
                covered.add(c)
        loose = [c for c in claims if c not in covered]

        if corrections:
            retired.append({"title": title, "corrections": corrections})
        if loose:
            unmapped.append({"title": title, "values": loose})
        if not corrections and not loose:
            clean += 1

    return {
        "notes": len(notes),
        "clean": clean,
        "with_retired_parameters": retired,
        "with_unmapped_values": unmapped,
        "stale_norms": [n.id for n in store.stale()],
    }


def format_report(result: dict) -> str:
    lines = [
        "# Vault audit",
        "",
        f"- notes scanned: **{result['notes']}**",
        f"- clean (no time-sensitive content): **{result['clean']}**",
        f"- notes citing a RETIRED parameter: **{len(result['with_retired_parameters'])}**",
        f"- notes with values not yet in the norms layer: **{len(result['with_unmapped_values'])}**",
        "",
    ]
    if result["with_retired_parameters"]:
        lines += ["## Retired parameters (fix these first)", ""]
        for r in result["with_retired_parameters"]:
            for c in r["corrections"]:
                lines.append(f"- `{r['title']}` — **{c['found']}** → **{c['superseded_by']}**")
        lines.append("")
    if result["with_unmapped_values"]:
        lines += [
            "## Values in prose with no norm record",
            "",
            "Each of these is a number a future reader could mistake for current guidance.",
            "",
        ]
        for r in result["with_unmapped_values"]:
            lines.append(f"- `{r['title']}` — {', '.join('`'+v+'`' for v in r['values'])}")
        lines.append("")
    if result["stale_norms"]:
        lines += ["## Norms past re-verification", ""]
        lines += [f"- `{s}`" for s in result["stale_norms"]]
    return "\n".join(lines)


if __name__ == "__main__":
    print(format_report(audit_vault()))
