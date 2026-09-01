"""Consolidation: turn the flat distiller output into a linked, de-duplicated vault.

The distiller writes one flat note per (encounter, pattern). Two defects make
the serving path (``Retriever`` -> ``load_notes``) lose information:

1. ``load_notes`` keys notes by **title**, but the distiller re-invents titles
   across encounters. Hundreds of notes share a title with a sibling, so a
   title-keyed dict silently keeps one and drops the rest.
2. ``Links:`` is empty in every note (the extraction model never saw the other
   notes), so ``Connected:`` is always ``-`` and graph traversal has no edges.

This module fixes both **without changing the on-disk format** the serving
path already parses (flat ``title:`` frontmatter + ``Content:``/``Links:``):

- **Merge**: exact-title groups collapse to one canonical note per title -
  union of sources/tags/applies_norm, longest content wins, extras removed.
- **Link**: candidate pairs from a rare-token inverted index, scored by
  Jaccard over the same weighted token vector the retriever builds
  (title*3 + tags*2 + content), top-N per note, mutual.

Lexical-only by design: the dense backend is optional elsewhere, so this pass
must run anywhere. A backup tarball must exist before ``--apply``.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from .index import normalize

FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.S)
LINK_LINE_RE = re.compile(r"^- \[\[(.+?)\]\]\s*$", re.M)
TAG_RE = re.compile(r"#([^\s,\]]+)")


def _parse_flat(text: str) -> dict:
    m = FRONTMATTER_RE.match(text)
    if not m:
        raise ValueError("not a flat note (no frontmatter)")
    fm_raw, body = m.group(1), m.group(2)
    fm: dict[str, str] = {}
    for line in fm_raw.splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            fm[k.strip()] = v.strip()
    content_m = re.search(r"Content:\s*(.*?)(?=\nLinks:|\Z)", body, re.S)
    return {
        "title": fm.get("title", "").strip(),
        "kind": fm.get("kind", "pattern"),
        "source": fm.get("source", "unknown"),
        "tags": fm.get("tags", "[]"),
        "applies_norm": fm.get("applies_norm", "[]"),
        "observed_on": fm.get("observed_on", ""),
        "extracted_by": fm.get("extracted_by", "Qwen27b"),
        "content": content_m.group(1).strip() if content_m else "",
        "links": LINK_LINE_RE.findall(body),
    }


def _render(note: dict, sources: list[str]) -> str:
    lines = [
        "---",
        f"title: {note['title']}",
        f"kind: {note['kind']}",
        f"source: {note['source']}",
    ]
    if len(sources) > 1:
        lines.append("sources: [" + ", ".join(sources) + "]")
    if note.get("applies_norm") and note["applies_norm"] != "[]":
        lines.append(f"applies_norm: {note['applies_norm']}")
    if note.get("observed_on"):
        lines.append(f"observed_on: {note['observed_on']}")
    lines += [
        f"extracted_by: {note['extracted_by']}",
        f"tags: {note['tags']}",
        "---",
        "",
        f"Title: {note['title']}",
        f"Kind: {note['kind']}",
        "Content:",
        note["content"],
        "Links:",
    ]
    lines += [f"- [[{x}]]" for x in note["links"]]
    lines.append("")
    return "\n".join(lines)


def _tok_vec(note: dict) -> set[str]:
    """Set form of the retriever's weighted token vector (weights don't change
    membership, only BM25 scores, so Jaccard here mirrors the index)."""
    toks = set(normalize(note["title"]))
    toks |= set(normalize(" ".join(TAG_RE.findall(note["tags"]))))
    toks |= set(normalize(note["content"]))
    return toks


def load_atomic(atomic: Path) -> dict[str, list[dict]]:
    """Group flat notes by title: title -> [member notes]."""
    out: dict[str, list[dict]] = {}
    for f in sorted(atomic.glob("*.md")):
        try:
            note = _parse_flat(f.read_text(encoding="utf-8"))
        except ValueError:
            continue
        if not note["title"]:
            continue
        note["file"] = f.name
        out.setdefault(note["title"], []).append(note)
    return out


def merge_groups(groups: dict[str, list[dict]]) -> dict[str, dict]:
    """Exact-title groups -> one canonical note each. Longest content wins."""
    canonical: dict[str, dict] = {}
    for title, members in groups.items():
        members = sorted(members, key=lambda m: m["file"])
        base = max(members, key=lambda m: len(m["content"]))
        tag_union: set[str] = set()
        norm_union: set[str] = set()
        for m in members:
            tag_union |= set(TAG_RE.findall(m["tags"]))
            norm_union |= set(TAG_RE.findall(m["applies_norm"]))
        canon = dict(base)
        canon["links"] = []
        canon["merged_from"] = [m["file"] for m in members if m["file"] != base["file"]]
        canon["sources"] = [m["source"] for m in members]
        # stored WITHOUT the key prefix — _render adds it
        canon["tags"] = "[" + ", ".join("#" + t for t in sorted(tag_union)) + "]"
        canon["applies_norm"] = "[" + ", ".join(sorted(norm_union)) + "]" if norm_union else "[]"
        for m in members:
            if not canon["observed_on"] and m["observed_on"]:
                canon["observed_on"] = m["observed_on"]
        canonical[title] = canon
    return canonical


def build_links(canon: dict[str, dict], top_n: int, min_jaccard: float):
    """Rare-token candidate pairs -> Jaccard -> top-N mutual links.

    Returns (links_by_title, pairs_scored, pairs_considered)."""
    vecs = {t: _tok_vec(c) for t, c in canon.items()}
    docs = {
        t: (normalize(c["title"]) * 3
            + normalize(" ".join(TAG_RE.findall(c["tags"]))) * 2
            + normalize(c["content"]))
        for t, c in canon.items()
    }
    df: Counter = Counter()
    for toks in docs.values():
        df.update(set(toks))
    n_docs = len(canon)
    rare_cut = max(5, int(0.02 * n_docs))
    inv: dict[str, list[str]] = defaultdict(list)
    for t, toks in docs.items():
        for tok in set(toks):
            if df[tok] <= rare_cut:
                inv[tok].append(t)
    pair_hits: Counter = Counter()
    for holders in inv.values():
        if not (2 <= len(holders) <= 400):
            continue
        for i in range(len(holders)):
            for j in range(i + 1, len(holders)):
                pair_hits[(holders[i], holders[j])] += 1
    scored_links: dict[str, list[tuple[float, str]]] = {t: [] for t in canon}
    scored = 0
    for (a, b), hits in pair_hits.items():
        if hits < 2:
            continue
        va, vb = vecs[a], vecs[b]
        inter = len(va & vb)
        if not inter:
            continue
        jac = inter / len(va | vb)
        if jac < min_jaccard:
            continue
        scored += 1
        scored_links[a].append((jac, b))
        scored_links[b].append((jac, a))
    out: dict[str, list[str]] = {}
    for t, cands in scored_links.items():
        cands.sort(key=lambda x: -x[0])
        out[t] = [other for _, other in cands[:top_n]]
    return out, scored, len(pair_hits)


def run(atomic: Path, top_n: int = 6, min_jaccard: float = 0.12,
        apply: bool = False) -> dict:
    groups = load_atomic(atomic)
    before = sum(len(v) for v in groups.values())
    canon = merge_groups(groups)
    links, scored, n_pairs = build_links(canon, top_n=top_n, min_jaccard=min_jaccard)

    report = {
        "notes_before": before,
        "titles_after": len(canon),
        "merged_away": before - len(canon),
        "candidate_pairs": n_pairs,
        "linked_pairs": scored,
        "edges": sum(len(v) for v in links.values()) // 2,
        "links_per_node": {str(k): v for k, v in sorted(Counter(len(v) for v in links.values()).items())},
    }
    if apply:
        for title, c in canon.items():
            c["links"] = links.get(title, [])
            (atomic / c["file"]).unlink()
            for old in c["merged_from"]:
                (atomic / old).unlink()
            slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:80]
            (atomic / f"{slug}.md").write_text(
                _render(c, c["sources"]), encoding="utf-8"
            )
        (atomic / ".consolidate-manifest.json").write_text(json.dumps(
            {
                "titles": len(canon),
                "merged_from": {t: c["merged_from"] for t, c in canon.items() if c["merged_from"]},
                "edges": report["edges"],
            }, indent=1), encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--atomic", default="vault/atomic")
    ap.add_argument("--top-n", type=int, default=6)
    ap.add_argument("--min-jaccard", type=float, default=0.12)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    report = run(Path(args.atomic), args.top_n, args.min_jaccard, apply=args.apply)
    print(json.dumps(report, indent=2))
    sys.exit(0)


if __name__ == "__main__":
    main()
