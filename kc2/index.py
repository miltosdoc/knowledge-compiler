"""Note loading, concept canonicalisation, and the lexical index.

v1 counted every ``[[link]]`` target as a graph node whether or not a note
backed it, which left ~90% of edges dangling and made traversal meaningless.
Canonicalisation here resolves link targets onto real note titles before the
graph is built.
"""
from __future__ import annotations

import difflib
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from . import config
from .schema import Note

STOP = set(
    "the a an of in on for for to and or is are was were be been with as at by from that "
    "this it its not no if then than into over under can may might will would should could "
    "do does did have has had but so such when while which who whom whose there here their "
    "his her they he she you i we".split()
)


def normalize(text: str) -> list[str]:
    """Lowercase, strip punctuation, drop stopwords, crude suffix stemming."""
    text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    out = []
    for w in text.split():
        if w in STOP or len(w) < 3:
            continue
        for suf in ("ies", "ing", "ed", "es", "s"):
            if len(w) > 4 and w.endswith(suf):
                w = w[: -len(suf)]
                break
        out.append(w)
    return out


def load_notes(atomic_dir: Path | None = None) -> dict[str, Note]:
    d = Path(atomic_dir or config.ATOMIC_DIR)
    notes: dict[str, Note] = {}
    if not d.exists():
        return notes
    for f in sorted(d.glob("*.md")):
        s = f.read_text(encoding="utf-8")
        m = re.search(r"^title:\s*(.+)$", s, re.M)
        if not m:
            continue
        title = m.group(1).strip()
        body = re.search(r"Content:\s*(.*?)(?=\nLinks:|\Z)", s, re.S)
        kind = re.search(r"^kind:\s*(\w+)$", s, re.M)
        notes[title] = Note(
            title=title,
            kind=kind.group(1) if kind else "pattern",
            content=(body.group(1).strip() if body else s.strip()),
            tags=re.findall(r"#([\w-]+)", s),
            links=[x.strip() for x in re.findall(r"\[\[(.+?)\]\]", s)],
            source=(re.search(r"^source:\s*(.+)$", s, re.M) or [None, None])[1]
            if re.search(r"^source:\s*(.+)$", s, re.M)
            else None,
        )
    return notes


def canonicalize(notes: dict[str, Note]) -> tuple[dict[str, set[str]], dict[str, str], int, int]:
    """Resolve link targets onto real titles. Returns (adjacency, alias, resolved, dangling)."""
    titles = list(notes)
    low = {t.lower(): t for t in titles}
    alias: dict[str, str] = {}
    resolved = dangling = 0

    for n in notes.values():
        fixed: list[str] = []
        for link in n.links:
            key = link.lower()
            target = low.get(key)
            if target is None:
                close = difflib.get_close_matches(key, list(low), n=1, cutoff=0.82)
                target = low[close[0]] if close else None
            if target is None:  # token-overlap fallback
                lt = set(normalize(link))
                best, best_score = None, 0.0
                for cand in titles:
                    ct = set(normalize(cand))
                    if not ct or not lt:
                        continue
                    j = len(lt & ct) / len(lt | ct)
                    if j > best_score:
                        best, best_score = cand, j
                if best_score >= 0.34:
                    target = best
            if target:
                fixed.append(target)
                alias[link] = target
                resolved += 1
            else:
                dangling += 1
        n.links_resolved = sorted(set(fixed) - {n.title})  # type: ignore[attr-defined]

    adjacency: dict[str, set[str]] = defaultdict(set)
    for n in notes.values():
        for target in getattr(n, "links_resolved", []):
            adjacency[n.title].add(target)
            adjacency[target].add(n.title)
    return adjacency, alias, resolved, dangling


class BM25:
    def __init__(self, docs: dict[str, list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.ids = list(docs)
        self.toks = docs
        self.len = {i: len(v) for i, v in docs.items()}
        self.avg = sum(self.len.values()) / max(len(self.len), 1)
        self.tf = {i: Counter(v) for i, v in docs.items()}
        self.df: Counter = Counter()
        for v in docs.values():
            self.df.update(set(v))
        self.N = len(docs)

    def score(self, query: list[str]) -> Counter:
        out: Counter = Counter()
        for term in query:
            df = self.df.get(term)
            if not df:
                continue
            idf = math.log(1 + (self.N - df + 0.5) / (df + 0.5))
            for i in self.ids:
                f = self.tf[i].get(term, 0)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg)
                out[i] += idf * f * (self.k1 + 1) / denom
        return out
