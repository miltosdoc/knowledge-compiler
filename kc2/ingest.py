"""Seamless ingestion: every new encounter merges into the existing graph.

The rule that makes the vault compound: nothing is written without first asking
what the vault already knows. An extraction resolves against existing concepts
before it is allowed to become a new node, so a pattern seen repeatedly
strengthens one node instead of spawning a synonym.

  MERGE   above the merge threshold -> attach as evidence to the existing
          concept, keeping the incoming title as an alias
  LINK    in the related band -> create the concept, but wire it to its nearest
          neighbours so it is never born isolated
  CREATE  genuinely novel -> a new node

Work is proportional to what changed: only the incoming concept and its
neighbourhood are re-embedded, never the whole vault.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from . import config
from .concepts import Concept, Observation, load_concepts, save_concept, slugify
from .embed import get_embedder

MERGE_THRESHOLD = 0.88
LINK_THRESHOLD = 0.72

ADJUDICATOR_PROMPT = """You decide whether two clinical notes describe THE SAME underlying reasoning concept, or two different ones.

Same concept: the same clinical pattern, principle or heuristic, even when the wording, the vocabulary or the illustrative case differs.
Different concepts: related or adjacent ideas that a clinician would want to keep as separate entries.

Answer with exactly one word: SAME or DIFFERENT."""


def adjudicate(a_title: str, a_body: str, b_title: str, b_body: str) -> bool | None:
    """Ask the local model whether two concepts are the same.

    Cosine similarity cannot settle the middle band: "Nitrate Response as a
    Diagnostic Test" scores 0.79 against "Nitroglycerin Responsiveness as a
    Diagnostic Anchor" - plainly the same concept, comfortably under any safe
    merge threshold. Rather than lower the threshold globally and start merging
    genuinely distinct ideas, only the ambiguous band is escalated.

    Returns True (same), False (different), or None when no model is reachable,
    in which case the caller keeps the conservative threshold behaviour.
    """
    import json
    import urllib.request

    payload = {
        "model": config.ADJUDICATOR_MODEL,
        "messages": [
            {"role": "system", "content": ADJUDICATOR_PROMPT},
            {
                "role": "user",
                "content": (
                    f"A) {a_title}\n{a_body[:700]}\n\nB) {b_title}\n{b_body[:700]}"
                ),
            },
        ],
        "temperature": 0,
        # Reasoning models (qwen3.8-flash-next among them) emit reasoning_content
        # first and leave content empty. A small budget is spent entirely on
        # thinking and returns finish_reason="length" with no answer at all - the
        # adjudicator then fails silently on every call and nothing ever merges.
        "max_tokens": 512,
    }
    req = urllib.request.Request(
        f"{config.ADJUDICATOR_BASE_URL.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.ADJUDICATOR_KEY}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=config.ADJUDICATOR_TIMEOUT) as r:
            message = json.loads(r.read())["choices"][0]["message"]
    except Exception:
        return None

    answer = (message.get("content") or "").strip()
    if not answer:
        # Fall back to the thinking trace, taking its LAST verdict: the model
        # weighs both options aloud, so only the final one is its conclusion.
        answer = message.get("reasoning_content") or ""

    hits = re.findall(r"\b(SAME|DIFFERENT)\b", answer.upper())
    if not hits:
        return None
    return hits[-1] == "SAME"


@dataclass
class IngestResult:
    action: str              # "merge" | "link" | "create" | "ambiguous"
    concept_id: str
    score: float = 0.0
    matched: str | None = None
    candidates: list = field(default_factory=list)
    pending: tuple | None = None


class ConceptStore:
    def __init__(self, directory: Path):
        self.dir = Path(directory)
        self.concepts: dict[str, Concept] = load_concepts(self.dir)
        self.embedder = get_embedder()
        self._vectors: dict[str, list[float]] = {}
        if self.embedder.available and self.concepts:
            self._embed_all()

    # -- embedding ---------------------------------------------------------
    def _text(self, c: Concept) -> str:
        return f"{c.title}. {c.content}"

    def _embed_all(self) -> None:
        ids = list(self.concepts)
        vecs = self.embedder.encode([self._text(self.concepts[i]) for i in ids])
        self._vectors = dict(zip(ids, vecs))

    def _embed_one(self, c: Concept) -> list[float]:
        return self.embedder.encode([self._text(c)])[0]

    # -- resolution --------------------------------------------------------
    def nearest(self, title: str, content: str, n: int = 5) -> list[tuple[float, str]]:
        if not self.embedder.available or not self._vectors:
            return []
        v = self.embedder.encode([f"{title}. {content}"])[0]
        scored = [
            (sum(x * y for x, y in zip(v, vec)), cid) for cid, vec in self._vectors.items()
        ]
        scored.sort(reverse=True)
        return scored[:n]

    # -- the ingest loop ---------------------------------------------------
    def ingest(
        self,
        title: str,
        content: str,
        source: str,
        observed_on: str | None = None,
        detail: str = "",
        tags: list[str] | None = None,
        write: bool = True,
        adjudicator: bool = True,
    ) -> IngestResult:
        near = self.nearest(title, content)
        obs = Observation(source=source, observed_on=observed_on, detail=detail or title)

        merge_with: tuple[float, str] | None = None
        if near and near[0][0] >= MERGE_THRESHOLD:
            merge_with = near[0]
        elif adjudicator and near and LINK_THRESHOLD <= near[0][0] < MERGE_THRESHOLD:
            score, cid = near[0]
            other = self.concepts[cid]
            mode = config.ADJUDICATOR_MODE
            if mode == "api":
                if adjudicate(title, content, other.title, other.content) is True:
                    merge_with = (score, cid)
            elif mode == "harness":
                # Do not guess and do not phone a second model. Hand the decision
                # back to the agent that is already driving this server; it calls
                # concept_merge if the two are in fact the same.
                return IngestResult(
                    "ambiguous", cid, score, other.title,
                    candidates=[(round(s2, 3), self.concepts[c2].title) for s2, c2 in near[:3]],
                    pending=(title, content, source, observed_on, detail, list(tags or [])),
                )

        if merge_with:
            score, cid = merge_with
            target = self.concepts[cid]
            if title != target.title and title not in target.aliases:
                target.aliases.append(title)
            target.evidence.append(obs)
            for t in tags or []:
                if t not in target.tags:
                    target.tags.append(t)
            if write:
                save_concept(self.dir, target)
            self._vectors[cid] = self._embed_one(target)
            return IngestResult("merge", cid, score, target.title)

        cid = slugify(title)
        suffix = 2
        while cid in self.concepts:
            cid = f"{slugify(title)}-{suffix}"
            suffix += 1
        concept = Concept(
            id=cid, title=title, content=content, tags=list(tags or []), evidence=[obs]
        )

        action = "create"
        matched = None
        related = [(s, c) for s, c in near if s >= LINK_THRESHOLD]
        if related:
            action = "link"
            matched = self.concepts[related[0][1]].title
            for _, other_id in related[:3]:
                other = self.concepts[other_id]
                if other.title not in concept.links:
                    concept.links.append(other.title)
                if concept.title not in other.links:
                    other.links.append(concept.title)
                    if write:
                        save_concept(self.dir, other)

        self.concepts[cid] = concept
        self._vectors[cid] = self._embed_one(concept)
        if write:
            save_concept(self.dir, concept)
        return IngestResult(action, cid, related[0][0] if related else 0.0, matched)

    def merge(self, keep_id: str, absorb_id: str, write: bool = True) -> bool:
        """Fold one concept into another. The decision is the caller's."""
        keep, gone = self.concepts.get(keep_id), self.concepts.get(absorb_id)
        if not keep or not gone or keep_id == absorb_id:
            return False
        keep.absorb(gone)
        self.concepts.pop(absorb_id, None)
        self._vectors.pop(absorb_id, None)
        if self.embedder.available:
            self._vectors[keep_id] = self._embed_one(keep)
        if write:
            save_concept(self.dir, keep)
            path = self.dir / f"{absorb_id}.md"
            if path.exists():
                path.unlink()
        return True

    def commit_pending(self, result: "IngestResult", target_id: str | None) -> IngestResult:
        """Resolve an ``ambiguous`` result once the agent has decided.

        ``target_id`` names the concept to merge into, or None to create a new one.
        """
        if result.pending is None:
            return result
        title, content, source, observed_on, detail, tags = result.pending
        if target_id and target_id in self.concepts:
            target = self.concepts[target_id]
            if title != target.title and title not in target.aliases:
                target.aliases.append(title)
            target.evidence.append(
                Observation(source=source, observed_on=observed_on, detail=detail or title)
            )
            for t in tags:
                if t not in target.tags:
                    target.tags.append(t)
            save_concept(self.dir, target)
            self._vectors[target_id] = self._embed_one(target)
            return IngestResult("merge", target_id, result.score, target.title)
        return self.ingest(
            title, content, source, observed_on, detail, tags, adjudicator=False
        )

    # -- consolidation of an existing vault --------------------------------
    def consolidate(self, threshold: float = MERGE_THRESHOLD, write: bool = True) -> list[tuple]:
        """Fold existing near-duplicates together. Retroactive compounding."""
        merged: list[tuple] = []
        ids = sorted(self.concepts, key=lambda i: (-self.concepts[i].strength, i))
        absorbed: set[str] = set()
        for cid in ids:
            if cid in absorbed or cid not in self.concepts:
                continue
            base = self.concepts[cid]
            for score, other_id in self.nearest(base.title, base.content, n=6):
                if other_id == cid or other_id in absorbed or score < threshold:
                    continue
                other = self.concepts[other_id]
                base.absorb(other)
                absorbed.add(other_id)
                merged.append((round(score, 3), base.title, other.title))
            if merged and write:
                save_concept(self.dir, base)
        for cid in absorbed:
            self.concepts.pop(cid, None)
            self._vectors.pop(cid, None)
            path = self.dir / f"{cid}.md"
            if write and path.exists():
                path.unlink()
        return merged

    # -- does the vault actually compound? ---------------------------------
    def compounding(self) -> dict:
        strengths = [c.strength for c in self.concepts.values()]
        evidence = sum(strengths)
        return {
            "concepts": len(self.concepts),
            "observations": evidence,
            "evidence_per_concept": round(evidence / max(len(self.concepts), 1), 2),
            "corroborated": sum(1 for s in strengths if s > 1),
            "single_source": sum(1 for s in strengths if s <= 1),
            "contested": sum(1 for c in self.concepts.values() if c.contested),
        }
