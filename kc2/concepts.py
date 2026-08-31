"""Concept nodes: the durable unit of the vault.

v1 made the *transcript* the unit - one encounter produced its own notes with
freshly invented titles, because the distiller never saw the existing graph.
The corpus therefore accumulated without compounding: measured on the recovered
notes, 12% sit in near-duplicate pairs ("The Warm-Up Phenomenon in Exertional
Chest Pain" beside "Warm-Up Phenomenon in Microvascular Angina"), each an
isolated node holding a single encounter's worth of support.

Here the *concept* is durable and encounters attach to it as evidence. Seeing a
pattern forty times yields one node with forty observations behind it, not forty
weak notes. That is what makes the vault compound.

Files stay Obsidian-native - YAML frontmatter, [[wikilinks]], plain headings -
so the vault can be opened, browsed and hand-edited directly, and those edits
read back in.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import yaml


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:80]


@dataclass
class Observation:
    """One encounter's contribution to a concept."""

    source: str                      # transcript id
    observed_on: str | None = None
    detail: str = ""
    contradicts: bool = False

    def to_line(self) -> str:
        mark = "⚠ CONTRADICTS — " if self.contradicts else ""
        when = f" {self.observed_on}" if self.observed_on else ""
        return f"- [[{self.source}]]{when} — {mark}{self.detail}".rstrip()

    @classmethod
    def from_line(cls, line: str) -> "Observation | None":
        m = re.match(r"-\s*\[\[(.+?)\]\]\s*(\d{4}-\d{2}-\d{2})?\s*—?\s*(.*)", line.strip())
        if not m:
            return None
        detail = m.group(3) or ""
        contradicts = detail.startswith("⚠ CONTRADICTS")
        detail = re.sub(r"^⚠ CONTRADICTS\s*—\s*", "", detail)
        return cls(source=m.group(1), observed_on=m.group(2), detail=detail,
                   contradicts=contradicts)


@dataclass
class Concept:
    id: str
    title: str
    kind: str = "pattern"
    content: str = ""
    tags: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)
    applies_norm: list[str] = field(default_factory=list)
    evidence: list[Observation] = field(default_factory=list)

    @property
    def strength(self) -> int:
        """How many independent encounters support this concept."""
        return len({o.source for o in self.evidence if not o.contradicts})

    @property
    def contested(self) -> int:
        return len({o.source for o in self.evidence if o.contradicts})

    @property
    def first_seen(self) -> str | None:
        dates = sorted(o.observed_on for o in self.evidence if o.observed_on)
        return dates[0] if dates else None

    @property
    def last_seen(self) -> str | None:
        dates = sorted(o.observed_on for o in self.evidence if o.observed_on)
        return dates[-1] if dates else None

    # -- Obsidian-native serialisation ------------------------------------
    def to_markdown(self) -> str:
        fm = {
            "id": self.id,
            "title": self.title,
            "kind": self.kind,
            "strength": self.strength,
            "tags": self.tags,
        }
        if self.aliases:
            fm["aliases"] = self.aliases
        if self.applies_norm:
            fm["applies_norm"] = self.applies_norm
        if self.first_seen:
            fm["first_seen"] = self.first_seen
        if self.last_seen:
            fm["last_seen"] = self.last_seen
        if self.contested:
            fm["contested"] = self.contested

        out = ["---", yaml.safe_dump(fm, sort_keys=False, allow_unicode=True).strip(), "---", ""]
        out += [self.content.strip(), ""]
        if self.evidence:
            out += [f"## Evidence ({self.strength})", ""]
            out += [o.to_line() for o in self.evidence]
            out.append("")
        if self.links:
            out += ["## Links", "", ", ".join(f"[[{x}]]" for x in self.links), ""]
        return "\n".join(out)

    @classmethod
    def from_markdown(cls, text: str, fallback_id: str = "") -> "Concept":
        fm: dict = {}
        body = text
        m = re.match(r"^---\n(.*?)\n---\n?(.*)$", text, re.S)
        if m:
            try:
                fm = yaml.safe_load(m.group(1)) or {}
            except yaml.YAMLError:
                fm = {}
            body = m.group(2)

        sections = re.split(r"^##\s+", body, flags=re.M)
        content = sections[0].strip()
        evidence, links = [], []
        for sec in sections[1:]:
            head, _, rest = sec.partition("\n")
            if head.lower().startswith("evidence"):
                for line in rest.splitlines():
                    obs = Observation.from_line(line)
                    if obs:
                        evidence.append(obs)
            elif head.lower().startswith("link"):
                links = [x.strip() for x in re.findall(r"\[\[(.+?)\]\]", rest)]

        title = fm.get("title") or fallback_id
        return cls(
            id=fm.get("id") or slugify(title) or fallback_id,
            title=title,
            kind=fm.get("kind", "pattern"),
            content=content,
            tags=list(fm.get("tags") or []),
            aliases=list(fm.get("aliases") or []),
            links=links,
            applies_norm=list(fm.get("applies_norm") or []),
            evidence=evidence,
        )

    # -- compounding -------------------------------------------------------
    def absorb(self, other: "Concept") -> None:
        """Fold another concept into this one, keeping both names findable."""
        for name in [other.title, *other.aliases]:
            if name != self.title and name not in self.aliases:
                self.aliases.append(name)
        known = {(o.source, o.detail) for o in self.evidence}
        self.evidence += [o for o in other.evidence if (o.source, o.detail) not in known]
        for t in other.tags:
            if t not in self.tags:
                self.tags.append(t)
        for l in other.links:
            if l not in self.links and l != self.title:
                self.links.append(l)
        for n in other.applies_norm:
            if n not in self.applies_norm:
                self.applies_norm.append(n)
        if len(other.content) > len(self.content):
            self.content, other.content = other.content, self.content


def load_concepts(directory: Path) -> dict[str, Concept]:
    out: dict[str, Concept] = {}
    if not Path(directory).exists():
        return out
    for f in sorted(Path(directory).glob("*.md")):
        c = Concept.from_markdown(f.read_text(encoding="utf-8"), fallback_id=f.stem)
        out[c.id] = c
    return out


def save_concept(directory: Path, concept: Concept) -> Path:
    Path(directory).mkdir(parents=True, exist_ok=True)
    path = Path(directory) / f"{concept.id}.md"
    path.write_text(concept.to_markdown(), encoding="utf-8")
    return path
