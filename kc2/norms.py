"""The norms layer: versioned clinical parameters, resolved at read time.

Every value that can change - score definitions, thresholds, targets, doses -
lives here and nowhere else. Notes reference norms by id. One record changes and
the whole corpus is current, without re-distilling anything.

Freshness is explicit: each norm declares a volatility, and a norm past its
re-verification date is flagged rather than served silently.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

from . import config

_VOL = {"m": 30, "mo": 30, "y": 365, "d": 1}


def _parse_volatility(v: str | None) -> timedelta:
    if not v:
        return timedelta(days=365)
    m = re.fullmatch(r"(\d+)\s*(mo|[mdy])", str(v).strip(), re.I)
    if not m:
        return timedelta(days=365)
    return timedelta(days=int(m.group(1)) * _VOL[m.group(2).lower()])


def _as_date(v: Any) -> date | None:
    if isinstance(v, date):
        return v
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, str):
        try:
            return datetime.strptime(v.strip(), "%Y-%m-%d").date()
        except ValueError:
            return None
    return None


@dataclass
class Norm:
    id: str
    current: str
    authority: str = ""
    valid_from: date | None = None
    last_verified: date | None = None
    volatility: str = "12mo"
    thresholds: dict[str, Any] = field(default_factory=dict)
    supersedes: list[dict[str, Any]] = field(default_factory=list)
    notes: str = ""
    aliases: list[str] = field(default_factory=list)

    # ---- freshness -------------------------------------------------------
    @property
    def expires_on(self) -> date | None:
        if not self.last_verified:
            return None
        return self.last_verified + _parse_volatility(self.volatility)

    def is_stale(self, today: date | None = None) -> bool:
        today = today or date.today()
        exp = self.expires_on
        return exp is None or exp < today

    # ---- supersession ----------------------------------------------------
    def superseded_names(self) -> list[str]:
        return [str(s.get("name", "")).strip() for s in self.supersedes if s.get("name")]

    def migration_for(self, old_name: str) -> dict[str, Any] | None:
        for s in self.supersedes:
            if str(s.get("name", "")).strip().lower() == old_name.strip().lower():
                return s
        return None

    def all_names(self) -> list[str]:
        return [self.current, *self.aliases, *self.superseded_names()]

    def is_retired_name(self, name: str) -> bool:
        """True only for names this norm has explicitly superseded.

        Aliases are alternate spellings of the *current* norm and must never be
        reported as stale.
        """
        low = name.strip().lower()
        return any(n.strip().lower() == low for n in self.superseded_names())


class NormStore:
    """Loads ``norms/*.yaml`` and answers read-time lookups."""

    def __init__(self, norms_dir: Path | None = None):
        self.dir = Path(norms_dir or config.NORMS_DIR)
        self.norms: dict[str, Norm] = {}
        self._by_name: dict[str, Norm] = {}
        self.load()

    def load(self) -> None:
        self.norms.clear()
        self._by_name.clear()
        if not self.dir.exists():
            return
        for f in sorted(self.dir.glob("*.yaml")) + sorted(self.dir.glob("*.yml")):
            data = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
            n = Norm(
                id=data["id"],
                current=data.get("current", ""),
                authority=data.get("authority", ""),
                valid_from=_as_date(data.get("valid_from")),
                last_verified=_as_date(data.get("last_verified")),
                volatility=str(data.get("volatility", "12mo")),
                thresholds=data.get("thresholds") or {},
                supersedes=data.get("supersedes") or [],
                notes=data.get("notes", "") or "",
                aliases=data.get("aliases") or [],
            )
            self.norms[n.id] = n
            for name in n.all_names():
                if name:
                    self._by_name[name.strip().lower()] = n

    # ---- the read-time API ----------------------------------------------
    def get(self, norm_id: str) -> Norm | None:
        return self.norms.get(norm_id)

    def resolve_name(self, name: str) -> tuple[Norm | None, bool]:
        """Map any name - current or retired - onto its live norm.

        Returns ``(norm, is_superseded)``. This is what turns a note that says
        "CHA2DS2-VASc" into today's CHA2DS2-VA.
        """
        n = self._by_name.get(name.strip().lower())
        if n is None:
            return None, False
        return n, n.is_retired_name(name)

    def search(self, query: str, limit: int = 5) -> list[Norm]:
        q = query.lower().strip()
        terms = [t for t in re.split(r"\W+", q) if len(t) > 2]
        scored: list[tuple[float, Norm]] = []
        for n in self.norms.values():
            hay = " ".join([n.id, n.current, n.authority, n.notes, *n.all_names()]).lower()
            s = sum(1.0 for t in terms if t in hay)
            if q and q in hay:
                s += 3.0
            if s:
                scored.append((s, n))
        scored.sort(key=lambda x: -x[0])
        return [n for _, n in scored[:limit]]

    def stale(self, today: date | None = None) -> list[Norm]:
        return [n for n in self.norms.values() if n.is_stale(today)]

    # ---- guardrail -------------------------------------------------------
    def audit_text(self, text: str) -> list[dict[str, Any]]:
        """Flag retired parameter names appearing in text.

        This is the gate that stops a 2024 note from pushing CHA2DS2-VASc into a
        2026 answer without a correction.
        """
        out: list[dict[str, Any]] = []
        for name, n in self._by_name.items():
            if re.search(rf"\b{re.escape(name)}\b", text, re.I):
                if n.is_retired_name(name):
                    mig = n.migration_for(name) or {}
                    out.append(
                        {
                            "found": name,
                            "superseded_by": n.current,
                            "norm_id": n.id,
                            "authority": n.authority,
                            "migration": mig.get("migration", ""),
                        }
                    )
        return out
