"""Read raw clinical sources - SQLite databases and pg_dump files.

Deliberately introspective rather than configured. Point it at a directory and
it reports what it found and which columns it believes hold the transcript and
the clinical note, so the mapping can be confirmed before anything is ingested.
v1's sieve.py hardcoded column offsets against one dump and split on tabs
without unescaping, so any transcript containing a tab silently corrupted the
row after it.

    from kc2.sources import discover, extract
    for src in discover("data/"):
        print(src.describe())
    records = extract(src, limit=5)      # inspect before committing
"""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

# Column names that plausibly hold each half of the pair, best guess first.
TRANSCRIPT_HINTS = ["transcript", "transcription", "dialogue", "conversation",
                    "text", "content", "body", "raw"]
NOTE_HINTS = ["note", "notes", "summary", "clinical_note", "report", "assessment",
              "conclusion", "journal"]
ID_HINTS = ["id", "uuid", "pk", "encounter_id", "visit_id"]


@dataclass
class Source:
    path: Path
    kind: str                      # "sqlite" | "pgdump"
    table: str = ""
    columns: list[str] = field(default_factory=list)
    id_col: str | None = None
    transcript_col: str | None = None
    note_col: str | None = None
    rows: int = 0

    def describe(self) -> str:
        return (
            f"{self.path.name} [{self.kind}] table={self.table!r} rows={self.rows}\n"
            f"    id         -> {self.id_col}\n"
            f"    transcript -> {self.transcript_col}\n"
            f"    note       -> {self.note_col}\n"
            f"    columns    : {', '.join(self.columns[:14])}"
            + (" ..." if len(self.columns) > 14 else "")
        )

    @property
    def usable(self) -> bool:
        return bool(self.transcript_col)


def _pick(columns: list[str], hints: list[str], exclude: set[str] = frozenset()) -> str | None:
    low = {c.lower(): c for c in columns if c not in exclude}
    for hint in hints:                       # exact match wins
        if hint in low:
            return low[hint]
    for hint in hints:                       # then substring
        for lc, orig in low.items():
            if hint in lc:
                return orig
    return None


# --- SQLite ----------------------------------------------------------------

def _scan_sqlite(path: Path) -> list[Source]:
    out: list[Source] = []
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error:
        return out
    try:
        tables = [
            r[0] for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table','view')"
            )
        ]
        for t in tables:
            if t.startswith("sqlite_"):
                continue
            try:
                cols = [r[1] for r in con.execute(f'PRAGMA table_info("{t}")')]
                rows = con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
            except sqlite3.Error:
                continue
            if not cols:
                continue
            tcol = _pick(cols, TRANSCRIPT_HINTS)
            ncol = _pick(cols, NOTE_HINTS, exclude={tcol} if tcol else set())
            if not tcol:
                continue
            out.append(Source(path, "sqlite", t, cols,
                              _pick(cols, ID_HINTS), tcol, ncol, rows))
    finally:
        con.close()
    # richest table first
    out.sort(key=lambda s: (s.note_col is not None, s.rows), reverse=True)
    return out


def _read_sqlite(src: Source, limit: int | None) -> Iterator[dict]:
    con = sqlite3.connect(f"file:{src.path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        cols = [c for c in (src.id_col, src.transcript_col, src.note_col) if c]
        sql = f'SELECT {", ".join(chr(34)+c+chr(34) for c in cols)} FROM "{src.table}"'
        if limit:
            sql += f" LIMIT {int(limit)}"
        for i, row in enumerate(con.execute(sql)):
            yield {
                "id": str(row[src.id_col]) if src.id_col else str(i),
                "transcript": row[src.transcript_col] or "",
                "notes": (row[src.note_col] or "") if src.note_col else "",
            }
    finally:
        con.close()


# --- pg_dump ---------------------------------------------------------------

_PG_ESCAPES = {"\\n": "\n", "\\t": "\t", "\\r": "\r", "\\\\": "\\"}


def _unescape_pg(value: str) -> str:
    if value == "\\N":
        return ""
    out = re.sub(r"\\[ntr\\]", lambda m: _PG_ESCAPES[m.group(0)], value)
    return out


def _scan_pgdump(path: Path) -> list[Source]:
    out: list[Source] = []
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            m = re.match(r"COPY\s+(?:[\w.\"]+\.)?\"?(\w+)\"?\s*\(([^)]*)\)", line)
            if not m:
                continue
            table = m.group(1)
            cols = [c.strip().strip('"') for c in m.group(2).split(",")]
            tcol = _pick(cols, TRANSCRIPT_HINTS)
            if not tcol:
                continue
            ncol = _pick(cols, NOTE_HINTS, exclude={tcol})
            out.append(Source(path, "pgdump", table, cols,
                              _pick(cols, ID_HINTS), tcol, ncol, 0))
    out.sort(key=lambda s: s.note_col is not None, reverse=True)
    return out


def _read_pgdump(src: Source, limit: int | None) -> Iterator[dict]:
    want = {c: src.columns.index(c) for c in
            (src.id_col, src.transcript_col, src.note_col) if c}
    started = False
    n = 0
    with src.path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if not started:
                if re.match(rf"COPY\s+(?:[\w.\"]+\.)?\"?{src.table}\"?\s*\(", line):
                    started = True
                continue
            if line.startswith("\\.") or not line.strip():
                break
            parts = line.rstrip("\n").split("\t")
            if len(parts) < len(src.columns):
                continue                      # malformed row, skip rather than shift
            rec = {
                "id": _unescape_pg(parts[want[src.id_col]]) if src.id_col else str(n),
                "transcript": _unescape_pg(parts[want[src.transcript_col]]),
                "notes": _unescape_pg(parts[want[src.note_col]]) if src.note_col else "",
            }
            yield rec
            n += 1
            if limit and n >= limit:
                break


# --- public API ------------------------------------------------------------

def discover(directory: str | Path) -> list[Source]:
    """Find every readable clinical source under a directory."""
    d = Path(directory)
    found: list[Source] = []
    if not d.exists():
        return found
    for f in sorted(d.rglob("*")):
        if not f.is_file():
            continue
        suffix = f.suffix.lower()
        if suffix in {".db", ".sqlite", ".sqlite3"}:
            found += _scan_sqlite(f)
        elif suffix in {".dump", ".sql", ".dmp"}:
            found += _scan_pgdump(f)
    return found


def extract(src: Source, limit: int | None = None, min_chars: int = 40) -> list[dict]:
    """Pull ``{id, transcript, notes}`` records, skipping empties."""
    reader = _read_sqlite if src.kind == "sqlite" else _read_pgdump
    out = []
    for rec in reader(src, None if limit is None else limit * 4):
        if len(rec["transcript"].strip()) < min_chars:
            continue
        out.append(rec)
        if limit and len(out) >= limit:
            break
    return out
