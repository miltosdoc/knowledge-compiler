"""Typed distillation: transcripts in, schema-valid notes out.

The missing v1->v2 module. Runs as a local batch job over ``{id, transcript,
notes}`` records and writes flat notes to ``vault/atomic/`` plus provenance
JSONL. Nothing here is interactive: no agent loop, no harness in the middle.

PHI rule: a scrubber rewrites the record with the app's own detected PII plus
a conservative regex net *before* any network call. The model only ever sees
anonymised text.

Validity rule: extraction returns constrained JSON; anything failing
``Note.validate()`` (a pattern carrying a threshold, unknown kind, empty
title) is quarantined, never written to the vault.

Resumability: processed record ids are checkpointed; a re-run skips them.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from . import config
from .schema import KINDS, Note, detect_normative_claims

# ---------------------------------------------------------------------------
# Configuration (env-driven like the rest of kc2)
# ---------------------------------------------------------------------------

DISTILL_BASE_URL = os.getenv("KC_DISTILL_BASE_URL", "")
DISTILL_MODEL = os.getenv("KC_DISTILL_MODEL", "")
DISTILL_KEY = os.getenv("KC_DISTILL_KEY", "")
DISTILL_TIMEOUT = int(os.getenv("KC_DISTILL_TIMEOUT", "180"))
DISTILL_RETRIES = int(os.getenv("KC_DISTILL_RETRIES", "3"))
DISTILL_NO_THINKING = os.getenv("KC_DISTILL_NO_THINKING", "1") == "1"
DISTILL_MAX_TOKENS = int(os.getenv("KC_DISTILL_MAX_TOKENS", "1600"))

STATE_DIR = Path(os.getenv("KC_DISTILL_STATE_DIR", str(config.ATOMIC_DIR.parent / ".distill")))
CHECKPOINT = STATE_DIR / "checkpoint.jsonl"
PROVENANCE = STATE_DIR / "provenance.jsonl"
QUARANTINE = STATE_DIR / "quarantine.jsonl"

# ---------------------------------------------------------------------------
# PHI scrubbing
# ---------------------------------------------------------------------------

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("personal-number", re.compile(r"\b(?:19|20)?\d{6}[-+]?\d{4}\b")),
    ("phone", re.compile(r"\b(?:\+46|0)\s?7[02369](?:[\s-]?\d){7}\b")),
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]{2,}\b")),
    ("url", re.compile(r"\bhttps?://\S+")),
    ("birth-date", re.compile(
        r"(\b(?:f[oö]dd|f[oö]dt|born|f[oö].dsel)\b\s*:?\s*)(\d{4,6}[\s-]?\d{2,5}|\d{4}-\d{2}-\d{2})", re.I)),
    ("journal-id", re.compile(
        r"\b(?:journal\s?nr|journalnummer|patient\s?id|person\s?nr|personnummer|ssn|mrn)\s*:?\s*\S*\d\S*", re.I)),
    ("street-address", re.compile(
        r"\b\w+\s?(?:gatan?|vägen|väg|stig|gränd|torget)\s+\d+\b", re.I)),
    ("post-code", re.compile(r"\b\d{3}\s?\d{2}\b")),
]


def scrub_phi(text: str, known: list[tuple[str, str]] | None = None) -> str:
    """Rewrite PII out of ``text``. ``known`` = (pii_type, value) pairs
    pre-detected by the app (its pii_patterns table) — exact replacements
    first, longest value first, then the regex net.
    """
    if known:
        for kind, value in sorted(known, key=lambda kv: -len(kv[1])):
            value = value.strip()
            if len(value) < 3:
                continue
            try:
                text = re.sub(re.escape(value), f"[REDACTED-{kind}]", text, flags=re.I)
            except re.error:
                continue
    for kind, rx in _PATTERNS:
        text = rx.sub(f"[REDACTED-{kind}]", text)
    return text


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["notes"],
    "properties": {
        "notes": {
            "type": "array",
            "minItems": 1,
            "maxItems": 6,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["kind", "title", "content", "tags", "links", "applies_norm"],
                "properties": {
                    "kind": {"type": "string", "enum": sorted(KINDS)},
                    "title": {"type": "string", "minLength": 3},
                    "content": {"type": "string", "minLength": 20},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "links": {"type": "array", "items": {"type": "string"}},
                    "applies_norm": {"type": "array", "items": {"type": "string"}},
                },
            },
        }
    },
}

SYSTEM_PROMPT = """You are extracting durable clinical reasoning from anonymised transcripts of one clinician (cardiology, internal medicine).

RULES:
- Extract 1-4 notes. Each must be ONE of:
  kind=pattern    - HOW the clinician reasons (what he attends to, rules out, weighs). Durable for decades. MUST NOT contain any number, unit, dose, score name or guideline reference. Reference such parameters via applies_norm instead.
  kind=protocol   - a procedural sequence the clinician follows. May reference norms.
  kind=vocabulary - how the patient phrases something -> clinical meaning. Never expires.
  kind=norm       - ONLY if the transcript states a parameter as current guidance; put the value in content.
- Write patterns as: "The clinician <behaviour>". Observation of practice, not advice. Never state that something IS correct.
- NEVER include patient identifiers. The text is already anonymised; keep it that way.
- Use the clinician's own clinical vocabulary. Keep each note 1-4 sentences.
- tags: 2-5 lowercase clinical topics. links: related note titles if obvious. applies_norm: norm ids only if a norm is named."""

_USER_TMPL = """Encounter date: {date}
Transcript (anonymised):
{transcript}

Clinical note produced afterwards (anonymised):
{notes}

Extract the durable reasoning."""


@dataclass
class DistillRecord:
    record_id: str
    status: str                      # ok | quarantined | api_error | skipped
    notes_ok: int = 0
    notes_quarantined: int = 0
    kinds: list[str] = field(default_factory=list)
    norm_candidates: list[str] = field(default_factory=list)
    latency_s: float = 0.0
    error: str | None = None
    model: str | None = None


def _chat(payload: dict) -> tuple[dict, float]:
    req = urllib.request.Request(
        f"{DISTILL_BASE_URL.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {DISTILL_KEY}",
        },
    )
    last: Exception | None = None
    for attempt in range(DISTILL_RETRIES):
        try:
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=DISTILL_TIMEOUT) as r:
                return json.loads(r.read()), time.time() - t0
        except Exception as e:  # noqa: BLE001 - batch wants max resilience
            last = e
            time.sleep(min(2 ** attempt * 2, 30))
    raise RuntimeError(f"extraction API failed after {DISTILL_RETRIES} attempts: {last}")


def extract_notes(record: dict) -> tuple[list[Note], DistillRecord]:
    """One record -> validated notes + metadata. Never raises for content."""
    rec_id = str(record["id"])
    meta = DistillRecord(record_id=rec_id, status="ok", model=DISTILL_MODEL or None)
    known = record.get("_pii_known") or []

    transcript = scrub_phi(record.get("transcript") or "", known)
    notes_text = scrub_phi(record.get("notes") or "", known)
    if len(transcript.strip()) < 40 and len(notes_text.strip()) < 40:
        meta.status = "skipped"
        return [], meta

    user = _USER_TMPL.format(
        date=record.get("date") or "unknown",
        transcript=transcript[:18_000],
        notes=notes_text[:8_000] or "(none)",
    )
    payload: dict = {
        "model": DISTILL_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        "max_tokens": DISTILL_MAX_TOKENS,
        "temperature": 0,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "notes", "strict": True, "schema": SCHEMA},
        },
    }
    if DISTILL_NO_THINKING:
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    try:
        r, latency = _chat(payload)
    except RuntimeError as e:
        meta.status, meta.error = "api_error", str(e)[:300]
        return [], meta
    meta.latency_s = round(latency, 2)

    msg = r["choices"][0]["message"]
    raw = (msg.get("content") or "").strip()
    if not raw:
        meta.status = "quarantined"
        meta.error = "empty content (thinking mode or truncated?)"
        _quarantine(rec_id, None, [meta.error])
        return [], meta
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        meta.status, meta.error = "quarantined", f"unparseable JSON: {e}"
        _quarantine(rec_id, None, [meta.error])
        return [], meta

    out: list[Note] = []
    for n_obj in obj.get("notes", []):
        note = Note(
            title=(n_obj.get("title") or "").strip(),
            kind=n_obj.get("kind", "pattern"),
            content=(n_obj.get("content") or "").strip(),
            tags=[str(t).lstrip("#").lower() for t in (n_obj.get("tags") or [])][:6],
            links=[],
            applies_norm=[str(x) for x in (n_obj.get("applies_norm") or [])][:6],
            source_transcript=rec_id,
            extracted_by=DISTILL_MODEL or None,
            observed_on=record.get("date"),
            source=record.get("source", "distill"),
        )
        errs = note.validate()
        if errs:
            meta.status = "quarantined"
            meta.notes_quarantined += 1
            meta.error = "; ".join(errs)[:200]
            _quarantine(rec_id, note, errs)
            continue
        if note.kind == "pattern":
            # values leaked despite the schema-free prompt contract -> quarantine
            leaked = detect_normative_claims(note.content)
            if leaked:
                meta.status = "quarantined"
                meta.notes_quarantined += 1
                meta.error = f"pattern carries values: {leaked[:3]}"
                _quarantine(rec_id, note, [meta.error])
                continue
        meta.notes_ok += 1
        meta.kinds.append(note.kind)
        meta.norm_candidates = list(
            dict.fromkeys(meta.norm_candidates + detect_normative_claims(note.content))
        )
        out.append(note)

    return out, meta


def _quarantine(rec_id: str, note: Note | None, errs: list[str]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with QUARANTINE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "record": rec_id,
            "errors": errs,
            "note": None if note is None else {
                "title": note.title, "kind": note.kind, "content": note.content[:400],
            },
        }, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Dataset preparation (from raw clinical sources)
# ---------------------------------------------------------------------------

def _parse_date_from_row(cols: dict[str, int], fields: list[str]) -> str | None:
    if "created_at" in cols and fields[cols["created_at"]][:4].isdigit():
        return fields[cols["created_at"]][:10]
    return None


def _scan_pgdump_full(dump_path: Path, table: str) -> tuple[list[str], Iterator[list[str]]]:
    """Column list + row stream for a COPY table, preserving ALL columns."""
    fields: list[str] = []
    text_lines: list[list[str]] = []
    started = False
    for line in dump_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not started:
            m = re.match(rf"COPY\s+(?:[\w.\"]+\.)?\"?{table}\"?\s*\(([^)]*)\)", line)
            if m:
                fields = [c.strip().strip('"') for c in m.group(1).split(",")]
                started = True
            continue
        if line.startswith("\\."):
            break
        parts = line.split("\t")
        if len(parts) == len(fields):
            text_lines.append(parts)
    return fields, iter(text_lines)


_PG_UNESCAPE = {"\\n": "\n", "\\t": "\t", "\\r": "\r", "\\\\": "\\"}


def _unescape(v: str) -> str:
    if v == "\\N":
        return ""
    return re.sub(r"\\[ntr\\]", lambda m: _PG_UNESCAPE[m.group(0)], v)


def _read_pgdump_transcriptions(dump_path: Path, pii_map: dict[str, list[tuple[str, str]]]) -> Iterator[dict]:
    cols, rows = _scan_pgdump_full(dump_path, "transcriptions")
    ci = {c: i for i, c in enumerate(cols)}
    ti, tri, ni, di = ci["id"], ci["transcript"], ci["medical_notes"], ci["created_at"]
    for f in rows:
        yield {
            "id": _unescape(f[ti]),
            "transcript": _unescape(f[tri]),
            "notes": _unescape(f[ni]),
            "date": _unescape(f[di])[:10] or None,
            "source": f"dump.sql:transcriptions#{_unescape(f[ti])}",
            "_pii_known": pii_map.get(_unescape(f[ti]), []),
        }


def _load_known_pii(dump_path: Path) -> dict[str, list[tuple[str, str]]]:
    """transcription_id -> [(pii_type, pattern)] from the app's pii_patterns."""
    txt = dump_path.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"COPY public\.pii_patterns \((.*?)\) FROM stdin;\n(.*?)\n\\\.", txt, re.S)
    out: dict[str, list[tuple[str, str]]] = {}
    if not m:
        return out
    cols = [c.strip() for c in m.group(1).split(",")]
    ti, pi = cols.index("transcription_id"), cols.index("pattern")
    ki = cols.index("pii_type")
    for row in m.group(2).split("\n"):
        f = row.split("\t")
        if len(f) < len(cols):
            continue
        tid, pat, ptype = f[ti], f[pi].strip(), f[ki]
        if tid != "\\N" and len(pat) >= 3:
            out.setdefault(tid, []).append((ptype, pat))
    return out


def prepare_dataset(dump_path: Path, out_path: Path, known_pii: bool = True) -> dict:
    """Stream the pg dump into scrub-ready JSONL, keeping usable encounters.

    Usable = transcript text >= 120 chars. Everything stays on disk; nothing
    is printed. The '' medical_notes empty-marker issue v1 had is impossible
    here because sources._read_pgdump unescapes and validates row width.
    """
    pii_map = _load_known_pii(dump_path) if known_pii else {}
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stats = {"total": 0, "usable": 0, "with_known_pii": 0, "with_notes": 0}
    with out_path.open("w", encoding="utf-8") as w:
        for rec in _read_pgdump_transcriptions(Path(dump_path), pii_map):
            stats["total"] += 1
            if rec["date"] is None:
                rec["date"] = None
            if len((rec["transcript"] or "").strip()) >= 120:
                stats["usable"] += 1
                if rec["_pii_known"]:
                    stats["with_known_pii"] += 1
                if (rec["notes"] or "").strip():
                    stats["with_notes"] += 1
                w.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return stats


def iter_records(dataset: Path) -> Iterator[dict]:
    """Read {id, transcript, notes, date, _pii_known} lines from a JSONL file."""
    with Path(dataset).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def _done_ids() -> set[str]:
    done: set[str] = set()
    if not CHECKPOINT.exists():
        return done
    for line in CHECKPOINT.read_text(encoding="utf-8").splitlines():
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("status") in ("ok", "skipped"):
            done.add(str(obj["record_id"]))
    return done


def _note_to_md(note: Note) -> str:
    """Flat note in the exact format kc2.index.load_notes() already parses:
    frontmatter with title/kind/source, then 'Title:/Kind:/Content:/Links:'."""
    lines = [
        "---",
        f"title: {note.title}",
        f"kind: {note.kind}",
        f"source: {note.source_transcript or 'unknown'}",
    ]
    if note.applies_norm:
        lines.append("applies_norm: [" + ", ".join(note.applies_norm) + "]")
    if note.observed_on:
        lines.append(f"observed_on: {note.observed_on}")
    if note.extracted_by:
        lines.append(f"extracted_by: {note.extracted_by}")
    lines += [
        "tags: [" + ", ".join(f"#{t}" for t in note.tags) + "]",
        "---",
        "",
        f"Title: {note.title}",
        f"Kind: {note.kind}",
        "Content:",
        note.content,
        "Links:",
        *[f"- [[{x}]]" for x in note.links],
        "",
    ]
    return "\n".join(lines)


def run_batch(dataset: Path, limit: int | None = None, quiet: bool = False,
              workers: int = 0) -> dict:
    """Iterate the dataset, extract, write atomic notes. Resumable.

    ``workers`` > 1 enables a thread pool over the API calls (I/O bound trust
    boundary stays the same: each worker sees only its own scrubbed record).
    0 = auto (4). Returns aggregate stats only (no content).
    """
    from concurrent.futures import ThreadPoolExecutor  # noqa: F401

    workers = workers or int(os.getenv("KC_DISTILL_WORKERS", "4"))
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    done = _done_ids()
    stats = {
        "processed": 0, "ok": 0, "quarantined": 0, "skipped": 0, "api_error": 0,
        "notes_ok": 0, "notes_quarantined": 0, "norm_candidates": 0,
        "median_latency_s": 0.0,
    }
    latencies: list[float] = []
    kind_counts: dict[str, int] = {}
    datestamp = datetime.now(timezone.utc).strftime("%Y%m%d")

    def _write_notes(notes_out: list[Note], rid: str) -> None:
        config.ATOMIC_DIR.mkdir(parents=True, exist_ok=True)
        for i, note in enumerate(notes_out):
            slug = re.sub(r"[^a-z0-9]+", "-", note.title.lower()).strip("-")[:70]
            fname = f"{datestamp}-{rid}-{i:02d}-{slug}.md"
            (config.ATOMIC_DIR / fname).write_text(_note_to_md(note), encoding="utf-8")

    def _account(meta: DistillRecord) -> None:
        stats["processed"] += 1
        if meta.status in stats:
            stats[meta.status] += 1
        if meta.status == "ok":
            stats["notes_ok"] += meta.notes_ok
            stats["norm_candidates"] += len(meta.norm_candidates)
            latencies.append(meta.latency_s)
        elif meta.status == "quarantined":
            stats["notes_quarantined"] += meta.notes_quarantined

    pending_records: list[dict] = []
    records: list[dict] = []
    for n_seen, rec in enumerate(iter_records(dataset), start=1):
        if limit and n_seen > limit:
            break
        if str(rec["id"]) in done:
            stats["skipped"] += 1
            continue
        records.append(rec)

    if workers <= 1 or len(records) <= 1:
        for rec in records:
            notes_out, meta = extract_notes(rec)
            if notes_out:
                _write_notes(notes_out, str(rec["id"]))
            with CHECKPOINT.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(asdict(meta), ensure_ascii=False) + "\n")
            _account(meta)
            if not quiet and stats["processed"] % 25 == 0:
                print(_progress(stats), flush=True)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(extract_notes, rec): rec for rec in records}
            for fut in _as_completed_in_order(futures):
                rec = futures[fut]
                notes_out, meta = fut.result()
                if notes_out:
                    _write_notes(notes_out, str(rec["id"]))
                with CHECKPOINT.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(asdict(meta), ensure_ascii=False) + "\n")
                _account(meta)
                if not quiet and stats["processed"] % 25 == 0:
                    print(_progress(stats), flush=True)

    if latencies:
        latencies.sort()
        stats["median_latency_s"] = round(latencies[len(latencies) // 2], 2)
    stats["kinds"] = kind_counts
    return stats


def _progress(stats: dict) -> str:
    return (f"  ...{stats['processed']} processed "
            f"({stats['ok']} ok / {stats['quarantined']} quarantined / {stats['api_error']} err)")


def _as_completed_in_order(futures: dict):
    """Yield futures in submission order (reads record order stable)."""
    from concurrent.futures import FIRST_COMPLETED, wait
    pending = set(futures)
    while pending:
        done_set, pending = wait(pending, return_when=FIRST_COMPLETED)
        for f in futures:
            if f in done_set:
                yield f


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(prog="kc2-distill")
    p.add_argument("dataset", help="prepared JSONL dataset")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()
    stats = run_batch(Path(args.dataset), limit=args.limit, quiet=args.quiet)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
