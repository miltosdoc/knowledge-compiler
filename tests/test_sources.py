"""Raw source import: SQLite and pg_dump, including the escaping v1 got wrong."""
import sqlite3

import pytest

from kc2.sources import discover, extract


@pytest.fixture
def sources_dir(tmp_path):
    con = sqlite3.connect(tmp_path / "patients.db")
    con.execute("CREATE TABLE users (id INTEGER, email TEXT)")
    con.execute(
        "CREATE TABLE transcriptions (id INTEGER PRIMARY KEY, created_at TEXT, "
        "transcription TEXT, clinical_note TEXT)"
    )
    con.executemany("INSERT INTO transcriptions VALUES (?,?,?,?)", [
        (2430, "2024-03-12", "Patient: my heart is jumping out of my chest, doctor.", "SVE."),
        (2455, "2024-05-19", "short", "skipped"),
    ])
    con.commit()
    con.close()

    (tmp_path / "dump.dump").write_text(
        "SET statement_timeout = 0;\n"
        "COPY public.transcriptions (id, created_at, transcript, notes) FROM stdin;\n"
        "3001\t2024-06-01\tPain radiating to BOTH arms.\\nDoctor: bilateral?\tHigh risk.\n"
        "3002\t2024-06-11\tHe said\\t(tab inside)\\tit wakes him at night.\tNocturnal.\n"
        "3003\t2024-07-02\t\\N\tno transcript\n"
        "\\.\n", encoding="utf-8")
    return tmp_path


def test_discovers_both_formats_and_skips_irrelevant_tables(sources_dir):
    found = discover(sources_dir)
    kinds = {s.kind for s in found}
    assert kinds == {"sqlite", "pgdump"}
    assert all(s.table == "transcriptions" for s in found), "users table must not match"


def test_infers_column_mapping_from_varied_names(sources_dir):
    sqlite_src = next(s for s in discover(sources_dir) if s.kind == "sqlite")
    assert sqlite_src.transcript_col == "transcription"
    assert sqlite_src.note_col == "clinical_note"
    assert sqlite_src.id_col == "id"


def test_pgdump_unescapes_newlines(sources_dir):
    src = next(s for s in discover(sources_dir) if s.kind == "pgdump")
    rec = next(r for r in extract(src) if r["id"] == "3001")
    assert "\n" in rec["transcript"], "\\n must become a real newline"


def test_embedded_tab_does_not_shift_columns(sources_dir):
    """v1 split on tabs with no unescaping, so one tab inside a transcript
    corrupted every later column on that row."""
    src = next(s for s in discover(sources_dir) if s.kind == "pgdump")
    rec = next(r for r in extract(src) if r["id"] == "3002")
    assert "\t" in rec["transcript"]
    assert rec["notes"] == "Nocturnal.", "the note column must not be shifted"


def test_null_and_short_transcripts_are_skipped(sources_dir):
    for src in discover(sources_dir):
        for rec in extract(src):
            assert len(rec["transcript"].strip()) >= 40
    pg = next(s for s in discover(sources_dir) if s.kind == "pgdump")
    assert "3003" not in {r["id"] for r in extract(pg)}


def test_limit_is_respected(sources_dir):
    pg = next(s for s in discover(sources_dir) if s.kind == "pgdump")
    assert len(extract(pg, limit=1)) == 1


def test_missing_directory_is_not_an_error(tmp_path):
    assert discover(tmp_path / "nope") == []
