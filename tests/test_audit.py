"""The vault auditor is the operator's view of decay."""
from kc2.audit import audit_vault, format_report


def _write(d, title, content):
    (d / f"{title.replace(' ', '-')}.md").write_text(
        f"---\ntitle: {title}\ntags: [#intuition]\n---\nContent: {content}\nLinks: \n",
        encoding="utf-8",
    )


def test_audit_separates_retired_from_unmapped_and_clean(tmp_path):
    _write(tmp_path, "Retired", "a CHA2DS2-VASc score mandates anticoagulation")
    _write(tmp_path, "Unmapped", "systolic response above 190 mmHg on exercise")
    _write(tmp_path, "Clean", "the palpitations are a nuisance; the stroke is the threat")

    r = audit_vault(tmp_path)
    assert r["notes"] == 3
    assert r["clean"] == 1
    assert len(r["with_retired_parameters"]) == 1
    assert r["with_retired_parameters"][0]["title"] == "Retired"
    assert any(x["title"] == "Unmapped" for x in r["with_unmapped_values"])


def test_report_renders_without_error(tmp_path):
    _write(tmp_path, "Retired", "CHA2DS2-VASc was applied")
    out = format_report(audit_vault(tmp_path))
    assert "CHA2DS2-VA" in out
