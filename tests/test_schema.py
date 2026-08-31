"""A durable note may never freeze a parameter into itself."""
from kc2.schema import Note, detect_normative_claims


def test_pattern_may_not_name_a_scoring_system():
    n = Note(title="AFib workup", content="a CHA2DS2-VASc score mandates anticoagulation")
    assert not n.is_valid
    assert any("scoring system" in e for e in n.validate())


def test_pattern_may_not_inline_a_threshold():
    n = Note(title="Lipids", content="target LDL below 1.4 mmol/L")
    assert not n.is_valid


def test_pattern_referencing_a_norm_is_valid():
    n = Note(
        title="Stroke Risk as the Primary Driver in AFib Workup",
        content="The palpitations are a nuisance; the stroke is the threat.",
        applies_norm=["norm:af-stroke-risk-score"],
    )
    assert n.is_valid, n.validate()


def test_vocabulary_notes_are_held_to_the_same_rule():
    assert not Note(title="v", kind="vocabulary", content="over 190 mmHg").is_valid


def test_detect_finds_both_names_and_values():
    claims = detect_normative_claims("CHA2DS2-VA and LDL 1.4 mmol/L and 5 mg")
    assert "CHA2DS2-VA" in claims
    assert any("1.4" in c for c in claims)


def test_bare_analyte_value_without_units_is_caught():
    """The hard case: 'aggressive lipid management (LDL 4.5 to 2.6)' has no unit,
    so a unit-based scan misses it. This is the miss that motivated v2."""
    claims = detect_normative_claims("aggressive lipid management (LDL from 4.5 to 2.6)")
    assert claims, "an analyte followed by a bare number must be detected"
    assert any("LDL" in c for c in claims)


def test_note_with_bare_analyte_value_fails_validation():
    n = Note(title="Lipids", content="LDL improved from 4.5 to 2.6, aggressive management")
    assert not n.is_valid


def test_prose_without_values_is_still_valid():
    n = Note(
        title="Stroke Risk as the Primary Driver in AFib Workup",
        content="The palpitations are a nuisance; the stroke is the threat.",
    )
    assert n.is_valid, n.validate()
