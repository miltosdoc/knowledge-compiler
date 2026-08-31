"""Temporal regression suite.

Cases whose CORRECT ANSWER CHANGED OVER TIME. v1 had no version of this, which
is exactly why a retired scoring system sat in the vault unnoticed for months.
A superseded parameter reaching an answer is a build failure here, not a
discovery made later by chance.
"""
from __future__ import annotations

from datetime import date

import pytest

from kc2.norms import NormStore


@pytest.fixture(scope="module")
def norms() -> NormStore:
    return NormStore()


# --- Case 1: CHA2DS2-VASc -> CHA2DS2-VA (ESC 2024) -------------------------

def test_vasc_is_recognised_as_retired(norms: NormStore):
    n, retired = norms.resolve_name("CHA2DS2-VASc")
    assert n is not None, "the AF stroke-risk norm must exist"
    assert retired is True, "CHA2DS2-VASc must be reported as superseded"
    assert n.current == "CHA2DS2-VA"


def test_va_is_current_not_retired(norms: NormStore):
    n, retired = norms.resolve_name("CHA2DS2-VA")
    assert retired is False
    assert n.authority.startswith("ESC 2024")


def test_migration_states_the_female_sex_rule(norms: NormStore):
    """The 62F case: a woman scoring 1 on sex alone maps to 0, so no OAC."""
    n, _ = norms.resolve_name("CHA2DS2-VASc")
    migration = n.migration_for("CHA2DS2-VASc")["migration"].lower()
    assert "sex" in migration
    assert "modifier" in migration or "maps to 0" in migration


def test_audit_catches_vasc_in_note_prose(norms: NormStore):
    """The literal sentence from the recovered vault must be flagged."""
    vault_text = (
        "A history of thrombocytopenia complicates the future necessity of "
        "anticoagulation (should AFib be confirmed and a CHA2DS2-VASc score mandate it)."
    )
    found = norms.audit_text(vault_text)
    assert any(f["superseded_by"] == "CHA2DS2-VA" for f in found)


def test_aliases_are_not_flagged_as_retired(norms: NormStore):
    """An alternate spelling of a CURRENT norm must never raise a correction."""
    _, retired = norms.resolve_name("HAS-BLED")
    assert retired is False


# --- Case 2: LDL targets ---------------------------------------------------

def test_very_high_risk_ldl_target_is_1_4(norms: NormStore):
    n = norms.get("norm:ldl-target")
    assert "1.4" in n.thresholds["very_high_risk"]


def test_2_6_is_the_moderate_target_not_aggressive(norms: NormStore):
    """The vault calls 'LDL 4.5 to 2.6' aggressive. 2.6 is the MODERATE target."""
    n = norms.get("norm:ldl-target")
    assert "2.6" in n.thresholds["moderate_risk"]
    assert "2.6" not in n.thresholds["very_high_risk"]
    mig = " ".join(s.get("migration", "") for s in n.supersedes).lower()
    assert "moderate" in mig


# --- Case 3: bleeding score must not gate anticoagulation ------------------

def test_bleeding_score_cannot_withhold_oac(norms: NormStore):
    n = norms.get("norm:af-bleeding-risk")
    assert "not" in n.thresholds["do_not"].lower()


# --- Freshness -------------------------------------------------------------

def test_every_norm_declares_verification_and_authority(norms: NormStore):
    for n in norms.norms.values():
        assert n.last_verified, f"{n.id} has no last_verified date"
        assert n.authority, f"{n.id} has no authority"


def test_freshness_gate_trips_when_a_norm_ages_out(norms: NormStore):
    """A norm past last_verified + volatility must report stale."""
    n = norms.get("norm:af-stroke-risk-score")
    assert n.is_stale(date(2099, 1, 1)) is True
    assert n.is_stale(n.last_verified) is False
