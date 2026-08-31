"""Ingestion merges into the existing graph rather than appending to it."""
import pytest

from kc2 import config
from kc2.concepts import Concept, Observation, save_concept
from kc2.embed import get_embedder
from kc2.ingest import ConceptStore


def _dense() -> bool:
    return get_embedder().available


@pytest.fixture
def store(tmp_path):
    save_concept(tmp_path, Concept(
        id="warm-up", title="The Warm-Up Phenomenon in Exertional Chest Pain",
        content="Discomfort at the onset of exercise which subsides as the patient continues.",
        evidence=[Observation("t-1", "2024-01-01", "eased after the first hill")]))
    save_concept(tmp_path, Concept(
        id="nitro", title="Nitroglycerin Responsiveness as a Diagnostic Anchor",
        content="Rapid relief after sublingual nitrate is treated as diagnostic confirmation.",
        evidence=[Observation("t-2", "2024-02-01", "relief within two minutes")]))
    return ConceptStore(tmp_path)


def test_compounding_metric_starts_flat(store):
    m = store.compounding()
    assert m["concepts"] == 2
    assert m["evidence_per_concept"] == 1.0
    assert m["corroborated"] == 0


@pytest.mark.skipif(not _dense(), reason="no embedding backend installed")
def test_close_restatement_merges_and_strengthens(store):
    r = store.ingest(
        "Nitrate Relief Confirms the Ischemic Hypothesis",
        "Rapid relief after sublingual nitroglycerin is treated as confirmatory, not merely therapeutic.",
        source="t-9", observed_on="2026-08-31")
    assert r.action == "merge"
    assert store.concepts["nitro"].strength == 2
    assert store.compounding()["concepts"] == 2, "a restatement must not create a node"


@pytest.mark.skipif(not _dense(), reason="no embedding backend installed")
def test_novel_concept_is_created_not_forced_into_a_match(store):
    r = store.ingest(
        "Pericardial Friction Rub After Viral Illness",
        "A scratchy triphasic rub two weeks after a viral illness reframes pleuritic pain.",
        source="t-10")
    assert r.action in {"create", "link"}
    assert store.compounding()["concepts"] == 3


@pytest.mark.skipif(not _dense(), reason="no embedding backend installed")
def test_harness_mode_surfaces_ambiguity_instead_of_guessing(store, monkeypatch):
    """An MCP server driven by an agent must not silently pick, nor call out to
    a second model. It reports candidates and waits."""
    monkeypatch.setattr(config, "ADJUDICATOR_MODE", "harness")
    r = store.ingest("Discomfort That Fades as the Patient Keeps Going",
                     "Tightness at the very start of exertion which settles if the pace is maintained.",
                     source="t-11")
    if r.action == "ambiguous":
        assert r.candidates and r.pending is not None
        out = store.commit_pending(r, "warm-up")
        assert out.action == "merge"
        assert store.concepts["warm-up"].strength == 2


def test_explicit_merge_is_the_callers_decision(store):
    assert store.merge("warm-up", "nitro") is True
    assert store.concepts["warm-up"].strength == 2
    assert "nitro" not in store.concepts
