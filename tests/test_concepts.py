"""Concept nodes must compound: evidence accretes onto durable concepts."""
from kc2.concepts import Concept, Observation, load_concepts, save_concept


def _c(cid, title, **kw):
    return Concept(id=cid, title=title, content=kw.pop("content", "body text here"), **kw)


def test_strength_counts_distinct_sources():
    c = _c("x", "X", evidence=[Observation("t1"), Observation("t1"), Observation("t2")])
    assert c.strength == 2, "the same encounter twice is one piece of support"


def test_absorb_keeps_both_names_findable():
    a = _c("a", "Warm-Up Phenomenon", evidence=[Observation("t1")])
    b = _c("b", "Second-Wind Angina", evidence=[Observation("t2")])
    a.absorb(b)
    assert a.strength == 2
    assert "Second-Wind Angina" in a.aliases


def test_absorb_is_idempotent_on_repeated_evidence():
    a = _c("a", "A", evidence=[Observation("t1", detail="d")])
    b = _c("b", "B", evidence=[Observation("t1", detail="d")])
    a.absorb(b)
    assert a.strength == 1


def test_contradiction_is_recorded_not_overwritten():
    c = _c("x", "X", evidence=[Observation("t1"), Observation("t2", contradicts=True)])
    assert c.strength == 1
    assert c.contested == 1


def test_markdown_roundtrip_is_obsidian_shaped(tmp_path):
    c = _c("warm-up", "Warm-Up Phenomenon", tags=["intuition"],
           links=["Exertional Symptom Trigger"],
           evidence=[Observation("t-2430", "2024-03-12", "eased after the first hill")])
    md = c.to_markdown()
    assert md.startswith("---")          # YAML frontmatter
    assert "[[Exertional Symptom Trigger]]" in md
    assert "## Evidence" in md

    save_concept(tmp_path, c)
    back = load_concepts(tmp_path)["warm-up"]
    assert back.title == c.title
    assert back.strength == c.strength
    assert back.links == c.links
