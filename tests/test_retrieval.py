"""Retrieval regression: the v1 substring failure must not come back.

Uses synthetic notes, never the real vault - clinical content stays off this
repo entirely (see .gitignore).
"""
from __future__ import annotations

import pytest

from kc2.compile import Compiler
from kc2.retrieve import Retriever

NOTES = {
    "Crescendo Angina and Exertional Threshold": (
        "tags: [#intuition]\n"
        "Content: A falling exertional threshold marks crescendo symptoms and "
        "demands urgent reassessment rather than routine follow-up.\n"
        "Links: [[Exertional Symptom Trigger]]\n"
    ),
    "Exertional Symptom Trigger": (
        "tags: [#protocol]\n"
        "Content: Symptoms reproduced by exertion and relieved by rest anchor an "
        "ischaemic mechanism.\n"
        "Links: [[Crescendo Angina and Exertional Threshold]]\n"
    ),
    "Angina Equivalents in Women": (
        "tags: [#vocabulary]\n"
        "Content: Fatigue and dyspnoea may stand in for chest pain; absence of "
        "classic pain does not lower the pre-test probability.\n"
        "Links: [[Exertional Symptom Trigger]]\n"
    ),
    "Vagal Atrial Fibrillation at Night": (
        "tags: [#arrhythmia]\n"
        "Content: Nocturnal onset after a heavy meal suggests a vagally mediated "
        "trigger in a structurally normal heart.\n"
        "Links: [[Stroke Risk in AFib]]\n"
    ),
    "Stroke Risk in AFib": (
        "tags: [#risk-stratification]\n"
        "Content: The palpitations are a nuisance; the stroke is the threat. This "
        "prioritisation drives the whole plan. A CHA2DS2-VASc score was applied.\n"
        "Links: [[Vagal Atrial Fibrillation at Night]]\n"
    ),
}


@pytest.fixture(scope="module")
def vault(tmp_path_factory):
    d = tmp_path_factory.mktemp("atomic")
    for title, body in NOTES.items():
        slug = title.replace(" ", "-")
        (d / f"{slug}.md").write_text(f"---\ntitle: {title}\n{body}", encoding="utf-8")
    return d


def test_multiword_query_no_longer_returns_nothing(vault):
    """v1: intuition_compile('refractory angina') -> 0 of 717. Never again."""
    hits = Retriever(vault).retrieve("refractory angina", k=5)
    assert hits, "a multi-word query must not return an empty set"
    assert any("Angina" in t for t, _ in hits)


def test_case_narrative_retrieves_relevant_notes(vault):
    """A pasted case, not a keyword - the real usage pattern."""
    hits = Retriever(vault).retrieve(
        "62 year old woman, palpitations at night after dinner, structurally normal heart",
        k=5,
    )
    assert any("Vagal" in t or "AFib" in t for t, _ in hits)


def test_graph_expansion_pulls_in_linked_concepts(vault):
    r = Retriever(vault)
    assert "Exertional Symptom Trigger" in r.neighbors(
        "Crescendo Angina and Exertional Threshold"
    )


def test_canonicalisation_resolves_links(vault):
    r = Retriever(vault)
    assert r.stats()["resolution_rate"] > 0.5


def test_compiled_prompt_forbids_answering_from_memory(vault):
    out = Compiler(vault).compile("atrial fibrillation stroke risk", k=5)
    assert "DO NOT state any numeric threshold" in out.prompt
    assert "norm_lookup" in out.prompt


def test_compiled_prompt_corrects_a_retired_parameter(vault):
    """The whole point: a 2024 note must not push CHA2DS2-VASc into a 2026 answer."""
    out = Compiler(vault).compile("atrial fibrillation stroke risk score", k=5)
    assert out.corrections, "retired parameter in the notes must raise a correction"
    assert "CHA2DS2-VA" in out.prompt
    assert out.prompt.index("SUPERSEDED") < out.prompt.index("###"), (
        "corrections must appear above the notes, where they cannot be missed"
    )
