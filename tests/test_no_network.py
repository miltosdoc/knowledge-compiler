"""The default path must make no outbound network call.

An MCP server is driven by a model already. In the default harness mode kc2
distils nothing itself and adjudicates nothing itself, so it has no reason to
contact an inference endpoint - and clinical text must not leave the machine.

An earlier version violated this quietly: get_embedder() fell through to an
OpenAI-compatible backend that probed localhost:8000 whenever the local encoder
was missing. Nothing in the configuration asked for that. These tests make the
guarantee enforceable instead of aspirational.
"""
from __future__ import annotations

import socket

import pytest

from kc2 import config
from kc2.concepts import Concept, Observation, save_concept


@pytest.fixture
def no_network(monkeypatch):
    """Any attempt to open a socket fails loudly."""
    def _forbidden(*a, **k):
        raise AssertionError("the default path attempted a network connection")

    monkeypatch.setattr(socket, "socket", _forbidden)
    monkeypatch.setattr(socket, "create_connection", _forbidden)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")


def test_defaults_are_harness_mode_with_nothing_configured():
    assert config.ADJUDICATOR_MODE == "harness"
    assert config.ADJUDICATOR_MODEL == "", "no inference model may be configured by default"
    assert config.EMBED_API_MODEL == "", "no remote embedding model may be configured by default"


def test_remote_embedder_is_never_probed_without_explicit_opt_in(no_network):
    """Constructing it must not touch the network; it simply reports unavailable."""
    from kc2.embed import OpenAICompatEmbedder

    assert OpenAICompatEmbedder().available is False


def test_adjudicator_returns_none_when_no_model_is_configured(no_network):
    from kc2.ingest import adjudicate

    assert adjudicate("A", "body a", "B", "body b") is None


def test_retrieval_and_compilation_make_no_network_call(no_network, tmp_path):
    from kc2.compile import Compiler

    (tmp_path / "n.md").write_text(
        "---\ntitle: Stroke Risk in AFib\ntags: [#intuition]\n---\n"
        "Content: The palpitations are a nuisance; the stroke is the threat.\nLinks: \n",
        encoding="utf-8")
    out = Compiler(tmp_path).compile("atrial fibrillation stroke", k=3)
    assert "norm_lookup" in out.prompt


def test_ingest_makes_no_network_call(no_network, tmp_path):
    from kc2.ingest import ConceptStore

    save_concept(tmp_path, Concept(
        id="warm-up", title="Warm-Up Phenomenon",
        content="Discomfort at the onset of exercise which subsides as the patient continues.",
        evidence=[Observation("t-1")]))
    store = ConceptStore(tmp_path)
    result = store.ingest("Second-Wind Angina",
                          "Tightness on setting off that eases if he keeps walking.",
                          source="t-2")
    assert result.action in {"merge", "link", "create", "ambiguous"}


def test_norm_lookup_makes_no_network_call(no_network):
    from kc2.norms import NormStore

    norm, retired = NormStore().resolve_name("CHA2DS2-VASc")
    assert retired is True and norm.current == "CHA2DS2-VA"
