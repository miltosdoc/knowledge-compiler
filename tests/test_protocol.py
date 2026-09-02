"""Protocol layer: the vault must drive a fresh agent that knows nothing.

The test target is behaviour, not source: call the MCP tool functions
directly (they are plain, mcp-free functions) and assert that their output
carries the operating contract.
"""
from __future__ import annotations

import json

import pytest

from kc2 import mcp_server as ms
from kc2.protocol import _missing_context, protocol_block


# --- the always-grade protocol block --------------------------------------


def test_protocol_block_has_the_operating_contract():
    p = protocol_block()
    assert p["grade"] == "always"
    # identity
    assert "reasoning" in p["nature"] and "norms" in p["nature"]
    # number sanctity
    assert "norm_lookup" in p["number_rule"] and "memory" in p["number_rule"]
    # role split (the läkaren rule, Miltos 2026-09-02)
    for role in ("läkaren", "sjuksköterskan", "personalen"):
        assert role in p["role_split"]
    assert "name" in p["role_split"]  # 'never the physician\'s name'
    # triage vs admin split
    assert "triage" in p["escalation_gear"].lower()


def test_every_tool_result_carries_protocol(tmp_path):
    """A fresh agent with zero memory must inherit the contract from output."""
    (tmp_path / "atomic").mkdir()
    (tmp_path / "atomic" / "Test Note.md").write_text(
        "---\ntitle: Test Note\ntags: [#test]\nContent: A test pattern.\n"
        "Links: []\n",
        encoding="utf-8",
    )
    (tmp_path / "norms").mkdir()
    ms._compiler = ms.Compiler(atomic_dir=tmp_path / "atomic", norms_dir=tmp_path / "norms")
    ms._norms = None

    try:
        for fn, kwargs in [
            (ms.intuition_search, {"query": "test"}),
            (ms.intuition_get_note, {"title": "Test Note"}),
            (ms.intuition_neighbors, {"concept": "test note"}),
            (ms.intuition_graph_stats, {}),
            (ms.norm_lookup, {"norm_id": "nope"}),
            (ms.norm_search, {"query": "blood pressure"}),
        ]:
            out = json.loads(fn(**kwargs))
            assert "protocol" in out, f"{fn.__name__} missing protocol block"
            assert "role_split" in out["protocol"]
    finally:
        ms._compiler = None


def test_intuition_compile_thin_context_fires_ask_next(tmp_path):
    (tmp_path / "atomic").mkdir()
    (tmp_path / "atomic" / "Tired.md").write_text(
        "---\ntitle: Tired\ntags: [#symptom]\nContent: Fatigue warrants a "
        "workup.\nLinks: []\n",
        encoding="utf-8",
    )
    (tmp_path / "norms").mkdir()
    ms._compiler = ms.Compiler(atomic_dir=tmp_path / "atomic", norms_dir=tmp_path / "norms")
    ms._norms = None
    try:
        # no context -> ask_next fires
        out = json.loads(ms.intuition_compile("tired patient", context=None))
        assert "ask_next" in out
        assert set(out["missing_context"]) == {"age", "duration", "meds", "workup", "goal"}
        # full context -> no nag
        out2 = json.loads(
            ms.intuition_compile(
                "tired patient",
                context={
                    "age": "71",
                    "duration": "2 weeks, worsening on stairs",
                    "meds": "metoprolol 50 mg",
                    "workup": "rest ECG done, normal",
                    "goal": "wants an earlier appointment",
                },
            )
        )
        assert "ask_next" not in out2
    finally:
        ms._compiler = None


def test_missing_context_normalisation():
    assert _missing_context(None) == ["age", "duration", "meds", "workup", "goal"]
    assert _missing_context({"AGE": "71", "duration": "2 weeks"}) == [
        "meds",
        "workup",
        "goal",
    ]
    # empty-string values count as unknown
    assert "age" in _missing_context({"age": ""})


def test_session_brief_returns_interview_and_coverage():
    out = json.loads(ms.session_brief())
    assert out["coverage"] == "0/5"
    assert set(out["missing"]) == {"age", "duration", "meds", "workup", "goal"}
    assert "ask (missing)" in out["brief"].lower()
    assert "syncope" in out["brief"]  # cardiac probes present

    out2 = json.loads(
        ms.session_brief({"age": "68", "duration": "3 days, at rest"})
    )
    assert out2["coverage"] == "2/5"
    assert "age" not in out2["missing"]


# --- the MCP server itself: instructions + prompt registration -------------


def test_server_advertises_instructions_and_prompts():
    server = ms.create_server()
    assert "session_brief" in (server.instructions or "")
    # prompts registered (Hermes exposes them via mcp__kc2__get_prompt)
    pm = getattr(server, "_prompt_manager", None)
    prompts = list(getattr(pm, "_prompts", {}) or {})
    assert "start_guidance" in prompts
    assert "role_rules" in prompts
