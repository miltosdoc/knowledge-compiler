"""The vault acting as a participant, not a lookup service.

Protocol layer for the MCP server. Every tool result carries a small
``protocol`` object so ANY agent driving this server — fresh session, no
memory, no skill — inherits the operating contract from the tool output
itself. Three grades:

- ``always``   : identity, role separation, number-sanctity, escalation gear.
                 Cheap (~120 tokens), appears on every tool response.
- ``contextual``: fired when the caller is proceeding on thin evidence —
                 no age, no duration/timeline, no meds, no plan. The
                 "self-aware" behaviour: the server notices what the agent
                 has NOT established and says so, with the exact questions
                 to put to the patient/owner, before numbers get compiled.
- ``checkpoint``: the start-of-case briefing, converted to a tool. Called
                 once per new case, optionally with what's already known;
                 returns interview questions and a coverage score, NOT a
                 verdict (that's intuition_compile's job, after).

Language rule is Miltos-facing policy (2026-09-02): patient-facing Swedish
names the ROLE — läkaren / sjuksköterskan / personalen — never the
physician's name. Tools speak English; the rule travels in the protocol
block so any agent picks it up without reading this file.
"""
from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Grade 1: carried on every response.
# ---------------------------------------------------------------------------

IDENTITY_BLOCK = {
    "server": "kc2 knowledge-compiler",
    "nature": (
        "not a medical database: a distilled graph of one cardiologist's "
        "reasoning patterns (5,600+ concepts) plus a norms layer of "
        "authoritative parameter values"
    ),
    "number_rule": (
        "numbers in notes are HISTORICAL; never state a threshold, dose, "
        "target or score definition from memory — resolve via norm_lookup / "
        "norm_search, else say the value is missing"
    ),
    "role_split": (
        "key facts/decisions -> 'läkaren'; nurse tasks -> 'sjuksköterskan'; "
        "clinic facilities -> 'personalen'. Patient-facing messages never "
        "carry the physician's personal or brand name. Signature line, "
        "legal/insurance/prescription documents excepted."
    ),
    "escalation_gear": (
        "patient calling about symptoms is triage — call intuition_compile "
        "on the case, then answer with red-flag questions. Admin (booking, "
        "results follow-up, complaints) — no compile; process question."
    ),
}

_CONTEXTUAL_TRIGGERS = ("age", "duration", "meds", "workup", "goal")

_CONTEXTUAL_MSG = (
    "This case is being answered on thin context. Before relying on this "
    "output, put the missing pieces to the patient/owner — you are THEIR "
    "questions, not verdicts: "
    "age; how long and how it evolved; current meds and recent changes; "
    "what has already been tried or ruled out (workup/tests); what the "
    "caller actually wants. If the conversation closes before you can ask, "
    "state explicitly which of these are unknown."
)

_CHECKPOINT_HINT = (
    "first contact with a new case? call session_brief first — it returns "
    "the interview list and coverage gaps"
)


def _missing_context(context: dict[str, Any] | None) -> list[str]:
    if not context or not isinstance(context, dict):
        return list(_CONTEXTUAL_TRIGGERS)
    norm = {k.strip().lower(): v for k, v in context.items() if v not in (None, "", [], {})}
    return [k for k in _CONTEXTUAL_TRIGGERS if k not in norm]


def protocol_block(
    *,
    contextual: bool = False,
    missing: list[str] | None = None,
    checkpoint: bool = False,
) -> dict[str, Any]:
    p: dict[str, Any] = {
        "grade": "contextual" if contextual else "always",
        **IDENTITY_BLOCK,
    }
    if contextual:
        p["missing_context"] = missing or []
        p["ask_next"] = _CONTEXTUAL_MSG
    if checkpoint:
        p["hint"] = _CHECKPOINT_HINT
    return p


# ---------------------------------------------------------------------------
# Grade 3: session_brief — the interview, as a tool.
# ---------------------------------------------------------------------------

_BRIEF_RULES = """\
- Compile patterns only AFTER these are asked, not before.
- Ask when momentum exists (person is present or mid-thread), and at any
  pivot where the next action would change without the answer.
- Never block urgent red flags (syncope, chest pain at rest, dyspnoea) on
  questionnaires — triage first.
- Patient-facing text in Swedish; keep 'läkaren / sjuksköterskan /
  personalen' — never a physician's name."""

_BASE_QUESTIONS = [
    "age",
    "duration_trend",
    "meds_changes",
    "prior_workup",
    "goal_now",
]


def session_brief(context: dict[str, Any] | None = None) -> str:
    """Interview scaffold. `context` = key/value of what's already known."""
    missing = _missing_context(context)
    have = [k for k in _CONTEXTUAL_TRIGGERS if k not in missing]

    core = {
        "age": "Age (adult years matter for risk stratification).",
        "duration": "How long, and the trend? days vs months? stable, worsening, episodic?",
        "meds": "Current meds and any recent change (new drug, dose change, missed doses).",
        "workup": "What has already been tried, tested or ruled out (ECG, labs, imaging, prior contact)?",
        "goal": "What does the caller want right now — reassurance, earlier slot, results, a plan?",
    }

    lines = [
        "# Session brief — before any intuition_compile",
        "",
        "Known already:",
        *([f"- {k}" for k in have] if have else ["- (nothing yet)"]),
        "",
        "Ask (missing):",
        *(f"- {i+1}. {k} — {core[k]}" for i, k in enumerate(missing)),
        "",
        "Cardiac-specific probes (only when symptoms suggest):",
        "- Exertional vs rest? Relieved by anything?",
        "- Syncope, presyncope, palpitations — any?",
        "- Orthopnoea, PND, ankle oedema?",
        "- Smoking, family history of sudden death / cardiomyopathy?",
        "",
        "Rules:",
        _BRIEF_RULES,
        "",
    ]

    import json as _json

    return _json.dumps(
        {
            "known": have,
            "missing": missing,
            "coverage": f"{len(have)}/{len(_CONTEXTUAL_TRIGGERS)}",
            "brief": "\n".join(lines),
            "next_step": (
                "ask the missing items; when you have them (or hit a natural "
                "pause), call intuition_compile with the full picture"
            ),
            "protocol": protocol_block(),
        },
        ensure_ascii=False,
        indent=2,
    )
