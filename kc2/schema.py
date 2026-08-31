"""Typed knowledge records.

The v1 defect this fixes: reasoning patterns and normative parameters were welded
into one flat prose blob, so a guideline change silently poisoned the corpus.
Here they are different types with different lifecycles, and a ``pattern`` is
forbidden from carrying a value.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

PATTERN = "pattern"
NORM = "norm"
VOCABULARY = "vocabulary"
PROTOCOL = "protocol"
KINDS = {PATTERN, NORM, VOCABULARY, PROTOCOL}

# Values that must never be inlined into a durable note.
_NUMERIC_CLAIM = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*(?:mmol/L|mg/dL|mmHg|ms|mL/min|bpm|mg|%)\b", re.I
)
_SCORE_NAME = re.compile(
    r"CHA2DS2[-\w]*|CHADS\d?|HAS-?BLED|GRACE|TIMI|EuroSCORE|Wells", re.I
)

# The hard case. "aggressive lipid management (LDL 4.5 to 2.6)" carries no unit,
# so a unit-based scan misses it entirely - the staleness sits in the judgement,
# not in the token. Anchor on the analyte name instead and accept a bare number.
_ANALYTES = (
    r"LDL|HDL|non-HDL|triglyceride|cholesterol|HbA1c|creatinine|eGFR|troponin|"
    r"BNP|NT-proBNP|potassium|sodium|haemoglobin|hemoglobin|EF|LVEF|PASP|"
    r"gradient|INR|CRP|TSH"
)
_BARE_ANALYTE = re.compile(
    rf"\b(?:{_ANALYTES})\b[^.\n]{{0,40}}?\b\d+(?:[.,]\d+)?\b", re.I
)


@dataclass
class Note:
    """A durable unit of clinical reasoning."""

    title: str
    kind: str = PATTERN
    content: str = ""
    tags: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)
    applies_norm: list[str] = field(default_factory=list)
    source_transcript: str | None = None
    observed_on: str | None = None
    extracted_by: str | None = None
    confidence: float | None = None
    source: str | None = None

    def validate(self) -> list[str]:
        """Return a list of schema violations (empty means valid)."""
        errs: list[str] = []
        if not self.title.strip():
            errs.append("title is empty")
        if self.kind not in KINDS:
            errs.append(f"unknown kind {self.kind!r}")
        if self.kind in (PATTERN, VOCABULARY):
            # A durable note may not freeze a parameter into itself.
            if _SCORE_NAME.search(self.content):
                errs.append(
                    "pattern/vocabulary note names a scoring system inline; "
                    "reference it via applies_norm instead"
                )
            if _NUMERIC_CLAIM.search(self.content) or _BARE_ANALYTE.search(self.content):
                errs.append(
                    "pattern/vocabulary note inlines a numeric threshold; "
                    "reference it via applies_norm instead"
                )
        return errs

    @property
    def is_valid(self) -> bool:
        return not self.validate()


def detect_normative_claims(text: str) -> list[str]:
    """Surface value-bearing spans that belong in the norms layer, not in prose."""
    return sorted(
        {m.group(0).strip() for m in _SCORE_NAME.finditer(text)}
        | {m.group(0).strip() for m in _NUMERIC_CLAIM.finditer(text)}
        | {re.sub(r"\s+", " ", m.group(0).strip()) for m in _BARE_ANALYTE.finditer(text)}
    )
