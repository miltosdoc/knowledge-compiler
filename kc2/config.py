"""Environment-driven configuration.

v1 hardcoded ``/Users/meditalks/knowledge-compiler`` into every module, which made
the repo unrunnable anywhere but one Mac. Everything here comes from the
environment with a sane default relative to the repo root.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _path(env: str, default: Path) -> Path:
    return Path(os.getenv(env, str(default))).expanduser()


BASE_DIR = _path("KC_BASE_DIR", REPO_ROOT)
VAULT_DIR = _path("KC_VAULT_DIR", BASE_DIR / "vault")
ATOMIC_DIR = _path("KC_ATOMIC_DIR", VAULT_DIR / "atomic")
COMPILED_DIR = _path("KC_COMPILED_DIR", VAULT_DIR / "compiled")
NORMS_DIR = _path("KC_NORMS_DIR", REPO_ROOT / "norms")

# Inference. Defaults to the local llama-server so PHI never leaves the box.
BASE_URL = os.getenv("KC_BASE_URL", "http://localhost:8000/v1")
MODEL = os.getenv("KC_MODEL", "qwen3.8-flash-next")

# Never hardcode a key. Absent is fine for a local llama-server with no --api-key.
API_KEY = os.getenv("KC_API_KEY") or os.getenv("XSILICO_API_KEY") or "sk-local"

# Concept-identity adjudication.
#   "harness" - return candidates and let the calling agent decide (default).
#               An MCP server already has a capable model driving it; calling a
#               second one over the network is slower, weaker, and sends clinical
#               text off the machine for no gain.
#   "api"     - headless batch ingestion, where no agent is in the loop.
#   "none"    - thresholds only.
ADJUDICATOR_MODE = os.getenv("KC_ADJUDICATOR_MODE", "harness")
ADJUDICATOR_BASE_URL = os.getenv("KC_ADJUDICATOR_BASE_URL", BASE_URL)
ADJUDICATOR_MODEL = os.getenv("KC_ADJUDICATOR_MODEL", MODEL)
ADJUDICATOR_KEY = os.getenv("KC_ADJUDICATOR_KEY") or API_KEY
ADJUDICATOR_TIMEOUT = int(os.getenv("KC_ADJUDICATOR_TIMEOUT", "60"))

MAX_PROMPT_TOKENS = int(os.getenv("KC_MAX_PROMPT_TOKENS", "8000"))
CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN
