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
CONCEPTS_DIR = _path("KC_CONCEPTS_DIR", VAULT_DIR / "concepts")
NORMS_DIR = _path("KC_NORMS_DIR", REPO_ROOT / "norms")
# Raw clinical sources. Anchored to the project root, not the working directory,
# so nothing depends on where the agent happens to be when it runs.
DATA_DIR = _path("KC_DATA_DIR", REPO_ROOT / "data")

# ---------------------------------------------------------------------------
# Concept-identity adjudication.
#
#   "harness"  (default) return candidates and let the agent driving this server
#              decide. It already has a capable model; calling a second one is
#              slower, weaker, and sends clinical text off the machine for no
#              gain.
#   "api"      headless batch ingestion, where no agent is in the loop.
#   "none"     similarity thresholds only.
# ---------------------------------------------------------------------------
ADJUDICATOR_MODE = os.getenv("KC_ADJUDICATOR_MODE", "harness")

# ---------------------------------------------------------------------------
# Remote inference endpoints.
#
# NOTHING ON THE DEFAULT PATH READS THESE. In harness mode kc2 makes no outbound
# network calls whatsoever: the driving model is the distiller and the
# adjudicator, and embeddings run locally. These settings are inert unless
# explicitly opted into:
#
#     KC_ADJUDICATOR_MODE=api   headless concept adjudication
#     KC_EMBED_API_MODEL=...    remote embeddings instead of the local encoder
#
# Do not read a default endpoint out of this block and conclude a model service
# is required. It is not, and no server needs to be running.
# ---------------------------------------------------------------------------
ADJUDICATOR_BASE_URL = os.getenv("KC_ADJUDICATOR_BASE_URL", "http://localhost:8000/v1")
ADJUDICATOR_MODEL = os.getenv("KC_ADJUDICATOR_MODEL", "")
ADJUDICATOR_KEY = os.getenv("KC_ADJUDICATOR_KEY", "")
ADJUDICATOR_TIMEOUT = int(os.getenv("KC_ADJUDICATOR_TIMEOUT", "60"))

# Remote embeddings. Unset means the OpenAI-compatible backend is never even
# probed - an earlier version reached for localhost:8000 whenever the local
# encoder was missing, which made a network call on the default path.
EMBED_API_MODEL = os.getenv("KC_EMBED_API_MODEL", "")
EMBED_API_BASE_URL = os.getenv("KC_EMBED_API_BASE_URL", ADJUDICATOR_BASE_URL)
EMBED_API_KEY = os.getenv("KC_EMBED_API_KEY", ADJUDICATOR_KEY)

MAX_PROMPT_TOKENS = int(os.getenv("KC_MAX_PROMPT_TOKENS", "8000"))
CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN
