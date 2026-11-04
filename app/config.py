"""Loads the YAML policy once and exposes it as a plain dict.

Keeping this tiny and separate means the rule engine stays a pure function of
(facts, policy) and can be unit-tested with an injected policy dict.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

_POLICY_PATH = Path(__file__).parent / "policy.yaml"


@lru_cache(maxsize=1)
def get_policy() -> dict:
    with open(_POLICY_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def get_gemini_model() -> str:
    return os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
