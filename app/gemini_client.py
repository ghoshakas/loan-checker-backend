"""Thin wrapper around the Google Gemini SDK.

Centralizes: client creation, model selection, JSON-mode calls, and — crucially —
graceful degradation when no API key is configured. The deterministic pipeline
must keep working without an LLM, so every call here is best-effort and returns
``None`` on any failure rather than raising into the request path.
"""

from __future__ import annotations

import json
import os
from typing import Optional

from .config import get_gemini_model

_client = None
_import_error: Optional[str] = None

try:  # SDK is optional at import time so tests/UI run without it.
    from google import genai
    from google.genai import types as genai_types
except Exception as exc:  # pragma: no cover - import environment dependent
    genai = None
    genai_types = None
    _import_error = str(exc)


def is_available() -> bool:
    """True only if the SDK is importable and an API key is set."""
    return genai is not None and bool(os.getenv("GEMINI_API_KEY"))


def _get_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    return _client


def generate_json(prompt: str, response_schema: type) -> Optional[dict]:
    """Call Gemini in JSON mode and return a parsed dict, or None on failure.

    `response_schema` is a Pydantic model class; the SDK uses it to constrain
    the model's output to valid JSON matching that schema.
    """
    if not is_available():
        return None
    try:
        client = _get_client()
        resp = client.models.generate_content(
            model=get_gemini_model(),
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=response_schema,
                temperature=0,  # determinism-friendly for extraction
            ),
        )
        text = (resp.text or "").strip()
        return json.loads(text) if text else None
    except Exception:
        return None


def generate_text(prompt: str) -> Optional[str]:
    """Plain text generation (used for the explanation step). None on failure."""
    if not is_available():
        return None
    try:
        client = _get_client()
        resp = client.models.generate_content(
            model=get_gemini_model(),
            contents=prompt,
            config=genai_types.GenerateContentConfig(temperature=0.2),
        )
        return (resp.text or "").strip() or None
    except Exception:
        return None
