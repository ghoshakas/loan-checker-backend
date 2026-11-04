"""Thin wrapper around the Google Gemini SDK.

Centralizes: client creation, model selection, JSON-mode calls, file (PDF/image)
attachments, and — crucially — graceful degradation when no API key is
configured. The deterministic pipeline must keep working without an LLM, so
every call here is best-effort and returns ``None`` on any failure rather than
raising into the request path. Failures are logged so they are not invisible
(e.g. a 429 quota error).
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Callable, Optional, Sequence

from .config import get_gemini_model

logger = logging.getLogger("loan_screening.gemini")

_client = None
_import_error: Optional[str] = None

try:  # SDK is optional at import time so tests/UI run without it.
    from google import genai
    from google.genai import types as genai_types
except Exception as exc:  # pragma: no cover - import environment dependent
    genai = None
    genai_types = None
    _import_error = str(exc)

# An attachment is (label, raw bytes, mime type), e.g. ("payslip.pdf", b"...", "application/pdf").
Attachment = tuple[str, bytes, str]


def is_available() -> bool:
    """True only if the SDK is importable and an API key is set."""
    return genai is not None and bool(os.getenv("GEMINI_API_KEY"))


def _get_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    return _client


# Temporary failures worth retrying: rate limit, server error, overloaded, timeout.
_TRANSIENT_CODES = {429, 500, 502, 503, 504}
_RETRY_DELAYS = (1.0, 3.0)   # seconds between attempts on the same model


def _models() -> list[str]:
    """Primary model, then the fallback (if different)."""
    primary = get_gemini_model()
    fallback = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-2.5-flash").strip()
    return [primary] + ([fallback] if fallback and fallback != primary else [])


def _call(make_request: Callable[[str], Any]) -> Any:
    """Run `make_request(model)` with retries on transient errors, then fall
    back to GEMINI_FALLBACK_MODEL. Raises the last error if everything fails."""
    last: Exception | None = None
    for model in _models():
        for attempt in range(len(_RETRY_DELAYS) + 1):
            try:
                return make_request(model)
            except Exception as exc:
                last = exc
                code = getattr(exc, "code", None)
                if code not in _TRANSIENT_CODES:
                    raise
                if attempt < len(_RETRY_DELAYS):
                    logger.info("Gemini %s returned %s; retrying in %.0fs", model, code,
                                _RETRY_DELAYS[attempt])
                    time.sleep(_RETRY_DELAYS[attempt])
        logger.warning("Gemini model %s unavailable (%s); trying fallback if configured",
                       model, getattr(last, "code", "?"))
    assert last is not None
    raise last


def _build_contents(prompt: str, attachments: Sequence[Attachment]) -> list | str:
    """Prompt first, then each file preceded by a text label so the model can
    refer to it by index."""
    if not attachments:
        return prompt
    contents: list = [prompt]
    for i, (label, data, mime) in enumerate(attachments):
        contents.append(f"--- FILE {i}: {label} ---")
        contents.append(genai_types.Part.from_bytes(data=data, mime_type=mime))
    return contents


def generate_json(
    prompt: str,
    response_schema: type,
    attachments: Sequence[Attachment] = (),
) -> Optional[dict]:
    """Call Gemini in JSON mode and return a parsed dict, or None on failure.

    `response_schema` is a Pydantic model class; the SDK uses it to constrain
    the model's output to valid JSON matching that schema. `attachments` are
    optional files (PDF/images) sent inline with the prompt.
    """
    if not is_available():
        return None
    try:
        client = _get_client()
        contents = _build_contents(prompt, attachments)
        config = genai_types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=response_schema,
            temperature=0,  # determinism-friendly for extraction
        )
        resp = _call(lambda model: client.models.generate_content(
            model=model, contents=contents, config=config))
        text = (resp.text or "").strip()
        return json.loads(text) if text else None
    except Exception as exc:
        logger.warning("Gemini JSON call failed: %s", str(exc)[:300])
        return None


def generate_text(prompt: str) -> Optional[str]:
    """Plain text generation (used for the explanation step). None on failure."""
    if not is_available():
        return None
    try:
        client = _get_client()
        config = genai_types.GenerateContentConfig(temperature=0.2)
        resp = _call(lambda model: client.models.generate_content(
            model=model, contents=prompt, config=config))
        return (resp.text or "").strip() or None
    except Exception as exc:
        logger.warning("Gemini text call failed: %s", str(exc)[:300])
        return None
