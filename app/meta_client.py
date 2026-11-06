"""Meta Llama API client — the backup LLM when Gemini is unavailable.

Uses Meta's OpenAI-compatible endpoint (``/compat/v1/chat/completions``) with
plain HTTP (httpx), so no extra SDK is needed. Like the Gemini wrapper, every
call is best-effort: it returns ``None`` on any failure instead of raising, so
the deterministic pipeline keeps working.

Llama models accept images but not PDFs, so PDF pages are rendered to JPEG
first (pypdfium2 + Pillow). If those libraries are missing, PDFs are skipped
and only image uploads are sent.

Settings (environment):
    META_API_KEY    required to enable this client (format ``LLM|<id>|<secret>``)
    META_MODEL      default ``Llama-4-Maverick-17B-128E-Instruct-FP8`` (multimodal)
    META_BASE_URL   default ``https://api.llama.com/compat/v1``
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import time
from typing import Any, Optional, Sequence

import httpx

logger = logging.getLogger("loan_screening.meta")

# (label, raw bytes, mime type) — same shape as the Gemini wrapper's attachments.
Attachment = tuple[str, bytes, str]

DEFAULT_MODEL = "Llama-4-Maverick-17B-128E-Instruct-FP8"
DEFAULT_BASE_URL = "https://api.llama.com/compat/v1"

_TRANSIENT_CODES = {429, 500, 502, 503, 504}
_RETRY_DELAYS = (1.0, 3.0)
_TIMEOUT = httpx.Timeout(90.0, connect=10.0)

MAX_IMAGES = 8          # per request, across all files
MAX_PDF_PAGES = 3       # per PDF; loan documents here are 1–2 pages
_PDF_RENDER_SCALE = 1.5  # ~108 dpi: readable text, modest payload


def is_available() -> bool:
    return bool(os.getenv("META_API_KEY", "").strip())


def model_name() -> str:
    return os.getenv("META_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL


# --------------------------------------------------------------------------- #
# Attachments -> image parts
# --------------------------------------------------------------------------- #
def _pdf_to_jpegs(data: bytes, max_pages: int) -> list[bytes]:
    try:
        import pypdfium2 as pdfium  # optional dependency
    except Exception:
        logger.warning("pypdfium2 not installed; PDF skipped for Meta Llama")
        return []
    pages: list[bytes] = []
    pdf = pdfium.PdfDocument(data)
    try:
        for i in range(min(len(pdf), max_pages)):
            img = pdf[i].render(scale=_PDF_RENDER_SCALE).to_pil().convert("RGB")
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=85)
            pages.append(buf.getvalue())
    finally:
        pdf.close()
    return pages


def _data_url(data: bytes, mime: str) -> dict:
    b64 = base64.b64encode(data).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}


def _build_content(prompt: str, attachments: Sequence[Attachment]) -> list[dict] | str:
    """Prompt, then per file a text label and its image(s) — same labels as the
    Gemini path, so DOCUMENT_PROMPT's file_index logic works unchanged."""
    if not attachments:
        return prompt
    parts: list[dict] = [{"type": "text", "text": prompt}]
    images_left = MAX_IMAGES
    for i, (label, data, mime) in enumerate(attachments):
        if mime == "application/pdf":
            images = [(p, "image/jpeg") for p in _pdf_to_jpegs(data, min(MAX_PDF_PAGES, images_left))]
        else:
            images = [(data, mime)] if images_left > 0 else []
        note = "" if images else " (could not be attached — treat as unreadable)"
        parts.append({"type": "text", "text": f"--- FILE {i}: {label} ---{note}"})
        for img, img_mime in images:
            parts.append(_data_url(img, img_mime))
        images_left -= len(images)
    return parts


# --------------------------------------------------------------------------- #
# Pydantic schema -> self-contained JSON schema
# --------------------------------------------------------------------------- #
def _inline_refs(node: Any, defs: dict) -> Any:
    """Replace every {"$ref": "#/$defs/X"} with the definition itself; some
    structured-output backends don't resolve $ref."""
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            return _inline_refs(defs[ref.split("/")[-1]], defs)
        return {k: _inline_refs(v, defs) for k, v in node.items() if k != "$defs"}
    if isinstance(node, list):
        return [_inline_refs(v, defs) for v in node]
    return node


def _response_format(schema_cls: type) -> dict:
    if hasattr(schema_cls, "model_json_schema"):
        raw = schema_cls.model_json_schema()
        schema = _inline_refs(raw, raw.get("$defs", {}))
        return {"type": "json_schema",
                "json_schema": {"name": schema_cls.__name__, "schema": schema}}
    return {"type": "json_object"}


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
def _post(body: dict) -> dict:
    """POST with retries on transient errors. Raises on final failure."""
    url = os.getenv("META_BASE_URL", DEFAULT_BASE_URL).rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {os.getenv('META_API_KEY', '').strip()}",
               "Content-Type": "application/json"}
    last: Exception | None = None
    for attempt in range(len(_RETRY_DELAYS) + 1):
        try:
            resp = httpx.post(url, json=body, headers=headers, timeout=_TIMEOUT)
        except httpx.TransportError as exc:   # timeouts, connection resets
            last = exc
            code: Any = "network"
        else:
            if resp.status_code < 400:
                return resp.json()
            code = resp.status_code
            last = RuntimeError(f"Meta Llama {code}: {resp.text[:300]}")
            if code not in _TRANSIENT_CODES:
                raise last
        if attempt < len(_RETRY_DELAYS):
            logger.info("Meta Llama returned %s; retrying in %.0fs", code, _RETRY_DELAYS[attempt])
            time.sleep(_RETRY_DELAYS[attempt])
    assert last is not None
    raise last


def _message_text(payload: dict) -> str:
    choices = payload.get("choices") or []
    content = (choices[0].get("message") or {}).get("content") if choices else None
    if isinstance(content, list):   # some backends return content parts
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return (content or "").strip()


def _strip_fences(text: str) -> str:
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0]
    return text.strip()


def generate_json(prompt: str, response_schema: type,
                  attachments: Sequence[Attachment] = ()) -> Optional[dict]:
    if not is_available():
        return None
    try:
        body = {
            "model": model_name(),
            "messages": [{"role": "user", "content": _build_content(prompt, attachments)}],
            "response_format": _response_format(response_schema),
            "temperature": 0,
        }
        text = _strip_fences(_message_text(_post(body)))
        return json.loads(text) if text else None
    except Exception as exc:
        logger.warning("Meta Llama JSON call failed: %s", str(exc)[:300])
        return None


def generate_text(prompt: str) -> Optional[str]:
    if not is_available():
        return None
    try:
        body = {"model": model_name(),
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2}
        return _message_text(_post(body)) or None
    except Exception as exc:
        logger.warning("Meta Llama text call failed: %s", str(exc)[:300])
        return None
