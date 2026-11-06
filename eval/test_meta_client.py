"""Meta Llama backup: request shape, retries, and fallback from Gemini.

No network: httpx.post is replaced by a fake.
Run:  cd backend && pytest eval/test_meta_client.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

from app import gemini_client as gc  # noqa: E402
from app import meta_client as mc  # noqa: E402
from app.schemas import DocumentAnalysis, ExtractedFacts  # noqa: E402

SAMPLE_PDF = (Path(__file__).resolve().parents[2] / "sample_documents"
              / "1_rahul_sharma_approve" / "income_proof_payslip_oct_2026.pdf")


class _Resp:
    def __init__(self, status: int, payload: dict | None = None, text: str = ""):
        self.status_code = status
        self._payload = payload or {}
        self.text = text or json.dumps(self._payload)

    def json(self):
        return self._payload


def _ok(content: str) -> _Resp:
    return _Resp(200, {"choices": [{"message": {"role": "assistant", "content": content}}]})


@pytest.fixture
def fake_http(monkeypatch):
    def install(script):
        calls: list[dict] = []

        def post(url, json=None, headers=None, timeout=None):
            calls.append({"url": url, "json": json, "headers": headers})
            return script.pop(0)

        monkeypatch.setenv("META_API_KEY", "LLM|1|secret")
        monkeypatch.setattr(mc.httpx, "post", post)
        monkeypatch.setattr(mc, "_RETRY_DELAYS", (0.0, 0.0))
        return calls
    return install


def test_json_call_sends_schema_and_parses(fake_http):
    calls = fake_http([_ok('{"annual_income": 300000}')])
    out = mc.generate_json("prompt", ExtractedFacts)
    assert out == {"annual_income": 300000}
    body = calls[0]["json"]
    assert calls[0]["url"].endswith("/compat/v1/chat/completions")
    assert calls[0]["headers"]["Authorization"] == "Bearer LLM|1|secret"
    assert body["response_format"]["type"] == "json_schema"
    assert body["temperature"] == 0


def test_nested_schema_has_no_refs():
    fmt = mc._response_format(DocumentAnalysis)
    assert "$ref" not in json.dumps(fmt) and "$defs" not in json.dumps(fmt)


def test_strips_code_fences(fake_http):
    fake_http([_ok('```json\n{"ok": 1}\n```')])
    assert mc.generate_json("p", ExtractedFacts) == {"ok": 1}


def test_retries_transient_then_succeeds(fake_http):
    calls = fake_http([_Resp(503), _ok('{"ok": 2}')])
    assert mc.generate_json("p", ExtractedFacts) == {"ok": 2}
    assert len(calls) == 2


def test_non_transient_error_returns_none(fake_http):
    calls = fake_http([_Resp(402, text="Billing Issue")])
    assert mc.generate_json("p", ExtractedFacts) is None
    assert len(calls) == 1


@pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="sample_documents/ not in this checkout")
def test_pdf_is_sent_as_labelled_image(fake_http):
    calls = fake_http([_ok('{"documents": []}')])
    mc.generate_json("read", DocumentAnalysis,
                     attachments=[("payslip.pdf", SAMPLE_PDF.read_bytes(), "application/pdf")])
    parts = calls[0]["json"]["messages"][0]["content"]
    assert parts[1] == {"type": "text", "text": "--- FILE 0: payslip.pdf ---"}
    assert parts[2]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_disabled_without_key(monkeypatch):
    monkeypatch.delenv("META_API_KEY", raising=False)
    assert mc.generate_text("p") is None


def test_gemini_failure_falls_back_to_meta(fake_http, monkeypatch):
    calls = fake_http([_ok('{"annual_income": 300000}')])
    monkeypatch.setenv("LLM_PROVIDERS", "gemini,meta")
    monkeypatch.setattr(gc, "_gemini_ready", lambda: True)
    monkeypatch.setattr(gc, "_gemini_json", lambda *a, **k: None)   # Gemini down
    assert gc.generate_json("p", ExtractedFacts) == {"annual_income": 300000}
    assert len(calls) == 1


def test_meta_not_called_when_gemini_works(fake_http, monkeypatch):
    calls = fake_http([])
    monkeypatch.setenv("LLM_PROVIDERS", "gemini,meta")
    monkeypatch.setattr(gc, "_gemini_ready", lambda: True)
    monkeypatch.setattr(gc, "_gemini_text", lambda p: "from gemini")
    assert gc.generate_text("p") == "from gemini"
    assert calls == []
