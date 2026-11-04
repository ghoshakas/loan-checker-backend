"""Retry / fallback behaviour of the Gemini wrapper, with a fake SDK client.

Run:  cd backend && pytest eval/test_gemini_client.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

from app import gemini_client as gc  # noqa: E402
from app.fact_extractor import normalize_employment_status  # noqa: E402


class _ApiError(Exception):
    def __init__(self, code: int):
        super().__init__(f"error {code}")
        self.code = code


class _Resp:
    def __init__(self, text: str):
        self.text = text


class _FakeModels:
    def __init__(self, script):
        self.script = list(script)     # list of (expected_model, outcome)
        self.calls: list[str] = []

    def generate_content(self, model, contents, config):
        self.calls.append(model)
        outcome = self.script.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return _Resp(outcome)


@pytest.fixture
def fake(monkeypatch):
    def install(script, primary="primary-model", fallback="fallback-model"):
        models = _FakeModels(script)
        monkeypatch.setattr(gc, "is_available", lambda: True)
        monkeypatch.setattr(gc, "_get_client", lambda: type("C", (), {"models": models})())
        monkeypatch.setattr(gc, "get_gemini_model", lambda: primary)
        monkeypatch.setattr(gc, "_RETRY_DELAYS", (0.0, 0.0))
        monkeypatch.setenv("GEMINI_FALLBACK_MODEL", fallback)
        return models
    return install


def test_retries_transient_error_then_succeeds(fake):
    m = fake([_ApiError(503), '{"ok": 1}'])
    assert gc.generate_json("p", dict) == {"ok": 1}
    assert m.calls == ["primary-model", "primary-model"]


def test_falls_back_when_primary_stays_overloaded(fake):
    m = fake([_ApiError(503)] * 3 + ['{"ok": 2}'])
    assert gc.generate_json("p", dict) == {"ok": 2}
    assert m.calls == ["primary-model"] * 3 + ["fallback-model"]


def test_non_transient_error_is_not_retried(fake):
    m = fake([_ApiError(400)])
    assert gc.generate_json("p", dict) is None
    assert m.calls == ["primary-model"]


def test_gives_up_after_all_models_fail(fake):
    m = fake([_ApiError(429)] * 6)
    assert gc.generate_text("p") is None
    assert m.calls == ["primary-model"] * 3 + ["fallback-model"] * 3


def test_no_fallback_when_same_model(fake):
    m = fake([_ApiError(503)] * 3, primary="x", fallback="x")
    assert gc.generate_text("p") is None
    assert m.calls == ["x"] * 3


@pytest.mark.parametrize("raw,expected", [
    ("Salaried, full-time (permanent)", "salaried"),
    ("Self-employed (proprietor)", "self-employed"),
    ("Government employee", "government"),
    ("Contract staff", "contract"),
    ("salaried", "salaried"),
    ("Unemployed", "unemployed"),
])
def test_employment_status_normalised(raw, expected):
    assert normalize_employment_status(raw) == expected


def test_honours_server_retry_delay(fake, monkeypatch):
    slept = []
    monkeypatch.setattr(gc.time, "sleep", lambda s: slept.append(s))
    err = _ApiError(429)
    err.args = ("429 RESOURCE_EXHAUSTED {'retryDelay': '12s'}",)
    m = fake([err, '{"ok": 3}'])
    assert gc.generate_json("p", dict) == {"ok": 3}
    assert slept == [13.0] and m.calls == ["primary-model", "primary-model"]


def test_too_long_retry_delay_skips_to_fallback(fake, monkeypatch):
    monkeypatch.setattr(gc.time, "sleep", lambda s: None)
    err = _ApiError(429)
    err.args = ("429 {'retryDelay': '300s'}",)
    m = fake([err, '{"ok": 4}'])
    assert gc.generate_json("p", dict) == {"ok": 4}
    assert m.calls == ["primary-model", "fallback-model"]
