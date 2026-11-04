"""Unit tests for document reading, merging and cross-checks.

The LLM is replaced by a stub returning a fixed DocumentAnalysis, so these test
the CODE around the model: normalisation, precedence, range checks, rule D2.

Run:  cd backend && pytest eval/test_documents.py -v
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

from app import document_analyzer as da  # noqa: E402
from app.config import get_policy  # noqa: E402
from app.fact_extractor import build_structured_facts  # noqa: E402
from app.rule_engine import compute_derived_facts, evaluate  # noqa: E402
from app.schemas import DecisionType, FieldSource, LoanApplication  # noqa: E402

POLICY = get_policy()
TODAY = date(2026, 11, 4)
PDF = b"%PDF-1.4 test"
FORM = dict(
    applicant_name="Rahul Sharma", age=35, requested_amount=1_000_000,
    annual_income=1_200_000, employment_status="salaried", employment_years=5,
    existing_monthly_debt=10_000, credit_score=720,
    documents_provided=["id_proof", "income_proof", "bank_statement"],
)


def _run(monkeypatch, form: dict, llm_docs: list[dict]):
    monkeypatch.setattr(da, "is_available", lambda: True)
    monkeypatch.setattr(da, "generate_json", lambda *a, **k: {"documents": llm_docs})
    facts = build_structured_facts(LoanApplication(**form))
    docs = [da.UploadedDocument(f"file{i}.pdf", "application/pdf", PDF) for i in range(len(llm_docs))]
    findings = da.analyze_documents(facts, docs, POLICY, today=TODAY)
    facts = compute_derived_facts(facts, POLICY)
    return facts, findings, evaluate(facts, POLICY)


def test_sniff_mime_uses_magic_bytes():
    assert da.sniff_mime(b"%PDF-1.7...") == "application/pdf"
    assert da.sniff_mime(b"\x89PNG\r\n\x1a\nxxxx") == "image/png"
    assert da.sniff_mime(b"\xff\xd8\xff\xe0xxxx") == "image/jpeg"
    assert da.sniff_mime(b"RIFF\x00\x00\x00\x00WEBPVP8 ") == "image/webp"
    assert da.sniff_mime(b"MZ\x90\x00 exe") is None


def test_validate_uploads_limits():
    with pytest.raises(da.UploadError):
        da.validate_uploads([("a.exe", b"MZ")])
    with pytest.raises(da.UploadError):
        da.validate_uploads([("empty.pdf", b"")])
    with pytest.raises(da.UploadError):
        da.validate_uploads([(f"{i}.pdf", PDF) for i in range(da.MAX_FILES + 1)])
    assert da.validate_uploads([("ok.pdf", PDF)])[0].mime_type == "application/pdf"


def test_compute_age_from_dob():
    assert da.compute_age("1990-11-05", TODAY) == 35   # birthday tomorrow
    assert da.compute_age("1990-11-04", TODAY) == 36
    assert da.compute_age("not a date", TODAY) is None
    assert da.compute_age("2030-01-01", TODAY) is None


def test_documents_fill_blank_fields_and_code_does_the_math(monkeypatch):
    form = {k: v for k, v in FORM.items() if k not in ("age", "annual_income")}
    facts, findings, out = _run(monkeypatch, form, [
        {"file_index": 0, "document_type": "id_proof", "name_on_document": "RAHUL K SHARMA",
         "date_of_birth": "1991-02-10"},
        {"file_index": 1, "document_type": "income_proof", "name_on_document": "Rahul Sharma",
         "monthly_income": 100_000, "employer_name": "Acme Ltd"},
    ])
    assert facts.age == 35 and facts.field_sources["age"] == FieldSource.DOCUMENT
    assert facts.annual_income == 1_200_000          # 100k x 12 computed in code
    assert facts.field_sources["annual_income"] == FieldSource.DOCUMENT
    assert set(facts.documents_uploaded) == {"id_proof", "income_proof"}
    assert findings[1].extracted["employer"] == "Acme Ltd"
    assert out["decision"] == DecisionType.APPROVE
    assert any(r.rule_id == "D2" and r.passed for r in out["reasons"])


def test_document_never_overrides_form_and_mismatch_triggers_review(monkeypatch):
    facts, _, out = _run(monkeypatch, FORM, [
        {"file_index": 0, "document_type": "income_proof", "name_on_document": "Rahul Sharma",
         "monthly_income": 50_000},   # 6L/yr vs form 12L/yr
    ])
    assert facts.annual_income == 1_200_000          # form value kept
    assert any("annual_income" in m for m in facts.document_mismatches)
    assert out["decision"] == DecisionType.REQUIRES_REVIEW
    assert any(r.rule_id == "D2" and not r.passed for r in out["reasons"])


def test_small_difference_within_tolerance_is_ok(monkeypatch):
    _, _, out = _run(monkeypatch, FORM, [
        {"file_index": 0, "document_type": "income_proof", "name_on_document": "Rahul Sharma",
         "monthly_income": 95_000},   # 11.4L vs 12L = 5% < 10%
    ])
    assert out["decision"] == DecisionType.APPROVE


def test_name_mismatch_triggers_review(monkeypatch):
    facts, _, out = _run(monkeypatch, FORM, [
        {"file_index": 0, "document_type": "id_proof", "name_on_document": "Priya Verma"},
    ])
    assert any("name" in m for m in facts.document_mismatches)
    assert out["decision"] == DecisionType.REQUIRES_REVIEW


def test_unreadable_or_other_documents_are_not_merged(monkeypatch):
    facts, findings, _ = _run(monkeypatch, {**FORM, "credit_score": None}, [
        {"file_index": 0, "document_type": "unreadable", "credit_score": 800},
    ])
    assert facts.credit_score is None
    assert facts.documents_uploaded == []
    assert findings[0].document_type == "unreadable"


def test_implausible_llm_values_are_discarded(monkeypatch):
    facts, _, out = _run(monkeypatch, {**FORM, "credit_score": None}, [
        {"file_index": 0, "document_type": "bank_statement", "credit_score": 5000},
    ])
    assert facts.credit_score is None
    assert any("implausible" in n for n in facts.extraction_notes)
    assert out["decision"] == DecisionType.INCOMPLETE


def test_file_in_wrong_slot_triggers_review(monkeypatch):
    monkeypatch.setattr(da, "is_available", lambda: True)
    monkeypatch.setattr(da, "generate_json", lambda *a, **k: {"documents": [
        {"file_index": 0, "document_type": "bank_statement", "name_on_document": "Rahul Sharma"}]})
    facts = build_structured_facts(LoanApplication(**FORM))
    docs = [da.UploadedDocument("payslip.pdf", "application/pdf", PDF, declared_type="income_proof")]
    da.analyze_documents(facts, docs, POLICY, today=TODAY)
    assert any("uploaded as Income proof but looks like Bank statement" in m
               for m in facts.document_mismatches)
    out = evaluate(compute_derived_facts(facts, POLICY), POLICY)
    assert out["decision"] == DecisionType.REQUIRES_REVIEW


def test_unreadable_file_in_required_slot_is_flagged(monkeypatch):
    monkeypatch.setattr(da, "is_available", lambda: True)
    monkeypatch.setattr(da, "generate_json", lambda *a, **k: {"documents": [
        {"file_index": 0, "document_type": "unreadable"}]})
    facts = build_structured_facts(LoanApplication(**FORM))
    da.analyze_documents(facts, [da.UploadedDocument("blur.jpg", "image/jpeg", PDF,
                                                     declared_type="id_proof")], POLICY, today=TODAY)
    out = evaluate(compute_derived_facts(facts, POLICY), POLICY)
    assert any(r.rule_id == "D2" and not r.passed for r in out["reasons"])


def test_provided_types_come_from_slots():
    docs = [da.UploadedDocument("a.pdf", "application/pdf", PDF, declared_type=t)
            for t in ("bank_statement", "other", "id_proof", "id_proof")]
    assert da.provided_document_types(docs) == ["id_proof", "bank_statement"]
    with pytest.raises(da.UploadError):
        da.validate_uploads([("a.pdf", PDF)], ["selfie"])
    with pytest.raises(da.UploadError):
        da.validate_uploads([("a.pdf", PDF), ("b.pdf", PDF)], ["id_proof"])


def test_llm_unavailable_skips_documents(monkeypatch):
    monkeypatch.setattr(da, "is_available", lambda: False)
    facts = build_structured_facts(LoanApplication(**FORM))
    findings = da.analyze_documents(facts, [da.UploadedDocument("a.pdf", "application/pdf", PDF)], POLICY)
    assert findings[0].status == "skipped"
    assert facts.documents_uploaded == []
