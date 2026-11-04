"""Unit tests for the deterministic core.

These run with NO API key and NO network. They prove the decision is a pure,
reproducible function of (facts, policy) — the central design claim.

Run:  cd backend && pytest eval/test_rule_engine.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_policy  # noqa: E402
from app.pipeline import screen  # noqa: E402
from app.rule_engine import compute_derived_facts, evaluate  # noqa: E402
from app.schemas import DecisionType, Facts, LoanApplication  # noqa: E402

POLICY = get_policy()


def _facts(**kw) -> Facts:
    base = dict(
        age=35,
        requested_amount=1_000_000,
        annual_income=1_200_000,
        employment_status="salaried",
        employment_years=5,
        existing_monthly_debt=10_000,
        credit_score=720,
        documents_provided=["id_proof", "income_proof", "bank_statement"],
    )
    base.update(kw)
    f = Facts(**base)
    return compute_derived_facts(f, POLICY)


def test_clean_approve():
    out = evaluate(_facts(), POLICY)
    assert out["decision"] == DecisionType.APPROVE
    assert out["missing_information"] == []


def test_missing_required_field_is_incomplete():
    out = evaluate(_facts(credit_score=None), POLICY)
    assert out["decision"] == DecisionType.INCOMPLETE
    assert any("Credit score" in m for m in out["missing_information"])


def test_low_credit_score_rejects():
    out = evaluate(_facts(credit_score=550), POLICY)
    assert out["decision"] == DecisionType.REJECT


def test_borderline_credit_score_reviews():
    # between reject_credit_score (580) and min_credit_score (650)
    out = evaluate(_facts(credit_score=610), POLICY)
    assert out["decision"] == DecisionType.REQUIRES_REVIEW


def test_high_dti_rejects():
    # Huge existing debt pushes DTI above review_dti (0.50) -> reject.
    out = evaluate(_facts(existing_monthly_debt=80_000), POLICY)
    assert out["decision"] == DecisionType.REJECT


def test_amount_over_income_cap_reviews():
    out = evaluate(_facts(requested_amount=10_000_000), POLICY)
    assert out["decision"] in (DecisionType.REQUIRES_REVIEW, DecisionType.REJECT)


def test_age_out_of_band_rejects():
    out = evaluate(_facts(age=18), POLICY)
    assert out["decision"] == DecisionType.REJECT


def test_missing_documents_triggers_review():
    out = evaluate(_facts(documents_provided=["id_proof"]), POLICY)
    assert out["decision"] == DecisionType.REQUIRES_REVIEW
    assert "D1" in out["policy_rules_used"]


def test_dti_computed_correctly():
    f = _facts(annual_income=1_200_000, existing_monthly_debt=10_000,
               requested_amount=1_000_000)
    # monthly income = 100000; EMI for 1,000,000 @11%/60m ~= 21742
    assert f.monthly_income == 100_000
    assert f.estimated_new_emi is not None and 21_000 < f.estimated_new_emi < 22_500
    assert f.dti_ratio is not None and 0.30 < f.dti_ratio < 0.33


def test_decision_is_deterministic():
    """Same input, many runs, identical decision (no explanation to avoid LLM)."""
    app = LoanApplication(
        age=35, requested_amount=1_000_000, annual_income=1_200_000,
        employment_status="salaried", employment_years=5,
        existing_monthly_debt=10_000, credit_score=720,
        documents_provided=["id_proof", "income_proof", "bank_statement"],
    )
    results = {screen(app, with_explanation=False).decision for _ in range(5)}
    assert results == {DecisionType.APPROVE}


def test_ltv_rule_rejects_review_when_overexposed():
    out = evaluate(
        _facts(requested_amount=1_000_000, collateral_value=1_000_000), POLICY
    )
    # LTV 100% > 80% max -> review
    assert out["decision"] == DecisionType.REQUIRES_REVIEW
    assert "R6" in out["policy_rules_used"]
