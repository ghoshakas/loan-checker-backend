"""Deterministic loan decision engine.

This module is the single authority on the APPROVE / REJECT / REQUIRES_REVIEW /
INCOMPLETE verdict. It is a pure function of (facts, policy): no LLM, no I/O, no
randomness. That is what makes the decision reproducible, auditable and testable,
and it is the central design claim of the project.

Pipeline within this module:
    compute_derived_facts()  -> fills monthly_income, EMI, DTI, LTV
    evaluate()               -> returns (DecisionType, reasons, rules_used, missing)
"""

from __future__ import annotations

from .exception_detector import (
    find_data_quality_flags,
    find_missing_documents,
    find_missing_information,
)
from .schemas import DecisionType, Facts, FieldSource, Reason


# --------------------------------------------------------------------------- #
# Derived facts (deterministic math)
# --------------------------------------------------------------------------- #
def _estimate_emi(principal: float, annual_rate: float, months: int) -> float:
    """Standard amortizing EMI formula. Used only to estimate DTI impact."""
    if principal <= 0 or months <= 0:
        return 0.0
    r = annual_rate / 12.0
    if r == 0:
        return principal / months
    factor = (1 + r) ** months
    return principal * r * factor / (factor - 1)


def compute_derived_facts(facts: Facts, policy: dict) -> Facts:
    """Return a copy of `facts` with derived numeric fields populated."""
    f = facts.model_copy(deep=True)

    if f.annual_income is not None:
        f.monthly_income = round(f.annual_income / 12.0, 2)
        f.field_sources["monthly_income"] = FieldSource.DERIVED

    if f.requested_amount is not None:
        emi_cfg = policy["emi_assumptions"]
        f.estimated_new_emi = round(
            _estimate_emi(
                f.requested_amount,
                emi_cfg["annual_interest_rate"],
                emi_cfg["tenure_months"],
            ),
            2,
        )
        f.field_sources["estimated_new_emi"] = FieldSource.DERIVED

    if (
        f.monthly_income
        and f.monthly_income > 0
        and f.existing_monthly_debt is not None
        and f.estimated_new_emi is not None
    ):
        total_debt = f.existing_monthly_debt + f.estimated_new_emi
        f.dti_ratio = round(total_debt / f.monthly_income, 4)
        f.field_sources["dti_ratio"] = FieldSource.DERIVED

    if (
        f.collateral_value
        and f.collateral_value > 0
        and f.requested_amount is not None
    ):
        f.loan_to_value = round(f.requested_amount / f.collateral_value, 4)
        f.field_sources["loan_to_value"] = FieldSource.DERIVED

    return f


# --------------------------------------------------------------------------- #
# Individual rule checks -> Reason
# --------------------------------------------------------------------------- #
def _rule_credit(facts: Facts, p: dict) -> tuple[Reason, str]:
    """Returns (Reason, severity) where severity in {ok, review, reject}."""
    t = p["thresholds"]
    cs = facts.credit_score
    if cs >= t["min_credit_score"]:
        return (
            Reason(rule_id="R1", description="Minimum credit score",
                   passed=True,
                   detail=f"Credit score {cs} meets minimum {t['min_credit_score']}."),
            "ok",
        )
    severity = "reject" if cs < t["reject_credit_score"] else "review"
    return (
        Reason(rule_id="R1", description="Minimum credit score", passed=False,
               detail=f"Credit score {cs} is below minimum {t['min_credit_score']}."),
        severity,
    )


def _rule_dti(facts: Facts, p: dict) -> tuple[Reason, str]:
    t = p["thresholds"]
    dti = facts.dti_ratio
    if dti is None:
        return (
            Reason(rule_id="R2", description="Debt-to-income ratio", passed=False,
                   detail="DTI could not be computed (missing income or debt data)."),
            "review",
        )
    pct = f"{dti * 100:.1f}%"
    if dti <= t["max_dti"]:
        return (
            Reason(rule_id="R2", description="Debt-to-income ratio", passed=True,
                   detail=f"DTI {pct} is within limit {t['max_dti'] * 100:.0f}%."),
            "ok",
        )
    severity = "reject" if dti > t["review_dti"] else "review"
    return (
        Reason(rule_id="R2", description="Debt-to-income ratio", passed=False,
               detail=f"DTI {pct} exceeds limit {t['max_dti'] * 100:.0f}%."),
        severity,
    )


def _rule_age(facts: Facts, p: dict) -> tuple[Reason, str]:
    t = p["thresholds"]
    age = facts.age
    if t["min_age"] <= age <= t["max_age"]:
        return (
            Reason(rule_id="R3", description="Eligible age band", passed=True,
                   detail=f"Age {age} is within {t['min_age']}-{t['max_age']}."),
            "ok",
        )
    return (
        Reason(rule_id="R3", description="Eligible age band", passed=False,
               detail=f"Age {age} is outside {t['min_age']}-{t['max_age']}."),
        "reject",
    )


def _rule_employment(facts: Facts, p: dict) -> tuple[Reason, str]:
    t = p["thresholds"]
    stable = (facts.employment_status or "").strip().lower() in {
        s.lower() for s in p["stable_employment_statuses"]
    }
    years = facts.employment_years if facts.employment_years is not None else 0
    if years >= t["min_employment_years"] or stable:
        return (
            Reason(rule_id="R4", description="Employment stability", passed=True,
                   detail=f"{years} yrs employment / status '{facts.employment_status}' "
                          f"satisfies stability requirement."),
            "ok",
        )
    return (
        Reason(rule_id="R4", description="Employment stability", passed=False,
               detail=f"{years} yrs employment is below {t['min_employment_years']} yrs "
                      f"and status '{facts.employment_status}' is not pre-approved as stable."),
        "review",
    )


def _rule_amount_vs_income(facts: Facts, p: dict) -> tuple[Reason, str]:
    t = p["thresholds"]
    cap = t["income_multiple"] * facts.annual_income
    if facts.requested_amount <= cap:
        return (
            Reason(rule_id="R5", description="Loan amount vs income", passed=True,
                   detail=f"Requested {facts.requested_amount:,.0f} is within "
                          f"{t['income_multiple']}x income cap ({cap:,.0f})."),
            "ok",
        )
    return (
        Reason(rule_id="R5", description="Loan amount vs income", passed=False,
               detail=f"Requested {facts.requested_amount:,.0f} exceeds "
                      f"{t['income_multiple']}x income cap ({cap:,.0f})."),
        "review",
    )


def _rule_ltv(facts: Facts, p: dict) -> tuple[Reason, str] | None:
    """Only applies when collateral was pledged."""
    if facts.loan_to_value is None:
        return None
    t = p["thresholds"]
    ltv = facts.loan_to_value
    if ltv <= t["max_ltv"]:
        return (
            Reason(rule_id="R6", description="Loan-to-value", passed=True,
                   detail=f"LTV {ltv * 100:.1f}% within limit {t['max_ltv'] * 100:.0f}%."),
            "ok",
        )
    return (
        Reason(rule_id="R6", description="Loan-to-value", passed=False,
               detail=f"LTV {ltv * 100:.1f}% exceeds limit {t['max_ltv'] * 100:.0f}%."),
        "review",
    )


# --------------------------------------------------------------------------- #
# Top-level evaluation with deterministic precedence
# --------------------------------------------------------------------------- #
def evaluate(facts: Facts, policy: dict) -> dict:
    """Run the full deterministic assessment.

    Returns a dict with keys: decision, reasons, policy_rules_used,
    missing_information. The caller attaches the LLM explanation separately.
    """
    missing = find_missing_information(facts, policy)

    # Precedence step 1: cannot decide without the required facts.
    if missing:
        return {
            "decision": DecisionType.INCOMPLETE,
            "reasons": [
                Reason(rule_id="R0", description="Required information present",
                       passed=False,
                       detail="Cannot evaluate: required information is missing.")
            ],
            "policy_rules_used": ["R0"],
            "missing_information": missing,
        }

    # All required facts present -> run the rules.
    checks = [
        _rule_credit(facts, policy),
        _rule_dti(facts, policy),
        _rule_age(facts, policy),
        _rule_employment(facts, policy),
        _rule_amount_vs_income(facts, policy),
    ]
    ltv = _rule_ltv(facts, policy)
    if ltv is not None:
        checks.append(ltv)

    reasons = [c[0] for c in checks]
    severities = [c[1] for c in checks]
    rules_used = [r.rule_id for r in reasons]

    # Missing documents and data-quality flags => soft, push toward review.
    missing_docs = find_missing_documents(facts, policy)
    quality_flags = find_data_quality_flags(facts, policy)
    if missing_docs:
        reasons.append(
            Reason(rule_id="D1", description="Required documents on file",
                   passed=False,
                   detail=f"Missing documents: {', '.join(missing_docs)}.")
        )
        rules_used.append("D1")
        severities.append("review")
    for flag in quality_flags:
        reasons.append(
            Reason(rule_id="Q1", description="Data quality check", passed=False,
                   detail=flag)
        )
        rules_used.append("Q1")
        severities.append("review")

    # Precedence step 2 & 3: reject beats review beats approve.
    if "reject" in severities:
        decision = DecisionType.REJECT
    elif "review" in severities:
        decision = DecisionType.REQUIRES_REVIEW
    else:
        decision = DecisionType.APPROVE

    return {
        "decision": decision,
        "reasons": reasons,
        "policy_rules_used": rules_used,
        "missing_information": [],
    }
