"""Deterministic gap and sanity analysis.

Runs BEFORE the rule engine's verdict logic. Answers two questions with code
(never the LLM):
  1. What required information is missing?  -> forces INCOMPLETE.
  2. Are any supplied values internally inconsistent or out of plausible range?
     -> surfaced as notes / review triggers.
"""

from __future__ import annotations

from .schemas import Facts


def find_missing_information(facts: Facts, policy: dict) -> list[str]:
    """Return human-readable labels for required fields that are null."""
    labels = {
        "age": "Applicant age",
        "requested_amount": "Requested loan amount",
        "annual_income": "Annual income",
        "credit_score": "Credit score",
        "employment_status": "Employment status",
        "employment_years": "Years of employment",
        "existing_monthly_debt": "Existing monthly debt obligations",
    }
    missing: list[str] = []
    for field in policy["required_fields"]:
        if getattr(facts, field, None) is None:
            missing.append(labels.get(field, field))
    return missing


def find_missing_documents(facts: Facts, policy: dict) -> list[str]:
    provided = {d.strip().lower() for d in facts.documents_provided}
    labels = {
        "id_proof": "ID proof",
        "income_proof": "Income proof",
        "bank_statement": "Bank statement",
    }
    missing = [
        labels.get(doc, doc)
        for doc in policy["required_documents"]
        if doc not in provided
    ]
    return missing


def find_data_quality_flags(facts: Facts, policy: dict) -> list[str]:
    """Out-of-range / implausible values that passed basic type validation but
    warrant a human look. These do not themselves decide the outcome."""
    flags: list[str] = []

    if facts.credit_score is not None and not (300 <= facts.credit_score <= 900):
        flags.append(
            f"Credit score {facts.credit_score} is outside the plausible 300-900 range."
        )
    if facts.requested_amount is not None and facts.requested_amount == 0:
        flags.append("Requested amount is zero.")
    if (
        facts.annual_income is not None
        and facts.existing_monthly_debt is not None
        and facts.annual_income > 0
        and facts.existing_monthly_debt * 12 > facts.annual_income
    ):
        flags.append(
            "Existing annual debt payments exceed annual income — figures may be inconsistent."
        )
    return flags
