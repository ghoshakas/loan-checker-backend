"""LLM step 1 — extract structured facts from unstructured applicant text.

Design guarantees enforced here (not left to the model):
  * The LLM only ever FILLS fields that are null in the structured input.
    A value the user typed into the form always wins.
  * If the LLM and the form disagree, the form value is kept and the conflict
    is recorded as an extraction note (and later surfaced for review).
  * The LLM is instructed to leave a field null when unsure; code never invents
    a number to replace a null.
"""

from __future__ import annotations

from .gemini_client import generate_json, is_available
from .schemas import ExtractedFacts, Facts, FieldSource, LoanApplication

# Final prompt (see docs/prompt-engineering.md for the V1 -> V2 evolution).
EXTRACTION_PROMPT = """You are a data-extraction component in a loan-screening \
system. Extract ONLY the facts explicitly stated in the applicant's notes below.

STRICT RULES:
- Return a value ONLY if the notes clearly state or unambiguously imply it.
- If a field is not mentioned, leave it null. NEVER guess, estimate, or infer a \
number that is not supported by the text.
- Do not perform any eligibility judgement. You extract facts; you do not decide.
- Monetary amounts: return plain numbers (e.g. "12 lakh" -> 1200000, "₹45,000" -> 45000).
- credit_score must be a number the text actually states; otherwise null.
- documents_provided: list only documents the text says are available, using the \
canonical labels id_proof, income_proof, bank_statement where they match.

Applicant notes:
\"\"\"
{free_text}
\"\"\"
"""


def extract_facts(application: LoanApplication) -> Facts:
    """Merge structured input with LLM-extracted facts into a validated Facts.

    Structured (form) values take precedence. The LLM only fills gaps.
    """
    facts = Facts(
        applicant_name=application.applicant_name,
        age=application.age,
        requested_amount=application.requested_amount,
        loan_purpose=application.loan_purpose,
        annual_income=application.annual_income,
        employment_status=application.employment_status,
        employment_years=application.employment_years,
        existing_monthly_debt=application.existing_monthly_debt,
        credit_score=application.credit_score,
        collateral_value=application.collateral_value,
        documents_provided=list(application.documents_provided),
    )

    # Record provenance of structured fields.
    for field in (
        "age", "requested_amount", "annual_income", "employment_status",
        "employment_years", "existing_monthly_debt", "credit_score",
        "collateral_value", "loan_purpose",
    ):
        if getattr(facts, field) is not None:
            facts.field_sources[field] = FieldSource.STRUCTURED

    if not application.free_text:
        return facts

    if not is_available():
        facts.extraction_notes.append(
            "Free text provided but LLM extraction unavailable (no API key); "
            "structured fields used as-is."
        )
        return facts

    raw = generate_json(
        EXTRACTION_PROMPT.format(free_text=application.free_text),
        ExtractedFacts,
    )
    if raw is None:
        facts.extraction_notes.append(
            "LLM extraction call failed; structured fields used as-is."
        )
        return facts

    try:
        extracted = ExtractedFacts.model_validate(raw)
    except Exception:
        facts.extraction_notes.append("LLM returned malformed data; ignored.")
        return facts

    _merge(facts, extracted)
    return facts


def _merge(facts: Facts, extracted: ExtractedFacts) -> None:
    """Fill null fields from extraction; flag conflicts instead of overwriting."""
    scalar_fields = [
        "age", "requested_amount", "loan_purpose", "annual_income",
        "employment_status", "employment_years", "existing_monthly_debt",
        "credit_score", "collateral_value",
    ]
    for field in scalar_fields:
        new_val = getattr(extracted, field)
        if new_val is None:
            continue
        cur_val = getattr(facts, field)
        if cur_val is None:
            setattr(facts, field, new_val)
            facts.field_sources[field] = FieldSource.EXTRACTED
        elif cur_val != new_val:
            # Form value wins; record the discrepancy for human review.
            facts.extraction_notes.append(
                f"Conflict on '{field}': form={cur_val} vs notes={new_val}; "
                f"kept form value."
            )

    # Union documents, de-duplicated and normalized.
    if extracted.documents_provided:
        have = {d.strip().lower() for d in facts.documents_provided}
        for doc in extracted.documents_provided:
            if doc.strip().lower() not in have:
                facts.documents_provided.append(doc.strip().lower())
