"""Pydantic data contracts for the loan screening pipeline.

These models are the single source of truth for the JSON contract between the
backend pipeline and the React frontend. They also enforce the project's core
design principle: *facts are structured and validated by code*, never trusted
raw from the LLM.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #
class DecisionType(str, Enum):
    APPROVE = "APPROVE"
    REJECT = "REJECT"
    REQUIRES_REVIEW = "REQUIRES_REVIEW"
    INCOMPLETE = "INCOMPLETE"


class FieldSource(str, Enum):
    """Where a fact value came from. Used for auditability and to flag which
    numbers an LLM produced vs. which the user typed directly."""
    STRUCTURED = "structured"      # typed into the form
    EXTRACTED = "extracted"        # pulled from free text by the LLM
    DERIVED = "derived"            # computed by deterministic code
    MISSING = "missing"            # not available from any source


# --------------------------------------------------------------------------- #
# Input
# --------------------------------------------------------------------------- #
class LoanApplication(BaseModel):
    """Raw input. Structured fields are optional because some may arrive only
    inside `free_text` and get extracted by the LLM. Code validates ranges."""

    applicant_name: Optional[str] = None
    age: Optional[int] = Field(default=None, ge=0, le=120)
    requested_amount: Optional[float] = Field(default=None, ge=0)
    loan_purpose: Optional[str] = None
    annual_income: Optional[float] = Field(default=None, ge=0)
    employment_status: Optional[str] = None
    employment_years: Optional[float] = Field(default=None, ge=0, le=80)
    existing_monthly_debt: Optional[float] = Field(default=None, ge=0)
    credit_score: Optional[int] = Field(default=None, ge=0, le=1000)
    collateral_value: Optional[float] = Field(default=None, ge=0)
    documents_provided: list[str] = Field(default_factory=list)

    # Unstructured applicant notes. If present, the LLM fact-extractor runs and
    # fills any null structured field it can confidently support.
    free_text: Optional[str] = None


# --------------------------------------------------------------------------- #
# Facts (normalized, post-extraction, post-derivation)
# --------------------------------------------------------------------------- #
class Facts(BaseModel):
    """The validated fact base the rule engine operates on. Every derived value
    is computed by deterministic code, not the LLM."""

    applicant_name: Optional[str] = None
    age: Optional[int] = None
    requested_amount: Optional[float] = None
    loan_purpose: Optional[str] = None
    annual_income: Optional[float] = None
    employment_status: Optional[str] = None
    employment_years: Optional[float] = None
    existing_monthly_debt: Optional[float] = None
    credit_score: Optional[int] = None
    collateral_value: Optional[float] = None
    documents_provided: list[str] = Field(default_factory=list)

    # Derived (code-computed) facts.
    monthly_income: Optional[float] = None
    estimated_new_emi: Optional[float] = None
    dti_ratio: Optional[float] = None
    loan_to_value: Optional[float] = None

    # Audit trail: which fields the LLM supplied, and any conflicts it found.
    field_sources: dict[str, FieldSource] = Field(default_factory=dict)
    extraction_notes: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Decision output
# --------------------------------------------------------------------------- #
class Reason(BaseModel):
    """A single rule evaluation result. The engine emits one per rule it ran."""

    rule_id: str
    description: str
    passed: bool
    detail: str          # e.g. "credit score 610 is below minimum 650"


class Decision(BaseModel):
    decision: DecisionType
    reasons: list[Reason] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    policy_rules_used: list[str] = Field(default_factory=list)
    explanation: str = ""          # LLM-generated prose; may be empty if no key
    facts_used: Facts


# --------------------------------------------------------------------------- #
# LLM structured-output schema for fact extraction
# --------------------------------------------------------------------------- #
class ExtractedFacts(BaseModel):
    """What the LLM is allowed to return from free text. Every field is optional
    and nullable so the model can leave values blank instead of guessing."""

    age: Optional[int] = None
    requested_amount: Optional[float] = None
    loan_purpose: Optional[str] = None
    annual_income: Optional[float] = None
    employment_status: Optional[str] = None
    employment_years: Optional[float] = None
    existing_monthly_debt: Optional[float] = None
    credit_score: Optional[int] = None
    collateral_value: Optional[float] = None
    documents_provided: list[str] = Field(default_factory=list)
