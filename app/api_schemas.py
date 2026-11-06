"""Request / response models for the applicant & admin API.

Kept separate from schemas.py (the screening pipeline's contracts) so the
pipeline stays independent of accounts and persistence.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from .schemas import Decision, LoanApplication


class RegisterIn(BaseModel):
    email: str = Field(max_length=255)
    password: str = Field(max_length=200)
    full_name: str = Field(min_length=1, max_length=200)


class LoginIn(BaseModel):
    email: str = Field(max_length=255)
    password: str = Field(max_length=200)


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    email: str
    full_name: str
    role: str


class AuthOut(BaseModel):
    token: str
    user: UserOut


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    filename: str
    mime_type: str
    size_bytes: int
    declared_type: Optional[str] = None
    uploaded_at: datetime


class EventOut(BaseModel):
    action: str
    detail: Optional[str] = None
    actor_name: Optional[str] = None
    created_at: datetime


class ApplicantApplicationOut(BaseModel):
    """What an applicant sees: their data, documents, status and the officer's
    decision/message. Internal AI analysis and recommendation are NOT exposed."""

    id: int
    status: str
    created_at: datetime
    updated_at: datetime
    application: LoanApplication
    documents: list[DocumentOut]
    final_decision: Optional[str] = None
    applicant_message: Optional[str] = None
    decided_at: Optional[datetime] = None
    # True while the AI screening runs in the background (it can take minutes).
    screening_in_progress: bool = False


class AdminApplicationSummary(BaseModel):
    id: int
    applicant_name: str
    applicant_email: str
    requested_amount: Optional[float] = None
    loan_purpose: Optional[str] = None
    status: str
    recommendation: Optional[str] = None
    risk_level: Optional[str] = None
    document_count: int
    created_at: datetime
    screening_in_progress: bool = False


class AdminApplicationOut(ApplicantApplicationOut):
    applicant: UserOut
    screening: Optional[Decision] = None      # rule recommendation + AI analysis
    recommendation: Optional[str] = None
    risk_level: Optional[str] = None
    screened_at: Optional[datetime] = None
    decision_note: Optional[str] = None
    decided_by: Optional[str] = None
    events: list[EventOut]


class DecisionIn(BaseModel):
    decision: Literal["approved", "rejected", "info_requested"]
    note: Optional[str] = Field(default=None, max_length=2000)                  # internal
    message_to_applicant: Optional[str] = Field(default=None, max_length=2000)  # shown to applicant


class StatsOut(BaseModel):
    total: int
    by_status: dict[str, int]
    by_recommendation: dict[str, int]
