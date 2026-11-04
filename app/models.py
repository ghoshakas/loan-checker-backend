"""Database tables.

    User ──< Application ──< StoredDocument
                │
                └──< ApplicationEvent   (audit trail of every action)

Application lifecycle (the FINAL decision is always a human admin's):

    pending_review ──admin──> approved | rejected | info_requested
    info_requested ──applicant adds documents──> pending_review
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base, utcnow

ROLE_APPLICANT = "applicant"
ROLE_ADMIN = "admin"

STATUS_PENDING = "pending_review"
STATUS_INFO = "info_requested"
STATUS_APPROVED = "approved"
STATUS_REJECTED = "rejected"
ALL_STATUSES = (STATUS_PENDING, STATUS_INFO, STATUS_APPROVED, STATUS_REJECTED)


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    full_name: Mapped[str] = mapped_column(String(200))
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(20), default=ROLE_APPLICANT)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    applications: Mapped[list["Application"]] = relationship(
        back_populates="user", foreign_keys="Application.user_id"
    )


class AuthToken(Base):
    """Login sessions. Only a SHA-256 hash of the bearer token is stored."""

    __tablename__ = "auth_tokens"

    token_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    user: Mapped[User] = relationship()


class Application(Base):
    __tablename__ = "applications"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    status: Mapped[str] = mapped_column(String(30), default=STATUS_PENDING, index=True)

    # What the applicant submitted (LoanApplication as JSON).
    application_data: Mapped[dict[str, Any]] = mapped_column(JSON)

    # Latest screening snapshot (Decision as JSON): rule recommendation, facts,
    # document findings, AI explanation and AI insights.
    screening: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)
    recommendation: Mapped[Optional[str]] = mapped_column(String(20), nullable=True, index=True)
    risk_level: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    screened_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Human decision.
    final_decision: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    decision_note: Mapped[Optional[str]] = mapped_column(Text, nullable=True)       # internal
    applicant_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)   # shown to applicant
    decided_by_id: Mapped[Optional[int]] = mapped_column(ForeignKey("users.id"), nullable=True)
    decided_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    user: Mapped[User] = relationship(back_populates="applications", foreign_keys=[user_id])
    decided_by: Mapped[Optional[User]] = relationship(foreign_keys=[decided_by_id])
    documents: Mapped[list["StoredDocument"]] = relationship(
        back_populates="application", cascade="all, delete-orphan", order_by="StoredDocument.id"
    )
    events: Mapped[list["ApplicationEvent"]] = relationship(
        back_populates="application", cascade="all, delete-orphan", order_by="ApplicationEvent.id"
    )


class StoredDocument(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    application_id: Mapped[int] = mapped_column(
        ForeignKey("applications.id", ondelete="CASCADE"), index=True
    )
    filename: Mapped[str] = mapped_column(String(255))
    mime_type: Mapped[str] = mapped_column(String(100))
    size_bytes: Mapped[int] = mapped_column(Integer)
    storage_key: Mapped[str] = mapped_column(String(255))   # path relative to UPLOAD_DIR
    # Slot the applicant uploaded it into: id_proof / income_proof /
    # bank_statement / other. The AI verifies it matches the file's content.
    declared_type: Mapped[Optional[str]] = mapped_column(String(30), nullable=True)
    uploaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    application: Mapped[Application] = relationship(back_populates="documents")


class ApplicationEvent(Base):
    __tablename__ = "application_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    application_id: Mapped[int] = mapped_column(
        ForeignKey("applications.id", ondelete="CASCADE"), index=True
    )
    actor_id: Mapped[Optional[int]] = mapped_column(ForeignKey("users.id"), nullable=True)
    action: Mapped[str] = mapped_column(String(50))
    detail: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    application: Mapped[Application] = relationship(back_populates="events")
    actor: Mapped[Optional[User]] = relationship()
