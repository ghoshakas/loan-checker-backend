"""Glue between the database, file storage and the screening pipeline."""

from __future__ import annotations

import logging
import threading
import uuid
from pathlib import Path
from typing import Optional

from fastapi import HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from .api_schemas import (
    AdminApplicationOut,
    AdminApplicationSummary,
    ApplicantApplicationOut,
    DocumentOut,
    EventOut,
    UserOut,
)
from .db import new_session, upload_dir, utcnow
from .document_analyzer import (
    MAX_FILE_BYTES,
    MAX_FILES,
    MAX_TOTAL_BYTES,
    UploadedDocument,
    UploadError,
    provided_document_types,
    validate_uploads,
)
from .models import STATUS_INFO, STATUS_PENDING, Application, ApplicationEvent, StoredDocument, User
from .pipeline import screen
from .schemas import Decision, LoanApplication

logger = logging.getLogger("loan_screening.services")

_EXT = {"application/pdf": ".pdf", "image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}


# --------------------------------------------------------------------------- #
# Uploads & storage
# --------------------------------------------------------------------------- #
def read_uploads(
    files: Optional[list[UploadFile]],
    document_types: Optional[list[str]],
) -> list[UploadedDocument]:
    """Read multipart files (size-capped) and validate them, or raise 400.

    `document_types[i]` is the upload slot of `files[i]` (id_proof,
    income_proof, bank_statement or other).
    """
    items: list[tuple[str, bytes]] = []
    for f in files or []:
        if not f.filename and not f.size:
            continue  # empty file input
        data = f.file.read(MAX_FILE_BYTES + 1)
        items.append((f.filename or "document", data))
    types = list(document_types or [])
    try:
        return validate_uploads(items, types if items or types else None)
    except UploadError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc


def _safe_path(storage_key: str) -> Path:
    root = upload_dir().resolve()
    path = (root / storage_key).resolve()
    if root not in path.parents:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid document path.")
    return path


def store_documents(db: Session, app_row: Application, docs: list[UploadedDocument]) -> None:
    for doc in docs:
        key = f"{app_row.id}/{uuid.uuid4().hex}{_EXT.get(doc.mime_type, '')}"
        path = _safe_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(doc.data)
        db.add(StoredDocument(application_id=app_row.id, filename=doc.filename,
                              mime_type=doc.mime_type, size_bytes=len(doc.data),
                              storage_key=key, declared_type=doc.declared_type))
    db.flush()
    db.refresh(app_row)


def document_path(doc: StoredDocument) -> Path:
    return _safe_path(doc.storage_key)


def check_document_capacity(app_row: Application, new_docs: list[UploadedDocument]) -> None:
    count = len(app_row.documents) + len(new_docs)
    size = sum(d.size_bytes for d in app_row.documents) + sum(len(d.data) for d in new_docs)
    if count > MAX_FILES:
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            f"An application can hold at most {MAX_FILES} documents.")
    if size > MAX_TOTAL_BYTES:
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            f"Total document size per application is limited to "
                            f"{MAX_TOTAL_BYTES // (1024 * 1024)} MB.")


# --------------------------------------------------------------------------- #
# Screening
# --------------------------------------------------------------------------- #
def log_event(db: Session, app_row: Application, actor: Optional[User],
              action: str, detail: Optional[str] = None) -> None:
    db.add(ApplicationEvent(application_id=app_row.id,
                            actor_id=actor.id if actor else None,
                            action=action, detail=detail))


def run_screening(db: Session, app_row: Application, actor: Optional[User]) -> None:
    """Run the full pipeline over the stored application + documents and save
    the snapshot. Never raises: a failure is logged and recorded as an event."""
    try:
        application = LoanApplication.model_validate(app_row.application_data)
        docs = [
            UploadedDocument(filename=d.filename, mime_type=d.mime_type,
                             data=document_path(d).read_bytes(),
                             declared_type=d.declared_type)
            for d in app_row.documents
        ]
        # A required document counts only if a file was uploaded into its slot.
        application.documents_provided = provided_document_types(docs)
        decision = screen(application, docs)
    except Exception:
        logger.exception("Screening failed for application %s", app_row.id)
        app_row.screened_at = utcnow()   # finished (unsuccessfully): no longer "in progress"
        log_event(db, app_row, actor, "screening_failed",
                  "Automated screening failed; an admin can re-run it.")
        return

    app_row.screening = decision.model_dump(mode="json")
    app_row.recommendation = decision.decision.value
    app_row.risk_level = decision.insights.risk_level.value if decision.insights else None
    app_row.screened_at = utcnow()
    detail = f"Recommendation: {decision.decision.value}"
    if decision.insights:
        detail += f"; AI risk level: {decision.insights.risk_level.value}"
    log_event(db, app_row, actor, "screened", detail)


# --------------------------------------------------------------------------- #
# Background screening
# --------------------------------------------------------------------------- #
# Screening makes several LLM calls and can take minutes, longer than a proxy
# will wait. So requests only save the data and mark the application as queued
# (screened_at = None); the screening itself runs after the response is sent.
_locks: dict[int, threading.Lock] = {}
_locks_guard = threading.Lock()


def _lock_for(app_id: int) -> threading.Lock:
    with _locks_guard:
        return _locks.setdefault(app_id, threading.Lock())


def queue_screening(app_row: Application) -> None:
    """Mark the application as waiting for a (re-)screening."""
    app_row.screened_at = None


def screening_in_progress(app_row: Application) -> bool:
    return app_row.screened_at is None


def screen_in_background(app_id: int, actor_id: Optional[int] = None) -> None:
    """Screen one application in its own DB session. Runs after the response.

    Screenings of the same application run one at a time, so a run started
    after new documents were added always sees them and finishes last."""
    with _lock_for(app_id):
        db = new_session()
        try:
            row = db.get(Application, app_id)
            if row is None:
                return
            actor = db.get(User, actor_id) if actor_id else None
            run_screening(db, row, actor=actor)
            db.commit()
        except Exception:
            logger.exception("Background screening failed for application %s", app_id)
            db.rollback()
        finally:
            db.close()


def resume_unfinished_screenings() -> None:
    """On startup: screen applications whose screening was cut off (e.g. by a
    restart or deploy), in a background thread."""
    db = new_session()
    try:
        # Only open applications: a decided one keeps the screening it was decided on.
        ids = list(db.scalars(select(Application.id).where(
            Application.screened_at.is_(None),
            Application.status.in_((STATUS_PENDING, STATUS_INFO)))))
    finally:
        db.close()
    if not ids:
        return
    logger.info("Resuming screening for %d application(s): %s", len(ids), ids)

    def run() -> None:
        for app_id in ids:
            screen_in_background(app_id)

    threading.Thread(target=run, name="resume-screening", daemon=True).start()


# --------------------------------------------------------------------------- #
# Serialisation
# --------------------------------------------------------------------------- #
def _documents(app_row: Application) -> list[DocumentOut]:
    return [DocumentOut.model_validate(d) for d in app_row.documents]


def to_applicant_out(app_row: Application) -> ApplicantApplicationOut:
    return ApplicantApplicationOut(
        id=app_row.id,
        status=app_row.status,
        created_at=app_row.created_at,
        updated_at=app_row.updated_at,
        application=LoanApplication.model_validate(app_row.application_data),
        documents=_documents(app_row),
        final_decision=app_row.final_decision,
        applicant_message=app_row.applicant_message,
        decided_at=app_row.decided_at,
        screening_in_progress=screening_in_progress(app_row),
    )


def to_admin_summary(app_row: Application) -> AdminApplicationSummary:
    data = app_row.application_data or {}
    return AdminApplicationSummary(
        id=app_row.id,
        applicant_name=data.get("applicant_name") or app_row.user.full_name,
        applicant_email=app_row.user.email,
        requested_amount=data.get("requested_amount"),
        loan_purpose=data.get("loan_purpose"),
        status=app_row.status,
        recommendation=app_row.recommendation,
        risk_level=app_row.risk_level,
        document_count=len(app_row.documents),
        created_at=app_row.created_at,
        screening_in_progress=screening_in_progress(app_row),
    )


def to_admin_out(app_row: Application) -> AdminApplicationOut:
    base = to_applicant_out(app_row)
    return AdminApplicationOut(
        **base.model_dump(),
        applicant=UserOut.model_validate(app_row.user),
        screening=Decision.model_validate(app_row.screening) if app_row.screening else None,
        recommendation=app_row.recommendation,
        risk_level=app_row.risk_level,
        screened_at=app_row.screened_at,
        decision_note=app_row.decision_note,
        decided_by=app_row.decided_by.full_name if app_row.decided_by else None,
        events=[
            EventOut(action=e.action, detail=e.detail,
                     actor_name=e.actor.full_name if e.actor else "System",
                     created_at=e.created_at)
            for e in app_row.events
        ],
    )
