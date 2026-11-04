"""HTTP routes for accounts, the applicant panel and the admin panel.

    /auth/*                 register, login, logout, me
    /applications/*         applicant: submit, list own, view own, add documents
    /admin/*                admin: queue, detail, re-screen, FINAL decision, stats
    /documents/{id}         download (owner or admin)

The automated pipeline only produces a RECOMMENDATION. The final approve /
reject / request-info decision is made by an admin via POST
/admin/applications/{id}/decision and is recorded in the audit trail.
"""

from __future__ import annotations

from typing import Annotated, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import FileResponse
from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .api_schemas import (
    AdminApplicationOut,
    AdminApplicationSummary,
    ApplicantApplicationOut,
    AuthOut,
    DecisionIn,
    LoginIn,
    RegisterIn,
    StatsOut,
    UserOut,
)
from .auth import (
    get_current_user,
    hash_password,
    issue_token,
    normalize_email,
    require_admin,
    require_applicant,
    revoke_token,
    validate_credentials,
    verify_password,
)
from .db import get_db, utcnow
from .models import (
    ALL_STATUSES,
    ROLE_ADMIN,
    ROLE_APPLICANT,
    STATUS_APPROVED,
    STATUS_INFO,
    STATUS_PENDING,
    STATUS_REJECTED,
    Application,
    StoredDocument,
    User,
)
from .schemas import LoanApplication
from .services import (
    check_document_capacity,
    document_path,
    log_event,
    read_uploads,
    run_screening,
    store_documents,
    to_admin_out,
    to_admin_summary,
    to_applicant_out,
)

DB = Annotated[Session, Depends(get_db)]
Files = Annotated[Optional[list[UploadFile]], File()]
# Upload slot of each file, same order as `files`.
DocTypes = Annotated[Optional[list[str]], Form(description="id_proof | income_proof | bank_statement | other, one per file")]

# --------------------------------------------------------------------------- #
# Auth
# --------------------------------------------------------------------------- #
auth_router = APIRouter(prefix="/auth", tags=["auth"])


@auth_router.post("/register", response_model=AuthOut, status_code=201)
def register(body: RegisterIn, db: DB) -> AuthOut:
    email = normalize_email(body.email)
    validate_credentials(email, body.password)
    if db.scalar(select(User).where(User.email == email)):
        raise HTTPException(status.HTTP_409_CONFLICT, "An account with this email already exists.")
    user = User(email=email, full_name=body.full_name.strip(),
                password_hash=hash_password(body.password), role=ROLE_APPLICANT)
    db.add(user)
    db.commit()
    return AuthOut(token=issue_token(db, user), user=UserOut.model_validate(user))


@auth_router.post("/login", response_model=AuthOut)
def login(body: LoginIn, db: DB) -> AuthOut:
    user = db.scalar(select(User).where(User.email == normalize_email(body.email)))
    if user is None or not verify_password(body.password, user.password_hash):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Incorrect email or password.")
    return AuthOut(token=issue_token(db, user), user=UserOut.model_validate(user))


@auth_router.post("/logout", status_code=204)
def logout(request: Request, db: DB, _: User = Depends(get_current_user)) -> None:
    token = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
    revoke_token(db, token)


@auth_router.get("/me", response_model=UserOut)
def me(user: User = Depends(get_current_user)) -> UserOut:
    return UserOut.model_validate(user)


# --------------------------------------------------------------------------- #
# Applicant panel
# --------------------------------------------------------------------------- #
applicant_router = APIRouter(prefix="/applications", tags=["applicant"])


def _own_application(db: Session, user: User, app_id: int) -> Application:
    row = db.get(Application, app_id)
    if row is None or row.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Application not found.")
    return row


@applicant_router.post("", response_model=ApplicantApplicationOut, status_code=201)
def submit_application(
    db: DB,
    application: Annotated[str, Form(description="LoanApplication as a JSON string")],
    files: Files = None,
    document_types: DocTypes = None,
    user: User = Depends(require_applicant),
) -> ApplicantApplicationOut:
    try:
        loan = LoanApplication.model_validate_json(application)
    except ValidationError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT,
                            exc.errors(include_url=False, include_context=False)) from exc
    if not loan.applicant_name:
        loan.applicant_name = user.full_name
    loan.documents_provided = []   # derived from the uploaded files, never claimed
    docs = read_uploads(files, document_types)

    row = Application(user_id=user.id, status=STATUS_PENDING,
                      application_data=loan.model_dump(mode="json"))
    db.add(row)
    db.flush()
    store_documents(db, row, docs)
    log_event(db, row, user, "submitted", f"{len(docs)} document(s) uploaded")
    run_screening(db, row, actor=None)
    db.commit()
    db.refresh(row)
    return to_applicant_out(row)


@applicant_router.get("", response_model=list[ApplicantApplicationOut])
def my_applications(db: DB, user: User = Depends(require_applicant)) -> list[ApplicantApplicationOut]:
    rows = db.scalars(
        select(Application).where(Application.user_id == user.id)
        .order_by(Application.created_at.desc())
    ).all()
    return [to_applicant_out(r) for r in rows]


@applicant_router.get("/{app_id}", response_model=ApplicantApplicationOut)
def my_application(app_id: int, db: DB, user: User = Depends(require_applicant)) -> ApplicantApplicationOut:
    return to_applicant_out(_own_application(db, user, app_id))


@applicant_router.post("/{app_id}/documents", response_model=ApplicantApplicationOut)
def add_documents(
    app_id: int,
    db: DB,
    files: Files = None,
    document_types: DocTypes = None,
    note: Annotated[Optional[str], Form(max_length=2000)] = None,
    user: User = Depends(require_applicant),
) -> ApplicantApplicationOut:
    """Add documents / a note to an open application (e.g. after the officer
    requested more information). Re-runs screening and returns it to the queue."""
    row = _own_application(db, user, app_id)
    if row.status not in (STATUS_PENDING, STATUS_INFO):
        raise HTTPException(status.HTTP_409_CONFLICT,
                            "This application has already been decided.")
    docs = read_uploads(files, document_types)
    if not docs and not (note and note.strip()):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Add at least one document or a note.")
    check_document_capacity(row, docs)

    if note and note.strip():
        data = dict(row.application_data)
        existing = (data.get("free_text") or "").strip()
        data["free_text"] = (existing + "\n\n" if existing else "") + note.strip()
        row.application_data = data
    store_documents(db, row, docs)
    was_info = row.status == STATUS_INFO
    row.status = STATUS_PENDING
    log_event(db, row, user, "updated",
              f"{len(docs)} document(s) added" + ("; note added" if note and note.strip() else "")
              + ("; returned to review queue" if was_info else ""))
    run_screening(db, row, actor=None)
    db.commit()
    db.refresh(row)
    return to_applicant_out(row)


# --------------------------------------------------------------------------- #
# Documents (owner or admin)
# --------------------------------------------------------------------------- #
documents_router = APIRouter(prefix="/documents", tags=["documents"])


@documents_router.get("/{doc_id}")
def download_document(doc_id: int, db: DB, user: User = Depends(get_current_user)) -> FileResponse:
    doc = db.get(StoredDocument, doc_id)
    if doc is None or (user.role != ROLE_ADMIN and doc.application.user_id != user.id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found.")
    path = document_path(doc)
    if not path.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document file is missing.")
    return FileResponse(path, media_type=doc.mime_type, filename=doc.filename,
                        headers={"X-Content-Type-Options": "nosniff"})


# --------------------------------------------------------------------------- #
# Admin panel
# --------------------------------------------------------------------------- #
admin_router = APIRouter(prefix="/admin", tags=["admin"])


def _get_application(db: Session, app_id: int) -> Application:
    row = db.get(Application, app_id)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Application not found.")
    return row


@admin_router.get("/applications", response_model=list[AdminApplicationSummary])
def list_applications(
    db: DB,
    status_filter: Annotated[Optional[str], Query(alias="status")] = None,
    recommendation: Optional[str] = None,
    _: User = Depends(require_admin),
) -> list[AdminApplicationSummary]:
    q = select(Application).order_by(Application.created_at.desc())
    if status_filter:
        if status_filter not in ALL_STATUSES:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Unknown status '{status_filter}'.")
        q = q.where(Application.status == status_filter)
    if recommendation:
        q = q.where(Application.recommendation == recommendation)
    return [to_admin_summary(r) for r in db.scalars(q).all()]


@admin_router.get("/applications/{app_id}", response_model=AdminApplicationOut)
def get_application(app_id: int, db: DB, _: User = Depends(require_admin)) -> AdminApplicationOut:
    return to_admin_out(_get_application(db, app_id))


@admin_router.post("/applications/{app_id}/rescreen", response_model=AdminApplicationOut)
def rescreen_application(app_id: int, db: DB, admin: User = Depends(require_admin)) -> AdminApplicationOut:
    row = _get_application(db, app_id)
    run_screening(db, row, actor=admin)
    db.commit()
    db.refresh(row)
    return to_admin_out(row)


_DECISION_TO_STATUS = {
    "approved": STATUS_APPROVED,
    "rejected": STATUS_REJECTED,
    "info_requested": STATUS_INFO,
}


@admin_router.post("/applications/{app_id}/decision", response_model=AdminApplicationOut)
def decide_application(app_id: int, body: DecisionIn, db: DB,
                       admin: User = Depends(require_admin)) -> AdminApplicationOut:
    """The FINAL human decision. Overriding the system recommendation requires
    a written justification, which is kept in the audit trail."""
    row = _get_application(db, app_id)
    note = (body.note or "").strip() or None
    message = (body.message_to_applicant or "").strip() or None
    rec = row.recommendation

    overrides = (
        (body.decision == "approved" and rec is not None and rec != "APPROVE")
        or (body.decision == "rejected" and rec == "APPROVE")
    )
    if overrides and not note:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"Your decision differs from the system recommendation ({rec}). "
            "Add an internal note explaining why.",
        )
    if body.decision == "info_requested" and not message:
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            "Tell the applicant what information is needed.")

    previous = row.final_decision
    row.status = _DECISION_TO_STATUS[body.decision]
    row.final_decision = body.decision if body.decision != "info_requested" else None
    row.decision_note = note
    row.applicant_message = message
    row.decided_by_id = admin.id
    row.decided_at = utcnow()

    detail = f"Decision: {body.decision} (system recommendation: {rec or 'n/a'})"
    if overrides:
        detail += " — OVERRIDE"
    if previous and previous != row.final_decision:
        detail += f"; changed from {previous}"
    if note:
        detail += f". Note: {note}"
    log_event(db, row, admin, "decision", detail)
    db.commit()
    db.refresh(row)
    return to_admin_out(row)


@admin_router.get("/stats", response_model=StatsOut)
def stats(db: DB, _: User = Depends(require_admin)) -> StatsOut:
    by_status = dict(db.execute(
        select(Application.status, func.count()).group_by(Application.status)).all())
    by_rec = dict(db.execute(
        select(Application.recommendation, func.count())
        .where(Application.recommendation.is_not(None))
        .group_by(Application.recommendation)).all())
    return StatsOut(total=sum(by_status.values()),
                    by_status={s: by_status.get(s, 0) for s in ALL_STATUSES},
                    by_recommendation=by_rec)
