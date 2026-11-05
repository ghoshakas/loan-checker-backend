"""End-to-end tests for accounts, the applicant panel and the admin panel.

Runs against a throwaway SQLite DB and upload dir, with the LLM disabled, so it
needs no API key and no network.

Run:  cd backend && pytest eval/test_platform.py -v
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

_TMP = tempfile.mkdtemp(prefix="loan-test-")
# Set TEST_DATABASE_URL to run against e.g. a throwaway Postgres database.
os.environ["DATABASE_URL"] = os.getenv("TEST_DATABASE_URL") or f"sqlite:///{_TMP}/test.db"
os.environ["UPLOAD_DIR"] = f"{_TMP}/uploads"
os.environ["GEMINI_API_KEY"] = ""          # LLM off (load_dotenv won't override)
os.environ["ADMIN_EMAIL"] = "officer@bank.test"
os.environ["ADMIN_PASSWORD"] = "admin-pass-123"
os.environ["PASSWORD_HASH_SECRET"] = "test-secret"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

PDF = b"%PDF-1.4\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF"
GOOD_APP = {
    "age": 35, "requested_amount": 1_000_000, "loan_purpose": "home",
    "annual_income": 1_200_000, "employment_status": "salaried", "employment_years": 5,
    "existing_monthly_debt": 10_000, "credit_score": 720,
}
# (upload slot, filename, bytes)
FULL_DOCS = [("id_proof", "aadhaar.pdf", PDF), ("income_proof", "payslip.pdf", PDF),
             ("bank_statement", "statement.pdf", PDF)]


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:   # runs lifespan: create tables + seed admin
        yield c


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _register(client, email: str, name: str = "Test User") -> str:
    r = client.post("/auth/register", json={"email": email, "password": "password123",
                                            "full_name": name})
    assert r.status_code == 201, r.text
    return r.json()["token"]


def _admin(client) -> str:
    r = client.post("/auth/login", json={"email": "officer@bank.test",
                                         "password": "admin-pass-123"})
    assert r.status_code == 200, r.text
    assert r.json()["user"]["role"] == "admin"
    return r.json()["token"]


def _multipart(docs):
    files = [("files", (name, data, "application/pdf")) for _, name, data in docs]
    types = [t for t, _, _ in docs]
    return files, types


def _submit(client, token, data=GOOD_APP, docs=FULL_DOCS):
    files, types = _multipart(docs)
    return client.post("/applications", headers=_auth(token),
                       data={"application": json.dumps(data), "document_types": types},
                       files=files)


# --------------------------------------------------------------------------- #
def test_register_login_and_me(client):
    token = _register(client, "asha@example.com", "Asha Rao")
    me = client.get("/auth/me", headers=_auth(token)).json()
    assert me["email"] == "asha@example.com" and me["role"] == "applicant"

    assert client.post("/auth/login", json={"email": "ASHA@example.com",
                                            "password": "password123"}).status_code == 200
    assert client.post("/auth/login", json={"email": "asha@example.com",
                                            "password": "wrong-pass"}).status_code == 401
    # duplicate email
    r = client.post("/auth/register", json={"email": "asha@example.com",
                                            "password": "password123", "full_name": "X"})
    assert r.status_code == 409


def test_weak_password_and_bad_email_rejected(client):
    assert client.post("/auth/register", json={"email": "a@b.co", "password": "short",
                                               "full_name": "A"}).status_code == 400
    assert client.post("/auth/register", json={"email": "not-an-email", "password": "password123",
                                               "full_name": "A"}).status_code == 400


def test_logout_revokes_token(client):
    token = _register(client, "temp@example.com")
    assert client.post("/auth/logout", headers=_auth(token)).status_code == 204
    assert client.get("/auth/me", headers=_auth(token)).status_code == 401


def test_requires_login(client):
    assert client.get("/applications").status_code == 401
    assert client.get("/admin/applications").status_code == 401


def test_applicant_cannot_use_admin_api(client):
    token = _register(client, "nosy@example.com")
    assert client.get("/admin/applications", headers=_auth(token)).status_code == 403


def test_submit_with_document_and_view(client):
    token = _register(client, "ravi@example.com", "Ravi Kumar")
    r = _submit(client, token, docs=[("income_proof", "payslip.pdf", PDF)])
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["status"] == "pending_review"
    assert body["application"]["applicant_name"] == "Ravi Kumar"   # defaulted from account
    assert len(body["documents"]) == 1
    assert body["documents"][0]["declared_type"] == "income_proof"
    # Applicant view must NOT leak internal AI analysis / recommendation.
    assert "screening" not in body and "recommendation" not in body

    mine = client.get("/applications", headers=_auth(token)).json()
    assert [a["id"] for a in mine] == [body["id"]]

    # Owner can download their document.
    doc_id = body["documents"][0]["id"]
    d = client.get(f"/documents/{doc_id}", headers=_auth(token))
    assert d.status_code == 200 and d.content == PDF


def test_upload_validation(client):
    token = _register(client, "files@example.com")
    r = _submit(client, token, docs=[("id_proof", "evil.pdf", b"MZ\x90\x00 not a pdf")])
    assert r.status_code == 400 and "supported file type" in r.json()["detail"]
    too_many = [("other", f"f{i}.pdf", PDF) for i in range(6)]
    assert _submit(client, token, docs=too_many).status_code == 400
    # Every file needs a valid slot type.
    r = client.post("/applications", headers=_auth(token), data={"application": json.dumps(GOOD_APP)},
                    files=[("files", ("x.pdf", PDF, "application/pdf"))])
    assert r.status_code == 400 and "document type" in r.json()["detail"]
    r = client.post("/applications", headers=_auth(token),
                    data={"application": json.dumps(GOOD_APP), "document_types": ["passport_photo"]},
                    files=[("files", ("x.pdf", PDF, "application/pdf"))])
    assert r.status_code == 400 and "Unknown document type" in r.json()["detail"]
    assert client.post("/applications", headers=_auth(token),
                       data={"application": "{not json"}).status_code == 422


def test_applicants_are_isolated(client):
    a = _register(client, "owner@example.com")
    b = _register(client, "other@example.com")
    created = _submit(client, a, docs=[("id_proof", "id.pdf", PDF)]).json()
    assert client.get(f"/applications/{created['id']}", headers=_auth(b)).status_code == 404
    doc_id = created["documents"][0]["id"]
    assert client.get(f"/documents/{doc_id}", headers=_auth(b)).status_code == 404


def test_admin_sees_recommendation_and_decides(client):
    token = _register(client, "meera@example.com", "Meera")
    app_id = _submit(client, token).json()["id"]
    admin = _admin(client)

    queue = client.get("/admin/applications", params={"status": "pending_review"},
                       headers=_auth(admin)).json()
    assert app_id in [a["id"] for a in queue]

    detail = client.get(f"/admin/applications/{app_id}", headers=_auth(admin)).json()
    assert detail["recommendation"] == "APPROVE"
    assert detail["screening"]["decision"] == "APPROVE"
    assert [e["action"] for e in detail["events"]] == ["submitted", "screened"]

    r = client.post(f"/admin/applications/{app_id}/decision", headers=_auth(admin),
                    json={"decision": "approved", "message_to_applicant": "Congratulations!"})
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "approved" and r.json()["decided_by"] == "Loan Officer"

    # The applicant sees the human decision and message.
    mine = client.get(f"/applications/{app_id}", headers=_auth(token)).json()
    assert mine["status"] == "approved" and mine["final_decision"] == "approved"
    assert mine["applicant_message"] == "Congratulations!"

    # Decided applications can't be modified by the applicant.
    r = client.post(f"/applications/{app_id}/documents", headers=_auth(token),
                    data={"note": "extra"})
    assert r.status_code == 409

    # ...nor re-decided or re-screened by the officer: the decision is final.
    r = client.post(f"/admin/applications/{app_id}/decision", headers=_auth(admin),
                    json={"decision": "rejected", "note": "changed my mind"})
    assert r.status_code == 409 and "final" in r.json()["detail"]
    r = client.post(f"/admin/applications/{app_id}/rescreen", headers=_auth(admin))
    assert r.status_code == 409


def test_override_requires_justification(client):
    token = _register(client, "lowscore@example.com")
    app_id = _submit(client, token, data={**GOOD_APP, "credit_score": 540}).json()["id"]
    admin = _admin(client)
    assert client.get(f"/admin/applications/{app_id}",
                      headers=_auth(admin)).json()["recommendation"] == "REJECT"

    r = client.post(f"/admin/applications/{app_id}/decision", headers=_auth(admin),
                    json={"decision": "approved"})
    assert r.status_code == 400 and "differs from the system recommendation" in r.json()["detail"]

    r = client.post(f"/admin/applications/{app_id}/decision", headers=_auth(admin),
                    json={"decision": "approved",
                          "note": "Verified recent credit report shows 700; bureau lag."})
    assert r.status_code == 200
    assert "OVERRIDE" in r.json()["events"][-1]["detail"]


def test_request_info_then_applicant_adds_document(client):
    token = _register(client, "incomplete@example.com")
    data = {k: v for k, v in GOOD_APP.items() if k != "credit_score"}
    app_id = _submit(client, token, data=data).json()["id"]
    admin = _admin(client)
    assert client.get(f"/admin/applications/{app_id}",
                      headers=_auth(admin)).json()["recommendation"] == "INCOMPLETE"

    # Must tell the applicant what is needed.
    assert client.post(f"/admin/applications/{app_id}/decision", headers=_auth(admin),
                       json={"decision": "info_requested"}).status_code == 400
    r = client.post(f"/admin/applications/{app_id}/decision", headers=_auth(admin),
                    json={"decision": "info_requested",
                          "message_to_applicant": "Please upload your credit report."})
    assert r.json()["status"] == "info_requested"

    r = client.post(f"/applications/{app_id}/documents", headers=_auth(token),
                    files=[("files", ("credit.pdf", PDF, "application/pdf"))],
                    data={"note": "My credit score is 760.", "document_types": ["other"]})
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "pending_review"
    detail = client.get(f"/admin/applications/{app_id}", headers=_auth(admin)).json()
    assert "My credit score is 760." in detail["application"]["free_text"]
    assert detail["events"][-1]["action"] == "screened"


def test_admin_stats_and_rescreen(client):
    admin = _admin(client)
    s = client.get("/admin/stats", headers=_auth(admin)).json()
    assert s["total"] >= 1 and set(s["by_status"]) == {
        "pending_review", "info_requested", "approved", "rejected"}
    any_id = client.get("/admin/applications", params={"status": "pending_review"},
                        headers=_auth(admin)).json()[0]["id"]
    r = client.post(f"/admin/applications/{any_id}/rescreen", headers=_auth(admin))
    assert r.status_code == 200 and r.json()["events"][-1]["action"] == "screened"


def test_tick_box_claims_are_ignored_documents_must_be_uploaded(client):
    """Claiming documents without uploading them no longer satisfies rule D1."""
    token = _register(client, "claims@example.com")
    data = {**GOOD_APP, "documents_provided": ["id_proof", "income_proof", "bank_statement"]}
    r = _submit(client, token, data=data, docs=[("id_proof", "id.pdf", PDF)])
    assert r.json()["application"]["documents_provided"] == []
    admin = _admin(client)
    detail = client.get(f"/admin/applications/{r.json()['id']}", headers=_auth(admin)).json()
    assert detail["recommendation"] == "REQUIRES_REVIEW"
    d1 = next(x for x in detail["screening"]["reasons"] if x["rule_id"] == "D1")
    assert "Income proof" in d1["detail"] and "Bank statement" in d1["detail"]
    assert detail["screening"]["facts_used"]["documents_provided"] == ["id_proof"]


def test_missing_slot_can_be_filled_later(client):
    token = _register(client, "later@example.com")
    app_id = _submit(client, token, docs=FULL_DOCS[:2]).json()["id"]
    admin = _admin(client)
    assert client.get(f"/admin/applications/{app_id}",
                      headers=_auth(admin)).json()["recommendation"] == "REQUIRES_REVIEW"
    files, types = _multipart([FULL_DOCS[2]])
    r = client.post(f"/applications/{app_id}/documents", headers=_auth(token),
                    files=files, data={"document_types": types})
    assert r.status_code == 200, r.text
    assert client.get(f"/admin/applications/{app_id}",
                      headers=_auth(admin)).json()["recommendation"] == "APPROVE"
