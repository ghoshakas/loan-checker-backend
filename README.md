# Loan Screening — Backend

FastAPI service for the AI Loan Application Screening Assistant. It stores
applications and documents, has Gemini **read** the uploaded documents and notes,
runs a **deterministic rule engine** to produce a recommendation, adds **advisory
AI risk insights**, and exposes an admin API where a **human loan officer makes
the final decision**.

> The AI never decides. It reads, explains and suggests. The recommendation comes
> from code in `app/rule_engine.py` driven by `app/policy.yaml`, and the final
> approve/reject comes from a person.

What the backend produces for the loan officer (rendered by the frontend):

| Rule results (code) | AI risk insights (advisory) |
|---|---|
| ![Rule results](docs/screenshots/09-reasons.png) | ![AI insights](docs/screenshots/07-ai-insights.png) |

---

## Contents

- [How a screening runs](#how-a-screening-runs)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [Project layout](#project-layout)
- [The rules](#the-rules)
- [AI usage and guardrails](#ai-usage-and-guardrails)
- [API](#api)
- [Database and migrations](#database-and-migrations)
- [Tests and evaluation](#tests-and-evaluation)
- [Security](#security)

---

## How a screening runs

`app/pipeline.py → screen()` is the single code path used by the API, the tests
and the evaluation harness.

```mermaid
flowchart LR
    A[LoanApplication<br/>+ uploaded files] --> B[build_structured_facts<br/>form values]
    B --> C[analyze_documents<br/>AI reads files]
    C --> D[extract_from_free_text<br/>AI reads notes]
    D --> E[compute_derived_facts<br/>EMI · DTI · LTV]
    E --> F[evaluate<br/>rules → RECOMMENDATION]
    F --> G[explain<br/>AI prose]
    G --> H[generate_insights<br/>AI risk read]
    H --> I[Decision JSON<br/>saved on the application]
    classDef ai fill:#2d3b55,stroke:#60a5fa,color:#e2e8f0;
    class C,D,G,H ai;
```

1. **Form facts:** values the applicant typed, tagged `structured`.
2. **Documents:** Gemini reads each file (multimodal, JSON schema). Code then:
   - works out age from the date of birth and annual from monthly income,
   - range-checks every value,
   - fills only fields that are still empty (precedence **form > document > notes**),
   - records mismatches with the form, the applicant's name, or the upload box (rule **D2**).
3. **Notes:** Gemini extracts facts from free text; lowest precedence.
4. **Derived facts:** monthly income, estimated new EMI (11% a year, 60 months),
   debt-to-income, loan-to-value.
5. **Rules:** produce `APPROVE`, `REJECT`, `REQUIRES_REVIEW` or `INCOMPLETE`,
   with a reason for every rule.
6. **Explanation and insights:** generated *after* the recommendation is fixed.

An admin then records the **final decision** (`approved`, `rejected` or
`info_requested`) through `POST /admin/applications/{id}/decision`.

![Audit trail](docs/screenshots/10-audit-trail.png)

---

## Quick start

Requires Python 3.12+ (tested on 3.14). PostgreSQL is optional, since SQLite is
used when `DATABASE_URL` is unset.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env               # then edit it (see below)
uvicorn app.main:app --reload      # http://127.0.0.1:8000
```

- Interactive API docs: <http://127.0.0.1:8000/docs>
- On startup the app applies pending **Alembic migrations** and creates the
  **admin account** from `ADMIN_EMAIL` / `ADMIN_PASSWORD` if it doesn't exist.
- Without `GEMINI_API_KEY` everything still works: documents are stored but not
  read, the explanation uses a template and the insights are skipped.

---

## Configuration

All settings come from `.env` (see [`.env.example`](.env.example)).

| Variable | Required | Purpose |
|---|---|---|
| `DATABASE_URL` | no | e.g. `postgresql+psycopg://postgres:postgres@localhost:5432/loan_check`. Unset = SQLite at `data/app.db` |
| `PASSWORD_HASH_SECRET` | **yes** | Long random secret mixed into password and token hashes. Changing it invalidates all logins. Generate with `python -c "import secrets; print(secrets.token_urlsafe(48))"` |
| `ADMIN_EMAIL`, `ADMIN_PASSWORD`, `ADMIN_NAME` | yes | Loan-officer account created on first start (nobody can self-register as admin) |
| `GEMINI_API_KEY` | no | Enables AI document reading, explanations and insights |
| `GEMINI_MODEL` | no | Default `gemini-2.5-flash` |
| `GEMINI_FALLBACK_MODEL` | no | Used after retries when the main model returns 429/5xx. Empty = no fallback |
| `UPLOAD_DIR` | no | Where files are stored. Default `data/uploads` |
| `TOKEN_TTL_HOURS` | no | Session lifetime, default 72 |
| `AUTO_MIGRATE` | no | `false` = don't migrate on startup; run `alembic upgrade head` yourself |
| `ALLOWED_ORIGINS` | no | Extra comma-separated CORS origins for a deployed frontend (`localhost:5173` is always allowed) |
| `LOG_LEVEL` | no | Default `INFO` |

---

## Project layout

```
backend/
├── app/
│   ├── main.py               FastAPI app, CORS, startup (migrations + admin seed), /health, /screen
│   ├── routes.py             /auth, /applications, /admin, /documents
│   ├── services.py           file storage, screening runs, response building
│   ├── auth.py               PBKDF2 passwords + pepper, HMAC'd bearer tokens, roles
│   ├── db.py / models.py     SQLAlchemy engine, tables, migration runner
│   ├── api_schemas.py        request/response models for the API
│   ├── schemas.py            pipeline contracts + AI output schemas
│   ├── pipeline.py           the screening pipeline
│   ├── document_analyzer.py  upload checks, AI document reading, normalisation, D2 checks
│   ├── fact_extractor.py     form facts, AI notes extraction, merge precedence, range checks
│   ├── rule_engine.py        derived numbers + rules → recommendation
│   ├── exception_detector.py missing fields/documents, data-quality flags
│   ├── explainer.py          AI explanation (template fallback)
│   ├── risk_analyst.py       AI risk insights (advisory)
│   ├── gemini_client.py      Gemini wrapper: retries, fallback model, never raises
│   ├── config.py             loads policy.yaml
│   └── policy.yaml           thresholds (policy as data)
├── migrations/               Alembic env + versions (0001 initial, 0002 document type)
├── eval/                     pytest suites, labelled cases, evaluation harness
├── data/                     SQLite DB + uploaded files (git-ignored)
├── alembic.ini
└── requirements.txt
```

---

## The rules

Thresholds are in [`app/policy.yaml`](app/policy.yaml). Priority:
**INCOMPLETE → REJECT → REQUIRES_REVIEW → APPROVE**.

| Rule | Checks | If it fails |
|---|---|---|
| R0 | Required fields known: age, amount, income, credit score, employment status, years employed, existing debt | INCOMPLETE |
| R1 | Credit score ≥ 650 | < 580 reject, otherwise review |
| R2 | Debt-to-income ≤ 40% | > 50% reject, otherwise review |
| R3 | Age 21–60 | reject |
| R4 | ≥ 2 years employed, or a stable status | review |
| R5 | Amount ≤ 5 × annual income | review |
| R6 | Loan-to-value ≤ 80% (only with collateral) | review |
| D1 | ID proof, income proof and bank statement **uploaded** (each in its own box) | review |
| D2 | Documents agree with the application: figures within 10%, name matches, right box, readable | review |
| Q1 | Plausible data (credit score 300–900, non-zero amount, debt below income) | review |

---

## AI usage and guardrails

| Job | Module | Output | Guardrails |
|---|---|---|---|
| Read documents | `document_analyzer.py` | `DocumentAnalysis` JSON | Read only printed values; document text treated as data; code does the maths; range checks; fills blanks only |
| Extract facts from notes | `fact_extractor.py` | `ExtractedFacts` JSON | `temperature=0`; `null` instead of guessing; lowest precedence |
| Explain | `explainer.py` | 2–4 sentences | Sees only the decision and rule results; may not change it; template fallback |
| Risk insights | `risk_analyst.py` | `RiskInsights` JSON | Runs after the recommendation; no applicant name; ignores protected characteristics; capped lists |

`gemini_client.py` never raises into a request. Temporary errors are retried and
then sent to `GEMINI_FALLBACK_MODEL`. If everything fails, the call returns
`None` and the pipeline degrades gracefully.

![Document findings](docs/screenshots/08-document-findings.png)

---

## API

| Method | Path | Auth | Purpose |
|---|---|---|---|
| GET | `/health` | none | Status and whether the AI is available |
| POST | `/screen` | none | Stateless screening (used by the Sandbox and the eval) |
| POST | `/auth/register` | none | Create an applicant account → token |
| POST | `/auth/login` | none | Log in → token |
| POST | `/auth/logout` · GET `/auth/me` | bearer | End session · current user |
| POST | `/applications` | applicant | Multipart: `application` (JSON string), `files[]`, `document_types[]` (one per file: `id_proof`, `income_proof`, `bank_statement`, `other`) |
| GET | `/applications` · `/applications/{id}` | applicant | Own applications (status and officer message only, no AI analysis) |
| POST | `/applications/{id}/documents` | applicant | Add files and/or a `note` to an open application; re-screens it |
| GET | `/documents/{id}` | owner or admin | Download a stored file |
| GET | `/admin/applications?status=&recommendation=` | admin | Review queue |
| GET | `/admin/applications/{id}` | admin | Full detail: screening, documents, audit trail |
| POST | `/admin/applications/{id}/rescreen` | admin | Re-run the AI and the rules |
| POST | `/admin/applications/{id}/decision` | admin | `{"decision": "approved" \| "rejected" \| "info_requested", "note", "message_to_applicant"}` |
| GET | `/admin/stats` | admin | Counts by status and recommendation |

Decision rules enforced by the server:
- Going **against the recommendation** (approving a non-APPROVE, or rejecting an
  APPROVE) requires an internal `note`.
- `info_requested` requires a `message_to_applicant`.
- Decided applications can't be changed by the applicant.

Example submission:
```bash
curl -X POST localhost:8000/applications -H "Authorization: Bearer $TOKEN" \
  -F 'application={"age":35,"requested_amount":1000000,"annual_income":1200000,"employment_status":"salaried","employment_years":5,"existing_monthly_debt":10000,"credit_score":720}' \
  -F files=@id.pdf      -F document_types=id_proof \
  -F files=@payslip.pdf -F document_types=income_proof \
  -F files=@bank.pdf    -F document_types=bank_statement
```

---

## Database and migrations

Tables: `users`, `auth_tokens`, `applications` (submitted data, screening
snapshot, recommendation, final decision), `documents` (file metadata and upload
box) and `application_events` (audit trail). Files live on disk under `UPLOAD_DIR`.

```bash
alembic upgrade head                            # apply migrations
alembic current                                 # current revision
alembic check                                   # models and database in sync?
alembic revision --autogenerate -m "add field"  # after editing app/models.py
alembic downgrade -1                            # roll back one step
```

Alembic reads the same `DATABASE_URL` as the app.

---

## Tests and evaluation

```bash
pytest eval/ -v            # no API key or network needed
python eval/run_eval.py    # metrics over 25 labelled screening cases
```

| Suite | Covers |
|---|---|
| `test_rule_engine.py` | Each rule, threshold boundaries, determinism |
| `test_documents.py` | Upload validation, DOB → age, merge precedence, D2 mismatches, wrong box, unreadable files (AI stubbed) |
| `test_platform.py` | End-to-end API on a temporary SQLite DB: accounts, roles, isolation, uploads, decisions, overrides, request-info round trip. Set `TEST_DATABASE_URL` to run it on Postgres |
| `test_gemini_client.py` | Retries, fallback model, employment-status normalisation (fake client) |

Fictional test documents for manual testing are in `../sample_documents/` in the
parent project.

---

## Security

- PBKDF2-SHA256 (310,000 iterations) with a per-user salt and an HMAC pepper (`PASSWORD_HASH_SECRET`).
- Session tokens are random; only their HMAC is stored, and they expire.
- Role checks on every endpoint; applicants can reach only their own data.
- Uploads: the type is detected from the file's content, with size and count
  limits, random storage names and path-traversal protection.
- `.env` and `data/` are git-ignored, so the database and uploaded documents
  never get committed.
