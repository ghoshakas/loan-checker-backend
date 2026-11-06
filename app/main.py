"""FastAPI application: screening pipeline + applicant and admin panels."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()  # load GEMINI_API_KEY, ADMIN_* etc. from backend/.env if present

from .auth import seed_admin  # noqa: E402  (after load_dotenv)
from .db import get_db, init_db  # noqa: E402
from .gemini_client import active_providers, is_available  # noqa: E402
from .pipeline import screen  # noqa: E402
from .services import resume_unfinished_screenings  # noqa: E402
from .routes import admin_router, applicant_router, auth_router, documents_router  # noqa: E402
from .schemas import Decision, LoanApplication  # noqa: E402

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    db = next(get_db())
    try:
        seed_admin(db)
    finally:
        db.close()
    resume_unfinished_screenings()
    yield


app = FastAPI(
    title="AI Loan Application Screening Assistant",
    description="Applicants submit applications and documents; the AI reads the "
    "documents and notes into facts; a deterministic rule engine produces a "
    "recommendation; the AI adds advisory insights; a human admin makes the "
    "final decision.",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS: local Vite dev server by default. For a deployed frontend that calls the
# backend directly (VITE_BACKEND_URL), add its origin via ALLOWED_ORIGINS,
# comma-separated, e.g. ALLOWED_ORIGINS=https://my-app.vercel.app
_DEFAULT_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]
_extra_origins = [
    o.strip().rstrip("/")
    for o in os.getenv("ALLOWED_ORIGINS", "").split(",")
    if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_DEFAULT_ORIGINS + _extra_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(applicant_router)
app.include_router(admin_router)
app.include_router(documents_router)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "llm_available": is_available(),
        "model": os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        "llm_providers": active_providers(),   # tried in this order, e.g. ["gemini", "meta"]
    }


@app.post("/screen", response_model=Decision, tags=["sandbox"])
def screen_endpoint(application: LoanApplication) -> Decision:
    """Stateless screening (no account, nothing stored). Used by the eval
    harness and the admin 'Sandbox' tab."""
    return screen(application)
