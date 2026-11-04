"""FastAPI application exposing the screening pipeline."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()  # load GEMINI_API_KEY from backend/.env if present

from .gemini_client import is_available  # noqa: E402  (after load_dotenv)
from .pipeline import screen  # noqa: E402
from .schemas import Decision, LoanApplication  # noqa: E402

app = FastAPI(
    title="AI Loan Application Screening Assistant",
    description="Facts are extracted and validated by code; the decision is made "
    "by a deterministic rule engine; the LLM only extracts facts and explains.",
    version="1.0.0",
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


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "llm_available": is_available(),
        "model": os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
    }


@app.post("/screen", response_model=Decision)
def screen_endpoint(application: LoanApplication) -> Decision:
    return screen(application)
