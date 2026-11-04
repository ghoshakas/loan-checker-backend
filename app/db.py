"""Database setup (SQLAlchemy 2.x).

Default: a local SQLite file at backend/data/app.db — zero setup. Point
DATABASE_URL at Postgres (e.g. postgresql+psycopg://user:pass@host/db) for a
real deployment. Uploaded files are stored on disk under UPLOAD_DIR; the DB only
holds their metadata.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

BACKEND_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BACKEND_DIR / "data"

# First migration (matches the schema the app used to create with create_all).
INITIAL_REVISION = "0001"


def database_url() -> str:
    url = os.getenv("DATABASE_URL")
    if url:
        return url
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{DATA_DIR / 'app.db'}"


def upload_dir() -> Path:
    path = Path(os.getenv("UPLOAD_DIR") or DATA_DIR / "uploads")
    path.mkdir(parents=True, exist_ok=True)
    return path


class Base(DeclarativeBase):
    pass


_engine = None
_SessionLocal: sessionmaker | None = None


def get_engine():
    global _engine, _SessionLocal
    if _engine is None:
        url = database_url()
        connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
        _engine = create_engine(url, connect_args=connect_args)
        _SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)
    return _engine


def _alembic_config():
    from alembic.config import Config

    cfg = Config(str(BACKEND_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND_DIR / "migrations"))
    cfg.attributes["skip_logging"] = True
    return cfg


def run_migrations() -> None:
    """Bring the schema up to date (`alembic upgrade head`).

    A database created before migrations existed (tables present, no
    alembic_version) is stamped at the initial revision first, then upgraded.
    """
    from alembic import command
    from sqlalchemy import inspect

    cfg = _alembic_config()
    tables = set(inspect(get_engine()).get_table_names())
    if "users" in tables and "alembic_version" not in tables:
        logging.getLogger("loan_screening.db").warning(
            "Existing schema without migration history; stamping it at %s.", INITIAL_REVISION)
        command.stamp(cfg, INITIAL_REVISION)
    command.upgrade(cfg, "head")


def init_db() -> None:
    """Called on app startup. Set AUTO_MIGRATE=false to manage migrations
    manually with `alembic upgrade head`."""
    if os.getenv("AUTO_MIGRATE", "true").lower() in ("0", "false", "no"):
        return
    run_migrations()


def get_db() -> Iterator[Session]:
    """FastAPI dependency: one session per request."""
    get_engine()
    assert _SessionLocal is not None
    db = _SessionLocal()
    try:
        yield db
    finally:
        db.close()


def utcnow() -> datetime:
    return datetime.now(timezone.utc)
