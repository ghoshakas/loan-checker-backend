"""Alembic environment.

Uses the SAME database URL as the app (DATABASE_URL from backend/.env, SQLite
fallback) and the app's SQLAlchemy models as the migration target, so
`alembic revision --autogenerate` diffs the models against the live schema.

Run from backend/:
    alembic upgrade head                              # apply migrations
    alembic revision --autogenerate -m "add x"        # after changing models.py
"""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from dotenv import load_dotenv
from sqlalchemy import engine_from_config, pool

load_dotenv()

from app import models  # noqa: E402,F401  (registers all tables on Base.metadata)
from app.db import Base, database_url  # noqa: E402

config = context.config
config.set_main_option("sqlalchemy.url", database_url().replace("%", "%%"))

# Skip Alembic's logging setup when migrations run from inside the app
# (app.db.run_migrations), so it doesn't override the app's logging config.
if config.config_file_name is not None and not config.attributes.get("skip_logging"):
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _is_sqlite(url: str) -> bool:
    return url.startswith("sqlite")


def run_migrations_offline() -> None:
    """Emit SQL to stdout instead of executing it (alembic upgrade head --sql)."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        render_as_batch=_is_sqlite(url),
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = config.attributes.get("connection")
    if connectable is None:
        connectable = engine_from_config(
            config.get_section(config.config_ini_section, {}),
            prefix="sqlalchemy.",
            poolclass=pool.NullPool,
        )
        with connectable.connect() as connection:
            _run(connection)
    else:
        _run(connectable)


def _run(connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        # SQLite can't ALTER most things; batch mode recreates tables instead.
        render_as_batch=connection.dialect.name == "sqlite",
    )
    with context.begin_transaction():
        context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
