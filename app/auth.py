"""Authentication & authorisation.

* Passwords: PBKDF2-HMAC-SHA256 with a per-user random salt (stdlib only),
  peppered with PASSWORD_HASH_SECRET from .env: the password is first
  HMAC'd with the secret, so a stolen database alone is not enough to run an
  offline password-cracking attack.
* Sessions: random bearer token returned at login; only an HMAC of it (keyed
  with the same secret) is stored, so a leaked database does not leak usable
  tokens.
* Changing PASSWORD_HASH_SECRET invalidates all passwords and sessions.
* Roles: "applicant" (self-registered) and "admin" (seeded from ADMIN_EMAIL /
  ADMIN_PASSWORD in .env — nobody can self-register as admin).
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import re
import secrets
from datetime import timedelta, timezone

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from .db import get_db, utcnow
from .models import ROLE_ADMIN, ROLE_APPLICANT, AuthToken, User

logger = logging.getLogger("loan_screening.auth")

_PBKDF2_ITERATIONS = 310_000
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MIN_PASSWORD_LENGTH = 8


def _token_ttl() -> timedelta:
    return timedelta(hours=float(os.getenv("TOKEN_TTL_HOURS", "72")))


_DEV_SECRET = "dev-only-insecure-secret"
_warned_secret = False


def _secret() -> bytes:
    """PASSWORD_HASH_SECRET from .env (falls back to a dev value with a warning)."""
    global _warned_secret
    value = os.getenv("PASSWORD_HASH_SECRET", "")
    if not value:
        if not _warned_secret:
            logger.warning("PASSWORD_HASH_SECRET is not set; using an insecure "
                           "development secret. Set it in backend/.env.")
            _warned_secret = True
        value = _DEV_SECRET
    return value.encode()


def _pepper(password: str) -> bytes:
    return hmac.new(_secret(), password.encode(), hashlib.sha256).digest()


# --------------------------------------------------------------------------- #
# Passwords
# --------------------------------------------------------------------------- #
def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", _pepper(password), salt, _PBKDF2_ITERATIONS)
    return f"pbkdf2_sha256${_PBKDF2_ITERATIONS}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, iterations, salt_hex, digest_hex = stored.split("$")
        digest = hashlib.pbkdf2_hmac(
            "sha256", _pepper(password), bytes.fromhex(salt_hex), int(iterations)
        )
        return hmac.compare_digest(digest.hex(), digest_hex)
    except (ValueError, TypeError):
        return False


def normalize_email(email: str) -> str:
    return email.strip().lower()


def validate_credentials(email: str, password: str) -> None:
    if not _EMAIL_RE.match(email):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Enter a valid email address.")
    if len(password) < MIN_PASSWORD_LENGTH:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"Password must be at least {MIN_PASSWORD_LENGTH} characters.",
        )


# --------------------------------------------------------------------------- #
# Tokens
# --------------------------------------------------------------------------- #
def _hash_token(token: str) -> str:
    return hmac.new(_secret(), token.encode(), hashlib.sha256).hexdigest()


def issue_token(db: Session, user: User) -> str:
    token = secrets.token_urlsafe(32)
    db.add(AuthToken(token_hash=_hash_token(token), user_id=user.id,
                     expires_at=utcnow() + _token_ttl()))
    # Opportunistic cleanup of expired sessions.
    db.execute(delete(AuthToken).where(AuthToken.expires_at < utcnow()))
    db.commit()
    return token


def revoke_token(db: Session, token: str) -> None:
    db.execute(delete(AuthToken).where(AuthToken.token_hash == _hash_token(token)))
    db.commit()


_bearer = HTTPBearer(auto_error=False)


def get_current_user(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
    db: Session = Depends(get_db),
) -> User:
    unauthorized = HTTPException(
        status.HTTP_401_UNAUTHORIZED, "Not signed in or session expired.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if creds is None or not creds.credentials:
        raise unauthorized
    row = db.get(AuthToken, _hash_token(creds.credentials))
    if row is None:
        raise unauthorized
    expires = row.expires_at
    if expires.tzinfo is None:  # SQLite returns naive datetimes
        expires = expires.replace(tzinfo=timezone.utc)
    if expires < utcnow():
        db.delete(row)
        db.commit()
        raise unauthorized
    return row.user


def require_admin(user: User = Depends(get_current_user)) -> User:
    if user.role != ROLE_ADMIN:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Admin access required.")
    return user


def require_applicant(user: User = Depends(get_current_user)) -> User:
    if user.role != ROLE_APPLICANT:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Applicant access required.")
    return user


# --------------------------------------------------------------------------- #
# Admin seeding
# --------------------------------------------------------------------------- #
def seed_admin(db: Session) -> None:
    """Create the admin account from ADMIN_EMAIL / ADMIN_PASSWORD if missing.

    An existing account is promoted to admin but its password is not changed.
    """
    email = normalize_email(os.getenv("ADMIN_EMAIL", ""))
    password = os.getenv("ADMIN_PASSWORD", "")
    if not email or not password:
        if db.scalar(select(User).where(User.role == ROLE_ADMIN)) is None:
            logger.warning(
                "No admin account exists. Set ADMIN_EMAIL and ADMIN_PASSWORD in "
                "backend/.env and restart to create one."
            )
        return
    user = db.scalar(select(User).where(User.email == email))
    if user is None:
        db.add(User(email=email, full_name=os.getenv("ADMIN_NAME", "Loan Officer"),
                    password_hash=hash_password(password), role=ROLE_ADMIN))
        logger.info("Created admin account %s", email)
    elif user.role != ROLE_ADMIN:
        user.role = ROLE_ADMIN
    db.commit()


__all__ = [
    "ROLE_ADMIN", "ROLE_APPLICANT", "get_current_user", "hash_password",
    "issue_token", "normalize_email", "require_admin", "require_applicant",
    "revoke_token", "seed_admin", "validate_credentials", "verify_password",
]
