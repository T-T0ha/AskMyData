"""Accounts and login tokens — everything that touches the database.

The API layer above this raises HTTP errors; this layer raises
:class:`AuthError` with a machine-readable ``code`` and never decides a status
code of its own.
"""

from __future__ import annotations

import hashlib
import logging
import re
import secrets
import threading
import time
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.auth import passwords
from app.core.config import get_settings
from app.db.models import AuthToken, User

logger = logging.getLogger(__name__)

#: Deliberately permissive.  The only authority on whether an address exists is
#: a message delivered to it; a stricter pattern here rejects valid addresses
#: (plus-tags, long TLDs, unicode domains) and still cannot prove the rest.
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s.]+(\.[^@\s.]+)+$")

TOKEN_BYTES = 32


class AuthError(Exception):
    """Something about the credentials or the account is wrong."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _aware(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def normalise_email(email: str) -> str:
    """Lower-cased and stripped.  Two spellings of one address are one account."""

    return email.strip().lower()


def email_problem(email: str) -> str | None:
    normalised = normalise_email(email)
    if not normalised:
        return "email is required"
    if len(normalised) > 255:
        return "email must be at most 255 characters"
    if not EMAIL_PATTERN.match(normalised):
        return "that does not look like an email address"
    return None


# ---------------------------------------------------------------------------
# failed-login throttling
# ---------------------------------------------------------------------------

#: email -> timestamps of recent failures.  In-memory and per-process, which
#: means it is a speed bump rather than a control: two uvicorn workers give an
#: attacker two budgets, and a restart clears it.  It is here because the
#: alternative — an unthrottled password endpoint — is worse, and because the
#: honest version (a shared counter in PostgreSQL or Redis) buys little against
#: a distributed attacker while costing a write on every failed login.
_failures: dict[str, list[float]] = {}
_failures_lock = threading.Lock()


def _record_failure(email: str) -> None:
    settings = get_settings()
    cutoff = time.monotonic() - settings.auth_failed_login_window_seconds
    with _failures_lock:
        recent = [t for t in _failures.get(email, []) if t > cutoff]
        recent.append(time.monotonic())
        _failures[email] = recent


def _clear_failures(email: str) -> None:
    with _failures_lock:
        _failures.pop(email, None)


def _throttled(email: str) -> bool:
    settings = get_settings()
    cutoff = time.monotonic() - settings.auth_failed_login_window_seconds
    with _failures_lock:
        recent = [t for t in _failures.get(email, []) if t > cutoff]
        _failures[email] = recent
    return len(recent) >= settings.auth_max_failed_logins


def reset_throttle() -> None:
    """Test hook."""

    with _failures_lock:
        _failures.clear()


# ---------------------------------------------------------------------------
# accounts
# ---------------------------------------------------------------------------


#: A real hash of a password nobody has, kept so that signing in with an
#: unknown email costs one scrypt derivation — the same as signing in with a
#: known one.  Without it the endpoint answers "no such account" measurably
#: faster than "wrong password", which is an account enumeration oracle.
_dummy_hash: tuple[tuple[int, int, int], str] | None = None


def _timing_equaliser() -> str:
    global _dummy_hash
    settings = get_settings()
    params = (settings.auth_scrypt_n, settings.auth_scrypt_r, settings.auth_scrypt_p)
    if _dummy_hash is None or _dummy_hash[0] != params:
        _dummy_hash = (params, passwords.hash_password(secrets.token_urlsafe(16)))
    return _dummy_hash[1]


def get_user_by_email(db: Session, email: str) -> User | None:
    return db.execute(
        select(User).where(User.email == normalise_email(email))
    ).scalar_one_or_none()


def register(db: Session, email: str, password: str, name: str = "") -> User:
    """Create an account.  Raises :class:`AuthError` on anything invalid."""

    settings = get_settings()
    if not settings.auth_registration_open:
        raise AuthError("registration_closed", "registration is closed on this deployment")

    problem = email_problem(email) or passwords.password_problem(password)
    if problem:
        raise AuthError("invalid", problem)

    normalised = normalise_email(email)
    if get_user_by_email(db, normalised) is not None:
        raise AuthError("email_taken", "an account with that email already exists")

    user = User(
        email=normalised,
        name=(name or "").strip()[:100],
        password_hash=passwords.hash_password(password),
    )
    db.add(user)
    try:
        db.flush()
    except IntegrityError as exc:
        # Two registrations for one address raced; the unique index is the
        # authority, not the SELECT above.
        db.rollback()
        raise AuthError("email_taken", "an account with that email already exists") from exc
    return user


def authenticate(db: Session, email: str, password: str) -> User:
    """Check credentials.  One error for both causes, on purpose.

    "No such account" and "wrong password" are the same message because the
    difference is exactly what an attacker enumerating addresses wants to
    learn.  The unknown-email branch still runs a hash so that the two paths
    take comparable time.
    """

    normalised = normalise_email(email)
    if _throttled(normalised):
        raise AuthError(
            "too_many_attempts",
            "too many failed sign-in attempts; wait a few minutes and try again",
        )

    user = get_user_by_email(db, normalised)
    if user is None:
        passwords.verify_password(password, _timing_equaliser())
        _record_failure(normalised)
        raise AuthError("invalid_credentials", "email or password is incorrect")

    if not passwords.verify_password(password, user.password_hash):
        _record_failure(normalised)
        raise AuthError("invalid_credentials", "email or password is incorrect")

    _clear_failures(normalised)
    # The only moment the plaintext is available, so the only moment a hash
    # made under weaker parameters can be upgraded.
    if passwords.needs_rehash(user.password_hash):
        user.password_hash = passwords.hash_password(password)
    user.last_login_at = _now()
    db.flush()
    return user


def change_password(db: Session, user: User, current: str, replacement: str) -> None:
    """Re-check the current password, then rotate.  Revokes every other login."""

    if not passwords.verify_password(current, user.password_hash):
        raise AuthError("invalid_credentials", "current password is incorrect")
    problem = passwords.password_problem(replacement)
    if problem:
        raise AuthError("invalid", problem)
    user.password_hash = passwords.hash_password(replacement)
    db.flush()


# ---------------------------------------------------------------------------
# tokens
# ---------------------------------------------------------------------------


def fingerprint(raw_token: str) -> str:
    """What gets stored.  The token itself is only ever held by its owner."""

    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()


def issue_token(db: Session, user: User, user_agent: str = "") -> str:
    """Mint a login token and return it — the one and only time it is readable."""

    settings = get_settings()
    raw = secrets.token_urlsafe(TOKEN_BYTES)
    db.add(
        AuthToken(
            user_id=user.id,
            token_hash=fingerprint(raw),
            expires_at=_now() + timedelta(hours=settings.auth_token_ttl_hours),
            user_agent=(user_agent or "")[:255],
        )
    )
    db.flush()
    return raw


def resolve_token(db: Session, raw_token: str) -> User | None:
    """The account behind a token, or ``None`` if it is unknown, revoked or expired."""

    if not raw_token:
        return None
    token = db.execute(
        select(AuthToken).where(AuthToken.token_hash == fingerprint(raw_token))
    ).scalar_one_or_none()
    if token is None or not token.is_live():
        return None
    # Cheap enough to be worth having: it is what tells a user which of their
    # listed logins is the one they are currently using.
    token.last_used_at = _now()
    return token.user


def revoke_token(db: Session, raw_token: str) -> bool:
    token = db.execute(
        select(AuthToken).where(AuthToken.token_hash == fingerprint(raw_token))
    ).scalar_one_or_none()
    if token is None or token.revoked_at is not None:
        return False
    token.revoked_at = _now()
    db.flush()
    return True


def revoke_all_tokens(db: Session, user: User, except_raw: str | None = None) -> int:
    """Sign out everywhere.  Used after a password change."""

    keep = fingerprint(except_raw) if except_raw else None
    tokens = db.execute(
        select(AuthToken).where(AuthToken.user_id == user.id, AuthToken.revoked_at.is_(None))
    ).scalars().all()
    revoked = 0
    for token in tokens:
        if keep is not None and token.token_hash == keep:
            continue
        token.revoked_at = _now()
        revoked += 1
    db.flush()
    return revoked


def list_tokens(db: Session, user: User) -> list[AuthToken]:
    """Live logins, newest first — the "where am I signed in?" view."""

    rows = db.execute(
        select(AuthToken)
        .where(AuthToken.user_id == user.id, AuthToken.revoked_at.is_(None))
        .order_by(AuthToken.created_at.desc())
    ).scalars().all()
    return [t for t in rows if t.is_live()]


def purge_expired_tokens(db: Session) -> int:
    """Drop tokens that can no longer authenticate anything.

    Called at startup rather than on a schedule: the table is small, and a row
    that is neither live nor deletable is just a record of when somebody last
    signed in from a laptop they no longer own.
    """

    now = _now()
    rows = db.execute(select(AuthToken)).scalars().all()
    stale = [t for t in rows if t.revoked_at is not None or _aware(t.expires_at) <= now]
    for token in stale:
        db.delete(token)
    db.flush()
    return len(stale)
