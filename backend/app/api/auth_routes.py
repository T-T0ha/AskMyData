

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.auth import passwords, service
from app.auth.deps import current_user, request_token
from app.cleaning.store import drop_store
from app.core.config import get_settings
from app.export.runner import drop_export
from app.db.base import session_scope
from app.db.models import User

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/auth", tags=["auth"])


# ---------------------------------------------------------------------------
# request bodies
# ---------------------------------------------------------------------------


class RegisterRequest(BaseModel):
    email: str = Field(min_length=3, max_length=255)
    password: str = Field(min_length=1, max_length=1024)
    name: str = Field(default="", max_length=100)


class LoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=255)
    password: str = Field(min_length=1, max_length=1024)


class PasswordChange(BaseModel):
    current_password: str = Field(min_length=1, max_length=1024)
    new_password: str = Field(min_length=1, max_length=1024)


class AccountDeletion(BaseModel):
    """Deleting an account destroys every dataset in it, so it asks again."""

    password: str = Field(min_length=1, max_length=1024)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

#: Which failures are the caller's fault in which way.  Kept in one place so a
#: new AuthError code cannot quietly become a 500.
_STATUS_FOR_CODE = {
    "invalid": 422,
    "email_taken": 409,
    "invalid_credentials": 401,
    "too_many_attempts": 429,
    "registration_closed": 403,
}


def _http(exc: service.AuthError) -> HTTPException:
    status = _STATUS_FOR_CODE.get(exc.code, 400)
    headers = {"WWW-Authenticate": "Bearer"} if status == 401 else None
    return HTTPException(status_code=status, detail=exc.message, headers=headers)


def _issue(db: Session, user: User, request: Request) -> dict[str, Any]:
    token = service.issue_token(db, user, user_agent=request.headers.get("user-agent", ""))
    return {
        "user": user.to_dict(),
        "access_token": token,
        "token_type": "bearer",
        "expires_in": get_settings().auth_token_ttl_hours * 3600,
    }


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------


@router.get("/config")
def auth_config() -> dict[str, Any]:
    """What the sign-in screen needs before anyone has signed in."""

    settings = get_settings()
    return {
        "registration_open": settings.auth_registration_open,
        "min_password_length": settings.auth_min_password_length,
    }


@router.post("/register", status_code=201)
def register(
    body: RegisterRequest,
    request: Request,
    db: Session = Depends(session_scope),
) -> dict[str, Any]:
    """Create an account and sign it in — registering then having to log in
    separately is a step with no purpose."""

    try:
        user = service.register(db, body.email, body.password, body.name)
    except service.AuthError as exc:
        raise _http(exc) from exc
    logger.info("registered account %s", user.email)
    return _issue(db, user, request)


@router.post("/login")
def login(
    body: LoginRequest,
    request: Request,
    db: Session = Depends(session_scope),
) -> dict[str, Any]:
    try:
        user = service.authenticate(db, body.email, body.password)
    except service.AuthError as exc:
        raise _http(exc) from exc
    return _issue(db, user, request)


@router.post("/logout", status_code=204)
def logout(
    token: str | None = Depends(request_token),
    _user: User = Depends(current_user),
    db: Session = Depends(session_scope),
) -> Response:
    """Revoke the token this request arrived with.  Other devices stay signed in."""

    if token:
        service.revoke_token(db, token)
    return Response(status_code=204)


@router.get("/me")
def me(user: User = Depends(current_user)) -> dict[str, Any]:
    return {"user": user.to_dict()}


@router.get("/sessions")
def list_logins(
    user: User = Depends(current_user),
    token: str | None = Depends(request_token),
    db: Session = Depends(session_scope),
) -> dict[str, Any]:
    """Every live login for this account, so the user can spot one they do not
    recognise.  "sessions" here means browser sessions, not datasets."""

    current = service.fingerprint(token) if token else None
    return {
        "logins": [
            {**t.to_dict(), "current": t.token_hash == current}
            for t in service.list_tokens(db, user)
        ]
    }


@router.delete("/sessions", status_code=200)
def revoke_other_logins(
    user: User = Depends(current_user),
    token: str | None = Depends(request_token),
    db: Session = Depends(session_scope),
) -> dict[str, Any]:
    """Sign out everywhere except here."""

    return {"revoked": service.revoke_all_tokens(db, user, except_raw=token)}


@router.post("/password", status_code=200)
def change_password(
    body: PasswordChange,
    user: User = Depends(current_user),
    token: str | None = Depends(request_token),
    db: Session = Depends(session_scope),
) -> dict[str, Any]:
    """Rotate the password and sign every other device out.

    A password change that leaves old logins working does not accomplish the
    thing people change their password for.
    """

    try:
        service.change_password(db, user, body.current_password, body.new_password)
    except service.AuthError as exc:
        raise _http(exc) from exc
    return {"revoked": service.revoke_all_tokens(db, user, except_raw=token)}


@router.post("/delete", status_code=200)
def delete_account(
    body: AccountDeletion,
    user: User = Depends(current_user),
    db: Session = Depends(session_scope),
) -> dict[str, Any]:
    """FR-15: remove the account and every dataset it owns.

    The database cascade takes the metadata; the working tables, the uploaded
    files and each dataset's materialized schema live outside it and are removed
    here, before the row goes, so that a failure halfway leaves an account that
    can be deleted again rather than orphaned data nothing points to.

    The materialized schema matters most of the three: it holds the rows
    themselves, in a database that stays running, and a ``ds_…`` left behind by
    a deleted account is readable by anything that can reach the server.
    """

    # Verified directly rather than through ``authenticate``: this is a
    # confirmation prompt, not a sign-in, and it should neither refresh
    # ``last_login_at`` nor spend the account's failed-login budget.
    if not passwords.verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=401, detail="password is incorrect")

    dataset_ids = [record.id for record in user.sessions]
    for dataset_id in dataset_ids:
        drop_store(dataset_id)
        drop_export(dataset_id)
    db.delete(user)
    db.flush()
    logger.info("deleted account %s and %d dataset(s)", user.email, len(dataset_ids))
    return {"deleted_datasets": len(dataset_ids)}
