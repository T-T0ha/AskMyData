"""FastAPI dependencies: who is calling, and may they touch this dataset.

``current_user`` is the gate on every non-public endpoint.  ``owned_session``
is the gate on every dataset-scoped one, and it answers 404 rather than 403 for
a dataset owned by somebody else — a 403 confirms that the id exists, which is
half of what an attacker enumerating ids is trying to find out.
"""

from __future__ import annotations

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.auth import service
from app.db.base import session_scope
from app.db.models import IngestionSession, User

#: ``auto_error=False`` so a missing header reaches our own handler and gets
#: the same shape of error body as everything else in the API.
bearer = HTTPBearer(auto_error=False)

UNAUTHENTICATED = HTTPException(
    status_code=401,
    detail="sign in to continue",
    headers={"WWW-Authenticate": "Bearer"},
)


def request_token(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
) -> str | None:
    """The raw bearer token, if the caller sent one."""

    if credentials is not None and credentials.scheme.lower() == "bearer":
        return credentials.credentials
    # EventSource and a plain <a download> cannot set headers; both are on the
    # roadmap for Phase 5/6, so the query fallback is here rather than bolted
    # on later.  It is read-only by convention and never logged.
    token = request.query_params.get("access_token")
    return token or None


def current_user(
    token: str | None = Depends(request_token),
    db: Session = Depends(session_scope),
) -> User:
    """The signed-in account, or 401."""

    if not token:
        raise UNAUTHENTICATED
    user = service.resolve_token(db, token)
    if user is None:
        raise UNAUTHENTICATED
    return user


def optional_user(
    token: str | None = Depends(request_token),
    db: Session = Depends(session_scope),
) -> User | None:
    """The signed-in account if there is one; never raises.

    Used by ``/api/status``, which has to answer before the user signs in
    because the login screen shows whether the backend is reachable at all.
    """

    if not token:
        return None
    return service.resolve_token(db, token)


def owned_session(db: Session, session_id: str, user: User) -> IngestionSession:
    """Load a dataset the caller owns, or 404.

    FR-14: ownership is checked before any metadata or data is read.  Note the
    ordering — the record is fetched, the owner is compared, and nothing about
    the dataset is touched in between.
    """

    record = db.get(IngestionSession, session_id)
    if record is None or record.user_id != user.id:
        raise HTTPException(status_code=404, detail=f"session {session_id} not found")
    return record
