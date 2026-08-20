"""SemanticLayer / AskMyData — FastAPI application.

Implements authentication and per-account dataset ownership, Phase 0
(ingestion and structural repair), Phase 1 (field semantic understanding),
Phase 2 (co-planned data cleaning) and Phase 3 (relationship detection and
interactive validation).  Phases 4–6 — PostgreSQL export, NL query and
dashboard — build on the validated semantic layer this application produces.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.auth_routes import router as auth_router
from app.api.routes import graph, router
from app.auth import service as auth_service
from app.core.config import get_settings
from app.db.base import get_session_factory, init_db, pending_schema_changes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SemanticLayer",
    description=(
        "Human-in-the-loop semantic data platform. Messy spreadsheets in, "
        "a validated semantic layer out."
    ),
    version="0.3.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(router)


@app.on_event("startup")
def on_startup() -> None:
    settings = get_settings()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    status = init_db()
    if not status.get("connected"):
        logger.warning(
            "Database is not reachable (%s). Start PostgreSQL with "
            "`docker compose up -d db`, or set DATABASE_URL to a local SQLite file.",
            status.get("error"),
        )
    else:
        logger.info(
            "Database ready (dialect=%s, pgvector=%s)",
            status["dialect"],
            status["pgvector"],
        )
        try:
            with get_session_factory()() as db:
                removed = auth_service.purge_expired_tokens(db)
                db.commit()
            if removed:
                logger.info("purged %d expired or revoked login token(s)", removed)
        except Exception as exc:
            logger.warning("could not purge expired tokens: %s", exc)

        # Build the LangGraph checkpointer now, while no request holds an open
        # transaction: PostgresSaver.setup() runs CREATE INDEX CONCURRENTLY,
        # which blocks until every other open transaction in the database
        # finishes. Triggering it lazily from inside a request (which already
        # holds its own open transaction via session_scope) deadlocks forever.
        try:
            graph()
            logger.info("Cleaning graph checkpointer ready")
        except Exception as exc:
            logger.warning("cleaning graph checkpointer failed to initialise: %s", exc)


@app.get("/health")
def health() -> dict[str, Any]:
    """``ok`` only when the database can actually serve the current code.

    A process that has been running since before a model changed answers every
    request with an "unknown column" error; reporting ``ok`` regardless sends
    whoever is debugging that straight past the cause.
    """

    pending = pending_schema_changes()
    if pending:
        return {
            "status": "degraded",
            "pending_schema_changes": pending,
            "detail": "restart the backend to apply the missing column(s)",
        }
    return {"status": "ok"}
