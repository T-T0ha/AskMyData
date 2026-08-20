"""Database engine, session factory and pgvector bootstrap."""

from __future__ import annotations

import logging
from typing import Any, Iterator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


_engine: Any = None
_session_factory: Any = None
_vector_ready: bool | None = None


def get_engine():
    global _engine
    if _engine is None:
        settings = get_settings()
        kwargs: dict[str, Any] = {"pool_pre_ping": True, "future": True}
        if settings.database_url.startswith("sqlite"):
            kwargs["connect_args"] = {"check_same_thread": False}
        _engine = create_engine(settings.database_url, **kwargs)
    return _engine


def get_session_factory():
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(bind=get_engine(), expire_on_commit=False, future=True)
    return _session_factory


def session_scope() -> Iterator[Session]:
    """FastAPI dependency: one transaction per request."""

    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def pgvector_available() -> bool:
    """True when the connected PostgreSQL has the ``vector`` extension enabled.

    Called once at startup; the result decides whether embeddings are stored in
    a real ``vector(384)`` column (with an HNSW index) or as JSON.
    """

    global _vector_ready
    if _vector_ready is not None:
        return _vector_ready
    settings = get_settings()
    if not settings.is_postgres:
        _vector_ready = False
        return _vector_ready
    try:
        with get_engine().begin() as connection:
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        _vector_ready = True
    except Exception as exc:  # pragma: no cover - depends on environment
        logger.warning("pgvector unavailable: %s", exc)
        _vector_ready = False
    return _vector_ready


def init_db() -> dict[str, Any]:
    """Create tables and report which backend features are live."""

    settings = get_settings()
    from app.db import models  # noqa: F401  (registers the mappers)

    status: dict[str, Any] = {
        "database_url": settings.database_url.split("@")[-1],
        "dialect": get_engine().dialect.name,
        "pgvector": False,
        "connected": False,
    }
    try:
        status["pgvector"] = pgvector_available()
        Base.metadata.create_all(bind=get_engine())
        status["added_columns"] = _add_missing_columns()
        status["pending_columns"] = pending_schema_changes()
        status["connected"] = True
        if status["pending_columns"]:
            logger.error(
                "the database is missing column(s) the code needs: %s — "
                "queries against those tables will fail until they are added",
                ", ".join(status["pending_columns"]),
            )
        if status["pgvector"]:
            _create_vector_index()
    except Exception as exc:
        logger.warning("database initialisation failed: %s", exc)
        status["error"] = str(exc)
    return status


#: Columns added to existing tables after the first release, as
#: ``table -> {column: DDL type}``.  ``create_all`` only creates tables that do
#: not exist yet, so without this a developer whose database predates a new
#: column gets an "unknown column" error on the next query instead of an
#: upgrade.  Additive only: this never drops or retypes anything, which is the
#: whole class of migration a project without Alembic can safely automate.
_ADDED_COLUMNS: dict[str, dict[str, str]] = {
    "sheet_records": {
        "source_kind": "VARCHAR(40)",
        "native_schema": "JSON",
        "key_analysis": "JSON",
    },
    # Ownership, added when authentication landed.  Declared without the
    # REFERENCES clause on purpose: SQLite cannot add a foreign key to an
    # existing table, and the constraint is already carried by the model for
    # every database created from scratch.  Rows that predate this column stay
    # NULL and are visible to nobody.
    "ingestion_sessions": {
        "user_id": "VARCHAR(32)",
    },
}


def pending_schema_changes() -> list[str]:
    """Columns the model expects that the connected database does not have.

    ``init_db`` applies these at startup, so anything still listed later means
    the running process is a version ahead of its database — every query
    against the affected table fails, and it fails in a way that looks like a
    problem with whatever the user just uploaded.  Naming the drift is what
    lets the API say so instead of blaming the file.
    """

    from sqlalchemy import inspect as sqlalchemy_inspect

    try:
        inspector = sqlalchemy_inspect(get_engine())
        existing_tables = set(inspector.get_table_names())
    except Exception:  # pragma: no cover - unreachable database, reported elsewhere
        return []

    missing: list[str] = []
    for table, columns in _ADDED_COLUMNS.items():
        if table not in existing_tables:
            continue  # create_all just made it, with every column
        present = {c["name"] for c in inspector.get_columns(table)}
        missing.extend(f"{table}.{column}" for column in columns if column not in present)
    return missing


def _add_missing_columns() -> list[str]:
    """Bring an existing database up to the current model.  Returns what it added."""

    engine = get_engine()
    added: list[str] = []

    for qualified in pending_schema_changes():
        table, column = qualified.split(".", 1)
        ddl_type = _ADDED_COLUMNS[table][column]
        try:
            with engine.begin() as connection:
                connection.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {ddl_type}"))
            added.append(qualified)
        except Exception as exc:  # pragma: no cover - depends on the database
            # Not a warning: until this succeeds, every read of the table 500s.
            logger.error("could not add %s: %s", qualified, exc)
    if added:
        logger.info("added missing column(s): %s", ", ".join(added))
    return added


def _create_vector_index() -> None:
    """HNSW index over the column-name embeddings (Phase 0 / Phase 4)."""

    statement = text(
        "CREATE INDEX IF NOT EXISTS idx_column_semantics_embedding "
        "ON column_semantics USING hnsw (embedding vector_cosine_ops)"
    )
    try:
        with get_engine().begin() as connection:
            connection.execute(statement)
    except Exception as exc:  # pragma: no cover - depends on environment
        logger.warning("could not create HNSW index: %s", exc)


def reset_engine() -> None:
    """Test hook."""

    global _engine, _session_factory, _vector_ready
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _session_factory = None
    _vector_ready = None
