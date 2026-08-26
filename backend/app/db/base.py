"""Database engine, session factory and pgvector bootstrap."""

from __future__ import annotations

import logging
from typing import Any, Callable, Iterator

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


def _embedding_ddl() -> str:
    """``vector(n)`` where pgvector is live, JSON everywhere else.

    The same choice :class:`~app.db.models.EmbeddingVector` makes for a table
    being created; made again here because ``ALTER TABLE`` is spelled by hand
    and a JSON column on PostgreSQL would be an embedding no index can reach.
    """

    if pgvector_available():
        return f"vector({get_settings().embedding_dim})"
    return "JSON"


def _timestamp_ddl() -> str:
    """PostgreSQL keeps the offset; SQLite has no timestamp type to qualify."""

    return "TIMESTAMP WITH TIME ZONE" if get_settings().is_postgres else "TIMESTAMP"


#: Columns added to existing tables after the first release, as
#: ``table -> {column: DDL type}``, where the type is either literal SQL or a
#: callable that spells it for the connected dialect.  ``create_all`` only creates tables that do
#: not exist yet, so without this a developer whose database predates a new
#: column gets an "unknown column" error on the next query instead of an
#: upgrade.  Additive only: this never drops or retypes anything, which is the
#: whole class of migration a project without Alembic can safely automate.
_ADDED_COLUMNS: dict[str, dict[str, "str | Callable[[], str]"]] = {
    "sheet_records": {
        "source_kind": "VARCHAR(40)",
        "native_schema": "JSON",
        "key_analysis": "JSON",
        # Table semantic profiling, added with Phase 4.
        "semantic_profile": "JSON",
        # The table's one-line description and its embedding, added when the
        # control plane was aligned with the ER model.
        "table_description": "TEXT",
        "llm_table_description": "TEXT",
        "table_embedding": _embedding_ddl,
    },
    # Key and reference roles, projected onto the column semantics so that the
    # control plane — not only the exported copy — knows which column is an
    # identifier and what it points at.
    "column_semantics": {
        "is_primary_key": "BOOLEAN DEFAULT FALSE",
        "is_foreign_key": "BOOLEAN DEFAULT FALSE",
        "references_table": "VARCHAR(255)",
        "references_column": "VARCHAR(255)",
    },
    "semantic_layer_exports": {
        "semantic_version": "INTEGER DEFAULT 0",
    },
    # Ownership, added when authentication landed.  Declared without the
    # REFERENCES clause on purpose: SQLite cannot add a foreign key to an
    # existing table, and the constraint is already carried by the model for
    # every database created from scratch.  Rows that predate this column stay
    # NULL and are visible to nobody.
    "ingestion_sessions": {
        "user_id": "VARCHAR(32)",
        # Materialization and versioning, added with the semantic layer.
        "db_schema_name": "VARCHAR(100)",
        "semantic_version": "INTEGER DEFAULT 0",
        "enriched_at": _timestamp_ddl,
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
        declared = _ADDED_COLUMNS[table][column]
        ddl_type = declared() if callable(declared) else declared
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
    """HNSW indexes over the two embedded columns of the control plane.

    ``column_semantics.embedding`` is what a question is matched against
    column by column; ``sheet_records.table_embedding`` is what narrows it to a
    handful of tables first.  Both are best-effort: without the index the same
    query still answers, by scanning.
    """

    statements = (
        "CREATE INDEX IF NOT EXISTS idx_column_semantics_embedding "
        "ON column_semantics USING hnsw (embedding vector_cosine_ops)",
        "CREATE INDEX IF NOT EXISTS idx_sheet_records_table_embedding "
        "ON sheet_records USING hnsw (table_embedding vector_cosine_ops)",
    )
    for statement in statements:
        try:
            with get_engine().begin() as connection:
                connection.execute(text(statement))
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
