"""LangGraph checkpointer selection.

PostgreSQL is the documented target: it is already in the stack, and a
Postgres-backed checkpoint means a cleaning session survives a browser close
or a server restart.  SQLite is used automatically when the app is configured
against a non-PostgreSQL database (local development, tests), and an in-memory
saver is the last resort so nothing hard-fails.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.core.config import get_settings

logger = logging.getLogger(__name__)

_checkpointer: Any = None
_context: Any = None


def _postgres_dsn(url: str) -> str:
    """SQLAlchemy URL -> libpq DSN (psycopg does not want the driver suffix)."""

    return url.replace("postgresql+psycopg://", "postgresql://").replace(
        "postgresql+psycopg2://", "postgresql://"
    )


def get_checkpointer() -> Any:
    """Process-wide checkpointer, created on first use."""

    global _checkpointer, _context
    if _checkpointer is not None:
        return _checkpointer

    settings = get_settings()
    if settings.is_postgres:
        try:
            from langgraph.checkpoint.postgres import PostgresSaver  # noqa: PLC0415

            _context = PostgresSaver.from_conn_string(_postgres_dsn(settings.database_url))
            _checkpointer = _context.__enter__()
            _checkpointer.setup()
            logger.info("LangGraph checkpointing: PostgreSQL")
            return _checkpointer
        except Exception as exc:
            logger.warning("PostgreSQL checkpointer unavailable (%s); falling back", exc)

    try:
        from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: PLC0415

        path = Path(settings.upload_dir).parent / "checkpoints.sqlite"
        path.parent.mkdir(parents=True, exist_ok=True)
        _context = SqliteSaver.from_conn_string(str(path))
        _checkpointer = _context.__enter__()
        _checkpointer.setup()
        logger.info("LangGraph checkpointing: SQLite at %s", path)
        return _checkpointer
    except Exception as exc:  # pragma: no cover - depends on environment
        logger.warning("SQLite checkpointer unavailable (%s); using in-memory", exc)

    from langgraph.checkpoint.memory import InMemorySaver  # noqa: PLC0415

    _checkpointer = InMemorySaver()
    return _checkpointer


def checkpointer_backend() -> str:
    return type(get_checkpointer()).__name__


def reset_checkpointer() -> None:
    """Test hook — closes the saver and forces re-creation."""

    global _checkpointer, _context
    if _context is not None:
        try:
            _context.__exit__(None, None, None)
        except Exception:  # pragma: no cover - best effort
            pass
    _checkpointer = None
    _context = None
