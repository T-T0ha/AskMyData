"""C-4 — the read-only role Phase 5's query execution connects as.

This is the one mechanism in the codebase a SQLite-backed suite cannot
exercise for real: there is no role or ``GRANT`` concept to test without a
live PostgreSQL server. The whole file is skipped unless one is reachable —
``docker compose up -d`` starts the one this repo ships, at localhost:5433.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import create_engine, text

from app.core.config import get_settings
from app.export.naming import schema_name
from app.export.runner import ExportTarget, _grant_readonly

ADMIN_URL = "postgresql+psycopg://semantic:semantic@localhost:5433/semanticlayer"
READONLY_URL = "postgresql+psycopg://semantic_readonly:semantic_readonly@localhost:5433/semanticlayer"


def _reachable(url: str) -> bool:
    try:
        engine = create_engine(url, future=True)
        with engine.connect():
            pass
        engine.dispose()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _reachable(ADMIN_URL),
    reason="no live PostgreSQL reachable at localhost:5433 — `docker compose up -d` starts one",
)


@pytest.fixture
def admin_engine():
    engine = create_engine(ADMIN_URL, future=True)
    yield engine
    engine.dispose()


def test_the_readonly_role_can_select_but_neither_write_nor_drop(admin_engine, monkeypatch):
    # The rest of the suite runs under a session-wide SQLite override (see
    # conftest.py), which would make _grant_readonly's own settings.is_postgres
    # check silently no-op. Point it at this real server for the life of this
    # test only — monkeypatch reverts the env vars afterward, and clearing the
    # cache on both sides of that makes sure every other test still sees the
    # SQLite settings it expects.
    monkeypatch.setenv("DATABASE_URL", ADMIN_URL)
    monkeypatch.setenv("READONLY_DATABASE_URL", READONLY_URL)
    get_settings.cache_clear()
    try:
        _run(admin_engine)
    finally:
        get_settings.cache_clear()


def _run(admin_engine):
    schema = schema_name(uuid.uuid4().hex)

    # The role itself, idempotently — this must work whether or not
    # ops/init-readonly-role.sql already ran against this server.
    with admin_engine.begin() as connection:
        connection.execute(
            text(
                "DO $$ BEGIN "
                "IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'semantic_readonly') THEN "
                "CREATE ROLE semantic_readonly WITH LOGIN PASSWORD 'semantic_readonly' "
                "NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT; "
                "END IF; END $$;"
            )
        )
        connection.execute(text("GRANT CONNECT ON DATABASE semanticlayer TO semantic_readonly"))
        connection.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        connection.execute(text(f'CREATE SCHEMA "{schema}"'))
        connection.execute(text(f'CREATE TABLE "{schema}".widgets (id INT PRIMARY KEY, name TEXT)'))
        connection.execute(text(f'INSERT INTO "{schema}".widgets VALUES (1, \'a\')'))

    # The real grant function — not a reimplementation of it.
    target = ExportTarget(engine=admin_engine, schema=schema, label=schema, dialect="postgresql")
    _grant_readonly(target)

    readonly_engine = create_engine(READONLY_URL, future=True)
    try:
        with readonly_engine.connect() as connection:
            connection.execute(text(f'SET search_path TO "{schema}"'))
            rows = [dict(r) for r in connection.execute(text("SELECT * FROM widgets")).mappings()]
            assert rows == [{"id": 1, "name": "a"}]

        with pytest.raises(Exception, match="permission denied"):
            with readonly_engine.begin() as connection:
                connection.execute(text(f'SET search_path TO "{schema}"'))
                connection.execute(text("INSERT INTO widgets VALUES (2, 'b')"))

        with pytest.raises(Exception, match="permission denied|must be owner"):
            with readonly_engine.begin() as connection:
                connection.execute(text(f'DROP TABLE "{schema}".widgets'))
    finally:
        readonly_engine.dispose()
        with admin_engine.begin() as connection:
            connection.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
