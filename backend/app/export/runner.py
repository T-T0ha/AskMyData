"""Creating the exported database, and reporting exactly what was created.

The target is a namespace of its own, never the application's own tables:

* on PostgreSQL, a schema named ``ds_<session id>`` in the same database, so
  the exported data sits beside the metadata without being able to collide
  with it, and so Phase 5 can query it with an ordinary connection;
* on SQLite — development and the test suite — a separate file per session,
  which is what a schema *is* in a database that has none.

Re-exporting replaces the previous export.  That is a destructive operation on
a namespace derived from a user-supplied session id, so it is guarded twice:
the name is rebuilt from the id by :func:`app.export.naming.schema_name` rather
than taken from any request, and it is checked against
:func:`~app.export.naming.is_managed_schema` immediately before the statement
that drops it.  Nothing outside ``ds_…`` is reachable from here.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
from sqlalchemy import MetaData, Table, create_engine, select, text
from sqlalchemy.engine import Engine

from app.core.config import get_settings
from app.export import metadata as semantic_metadata
from app.export.naming import is_managed_schema, schema_name
from app.export.schema import ExportPlan, TableSpec, build_metadata, render_ddl, type_summary
from app.export.types import coerce_series, to_records

logger = logging.getLogger(__name__)

#: Rows per ``INSERT``.  Large enough that the round trips stop mattering,
#: small enough that one statement's parameters stay well inside PostgreSQL's
#: 65535-parameter limit for any realistic column count.
CHUNK_ROWS = 1000


@dataclass(slots=True)
class ExportTarget:
    """Where an export goes, and how to reach it."""

    engine: Engine
    schema: str | None
    #: Human-readable, and deliberately free of any credential: the schema
    #: name on PostgreSQL, the file name on SQLite.
    label: str
    dialect: str
    #: True when the engine was created for this export and must be disposed.
    owned: bool = False
    path: Path | None = None

    def dispose(self) -> None:
        if self.owned:
            self.engine.dispose()


@dataclass(slots=True)
class ExportReport:
    """What the export did, in the words the user will read."""

    session_id: str
    target: str
    dialect: str
    tables: list[dict[str, Any]] = field(default_factory=list)
    foreign_keys: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    skipped: list[dict[str, str]] = field(default_factory=list)
    widened_types: list[dict[str, str]] = field(default_factory=list)
    renamed: list[dict[str, str]] = field(default_factory=list)
    ddl: str = ""
    row_count: int = 0
    #: Columns described in ``sem_metadata``, and how many of those carry a
    #: vector.  They differ when no embedding model could be loaded, which
    #: costs Phase 5 its ranking but not its answers.
    metadata_rows: int = 0
    embedded_rows: int = 0
    vector_index: bool = False
    duration_seconds: float = 0.0

    @property
    def table_count(self) -> int:
        return len(self.tables)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "target": self.target,
            "dialect": self.dialect,
            "table_count": self.table_count,
            "row_count": self.row_count,
            "tables": self.tables,
            "foreign_keys": self.foreign_keys,
            "enforced_foreign_keys": sum(1 for fk in self.foreign_keys if fk["enforced"]),
            "warnings": self.warnings,
            "skipped": self.skipped,
            "widened_types": self.widened_types,
            "renamed": self.renamed,
            "metadata_rows": self.metadata_rows,
            "embedded_rows": self.embedded_rows,
            "vector_index": self.vector_index,
            "ddl": self.ddl,
            "duration_seconds": round(self.duration_seconds, 3),
        }


# ---------------------------------------------------------------------------
# target resolution
# ---------------------------------------------------------------------------


def export_root() -> Path:
    """``var/exports`` — beside ``var/sessions`` and ``var/uploads``."""

    root = get_settings().upload_dir.parent / "exports"
    root.mkdir(parents=True, exist_ok=True)
    return root


def resolve_target(session_id: str) -> ExportTarget:
    """The namespace this session's export owns.

    ``schema_name`` is what makes this safe: it rebuilds the name from the
    session id, keeping only hex characters, so no request can steer it at
    ``public`` or at a path outside ``var/exports``.
    """

    name = schema_name(session_id)
    settings = get_settings()

    if settings.is_postgres:
        from app.db.base import get_engine  # local import: avoids a cycle

        return ExportTarget(
            engine=get_engine(), schema=name, label=name, dialect="postgresql"
        )

    path = export_root() / f"{name}.db"
    engine = create_engine(f"sqlite:///{path}", future=True)
    return ExportTarget(
        engine=engine,
        schema=None,
        label=path.name,
        dialect="sqlite",
        owned=True,
        path=path,
    )


def reset_target(target: ExportTarget) -> None:
    """Erase whatever the last export left, and nothing else.

    The name is checked here rather than only at construction because this is
    the function that runs ``DROP``: a guard is worth most immediately before
    the statement it guards.
    """

    if target.schema is not None:
        if not is_managed_schema(target.schema):
            raise ValueError(f"refusing to drop schema {target.schema!r}")
        with target.engine.begin() as connection:
            # Identifier interpolation, which is why the guard above is not
            # optional: PostgreSQL has no parameter form for a schema name.
            connection.execute(text(f'DROP SCHEMA IF EXISTS "{target.schema}" CASCADE'))
            connection.execute(text(f'CREATE SCHEMA "{target.schema}"'))
        return

    if target.path is not None:
        if not is_managed_schema(target.path.stem):
            raise ValueError(f"refusing to delete {target.path}")
        target.engine.dispose()
        target.path.unlink(missing_ok=True)


def drop_export(session_id: str) -> None:
    """Remove a session's exported database — used when the session is deleted."""

    try:
        target = resolve_target(session_id)
    except ValueError:
        return
    try:
        if target.schema is not None:
            if is_managed_schema(target.schema):
                with target.engine.begin() as connection:
                    connection.execute(
                        text(f'DROP SCHEMA IF EXISTS "{target.schema}" CASCADE')
                    )
        elif target.path is not None and is_managed_schema(target.path.stem):
            target.engine.dispose()
            target.path.unlink(missing_ok=True)
    except Exception as exc:  # pragma: no cover - depends on the server
        logger.warning("could not drop the export for %s: %s", session_id, exc)
    finally:
        target.dispose()


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------


def prepare_frame(spec: TableSpec, df: pd.DataFrame) -> pd.DataFrame:
    """The table as its exported columns: renamed, retyped, blanks intact."""

    data = {}
    for column in spec.columns:
        if column.source_name not in df.columns:
            continue
        data[column.name] = coerce_series(df[column.source_name], column.column_type)
    return pd.DataFrame(data, index=df.index)


def _insert(connection: Any, table: Table, frame: pd.DataFrame) -> int:
    records = to_records(frame)
    for start in range(0, len(records), CHUNK_ROWS):
        chunk = records[start : start + CHUNK_ROWS]
        if chunk:
            connection.execute(table.insert(), chunk)
    return len(records)


def run_export(
    session_id: str,
    plan: ExportPlan,
    tables: Mapping[str, pd.DataFrame],
    with_semantics: bool = True,
    target: ExportTarget | None = None,
) -> ExportReport:
    """Create the schema, the tables, the rows and the semantic layer.

    ``sem_metadata`` is created and filled in the same transaction as the data
    it describes, so a database that has the rows always has the metadata that
    explains them.  ``with_semantics=False`` exists for tests that are only
    about the tables.
    """

    started = time.perf_counter()
    owned_target = target is None
    target = target or resolve_target(session_id)
    report = ExportReport(
        session_id=session_id,
        target=target.label,
        dialect=target.dialect,
        warnings=list(plan.warnings),
        skipped=list(plan.skipped),
        widened_types=type_summary(plan),
        renamed=[
            {"kind": "table", "from": spec.source_name, "to": spec.name}
            for spec in plan.tables
            if spec.source_name != spec.name
        ]
        + [
            {"kind": "column", "table": spec.name, "from": column.source_name, "to": column.name}
            for spec in plan.tables
            for column in spec.columns
            if column.renamed
        ],
        ddl=render_ddl(plan, schema=target.schema),
    )

    try:
        reset_target(target)
        if target.owned and target.path is not None:
            # The engine was disposed with the file it pointed at.
            target.engine = create_engine(f"sqlite:///{target.path}", future=True)

        metadata = build_metadata(plan, schema=target.schema)
        semantic_table: Table | None = None
        semantic_rows: list[dict[str, Any]] = []
        if with_semantics:
            semantic_table = semantic_metadata.build_table(metadata)
            semantic_rows = semantic_metadata.build_rows(plan)
            report.metadata_rows = len(semantic_rows)
            report.embedded_rows = semantic_metadata.embed_rows(semantic_rows)
        metadata.create_all(bind=target.engine)

        with target.engine.begin() as connection:
            for spec in plan.tables:
                table = metadata.tables[
                    f"{target.schema}.{spec.name}" if target.schema else spec.name
                ]
                frame = prepare_frame(spec, tables[spec.source_name])
                written = _insert(connection, table, frame)
                report.row_count += written
                report.tables.append(
                    {
                        **spec.to_dict(),
                        "rows_written": written,
                    }
                )
            if semantic_table is not None and semantic_rows:
                for start in range(0, len(semantic_rows), CHUNK_ROWS):
                    chunk = semantic_rows[start : start + CHUNK_ROWS]
                    connection.execute(semantic_table.insert(), chunk)

        report.foreign_keys = [
            fk.to_dict() for spec in plan.tables for fk in spec.foreign_keys
        ]
        if semantic_table is not None and report.embedded_rows:
            report.vector_index = semantic_metadata.create_vector_index(
                target.engine, target.schema
            )
    finally:
        if owned_target:
            target.dispose()

    report.duration_seconds = time.perf_counter() - started
    return report


def read_back(session_id: str, statement: str, parameters: Mapping[str, Any] | None = None):
    """Run one read against a session's exported database.

    Used by the tests and, later, by Phase 5.  It takes a whole statement
    because the caller composes it; every identifier in it comes from the
    export plan, never from a request.
    """

    target = resolve_target(session_id)
    try:
        with target.engine.connect() as connection:
            result = connection.execute(text(statement), dict(parameters or {}))
            return [dict(row) for row in result.mappings()]
    finally:
        target.dispose()


def read_semantic_layer(session_id: str) -> list[dict[str, Any]]:
    """Every ``sem_metadata`` row of a session's export, fully typed.

    Read through the table definition rather than as raw SQL so that the JSON
    and vector columns arrive as lists rather than as the text a driver stores
    them in.  This is the reader Phase 5's retrieval step will use.
    """

    target = resolve_target(session_id)
    try:
        metadata = MetaData(schema=target.schema)
        table = semantic_metadata.build_table(metadata)
        with target.engine.connect() as connection:
            return [dict(row) for row in connection.execute(select(table)).mappings()]
    finally:
        target.dispose()


def qualified(target: ExportTarget, table: str) -> str:
    """``schema.table``, quoted, for a statement composed by hand."""

    if not re.fullmatch(r"[a-z_][a-z0-9_]*", table or ""):
        raise ValueError(f"{table!r} is not an exported table name")
    return f'"{target.schema}"."{table}"' if target.schema else f'"{table}"'
