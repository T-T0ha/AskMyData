"""Phase 0 — ingestion from relational sources.

The proposal accepts five kinds of input: Excel workbooks, CSV files, SQLite
database files, live PostgreSQL connections and SQL dump files.  The first two
arrive as a grid of cells and need structural repair (:mod:`app.ingestion.headers`).
The last three arrive as *tables that already have a schema*, and that changes
what Phase 0 should do with them:

* **Column names are given, not inferred.**  There is no header block to find,
  no merged cells to flatten and no banner row to skip.  Names are sanitised to
  the same identifier shape the spreadsheet path produces, and the original is
  kept so the UI can show both.
* **Declared types are authoritative, so nothing is re-typed.**  The
  spreadsheet cleaner converts a text column to numeric when 80 % of its values
  parse, which is right for a sheet where everything is a string, and wrong for
  a database that has already committed to ``VARCHAR``.  Re-typing one side of
  a declared foreign key and not the other would break exactly the joins the
  source told us about.  Null markers are still normalised and whitespace still
  trimmed — those change no type.  Phase 1 detects semantic types from *values*
  regardless of dtype, so nothing is lost by leaving them alone.
* **The schema itself is evidence.**  Declared primary keys, foreign keys and
  nullability are ground truth that Phase 3 would otherwise have to rediscover
  statistically, so they are captured into :class:`TableSchema` and persisted
  alongside the data rather than thrown away.

Everything here only ever issues ``SELECT`` and schema reflection.  No DDL, no
writes, and on PostgreSQL the session is explicitly opened read-only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from sqlalchemy import MetaData, Table, UniqueConstraint, create_engine, func, select
from sqlalchemy.engine import Engine
from sqlalchemy.engine.url import URL, make_url
from sqlalchemy.exc import SQLAlchemyError

from app.core.config import get_settings
from app.core.schemas import SheetIssue, TriageStatus
from app.ingestion.cleaning import clean_dataframe
from app.ingestion.headers import HeaderAnalysis, sanitize_name
from app.ingestion.keys import discover_keys
from app.ingestion.triage import triage_sheet

logger = logging.getLogger(__name__)

#: File extensions that hold a SQLite database rather than a spreadsheet.
SQLITE_SUFFIXES = {".db", ".sqlite", ".sqlite3", ".db3"}

#: Database backends this platform will connect to.  An allowlist rather than
#: "whatever SQLAlchemy can parse": a URL is user input, and SQLAlchemy will
#: happily load an arbitrary dialect plugin named in one.
ALLOWED_BACKENDS: frozenset[str] = frozenset(
    {"postgresql", "postgres", "mysql", "mariadb", "sqlite"}
)

#: System schemas that are never business data.
INTERNAL_SCHEMAS: frozenset[str] = frozenset(
    {"information_schema", "pg_catalog", "pg_toast", "performance_schema", "mysql", "sys"}
)

#: Table names SQLite and friends create for their own bookkeeping.
INTERNAL_TABLE_PREFIXES: tuple[str, ...] = ("sqlite_", "pg_")


class SourceError(ValueError):
    """The source cannot be read: bad URL, unreachable server, missing file."""


@dataclass(slots=True)
class ColumnSchema:
    """One column as the *source database* describes it."""

    name: str
    source_name: str
    data_type: str
    nullable: bool = True
    default: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source_name": self.source_name,
            "data_type": self.data_type,
            "nullable": self.nullable,
            "default": self.default,
        }


@dataclass(slots=True)
class TableSchema:
    """A source table's declared structure — Phase 3's ground truth.

    Primary and foreign keys here are *declared*, not detected.  Phase 3 still
    runs its own key discovery (a declared key can be missing, and a dump can
    drop constraints), but a relationship the database itself asserts does not
    need to be inferred from value overlap, and it should never be contradicted
    by a weaker statistical guess.
    """

    name: str
    source_name: str
    schema: str | None = None
    columns: list[ColumnSchema] = field(default_factory=list)
    primary_key: list[str] = field(default_factory=list)
    foreign_keys: list[dict[str, Any]] = field(default_factory=list)
    unique_constraints: list[list[str]] = field(default_factory=list)
    row_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source_name": self.source_name,
            "schema": self.schema,
            "columns": [c.to_dict() for c in self.columns],
            "primary_key": self.primary_key,
            "foreign_keys": self.foreign_keys,
            "unique_constraints": self.unique_constraints,
            "row_count": self.row_count,
        }


@dataclass(slots=True)
class SourceInspection:
    """What a source contains, shown to the user before anything is ingested."""

    kind: str
    dialect: str
    database: str | None
    display_url: str
    tables: list[TableSchema] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "dialect": self.dialect,
            "database": self.database,
            "display_url": self.display_url,
            "table_count": len(self.tables),
            "tables": [t.to_dict() for t in self.tables],
        }


# ---------------------------------------------------------------------------
# connecting
# ---------------------------------------------------------------------------


def _validate_url(raw_url: str) -> URL:
    try:
        url = make_url(raw_url.strip())
    except Exception as exc:  # sqlalchemy raises ArgumentError subclasses
        raise SourceError(f"that is not a valid database URL: {exc}") from exc

    backend = url.get_backend_name()
    if backend not in ALLOWED_BACKENDS:
        raise SourceError(
            f"unsupported database type {backend!r}; this platform reads "
            + ", ".join(sorted(ALLOWED_BACKENDS))
        )
    if backend == "sqlite":
        if not url.database:
            raise SourceError("a SQLite URL must point at a file")
        if not Path(url.database).exists():
            raise SourceError(f"no SQLite database at {url.database}")
    elif not url.database:
        raise SourceError("the URL does not name a database")
    return url


def display_url(url: str | URL) -> str:
    """The URL with the password removed — safe to log, store and return."""

    parsed = url if isinstance(url, URL) else make_url(url)
    return parsed.render_as_string(hide_password=True)


def source_kind(url: URL) -> str:
    backend = url.get_backend_name()
    return "sqlite" if backend == "sqlite" else backend


def make_source_engine(raw_url: str) -> Engine:
    """Open a **read-only** engine against a user-supplied database URL."""

    settings = get_settings()
    url = _validate_url(raw_url)
    backend = url.get_backend_name()
    kwargs: dict[str, Any] = {"pool_pre_ping": True, "future": True}

    if backend in {"postgresql", "postgres"}:
        kwargs["connect_args"] = {
            "connect_timeout": settings.source_connect_timeout,
            # Belt and braces: even though this module only ever emits SELECT,
            # the server refuses a write on this session, and a runaway query
            # cannot hold the user's production database open indefinitely.
            "options": (
                "-c default_transaction_read_only=on "
                f"-c statement_timeout={settings.source_statement_timeout * 1000}"
            ),
        }
    elif backend in {"mysql", "mariadb"}:
        kwargs["connect_args"] = {
            "connect_timeout": settings.source_connect_timeout,
            "read_timeout": settings.source_statement_timeout,
        }

    try:
        return create_engine(url, **kwargs)
    except SQLAlchemyError as exc:
        raise SourceError(f"could not open the database: {exc}") from exc


def sqlite_url(path: str | Path) -> str:
    return f"sqlite:///{Path(path).resolve()}"


# ---------------------------------------------------------------------------
# reflection
# ---------------------------------------------------------------------------


def _unique_names(source_names: Iterable[str], fallback: str = "col") -> dict[str, str]:
    """Sanitised, de-duplicated identifier for every source name.

    Two source columns can sanitise to the same identifier (``Order ID`` and
    ``order_id``); the map keeps them distinct so no column is silently lost.
    """

    mapping: dict[str, str] = {}
    used: set[str] = set()
    for index, source in enumerate(source_names):
        base = sanitize_name(source) or f"{fallback}_{index + 1}"
        name = base
        counter = 2
        while name in used:
            name = f"{base}_{counter}"
            counter += 1
        used.add(name)
        mapping[source] = name
    return mapping


def _reflect_table(engine: Engine, name: str, schema: str | None) -> tuple[Table, TableSchema]:
    metadata = MetaData(schema=schema)
    table = Table(name, metadata, autoload_with=engine)

    column_names = _unique_names([str(c.name) for c in table.columns])
    columns = [
        ColumnSchema(
            name=column_names[str(column.name)],
            source_name=str(column.name),
            data_type=str(column.type),
            nullable=bool(column.nullable),
            default=str(column.server_default.arg) if column.server_default is not None else None,
        )
        for column in table.columns
    ]

    primary_key = [
        column_names[str(c.name)] for c in table.primary_key.columns if str(c.name) in column_names
    ]
    foreign_keys: list[dict[str, Any]] = []
    for constraint in table.foreign_key_constraints:
        elements = list(constraint.elements)
        foreign_keys.append(
            {
                "columns": [
                    column_names.get(str(e.parent.name), str(e.parent.name)) for e in elements
                ],
                "references_table": sanitize_name(str(elements[0].column.table.name))
                if elements
                else None,
                "references_columns": [
                    sanitize_name(str(e.column.name)) or str(e.column.name) for e in elements
                ],
                "source": "declared",
            }
        )

    unique_constraints = [
        [column_names[str(c.name)] for c in constraint.columns if str(c.name) in column_names]
        for constraint in table.constraints
        if isinstance(constraint, UniqueConstraint)
    ]

    schema_info = TableSchema(
        name=sanitize_name(name) or "table",
        source_name=name,
        schema=schema,
        columns=columns,
        primary_key=primary_key,
        foreign_keys=foreign_keys,
        unique_constraints=[u for u in unique_constraints if u],
    )
    return table, schema_info


def _row_count(engine: Engine, table: Table) -> int | None:
    """Exact row count, or ``None`` when the server will not give one cheaply."""

    try:
        with engine.connect() as connection:
            return int(connection.execute(select(func.count()).select_from(table)).scalar_one())
    except SQLAlchemyError as exc:  # pragma: no cover - depends on the server
        logger.warning("could not count rows of %s: %s", table.name, exc)
        return None


def list_source_tables(engine: Engine) -> list[tuple[str, str | None]]:
    """Every user table in the source, as ``(name, schema)`` pairs."""

    from sqlalchemy import inspect as sqlalchemy_inspect

    inspector = sqlalchemy_inspect(engine)
    schemas: list[str | None]
    if engine.dialect.name == "sqlite":
        schemas = [None]
    else:
        schemas = [s for s in inspector.get_schema_names() if s not in INTERNAL_SCHEMAS]
        default_schema = inspector.default_schema_name
        if default_schema in schemas:  # list the default schema first
            schemas.remove(default_schema)
            schemas.insert(0, default_schema)

    found: list[tuple[str, str | None]] = []
    for schema in schemas:
        for name in inspector.get_table_names(schema=schema):
            if name.startswith(INTERNAL_TABLE_PREFIXES):
                continue
            found.append((name, schema))
        for name in inspector.get_view_names(schema=schema):
            # Views are legitimate business data — a reporting view is often
            # exactly what the user wants to query — and read the same way.
            if not name.startswith(INTERNAL_TABLE_PREFIXES):
                found.append((name, schema))
    return found


def inspect_source(raw_url: str, kind: str | None = None) -> SourceInspection:
    """List a source's tables, columns and keys without ingesting anything."""

    url = _validate_url(raw_url)
    engine = make_source_engine(raw_url)
    try:
        tables: list[TableSchema] = []
        for name, schema in list_source_tables(engine):
            try:
                table, info = _reflect_table(engine, name, schema)
            except SQLAlchemyError as exc:
                logger.warning("could not reflect %s: %s", name, exc)
                continue
            info.row_count = _row_count(engine, table)
            tables.append(info)
        return SourceInspection(
            kind=kind or source_kind(url),
            dialect=engine.dialect.name,
            database=url.database,
            display_url=display_url(url),
            tables=tables,
        )
    except SQLAlchemyError as exc:
        raise SourceError(f"could not read the database: {exc}") from exc
    finally:
        engine.dispose()


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def _read_table(engine: Engine, table: Table, row_limit: int) -> tuple[pd.DataFrame, bool]:
    """Read up to ``row_limit`` rows.  Returns the frame and a truncation flag.

    One row beyond the limit is requested so truncation is *known* rather than
    guessed from a count that may have changed since it was taken.
    """

    statement = select(table).limit(row_limit + 1)
    with engine.connect() as connection:
        frame = pd.read_sql_query(statement, connection)
    truncated = len(frame) > row_limit
    if truncated:
        frame = frame.head(row_limit)
    return frame, truncated


def load_relational_tables(
    raw_url: str,
    tables: list[str] | None = None,
    kind: str | None = None,
    row_limit: int | None = None,
) -> list["SheetLoadResult"]:
    """Load selected tables from a relational source through Phase 0.

    ``tables`` names tables as the *source* names them; ``None`` loads every
    user table, up to ``max_tables_per_source``.
    """

    from app.ingestion.loader import SheetLoadResult  # local import: avoids a cycle

    settings = get_settings()
    url = _validate_url(raw_url)
    resolved_kind = kind or source_kind(url)
    limit = row_limit or settings.max_rows_per_table
    engine = make_source_engine(raw_url)

    try:
        available = list_source_tables(engine)
        by_name = {name: schema for name, schema in available}
        if tables:
            missing = [t for t in tables if t not in by_name]
            if missing:
                raise SourceError(
                    f"table(s) not found in the source: {', '.join(sorted(missing))}"
                )
            selected = [(name, by_name[name]) for name in tables]
        else:
            selected = available

        if not selected:
            raise SourceError("the source contains no readable tables")
        if len(selected) > settings.max_tables_per_source:
            raise SourceError(
                f"{len(selected)} tables selected; this platform ingests at most "
                f"{settings.max_tables_per_source} at a time — choose the ones you need"
            )

        results: list[SheetLoadResult] = []
        for name, schema in selected:
            results.append(
                _load_one_table(engine, name, schema, resolved_kind, limit, display_url(url))
            )
        return results
    except SQLAlchemyError as exc:
        raise SourceError(f"could not read the database: {exc}") from exc
    finally:
        engine.dispose()


def _load_one_table(
    engine: Engine,
    name: str,
    schema: str | None,
    kind: str,
    row_limit: int,
    source_label: str,
) -> "SheetLoadResult":
    from app.ingestion.loader import SheetLoadResult

    table, info = _reflect_table(engine, name, schema)
    frame, truncated = _read_table(engine, table, row_limit)

    rename = {
        str(column.source_name): column.name
        for column in info.columns
        if str(column.source_name) in frame.columns
    }
    frame = frame.rename(columns=rename)

    # retype=False: the source already declared these types (see module docstring).
    cleaned, reports = clean_dataframe(frame, retype=False)
    info.row_count = int(len(cleaned)) if not truncated else info.row_count

    header = HeaderAnalysis(
        header_rows=[],
        data_start_row=1,
        columns=[str(c) for c in cleaned.columns],
        original_labels=[[c.source_name] for c in info.columns],
        confidence=1.0,
        notes=[f"column names and types came from the {kind} schema"],
    )

    # The declared key is trusted, not re-derived — but it is still checked
    # against the rows that arrived, because a dump can lose its constraints.
    keys = discover_keys(cleaned, declared_primary_key=info.primary_key)
    triage, issues = triage_sheet(cleaned, header, reports, keys)
    if truncated:
        issues.append(
            SheetIssue(
                "truncated_read",
                "warning",
                f"Only the first {row_limit:,} rows were read from '{name}'. Detection and "
                "profiling describe those rows, not the whole table.",
            )
        )
        if triage is TriageStatus.CLEAN:
            triage = TriageStatus.NEEDS_ATTENTION

    notes = [f"read from {source_label}"]
    if info.primary_key:
        notes.append("declared primary key: " + ", ".join(info.primary_key))
    for foreign_key in info.foreign_keys:
        notes.append(
            "declared foreign key: "
            + ", ".join(foreign_key["columns"])
            + f" → {foreign_key['references_table']}."
            + ", ".join(foreign_key["references_columns"])
        )

    qualified = f"{schema}.{name}" if schema else name
    return SheetLoadResult(
        name=info.name,
        source_name=qualified,
        dataframe=cleaned,
        header=header,
        clean_reports=reports,
        triage=triage,
        issues=issues,
        row_count=int(len(cleaned)),
        column_count=int(len(cleaned.columns)),
        span=f"{len(cleaned.columns)} columns x {len(cleaned)} rows",
        source_kind=kind,
        native_schema=info.to_dict(),
        key_analysis=keys,
        notes=notes,
    )
