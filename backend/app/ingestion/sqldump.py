"""Phase 0 — ingestion from SQL dump files.

A ``.sql`` dump is a script, not a database: to read it as tables it has to be
replayed somewhere first.  This module replays the parts that carry structure
and data — ``CREATE TABLE``, ``INSERT`` and PostgreSQL's ``COPY … FROM stdin``
— into a throwaway SQLite database, which the relational loader then reads like
any other source.

Two decisions worth defending:

**sqlglot transpiles, it does not merely parse.**  Dumps are written in a
specific dialect: ``AUTO_INCREMENT`` and backtick quoting from MySQL,
``SERIAL`` and ``::text`` casts from PostgreSQL.  Replaying that text verbatim
against SQLite fails on the first statement.  sqlglot (already named in the
project proposal's tool list) parses in the source dialect and re-emits SQLite,
so the dump's own syntax is not something the user has to care about.

**``COPY … FROM stdin`` is handled before sqlglot sees it.**  This is the
detail that decides whether the feature works on real files: ``pg_dump``'s
default plain-text format writes table data as ``COPY`` blocks with
tab-separated rows terminated by ``\\.``, not as ``INSERT`` statements.  A dump
reader that only understands ``INSERT`` loads every table's schema and none of
its rows — structurally successful, completely empty, and the failure looks
like the user's file is broken.  The blocks are extracted first and replayed as
parameterised inserts.

Anything else in the dump (roles, grants, sequences, index and trigger
definitions, ``SET`` statements) is skipped and counted.  The count is reported
so an empty result is explainable rather than mysterious.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Dialects tried against a dump, most specific marker first.
DIALECTS: tuple[str, ...] = ("postgres", "mysql", "sqlite")

_COPY_RE = re.compile(
    r"^\s*COPY\s+(?P<table>[^\s(]+)\s*(?:\((?P<columns>[^)]*)\))?\s*FROM\s+(?:stdin|STDIN)\s*;\s*$"
)

#: PostgreSQL's COPY escapes, in the text format's own encoding.
_COPY_ESCAPES = {
    r"\t": "\t",
    r"\n": "\n",
    r"\r": "\r",
    r"\\": "\\",
}

_MYSQL_MARKERS = ("engine=innodb", "auto_increment", "`", "unlock tables")
_POSTGRES_MARKERS = ("copy ", "serial", "owner to", "::", "search_path", "\\connect")


class DumpError(ValueError):
    """The dump could not be replayed into a readable database."""


@dataclass(slots=True)
class DumpReport:
    """What replaying the dump actually produced."""

    dialect: str
    tables_created: list[str] = field(default_factory=list)
    rows_inserted: int = 0
    statements_executed: int = 0
    statements_skipped: int = 0
    skipped_examples: list[str] = field(default_factory=list)
    #: Keys declared by trailing ``ALTER TABLE … ADD CONSTRAINT`` statements,
    #: as ``table -> {"primary_key": [...], "foreign_keys": [...]}``.
    constraints: dict[str, dict[str, Any]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dialect": self.dialect,
            "tables_created": self.tables_created,
            "rows_inserted": self.rows_inserted,
            "statements_executed": self.statements_executed,
            "statements_skipped": self.statements_skipped,
            "skipped_examples": self.skipped_examples,
            "constraints": self.constraints,
            "notes": self.notes,
        }


def guess_dialect(sql: str) -> str:
    """Pick the dump's dialect from the syntax only it would produce."""

    sample = sql[:200_000].lower()
    mysql_hits = sum(1 for marker in _MYSQL_MARKERS if marker in sample)
    postgres_hits = sum(1 for marker in _POSTGRES_MARKERS if marker in sample)
    if mysql_hits > postgres_hits:
        return "mysql"
    if postgres_hits > 0:
        return "postgres"
    return "sqlite"


def _unquote_identifier(raw: str) -> str:
    name = raw.strip().strip(";")
    for quote in ('"', "`", "[", "]"):
        name = name.replace(quote, "")
    # pg_dump qualifies tables as public.orders; SQLite has no schemas.
    return name.rsplit(".", 1)[-1]


def _decode_copy_value(value: str) -> Any:
    if value == r"\N":
        return None
    if "\\" not in value:
        return value
    out: list[str] = []
    index = 0
    while index < len(value):
        if value[index] == "\\" and index + 1 < len(value):
            pair = value[index : index + 2]
            out.append(_COPY_ESCAPES.get(pair, pair[1]))
            index += 2
            continue
        out.append(value[index])
        index += 1
    return "".join(out)


def extract_copy_blocks(sql: str) -> tuple[str, list[tuple[str, list[str], list[list[Any]]]]]:
    """Pull ``COPY … FROM stdin`` blocks out of a dump.

    Returns the SQL with those blocks removed (so sqlglot never sees raw data
    lines) plus ``(table, columns, rows)`` for each block.
    """

    remaining: list[str] = []
    blocks: list[tuple[str, list[str], list[list[Any]]]] = []
    lines = sql.splitlines()
    index = 0

    while index < len(lines):
        match = _COPY_RE.match(lines[index])
        if match is None:
            remaining.append(lines[index])
            index += 1
            continue

        table = _unquote_identifier(match.group("table"))
        raw_columns = match.group("columns") or ""
        columns = [_unquote_identifier(c) for c in raw_columns.split(",") if c.strip()]
        rows: list[list[Any]] = []
        index += 1
        while index < len(lines) and lines[index].rstrip() != r"\.":
            line = lines[index]
            if line:  # a blank line inside a COPY block is not a row
                rows.append([_decode_copy_value(v) for v in line.split("\t")])
            index += 1
        index += 1  # step over the terminating "\."
        blocks.append((table, columns, rows))

    return "\n".join(remaining), blocks


def _strip_schema_qualifiers(statement: Any) -> Any:
    """Rewrite ``public.orders`` to ``orders`` throughout a statement.

    ``pg_dump`` qualifies every table with its schema. SQLite reads a qualifier
    as an *attached database*, so replaying the statement verbatim fails with
    "unknown database public" — for every table in the file. Since the replay
    target is a single throwaway database, dropping the qualifier is both safe
    and necessary.
    """

    from sqlglot import expressions as exp

    for table in statement.find_all(exp.Table):
        table.set("db", None)
        table.set("catalog", None)
    return statement


def _constraint_columns(node: Any) -> list[str]:
    from sqlglot import expressions as exp

    names: list[str] = []
    for column in node.find_all(exp.Identifier):
        name = _unquote_identifier(column.name)
        if name and name not in names:
            names.append(name)
    return names


def _collect_constraints(statement: Any, report: DumpReport) -> bool:
    """Record keys from ``ALTER TABLE … ADD CONSTRAINT``.  True when it found any.

    SQLite cannot add a primary or foreign key to an existing table, and
    ``pg_dump`` declares nearly all of them that way — so replaying the file
    would lose every relationship the source database documented. The keys are
    kept as metadata instead: the replay target only has to *hold* the rows,
    while Phase 3 needs the constraints, and it reads them from here.
    """

    from sqlglot import expressions as exp

    # Drop schema qualifiers first, so 'public' in "public.customers" is not
    # mistaken for a key column when the reference's identifiers are read.
    statement = _strip_schema_qualifiers(statement)
    table_node = statement.find(exp.Table)
    if table_node is None:
        return False
    table = _unquote_identifier(table_node.name)
    entry = report.constraints.setdefault(table, {"primary_key": [], "foreign_keys": []})
    found = False

    for foreign_key in statement.find_all(exp.ForeignKey):
        reference = foreign_key.args.get("reference")
        referenced_table = reference.find(exp.Table) if reference is not None else None
        local = _constraint_columns(exp.Tuple(expressions=foreign_key.expressions))
        target_columns = (
            [c for c in _constraint_columns(reference) if c != _unquote_identifier(referenced_table.name)]
            if reference is not None and referenced_table is not None
            else []
        )
        if not local or referenced_table is None:
            continue
        entry["foreign_keys"].append(
            {
                "columns": local,
                "references_table": _unquote_identifier(referenced_table.name),
                "references_columns": target_columns or local,
                "source": "declared",
            }
        )
        found = True

    for primary_key in statement.find_all(exp.PrimaryKey):
        columns = _constraint_columns(primary_key)
        if columns:
            entry["primary_key"] = columns
            found = True

    if not found and not entry["primary_key"] and not entry["foreign_keys"]:
        report.constraints.pop(table, None)
    return found


def _statements(sql: str, dialect: str) -> list[Any]:
    import sqlglot

    try:
        return [s for s in sqlglot.parse(sql, read=dialect) if s is not None]
    except Exception as exc:  # sqlglot raises ParseError and friends
        raise DumpError(f"could not parse the dump as {dialect} SQL: {exc}") from exc


def replay_dump(path: str | Path, target: str | Path | None = None) -> tuple[Path, DumpReport]:
    """Replay a dump into a SQLite file.  Returns its path and what happened."""

    import sqlglot
    from sqlglot import expressions as exp

    source = Path(path)
    text = source.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        raise DumpError("the dump file is empty")

    dialect = guess_dialect(text)
    body, copy_blocks = extract_copy_blocks(text)
    report = DumpReport(dialect=dialect)

    try:
        statements = _statements(body, dialect)
    except DumpError:
        # A dump that does not parse in the guessed dialect is worth one more
        # try in the others before telling the user their file is unreadable.
        statements = []
        for candidate in (d for d in DIALECTS if d != dialect):
            try:
                statements = _statements(body, candidate)
                dialect = candidate
                report.dialect = candidate
                break
            except DumpError:
                continue
        if not statements:
            raise

    if target is None:
        handle = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
        handle.close()
        target = handle.name
    target_path = Path(target)

    connection = sqlite3.connect(target_path)
    try:
        for statement in statements:
            if isinstance(statement, exp.Alter):
                # Not replayable on SQLite, but the keys it declares are the
                # point of reading a dump at all.
                _collect_constraints(statement, report)
                report.statements_skipped += 1
                continue
            if isinstance(statement, exp.Create) and str(
                statement.args.get("kind") or ""
            ).upper() not in {"TABLE", ""}:
                report.statements_skipped += 1
                continue
            if not isinstance(statement, (exp.Create, exp.Insert)):
                report.statements_skipped += 1
                if len(report.skipped_examples) < 5:
                    report.skipped_examples.append(statement.sql(dialect=dialect)[:120])
                continue
            try:
                rendered = _strip_schema_qualifiers(statement).sql(dialect="sqlite")
                connection.execute(rendered)
                report.statements_executed += 1
                if isinstance(statement, exp.Create):
                    name = statement.find(exp.Table)
                    if name is not None:
                        report.tables_created.append(_unquote_identifier(name.name))
                else:
                    report.rows_inserted += max(1, len(list(statement.find_all(exp.Tuple))))
            except (sqlite3.Error, sqlglot.errors.SqlglotError) as exc:
                report.statements_skipped += 1
                if len(report.skipped_examples) < 5:
                    report.skipped_examples.append(f"{exc}")

        for table, columns, rows in copy_blocks:
            if not rows:
                continue
            width = len(columns) if columns else len(rows[0])
            # A row whose field count does not match the block header is
            # malformed; dropping just that row beats failing the whole block
            # and leaving the user with a table that is silently empty.
            usable = [row for row in rows if len(row) == width]
            malformed = len(rows) - len(usable)
            if malformed:
                report.notes.append(
                    f"{malformed} row(s) in the COPY block for '{table}' had the wrong number "
                    f"of fields ({width} expected) and were skipped"
                )
            if not usable:
                report.statements_skipped += 1
                continue

            column_sql = (
                " (" + ", ".join('"' + c + '"' for c in columns) + ")" if columns else ""
            )
            placeholders = ", ".join("?" * width)
            try:
                connection.executemany(
                    f'INSERT INTO "{table}"{column_sql} VALUES ({placeholders})', usable
                )
                report.rows_inserted += len(usable)
                report.statements_executed += 1
            except sqlite3.Error as exc:
                report.statements_skipped += 1
                if len(report.skipped_examples) < 5:
                    report.skipped_examples.append(f"COPY into {table}: {exc}")

        connection.commit()
    finally:
        connection.close()

    if not report.tables_created:
        raise DumpError(
            "no CREATE TABLE statement in this dump could be replayed — "
            f"{report.statements_skipped} statement(s) were skipped"
        )
    if report.statements_skipped:
        report.notes.append(
            f"{report.statements_skipped} statement(s) were skipped: this reader replays "
            "table definitions and row data, and ignores roles, grants, sequences, "
            "indexes and triggers."
        )
    return target_path, report


def load_sql_dump(path: str | Path) -> list["SheetLoadResult"]:
    """Replay a ``.sql`` dump and load every table it defines."""

    from app.ingestion.loader import SheetLoadResult  # local import: avoids a cycle
    from app.ingestion.relational import load_relational_tables, sqlite_url

    replayed, report = replay_dump(path)
    try:
        results: list[SheetLoadResult] = load_relational_tables(
            sqlite_url(replayed), kind="sql_dump"
        )
    finally:
        replayed.unlink(missing_ok=True)

    label = Path(path).name
    for result in results:
        _merge_dump_constraints(result, report)
        result.notes = [
            f"replayed from the {report.dialect} dump {label}",
            *report.notes,
            *[note for note in result.notes if not note.startswith("read from ")],
        ]
    return results


def _merge_dump_constraints(result: "SheetLoadResult", report: DumpReport) -> None:
    """Restore keys the replay could not enforce onto the table's schema."""

    declared = report.constraints.get(result.native_schema.get("source_name", "")) or (
        report.constraints.get(result.name)
    )
    if not declared:
        return

    known_columns = {c["name"] for c in result.native_schema.get("columns", [])}
    if not result.native_schema.get("primary_key"):
        primary_key = [c for c in declared.get("primary_key", []) if c in known_columns]
        if primary_key:
            result.native_schema["primary_key"] = primary_key
            result.notes.append("declared primary key: " + ", ".join(primary_key))

    existing = {tuple(fk["columns"]) for fk in result.native_schema.get("foreign_keys", [])}
    for foreign_key in declared.get("foreign_keys", []):
        columns = [c for c in foreign_key["columns"] if c in known_columns]
        if not columns or tuple(columns) in existing:
            continue
        result.native_schema.setdefault("foreign_keys", []).append({**foreign_key, "columns": columns})
        result.notes.append(
            "declared foreign key: "
            + ", ".join(columns)
            + f" → {foreign_key['references_table']}."
            + ", ".join(foreign_key["references_columns"])
        )
