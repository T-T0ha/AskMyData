"""From validated analysis to a set of SQL tables.

This module decides *what the database will look like* and nothing else: it
touches no server, opens no connection, and can be run to show the user the
DDL before anything is created.  :mod:`app.export.runner` does the creating.

Three decisions live here, and each one is a place where being permissive would
produce an export that fails halfway or, worse, succeeds while meaning
something different from the spreadsheet it came from:

**Which columns exist, and under what names.**  Headers are sanitised
(:mod:`app.export.naming`) and typed against their own values
(:mod:`app.export.types`).  The original header survives in the metadata.

**What the primary key is.**  Whatever Phase 0 detected or the user confirmed —
re-verified here against the *cleaned* table, because a cleaning step can
deduplicate a table into having a key, and a merge can take one away.  A key
that no longer holds is dropped with a note rather than emitted as a constraint
the data cannot satisfy.

**Which references become constraints.**  Only relationships the user
confirmed, and only those that would actually hold: the target has to be a real
key, the types have to match, and every child value has to exist in the parent.
A confirmed reference that fails those checks is not discarded — it stays in
the semantic layer as a join hint for Phase 5, and the report says why it is
not enforced.  The alternative is an export that dies at ``INSERT`` on data the
user was told was fine.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import pandas as pd
from sqlalchemy import Column, ForeignKeyConstraint, MetaData, PrimaryKeyConstraint, Table
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateTable

from app.core.schemas import (
    ADDITIVE_LABELS,
    ColumnType,
    RelationshipStatus,
    RelationshipType,
)
from app.export.naming import sanitize_identifier, unique_identifiers
from app.export.types import TypePlan, plan_type, render_type

logger = logging.getLogger(__name__)

#: Orphan values listed in a report before it stops naming them individually.
MAX_REPORTED_ORPHANS = 5


@dataclass(slots=True)
class ColumnSpec:
    """One column of one exported table."""

    source_name: str
    name: str
    column_type: ColumnType
    type_plan: TypePlan
    nullable: bool
    taxonomy_label: str = "unknown"
    is_primary_key: bool = False
    is_foreign_key: bool = False
    references_table: str | None = None
    references_column: str | None = None
    is_additive: bool = False
    null_ratio: float = 0.0
    sample_values: list[Any] = field(default_factory=list)
    description: str = ""

    @property
    def renamed(self) -> bool:
        return self.source_name != self.name

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_name": self.source_name,
            "name": self.name,
            "renamed": self.renamed,
            "column_type": self.column_type.value,
            "sql_type": self.type_plan.render(),
            "type_notes": self.type_plan.notes,
            "nullable": self.nullable,
            "taxonomy_label": self.taxonomy_label,
            "is_primary_key": self.is_primary_key,
            "is_foreign_key": self.is_foreign_key,
            "references_table": self.references_table,
            "references_column": self.references_column,
            "is_additive": self.is_additive,
            "null_ratio": round(self.null_ratio, 4),
        }


@dataclass(slots=True)
class ForeignKeySpec:
    """A confirmed reference, and whether the database will enforce it."""

    source_table: str
    source_column: str
    target_table: str
    target_column: str
    #: False when the reference is real but cannot be a constraint.  It still
    #: reaches ``sem_metadata``, so Phase 5 can join on it.
    enforced: bool = True
    reason: str = ""
    #: Set for a self-reference or a cycle: the constraint is added after the
    #: tables exist and checked at commit rather than per row.
    deferrable: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_table": self.source_table,
            "source_column": self.source_column,
            "target_table": self.target_table,
            "target_column": self.target_column,
            "enforced": self.enforced,
            "deferrable": self.deferrable,
            "reason": self.reason,
        }


@dataclass(slots=True)
class TableSpec:
    """One exported table."""

    source_name: str
    name: str
    columns: list[ColumnSpec]
    primary_key: list[str] = field(default_factory=list)
    foreign_keys: list[ForeignKeySpec] = field(default_factory=list)
    table_type: str = "unknown"
    #: The stored one-line description of the table.  Carried through the plan
    #: so that the JSON bundle and the Markdown documentation say what a table
    #: is for, not only what shape it has.
    description: str = ""
    row_count: int = 0
    notes: list[str] = field(default_factory=list)

    def column(self, name: str) -> ColumnSpec | None:
        return next((c for c in self.columns if c.name == name), None)

    @property
    def rename_map(self) -> dict[str, str]:
        return {column.source_name: column.name for column in self.columns}

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_name": self.source_name,
            "name": self.name,
            "renamed": self.source_name != self.name,
            "table_type": self.table_type,
            "description": self.description,
            "row_count": self.row_count,
            "primary_key": self.primary_key,
            "columns": [column.to_dict() for column in self.columns],
            "foreign_keys": [fk.to_dict() for fk in self.foreign_keys],
            "notes": self.notes,
        }


@dataclass(slots=True)
class ExportPlan:
    """Every table, in an order that can be inserted, plus what was downgraded."""

    tables: list[TableSpec] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    skipped: list[dict[str, str]] = field(default_factory=list)

    def table(self, name: str) -> TableSpec | None:
        return next((t for t in self.tables if t.name == name), None)

    @property
    def source_map(self) -> dict[str, TableSpec]:
        return {table.source_name: table for table in self.tables}

    @property
    def enforced_keys(self) -> list[ForeignKeySpec]:
        return [fk for table in self.tables for fk in table.foreign_keys if fk.enforced]

    @property
    def unenforced_keys(self) -> list[ForeignKeySpec]:
        return [fk for table in self.tables for fk in table.foreign_keys if not fk.enforced]

    def to_dict(self) -> dict[str, Any]:
        return {
            "tables": [table.to_dict() for table in self.tables],
            "warnings": self.warnings,
            "skipped": self.skipped,
            "table_count": len(self.tables),
            "row_count": sum(table.row_count for table in self.tables),
            "enforced_foreign_keys": len(self.enforced_keys),
            "unenforced_foreign_keys": len(self.unenforced_keys),
        }


# ---------------------------------------------------------------------------
# building the plan
# ---------------------------------------------------------------------------


def _column_type(fact: Mapping[str, Any]) -> ColumnType:
    raw = str(fact.get("effective_type") or fact.get("column_type") or ColumnType.TEXT.value)
    try:
        return ColumnType(raw)
    except ValueError:
        return ColumnType.TEXT


def _label(fact: Mapping[str, Any]) -> str:
    return str(fact.get("effective_label") or fact.get("taxonomy_label") or "unknown")


def _key_holds(df: pd.DataFrame, columns: Sequence[str]) -> tuple[bool, str]:
    """Re-verify a primary key against the table as it is *now*.

    Cleaning runs between key discovery and the export: a deduplicate step can
    make a key hold that did not, and a merge can break one that did.  The
    constraint is only as good as the data at the moment it is created.
    """

    missing = [column for column in columns if column not in df.columns]
    if missing:
        return False, f"{', '.join(missing)} no longer exists after cleaning"
    subset = df.loc[:, list(columns)]
    blanks = int(subset.isna().any(axis=1).sum())
    if blanks:
        return False, f"{blanks:,} row(s) leave it empty, and NULL cannot identify a row"
    repeats = int(subset.duplicated().sum())
    if repeats:
        return False, f"{repeats:,} row(s) repeat the same value"
    return True, ""


def _reference_holds(
    child: pd.Series, parent: pd.Series
) -> tuple[bool, str]:
    """Would this foreign key survive being enforced?

    Checks only what the database will check: every non-null child value has to
    exist in the parent column.  NULL is not a violation — an order with no
    customer recorded is unknown, not wrong — which is the same reading the
    rest of the platform gives a blank.
    """

    child_values = {_key(value) for value in child.dropna()}
    parent_values = {_key(value) for value in parent.dropna()}
    orphans = sorted(child_values - parent_values, key=str)
    if not orphans:
        return True, ""
    shown = ", ".join(str(value) for value in orphans[:MAX_REPORTED_ORPHANS])
    more = "" if len(orphans) <= MAX_REPORTED_ORPHANS else f" and {len(orphans) - MAX_REPORTED_ORPHANS:,} more"
    return False, f"{len(orphans):,} value(s) have no match in the parent: {shown}{more}"


def _key(value: Any) -> str:
    """Compare values the way a database compares keys.

    ``1001`` read from one sheet as an integer and from another as text is one
    key, not two — and the join the user confirmed is the one that treats them
    as the same.
    """

    if isinstance(value, float) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def build_plan(
    tables: Mapping[str, pd.DataFrame],
    columns_by_table: Mapping[str, Sequence[Mapping[str, Any]]],
    key_analyses: Mapping[str, Mapping[str, Any]] | None = None,
    relationships: Sequence[Mapping[str, Any]] = (),
    profiles: Mapping[str, Mapping[str, Any]] | None = None,
    descriptions: Mapping[str, str] | None = None,
) -> ExportPlan:
    """Decide the whole schema.  Reads data; writes nothing anywhere."""

    keys = key_analyses or {}
    profile_map = profiles or {}
    description_map = descriptions or {}
    plan = ExportPlan()

    exportable = {name: df for name, df in tables.items() if len(df.columns)}
    for name in sorted(set(tables) - set(exportable)):
        plan.skipped.append({"table": name, "reason": "the table has no columns"})

    table_names = unique_identifiers(sorted(exportable), fallback="table")
    specs: dict[str, TableSpec] = {}

    for source_name in sorted(exportable):
        df = exportable[source_name]
        facts = {
            str(fact.get("name") or fact.get("column_name")): fact
            for fact in columns_by_table.get(source_name, ())
        }
        column_names = unique_identifiers([str(c) for c in df.columns], fallback="column")

        columns: list[ColumnSpec] = []
        for source_column in (str(c) for c in df.columns):
            fact = facts.get(source_column, {})
            column_type = _column_type(fact)
            series = df[source_column]
            columns.append(
                ColumnSpec(
                    source_name=source_column,
                    name=column_names[source_column],
                    column_type=column_type,
                    type_plan=plan_type(column_type, series),
                    nullable=True,
                    taxonomy_label=_label(fact),
                    is_additive=bool(
                        fact.get("is_additive", _label(fact) in ADDITIVE_LABELS)
                    ),
                    null_ratio=float(fact.get("null_ratio") or 0.0),
                    sample_values=list(fact.get("sample_values") or [])[:5],
                    description=str(fact.get("semantic_description") or ""),
                )
            )

        spec = TableSpec(
            source_name=source_name,
            name=table_names[source_name],
            columns=columns,
            row_count=int(len(df)),
            table_type=str(
                (profile_map.get(source_name) or {}).get("effective_type", "unknown")
            ),
            description=str(description_map.get(source_name) or ""),
        )
        _apply_primary_key(spec, df, keys.get(source_name) or {}, plan)
        specs[source_name] = spec

    _apply_foreign_keys(specs, exportable, relationships, plan)

    ordered = _insert_order(specs, plan)
    plan.tables = [specs[name] for name in ordered]
    return plan


def _apply_primary_key(
    spec: TableSpec, df: pd.DataFrame, key_analysis: Mapping[str, Any], plan: ExportPlan
) -> None:
    declared = [str(column) for column in key_analysis.get("primary_key") or []]
    if not declared:
        spec.notes.append(
            "no primary key: the table is exported without one, and nothing can reference it"
        )
        plan.warnings.append(f"{spec.name} has no primary key")
        return

    holds, reason = _key_holds(df, declared)
    if not holds:
        spec.notes.append(f"the primary key {' + '.join(declared)} was dropped — {reason}")
        plan.warnings.append(f"{spec.name}: primary key dropped because {reason}")
        return

    rename = spec.rename_map
    spec.primary_key = [rename[column] for column in declared]
    for name in spec.primary_key:
        column = spec.column(name)
        if column is not None:
            column.is_primary_key = True
            column.nullable = False  # PRIMARY KEY implies NOT NULL; say it anyway


def _apply_foreign_keys(
    specs: Mapping[str, TableSpec],
    tables: Mapping[str, pd.DataFrame],
    relationships: Sequence[Mapping[str, Any]],
    plan: ExportPlan,
) -> None:
    """Turn confirmed references into constraints, or into explained hints."""

    for relation in relationships:
        if relation.get("rel_type") != RelationshipType.FOREIGN_KEY.value:
            continue
        if relation.get("status") != RelationshipStatus.CONFIRMED.value:
            continue

        child_spec = specs.get(str(relation.get("from_table")))
        parent_spec = specs.get(str(relation.get("to_table")))
        if child_spec is None or parent_spec is None:
            continue

        child_source = str(relation.get("from_column"))
        parent_source = str(relation.get("to_column"))
        child_column = child_spec.rename_map.get(child_source)
        parent_column = parent_spec.rename_map.get(parent_source)
        if child_column is None or parent_column is None:
            plan.warnings.append(
                f"{child_spec.name}.{child_source} → {parent_spec.name}.{parent_source} was "
                "confirmed but one of its columns no longer exists after cleaning"
            )
            continue

        fk = ForeignKeySpec(
            source_table=child_spec.name,
            source_column=child_column,
            target_table=parent_spec.name,
            target_column=parent_column,
        )

        if parent_spec.primary_key != [parent_column]:
            fk.enforced = False
            fk.reason = (
                f"{parent_spec.name}.{parent_column} is not the primary key of its table, and "
                "a foreign key can only point at a key"
            )
        else:
            holds, reason = _reference_holds(
                tables[child_spec.source_name][child_source],
                tables[parent_spec.source_name][parent_source],
            )
            if not holds:
                fk.enforced = False
                fk.reason = reason

        if fk.enforced and child_spec is parent_spec:
            # A row whose parent is inserted later would fail an immediate
            # check; deferring moves it to the commit, by which time every row
            # is in.  The values were verified above, so the commit passes.
            fk.deferrable = True

        child_spec.foreign_keys.append(fk)
        column = child_spec.column(child_column)
        if column is not None:
            column.is_foreign_key = True
            column.references_table = parent_spec.name
            column.references_column = parent_column
        if not fk.enforced:
            plan.warnings.append(
                f"{fk.source_table}.{fk.source_column} → {fk.target_table}.{fk.target_column} "
                f"is recorded in the semantic layer but not enforced: {fk.reason}"
            )


def _insert_order(specs: Mapping[str, TableSpec], plan: ExportPlan) -> list[str]:
    """Parents before children, so an enforced reference is satisfiable.

    Kahn's algorithm over the enforced foreign keys.  A cycle has no valid
    order, so the constraints that close it are deferred to the commit instead
    of being checked row by row — the values behind them were verified before
    the plan was built, so the commit succeeds.
    """

    by_name = {spec.name: source for source, spec in specs.items()}
    dependencies: dict[str, set[str]] = {spec.name: set() for spec in specs.values()}
    for spec in specs.values():
        for fk in spec.foreign_keys:
            if fk.enforced and fk.target_table != spec.name:
                dependencies[spec.name].add(fk.target_table)

    ordered: list[str] = []
    remaining = dict(dependencies)
    while remaining:
        ready = sorted(name for name, deps in remaining.items() if not (deps & set(remaining)))
        if not ready:
            cycle = sorted(remaining)
            plan.warnings.append(
                "circular references between "
                + ", ".join(cycle)
                + ": their constraints are checked when the export commits rather than row "
                "by row, because no insert order satisfies a cycle"
            )
            for name in cycle:
                spec = specs[by_name[name]]
                for fk in spec.foreign_keys:
                    if fk.enforced and fk.target_table in remaining:
                        fk.deferrable = True
            ready = cycle
        for name in ready:
            ordered.append(name)
            remaining.pop(name, None)
    return [by_name[name] for name in ordered]


# ---------------------------------------------------------------------------
# the plan as SQLAlchemy, and as text
# ---------------------------------------------------------------------------


def build_metadata(plan: ExportPlan, schema: str | None = None) -> MetaData:
    """The plan as SQLAlchemy objects, which is what quotes every identifier."""

    metadata = MetaData(schema=schema)
    for spec in plan.tables:
        columns = [
            Column(column.name, column.type_plan.sql_type, nullable=column.nullable)
            for column in spec.columns
        ]
        constraints: list[Any] = []
        if spec.primary_key:
            constraints.append(PrimaryKeyConstraint(*spec.primary_key))
        for fk in spec.foreign_keys:
            if not fk.enforced:
                continue
            target = f"{fk.target_table}.{fk.target_column}"
            constraints.append(
                ForeignKeyConstraint(
                    [fk.source_column],
                    [f"{schema}.{target}" if schema else target],
                    name=sanitize_identifier(
                        f"fk_{fk.source_table}_{fk.source_column}", fallback="fk"
                    ),
                    # A deferred constraint is added after the tables exist,
                    # which is also what makes a cycle creatable at all.
                    use_alter=fk.deferrable,
                    deferrable=fk.deferrable or None,
                    initially="DEFERRED" if fk.deferrable else None,
                )
            )
        Table(spec.name, metadata, *columns, *constraints)
    return metadata


def render_ddl(plan: ExportPlan, schema: str | None = None) -> str:
    """The ``CREATE TABLE`` script as PostgreSQL would spell it.

    Shown to the user and offered as a download: the export is meant to be
    something they can take somewhere else, and this is the part that says
    exactly what was built.
    """

    dialect = postgresql.dialect()
    metadata = build_metadata(plan, schema=schema)
    statements: list[str] = []
    if schema:
        statements.append(f"CREATE SCHEMA IF NOT EXISTS {schema};")
    for spec in plan.tables:
        table = metadata.tables[f"{schema}.{spec.name}" if schema else spec.name]
        statements.append(str(CreateTable(table).compile(dialect=dialect)).strip() + ";")
    return "\n\n".join(statements) + "\n"


def type_summary(plan: ExportPlan) -> list[dict[str, str]]:
    """Every column whose declared type had to be widened to fit its values."""

    return [
        {
            "table": spec.name,
            "column": column.name,
            "sql_type": column.type_plan.render(),
            "note": note,
        }
        for spec in plan.tables
        for column in spec.columns
        for note in column.type_plan.notes
    ]


__all__ = [
    "ColumnSpec",
    "ExportPlan",
    "ForeignKeySpec",
    "TableSpec",
    "build_metadata",
    "build_plan",
    "render_ddl",
    "render_type",
    "type_summary",
]
