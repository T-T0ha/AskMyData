"""``sem_metadata`` — the semantic layer, written into the exported database.

The point of the whole platform is that the database it produces carries its
own meaning.  A ``VARCHAR(100)`` called ``cust_ref`` says nothing; the row in
``sem_metadata`` beside it says that it is a foreign identifier, that it
references ``customers.customer_id``, that it is never summed, that 4 % of it is
blank, and — in one sentence, embedded as a vector — what it is *for*.

Two consequences follow from putting the table **inside** the exported schema
rather than in the application's own database:

* the export is portable.  A user who takes the schema somewhere else takes the
  semantics with it, which is what makes the JSON bundle and this table two
  views of one thing rather than two separate exports;
* Phase 5 needs one connection, not two.  Its retrieval step is a nearest
  neighbour query against ``sem_metadata.embedding`` followed by a SQL query
  against the tables described in the same schema.

The description sentence is composed here, deterministically, rather than
reused from ``ColumnSemantics.semantic_description``.  They are different
sentences for different jobs: Phase 1's is about a column *name*, and is what
cross-sheet equivalence matches on; this one is about a column *in its finished
table*, and carries the physical type, the key role and the table type that
only exist once the export has decided them.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    text,
)

from app.core.config import get_settings
from app.db.models import EmbeddingVector
from app.export.naming import is_managed_schema
from app.export.schema import ColumnSpec, ExportPlan, TableSpec
from app.semantics.embeddings import get_embedder

logger = logging.getLogger(__name__)

#: The name is fixed: Phase 5 and anything else reading an exported database
#: has to be able to find it without being told.
SEM_METADATA_TABLE = "sem_metadata"

#: Sample values carried per column.  The same five the LLM prompt is allowed
#: to see — enough to disambiguate a code from a name, far short of the data.
SAMPLE_VALUES = 5


def build_table(metadata: MetaData) -> Table:
    """Define ``sem_metadata`` inside the export's own MetaData.

    ``embedding`` is a real ``vector(384)`` on PostgreSQL and a JSON array
    elsewhere — the same dialect-aware column ``column_semantics`` uses, so a
    development export on SQLite has the same shape as a production one.
    """

    return Table(
        SEM_METADATA_TABLE,
        metadata,
        Column("table_name", String(63), primary_key=True),
        Column("column_name", String(63), primary_key=True),
        # What the spreadsheet called them.  Kept so a user can find their own
        # column after it was renamed to something SQL will accept.
        Column("source_table", Text, nullable=False, default=""),
        Column("source_column", Text, nullable=False, default=""),
        Column("table_type", String(40), nullable=False, default="unknown"),
        Column("taxonomy_label", String(60), nullable=False, default="unknown"),
        Column("data_type", String(40), nullable=False, default="text"),
        Column("sql_type", String(60), nullable=False, default="TEXT"),
        Column("is_primary_key", Boolean, nullable=False, default=False),
        Column("is_foreign_key", Boolean, nullable=False, default=False),
        Column("references_table", String(63), nullable=True),
        Column("references_column", String(63), nullable=True),
        #: False for a reference that is real but not enforced by a constraint
        #: — Phase 5 may still join on it, and should know it is not policed.
        Column("reference_enforced", Boolean, nullable=False, default=False),
        Column("is_additive", Boolean, nullable=False, default=False),
        Column("is_nullable", Boolean, nullable=False, default=True),
        Column("null_ratio", Float, nullable=False, default=0.0),
        Column("row_count", Integer, nullable=False, default=0),
        Column("sample_values", JSON, nullable=False, default=list),
        Column("semantic_description", Text, nullable=False, default=""),
        Column("embedding", EmbeddingVector(get_settings().embedding_dim), nullable=True),
    )


def describe(column: ColumnSpec, table: TableSpec) -> str:
    """The sentence that gets embedded, and that a person can read.

    Built from what the analysis established, in the order a reader needs it:
    what it is, where it lives, what kind of thing it holds, what role it plays
    and what it looks like.
    """

    # A table the decision tree would not name is described as an ordinary
    # data table rather than as "unknown": the sentence is read by a person and
    # embedded for retrieval, and "unknown table" is worse than silence at both.
    table_type = table.table_type if table.table_type != "unknown" else "data"

    head = (
        f"{column.name.replace('_', ' ')} in {table.name.replace('_', ' ')} "
        f"({column.taxonomy_label.replace('_', ' ')}, {column.type_plan.render()})"
        f" — {table_type.replace('_', ' ')} table"
    )
    clauses: list[str] = []
    if column.is_primary_key:
        clauses.append("identifies the row")
    if column.is_foreign_key and column.references_table:
        clauses.append(f"references {column.references_table}.{column.references_column}")
    if column.is_additive:
        clauses.append("can be summed")
    if column.null_ratio > 0:
        clauses.append(f"{column.null_ratio:.0%} blank")
    samples = [str(value) for value in column.sample_values[:SAMPLE_VALUES] if value is not None]
    if samples:
        clauses.append("example values: " + ", ".join(samples))
    return "; ".join([head, *clauses])


def build_rows(plan: ExportPlan) -> list[dict[str, Any]]:
    """One row per exported column, in table order.  No embeddings yet."""

    enforced: dict[tuple[str, str], bool] = {
        (fk.source_table, fk.source_column): fk.enforced
        for table in plan.tables
        for fk in table.foreign_keys
    }

    rows: list[dict[str, Any]] = []
    for table in plan.tables:
        for column in table.columns:
            rows.append(
                {
                    "table_name": table.name,
                    "column_name": column.name,
                    "source_table": table.source_name,
                    "source_column": column.source_name,
                    "table_type": table.table_type,
                    "taxonomy_label": column.taxonomy_label,
                    "data_type": column.column_type.value,
                    "sql_type": column.type_plan.render(),
                    "is_primary_key": column.is_primary_key,
                    "is_foreign_key": column.is_foreign_key,
                    "references_table": column.references_table,
                    "references_column": column.references_column,
                    "reference_enforced": bool(
                        enforced.get((table.name, column.name), False)
                    ),
                    "is_additive": column.is_additive,
                    "is_nullable": column.nullable,
                    "null_ratio": float(column.null_ratio),
                    "row_count": table.row_count,
                    "sample_values": [
                        _json_safe(value) for value in column.sample_values[:SAMPLE_VALUES]
                    ],
                    "semantic_description": describe(column, table),
                    "embedding": None,
                }
            )
    return rows


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def embed_rows(rows: Sequence[dict[str, Any]]) -> int:
    """Fill in ``embedding`` for every row.  Returns how many were embedded.

    A failure here is not a failed export.  The tables, the constraints and
    every metadata field except the vector are already correct; what is lost is
    the *ranking* of Phase 5's retrieval, which then falls back to reading all
    of the metadata rather than the nearest fifteen rows.  Refusing to export
    over that would be the wrong trade.
    """

    if not rows:
        return 0
    try:
        vectors = get_embedder().encode([row["semantic_description"] for row in rows])
    except Exception as exc:  # pragma: no cover - depends on the environment
        logger.warning("semantic descriptions were not embedded: %s", exc)
        return 0

    array = np.asarray(vectors, dtype="float32")
    if array.ndim != 2 or len(array) != len(rows):
        logger.warning("embedder returned %s vectors for %s rows", len(array), len(rows))
        return 0
    for row, vector in zip(rows, array):
        row["embedding"] = [float(value) for value in vector]
    return len(rows)


def attach(plan: ExportPlan, embed: bool = True):
    """The ``extra`` callback :func:`app.export.runner.run_export` expects.

    Returns a function that adds ``sem_metadata`` to the export's MetaData and
    hands back its rows, so the metadata is created and filled in the same
    transaction as the data it describes.
    """

    def build(metadata: MetaData) -> list[tuple[Table, list[dict[str, Any]]]]:
        table = build_table(metadata)
        rows = build_rows(plan)
        if embed:
            embed_rows(rows)
        return [(table, rows)]

    return build


def create_vector_index(engine, schema: str | None) -> bool:
    """HNSW index over ``sem_metadata.embedding``.  PostgreSQL only.

    Returns whether the index exists afterwards.  Like the embedding itself,
    failing to build it costs speed rather than correctness — a sequential scan
    over a few hundred metadata rows answers the same query.
    """

    if schema is None:
        return False  # SQLite has no vector type to index
    if not is_managed_schema(schema):
        raise ValueError(f"refusing to index inside schema {schema!r}")
    statement = text(
        f'CREATE INDEX IF NOT EXISTS idx_{schema}_sem_metadata_embedding '
        f'ON "{schema}"."{SEM_METADATA_TABLE}" USING hnsw (embedding vector_cosine_ops)'
    )
    try:
        with engine.begin() as connection:
            connection.execute(statement)
        return True
    except Exception as exc:  # pragma: no cover - depends on the server
        logger.warning("could not create the HNSW index on %s: %s", schema, exc)
        return False


def bundle(plan: ExportPlan, rows: Iterable[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """The semantic layer as portable JSON — no vectors, no rows of data.

    Deliberately tool-agnostic: table and column names, types, keys,
    references, labels and descriptions.  The embeddings are left out because
    they are reproducible from the descriptions by anything that has the same
    model, and including 384 floats per column would multiply the file size for
    no information.
    """

    payload = list(rows) if rows is not None else build_rows(plan)
    return {
        "format": "semanticlayer/1",
        "tables": [
            {
                "name": table.name,
                "source_name": table.source_name,
                "table_type": table.table_type,
                "description": table.description,
                "row_count": table.row_count,
                "primary_key": table.primary_key,
                "foreign_keys": [fk.to_dict() for fk in table.foreign_keys],
                "notes": table.notes,
                "columns": [
                    {
                        key: value
                        for key, value in row.items()
                        if key not in {"embedding", "table_name"}
                    }
                    for row in payload
                    if row["table_name"] == table.name
                ],
            }
            for table in plan.tables
        ],
        "warnings": plan.warnings,
        "skipped": plan.skipped,
    }
