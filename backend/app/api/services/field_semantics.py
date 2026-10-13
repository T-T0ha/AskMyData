"""Phase 1 — field semantics: column types, taxonomy labels and descriptions."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.api.services.ingestion import analyzable_tables
from app.api.services.profiling import sync_key_roles, table_type_of
from app.cleaning.store import get_store
from app.core.config import get_settings
from app.db.models import (
    ColumnSemantics,
    EquivalenceCandidateRecord,
    IngestionSession,
    SheetRecord,
)
from app.semantics.descriptions import is_template_description, template_description
from app.semantics.embeddings import get_embedder
from app.semantics.pipeline import analyze_tables, statistical_summary

logger = logging.getLogger(__name__)


def _semantic_description(column: ColumnSemantics, table_type: str = "data table") -> str:
    """Deterministic fallback description, used when Claude is unavailable.

    Mechanical but always correct.  :func:`_describe_columns` upgrades these to
    Claude-written sentences when an API key is configured.
    """

    return template_description(
        column.column_name,
        column.table_name,
        column.effective_label,
        column.effective_type,
        table_type,
        column.sample_values or [],
    )


#: Columns per description request.  A model's latency is dominated by how much
#: it has to write, so a few shorter requests in parallel finish well before one
#: long one — but every request also counts against the model's quota, and a
#: free-tier key allows only a few dozen a day.  Sixty columns keeps a typical
#: dataset (a hundred-odd columns) to two or three requests.
DESCRIBE_CHUNK_COLUMNS = 60
DESCRIBE_MAX_WORKERS = 4


def _describe_in_chunks(
    claude, tables_payload: list[dict[str, Any]]
) -> dict[str, dict[str, str]]:
    """``claude.describe_columns_dataset`` split into parallel column chunks.

    A chunk may hold part of a table; the results are merged per
    ``table -> column``.  A chunk that fails costs only its own columns their
    Claude-written description — they keep the deterministic template.
    """

    chunks: list[list[dict[str, Any]]] = [[]]
    room = DESCRIBE_CHUNK_COLUMNS
    for table in tables_payload:
        columns = list(table["columns"])
        while columns:
            take, columns = columns[:room], columns[room:]
            chunks[-1].append({"table": table["table"], "columns": take})
            room -= len(take)
            if room == 0:
                chunks.append([])
                room = DESCRIBE_CHUNK_COLUMNS
    chunks = [chunk for chunk in chunks if chunk]

    def run(chunk: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
        try:
            return claude.describe_columns_dataset(chunk) or {}
        except Exception as exc:  # pragma: no cover - defensive, network dependent
            logger.warning("column description failed: %s", exc)
            return {}

    if len(chunks) == 1:
        results = [run(chunks[0])]
    else:
        with ThreadPoolExecutor(max_workers=min(DESCRIBE_MAX_WORKERS, len(chunks))) as pool:
            results = list(pool.map(run, chunks))

    merged: dict[str, dict[str, str]] = {}
    for result in results:
        for table_name, columns in result.items():
            merged.setdefault(table_name, {}).update(columns)
    return merged


def _describe_columns(rows: list[ColumnSemantics], claude=None) -> None:
    """Phase 1's final step — rich semantic descriptions for pgvector.

    One Claude call for the WHOLE dataset (never per table, never per
    column).  Claude sees column names, detected types, taxonomy labels and
    five sample values — never a data row.  Any column Claude does not
    return keeps its deterministic description, so a partial response
    degrades instead of leaving a gap.
    """

    if not rows or claude is None or not claude.available:
        return

    by_table: dict[str, list[ColumnSemantics]] = {}
    for row in rows:
        by_table.setdefault(row.table_name, []).append(row)

    settings = get_settings()
    tables_payload = [
        {
            "table": table_name,
            "columns": [
                {
                    "name": column.column_name,
                    "detected_type": column.effective_type,
                    "taxonomy": column.effective_label,
                    "null_ratio": round(column.null_ratio, 3),
                    "is_additive": column.is_additive,
                    "sample_values": (column.sample_values or [])[: settings.plan_sample_values],
                }
                for column in columns
            ],
        }
        for table_name, columns in by_table.items()
    ]
    descriptions = _describe_in_chunks(claude, tables_payload)
    if not descriptions:
        return
    for table_name, columns in by_table.items():
        table_descriptions = descriptions.get(table_name, {})
        for column in columns:
            described = table_descriptions.get(column.column_name)
            if described:
                column.semantic_description = described


def run_field_semantics(
    db: Session, record: IngestionSession, claude=None
) -> list[ColumnSemantics]:
    """Phase 1 over every analyzable table; replaces prior results."""

    tables = analyzable_tables(db, record.id)
    if not tables:
        return []

    reports = {
        sheet.table_name: sheet.clean_reports or []
        for sheet in db.execute(
            select(SheetRecord).where(SheetRecord.session_id == record.id)
        )
        .scalars()
        .all()
    }

    class _Report:  # minimal adapter so the pipeline can read stored dicts
        __slots__ = ("column", "detected_currency_symbol")

        def __init__(self, payload: dict[str, Any]) -> None:
            self.column = payload.get("column", "")
            self.detected_currency_symbol = payload.get("detected_currency_symbol")

    adapted = {
        name: [_Report(r) for r in payload] for name, payload in reports.items()
    }

    # Preserve the user's manual overrides across a re-run.
    overrides = {
        (c.table_name, c.column_name): (c.user_column_type, c.user_taxonomy_label, c.validated)
        for c in db.execute(
            select(ColumnSemantics).where(ColumnSemantics.session_id == record.id)
        )
        .scalars()
        .all()
    }
    db.execute(delete(ColumnSemantics).where(ColumnSemantics.session_id == record.id))

    semantics = analyze_tables(tables, adapted, claude=claude)
    # What each table *is* goes into every one of its column descriptions, and
    # those descriptions are what Phase 5 retrieves on.  A session profiled
    # earlier already knows; one that has not reached Phase 3 yet says "data
    # table" and is rewritten when profiling runs.
    table_types = {name: table_type_of(db, record.id, name) for name in semantics}
    rows: list[ColumnSemantics] = []
    for table_name, table in semantics.items():
        for profile in table.columns:
            key = (table_name, profile.name)
            user_type, user_label, validated = overrides.get(key, (None, None, False))
            row = ColumnSemantics(
                session_id=record.id,
                table_name=table_name,
                column_name=profile.name,
                original_name=profile.original_name,
                column_type=profile.column_type.value,
                type_confidence=profile.type_confidence,
                type_evidence=profile.type_evidence,
                taxonomy_label=profile.taxonomy_label,
                taxonomy_confidence=profile.taxonomy_confidence,
                taxonomy_source=profile.taxonomy_source,
                taxonomy_rule=profile.taxonomy_rule,
                user_column_type=user_type,
                user_taxonomy_label=user_label,
                validated=validated,
                nullable=profile.nullable,
                null_ratio=profile.null_ratio,
                unique_count=profile.unique_count,
                unique_ratio=profile.unique_ratio,
                row_count=profile.row_count,
                is_additive=profile.is_additive,
                sample_values=profile.sample_values,
                distribution=profile.distribution,
            )
            row.semantic_description = _semantic_description(
                row, table_type=table_types.get(table_name, "data table")
            )
            db.add(row)
            rows.append(row)

    # Upgrade the template descriptions to Claude-written ones before embedding:
    # the embedding is only ever as good as the sentence it is built from.
    _describe_columns(rows, claude=claude)
    _embed_columns(rows)
    record.status = "semantics_ready"
    db.flush()
    # These rows were just replaced, so the key and reference roles that were
    # projected onto the old ones went with them.  They are derived from
    # decisions that outlive Phase 1, so they are put back rather than lost.
    sync_key_roles(db, record.id)
    return rows


def _embed_columns(rows: list[ColumnSemantics]) -> None:
    """Embed each column's semantic description (pgvector column)."""

    if not rows:
        return
    try:
        vectors = get_embedder().encode([r.semantic_description for r in rows])
    except Exception as exc:  # pragma: no cover - depends on environment
        logger.warning("embedding failed: %s", exc)
        return
    for row, vector in zip(rows, np.asarray(vectors), strict=False):
        row.embedding = [float(v) for v in vector]


def semantics_payload(db: Session, session_id: str) -> dict[str, Any]:
    columns = list(
        db.execute(
            select(ColumnSemantics)
            .where(ColumnSemantics.session_id == session_id)
            .order_by(ColumnSemantics.table_name, ColumnSemantics.id)
        )
        .scalars()
        .all()
    )
    tables: dict[str, list[dict[str, Any]]] = {}
    for column in columns:
        tables.setdefault(column.table_name, []).append(column.to_dict())
    return {
        "tables": [
            {"table": name, "columns": entries, "column_count": len(entries)}
            for name, entries in tables.items()
        ],
        "unknown_count": sum(1 for c in columns if c.effective_label == "unknown"),
        "validated_count": sum(1 for c in columns if c.validated),
        "total_columns": len(columns),
        # Columns whose description is still the mechanical template — the
        # language model was unavailable (a rate limit, no key) when they were
        # described.  Questions find columns through these descriptions, so the
        # screen says so and offers to run it again.
        "basic_description_count": sum(
            1 for c in columns if is_template_description(c.semantic_description)
        ),
    }


def build_summary(db: Session, record: IngestionSession) -> dict[str, Any]:
    """The row-free statistical summary handed to Claude in Phase 2."""

    from app.semantics.pipeline import TableSemantics  # local import: avoids cycle
    from app.core.schemas import ColumnProfile, ColumnType

    tables = analyzable_tables(db, record.id)
    columns = list(
        db.execute(select(ColumnSemantics).where(ColumnSemantics.session_id == record.id))
        .scalars()
        .all()
    )
    semantics: dict[str, TableSemantics] = {}
    for column in columns:
        table = semantics.setdefault(
            column.table_name,
            TableSemantics(table=column.table_name, row_count=column.row_count),
        )
        table.columns.append(
            ColumnProfile(
                table=column.table_name,
                name=column.column_name,
                original_name=column.original_name,
                column_type=ColumnType(column.effective_type),
                type_confidence=column.type_confidence,
                taxonomy_label=column.effective_label,
                taxonomy_confidence=column.taxonomy_confidence,
                taxonomy_source=column.taxonomy_source,
                null_ratio=column.null_ratio,
                unique_count=column.unique_count,
                unique_ratio=column.unique_ratio,
                row_count=column.row_count,
                sample_values=column.sample_values or [],
                distribution=column.distribution or {},
                is_additive=column.is_additive,
            )
        )

    equivalences = [
        e.to_dict()
        for e in db.execute(
            select(EquivalenceCandidateRecord).where(
                EquivalenceCandidateRecord.session_id == record.id
            )
        )
        .scalars()
        .all()
    ]
    duplicates = {name: int(df.duplicated().sum()) for name, df in tables.items()}
    return statistical_summary(semantics, equivalences, duplicates)


def table_preview(session_id: str, table: str, limit: int = 25) -> dict[str, Any]:
    from app.semantics.profiling import _json_safe  # local import: avoids cycle

    store = get_store(session_id)
    df = store.get(table)
    head = df.head(limit)
    return {
        "table": table,
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "columns": [str(c) for c in df.columns],
        "dtypes": {str(c): str(t) for c, t in df.dtypes.items()},
        "rows": [
            {str(k): _json_safe(v) for k, v in record.items()}
            for record in head.to_dict(orient="records")
        ],
    }
