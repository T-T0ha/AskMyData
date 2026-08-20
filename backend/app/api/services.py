"""Service layer — the phase orchestration the API routes call into.

Keeps FastAPI handlers thin: a route parses the request, calls one function
here, and serialises the result.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.cleaning.store import get_store
from app.core.config import get_settings
from app.core.schemas import (
    RelationshipOrigin,
    RelationshipStatus,
    RelationshipType,
    TriageStatus,
)
from app.db.models import (
    ColumnSemantics,
    EquivalenceCandidateRecord,
    IngestionSession,
    RelationshipRecord,
    SheetRecord,
)
from app.ingestion.equivalence import detect_equivalences
from app.ingestion.keys import KeyAnalysis, confirm_key, decline_key, is_unique_key
from app.ingestion.loader import load_file_source
from app.ingestion.relational import display_url, load_relational_tables
from app.ingestion.triage import classify_issues
from app.relationships.dependencies import detect_all_dependencies
from app.relationships.evidence import build_evidence
from app.relationships.foreign_keys import (
    detect_foreign_keys,
    skipped_tables as fk_skipped_tables,
)
from app.semantics.embeddings import get_embedder
from app.semantics.pipeline import analyze_tables, statistical_summary

logger = logging.getLogger(__name__)

#: Sheets below this row count are ingested but excluded from semantic
#: analysis — statistics over two rows are noise, not signal.
MIN_ANALYZABLE_ROWS = 3


def create_session(db: Session, name: str, owner_id: str) -> IngestionSession:
    record = IngestionSession(
        name=name or "Untitled session", status="created", user_id=owner_id
    )
    db.add(record)
    db.flush()
    return record


# Loading one dataset by id is deliberately *not* offered here: it is an
# authorisation decision, and there is exactly one implementation of it —
# ``app.auth.deps.owned_session``, which every dataset-scoped route depends on.
# A second, owner-optional lookup in this module would be the one a future
# caller reaches for by accident.


def list_sessions(db: Session, owner_id: str) -> list[IngestionSession]:
    return list(
        db.execute(
            select(IngestionSession)
            .where(IngestionSession.user_id == owner_id)
            .order_by(IngestionSession.created_at.desc())
        )
        .scalars()
        .all()
    )


# ---------------------------------------------------------------------------
# Phase 0
# ---------------------------------------------------------------------------


def ingest_file(db: Session, record: IngestionSession, path: Path) -> list[SheetRecord]:
    """Run Phase 0 over one uploaded file and persist the results."""

    return _persist_results(db, record, load_file_source(path), source_label=path.name)


def ingest_database(
    db: Session,
    record: IngestionSession,
    url: str,
    tables: list[str] | None = None,
) -> list[SheetRecord]:
    """Run Phase 0 over selected tables of a live database connection.

    The connection string is used and discarded.  What is persisted is the
    password-free display form, because a session record that stores database
    credentials is a liability the platform has no reason to take on.
    """

    results = load_relational_tables(url, tables=tables)
    return _persist_results(db, record, results, source_label=display_url(url))


def _persist_results(
    db: Session,
    record: IngestionSession,
    results: list[Any],
    source_label: str,
) -> list[SheetRecord]:
    """Store one source's tables, whatever kind of source produced them."""

    store = get_store(record.id)
    existing = {s.table_name for s in record.sheets}
    sheets: list[SheetRecord] = []

    for result in results:
        # Sheet names collide across files; disambiguate rather than overwrite.
        name = result.name
        suffix = 2
        while name in existing:
            name = f"{result.name}_{suffix}"
            suffix += 1
        existing.add(name)
        result.name = name

        if not result.skipped and not result.dataframe.empty:
            store.put(name, result.dataframe)

        sheet = SheetRecord(
            session_id=record.id,
            table_name=name,
            source_name=result.source_name,
            source_file=source_label,
            source_kind=result.source_kind,
            native_schema=result.native_schema,
            key_analysis=result.key_analysis.to_dict(),
            triage=result.triage.value,
            row_count=result.row_count,
            column_count=result.column_count,
            skipped=result.skipped,
            skip_reason=result.skip_reason,
            header_info=result.summary()["header"],
            clean_reports=[r.to_dict() for r in result.clean_reports],
            issues=[i.to_dict() for i in result.issues],
        )
        db.add(sheet)
        sheets.append(sheet)

    store.persist()
    files = list(record.source_files or [])
    if source_label not in files:
        files.append(source_label)
    record.source_files = files
    record.status = "ingested"
    record.stats = {**(record.stats or {}), **triage_stats(db, record)}
    db.flush()
    return sheets


def triage_stats(db: Session, record: IngestionSession) -> dict[str, Any]:
    counts = {status.value: 0 for status in TriageStatus}
    total_rows = 0
    sheets = list(
        db.execute(select(SheetRecord).where(SheetRecord.session_id == record.id))
        .scalars()
        .all()
    )
    for sheet in sheets:
        counts[sheet.triage] = counts.get(sheet.triage, 0) + 1
        total_rows += sheet.row_count
    return {
        "sheet_count": len(sheets),
        "total_rows": total_rows,
        "triage_counts": counts,
    }


def analyzable_tables(db: Session, session_id: str) -> dict[str, Any]:
    """Tables large enough for semantic analysis."""

    store = get_store(session_id)
    skipped = {
        s.table_name
        for s in db.execute(select(SheetRecord).where(SheetRecord.session_id == session_id))
        .scalars()
        .all()
        if s.skipped or s.row_count < MIN_ANALYZABLE_ROWS
    }
    return {name: df for name, df in store.tables().items() if name not in skipped}


def table_keys(db: Session, session_id: str) -> dict[str, dict[str, Any]]:
    """Every table's key analysis, as the planner and Phase 3 need it."""

    return {
        sheet.table_name: sheet.key_analysis or {}
        for sheet in db.execute(select(SheetRecord).where(SheetRecord.session_id == session_id))
        .scalars()
        .all()
    }


def set_table_key(
    db: Session,
    record: IngestionSession,
    table: str,
    columns: list[str] | None,
) -> SheetRecord:
    """Record the user's answer to "what identifies a row in this table?".

    An empty ``columns`` list is a rejection — the user is saying none of the
    candidates is the real key — and hands the table to the synthetic-key
    branch of the plan.  A non-empty one is verified against every row before
    it is accepted: confirming a key that does not hold would put a
    ``PRIMARY KEY`` on the export that the data cannot satisfy.
    """

    sheet = db.execute(
        select(SheetRecord)
        .where(SheetRecord.session_id == record.id)
        .where(SheetRecord.table_name == table)
    ).scalar_one_or_none()
    if sheet is None:
        raise KeyError(f"session {record.id} has no table {table!r}")

    analysis = KeyAnalysis.from_dict(sheet.key_analysis)

    if columns:
        df = get_store(record.id).get(table)
        missing = [c for c in columns if c not in {str(x) for x in df.columns}]
        if missing:
            raise ValueError(f"{table} has no column(s) {', '.join(missing)}")
        if not is_unique_key(df, columns):
            subset = df.loc[:, columns]
            empty = int(subset.isna().any(axis=1).sum())
            repeated = int(subset.duplicated().sum())
            reason = (
                f"{repeated:,} row(s) repeat the same value" if repeated else ""
            ) or f"{empty:,} row(s) leave it empty"
            raise ValueError(
                f"{' + '.join(columns)} cannot be the key of {table}: {reason}"
            )
        confirm_key(analysis, columns)
    else:
        decline_key(analysis)

    sheet.key_analysis = analysis.to_dict()
    # The question has been answered, so the issue that asked it goes away and
    # the sheet is re-classified on what is left.
    issues = [
        issue
        for issue in (sheet.issues or [])
        if issue.get("code") not in {"composite_key_candidate", "no_primary_key"}
    ]
    if analysis.needs_synthetic_key:
        issues.append(
            {
                "code": "no_primary_key",
                "severity": "info",
                "message": (
                    "No candidate key was accepted for this table; the cleaning plan will "
                    "offer to add a numbered row_id."
                ),
                "columns": [],
            }
        )
    sheet.issues = issues
    sheet.triage = classify_issues(issues).value
    record.stats = {**(record.stats or {}), **triage_stats(db, record)}
    db.flush()
    return sheet


def detect_and_store_equivalences(
    db: Session, record: IngestionSession
) -> list[EquivalenceCandidateRecord]:
    """Phase 0 cross-sheet equivalence detection."""

    tables = analyzable_tables(db, record.id)
    db.execute(
        delete(EquivalenceCandidateRecord).where(
            EquivalenceCandidateRecord.session_id == record.id
        )
    )
    candidates = detect_equivalences(tables)
    rows: list[EquivalenceCandidateRecord] = []
    for candidate in candidates:
        payload = candidate.to_dict()
        row = EquivalenceCandidateRecord(
            session_id=record.id,
            left_table=payload["left_table"],
            left_column=payload["left_column"],
            right_table=payload["right_table"],
            right_column=payload["right_column"],
            score=payload["score"],
            embedding_similarity=payload["embedding_similarity"],
            lexical_similarity=payload["lexical_similarity"],
            value_overlap=payload["value_overlap"],
            type_compatible=payload["type_compatible"],
            explanation=payload["explanation"],
            confirmed=None,
        )
        db.add(row)
        rows.append(row)
    db.flush()
    return rows


# ---------------------------------------------------------------------------
# Phase 1
# ---------------------------------------------------------------------------


def _semantic_description(column: ColumnSemantics, table_type: str = "data table") -> str:
    """Deterministic fallback description, used when Claude is unavailable.

    Mechanical but always correct.  :func:`_describe_columns` upgrades these to
    Claude-written sentences when an API key is configured.
    """

    samples = ", ".join(str(v) for v in (column.sample_values or [])[:5])
    parts = [
        column.column_name.replace("_", " "),
        f"in table {column.table_name}",
        f"({column.effective_label.replace('_', ' ')}, {column.effective_type})",
        f"— {table_type}",
    ]
    if samples:
        parts.append(f"example values: {samples}")
    return " ".join(parts)


def _describe_columns(rows: list[ColumnSemantics], claude=None) -> None:
    """Phase 1's final step — rich semantic descriptions for pgvector.

    One Claude call per table (never per column).  Claude sees column names,
    detected types, taxonomy labels and five sample values — never a data row.
    Any column Claude does not return keeps its deterministic description, so a
    partial response degrades instead of leaving a gap.
    """

    if not rows or claude is None or not claude.available:
        return

    by_table: dict[str, list[ColumnSemantics]] = {}
    for row in rows:
        by_table.setdefault(row.table_name, []).append(row)

    settings = get_settings()
    for table_name, columns in by_table.items():
        payload = [
            {
                "name": column.column_name,
                "detected_type": column.effective_type,
                "taxonomy": column.effective_label,
                "null_ratio": round(column.null_ratio, 3),
                "is_additive": column.is_additive,
                "sample_values": (column.sample_values or [])[: settings.plan_sample_values],
            }
            for column in columns
        ]
        try:
            descriptions = claude.describe_columns(table_name, payload)
        except Exception as exc:  # pragma: no cover - defensive, network dependent
            logger.warning("column description failed for %s: %s", table_name, exc)
            continue
        if not descriptions:
            continue
        for column in columns:
            described = descriptions.get(column.column_name)
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
            row.semantic_description = _semantic_description(row)
            db.add(row)
            rows.append(row)

    # Upgrade the template descriptions to Claude-written ones before embedding:
    # the embedding is only ever as good as the sentence it is built from.
    _describe_columns(rows, claude=claude)
    _embed_columns(rows)
    record.status = "semantics_ready"
    db.flush()
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


# ---------------------------------------------------------------------------
# Phase 3
# ---------------------------------------------------------------------------

#: Score band in which a candidate is worth explaining in prose.  Above it the
#: arithmetic speaks for itself; below it the candidate is barely proposed.
LLM_EXPLAIN_BAND = (0.50, 0.65)


def _key_analyses(db: Session, session_id: str) -> dict[str, KeyAnalysis]:
    return {
        table: KeyAnalysis.from_dict(payload)
        for table, payload in table_keys(db, session_id).items()
    }


def _native_schemas(db: Session, session_id: str) -> dict[str, dict[str, Any]]:
    return {
        sheet.table_name: sheet.native_schema or {}
        for sheet in db.execute(select(SheetRecord).where(SheetRecord.session_id == session_id))
        .scalars()
        .all()
    }


def _explain_candidate(db: Session, session_id: str, payload: dict[str, Any], claude: Any) -> str | None:
    """Ask Claude to read a borderline candidate, if there is one to ask."""

    if claude is None or not getattr(claude, "available", False):
        return None
    columns = {
        (c.table_name, c.column_name): c
        for c in db.execute(
            select(ColumnSemantics).where(ColumnSemantics.session_id == session_id)
        )
        .scalars()
        .all()
    }

    def describe(table: str, column: str) -> dict[str, Any]:
        row = columns.get((table, column))
        return {
            "table": table,
            "column": column,
            "meaning": row.effective_label if row else None,
            "type": row.effective_type if row else None,
            # Five values, as everywhere else: the model reasons about the
            # shape of a column, and never needs the column itself.
            "sample_values": (row.sample_values or [])[:5] if row else [],
        }

    return claude.explain_relationship(
        {
            "relationship": payload["rel_type"],
            "source": describe(payload["from_table"], payload["from_column"]),
            "target": describe(payload["to_table"], payload["to_column"]),
            "statistics": payload.get("evidence", {}),
            "score": payload.get("score"),
        }
    )


def _upsert_relationship(
    db: Session,
    session_id: str,
    payload: dict[str, Any],
) -> tuple[RelationshipRecord, bool]:
    """Store one candidate.  Returns ``(record, is_new)``.

    A relationship the user has already decided on is left exactly as it is.
    Re-running detection is something a user does after cleaning changed the
    data, and having it quietly resurrect a rejected edge — or downgrade a
    confirmed one — would make the confirm button meaningless.
    """

    existing = db.execute(
        select(RelationshipRecord).where(
            RelationshipRecord.session_id == session_id,
            RelationshipRecord.rel_type == payload["rel_type"],
            RelationshipRecord.from_table == payload["from_table"],
            RelationshipRecord.from_column == payload["from_column"],
            RelationshipRecord.to_table == payload["to_table"],
            RelationshipRecord.to_column == payload["to_column"],
        )
    ).scalar_one_or_none()

    if existing is not None:
        if existing.status == RelationshipStatus.PROPOSED.value:
            existing.score = payload["score"]
            existing.origin = payload["origin"]
            existing.evidence = payload.get("evidence", {})
            existing.explanation = payload.get("explanation", "")
        return existing, False

    record = RelationshipRecord(
        session_id=session_id,
        rel_type=payload["rel_type"],
        from_table=payload["from_table"],
        from_column=payload["from_column"],
        to_table=payload["to_table"],
        to_column=payload["to_column"],
        score=payload["score"],
        origin=payload["origin"],
        status=RelationshipStatus.PROPOSED.value,
        evidence=payload.get("evidence", {}),
        explanation=payload.get("explanation", ""),
    )
    db.add(record)
    return record, True


def detect_relationships(
    db: Session,
    record: IngestionSession,
    claude: Any = None,
) -> dict[str, Any]:
    """Phase 3: find foreign keys and functional dependencies, and persist them.

    Detection is algorithmic throughout.  ``claude`` is optional and is used
    only to add a prose reading of candidates that landed in the borderline
    band — it can neither create a candidate nor change a score.
    """

    tables = analyzable_tables(db, record.id)
    keys = _key_analyses(db, record.id)
    key_columns = {table: analysis.primary_key for table, analysis in keys.items()}

    candidates = detect_foreign_keys(
        tables, keys, native_schemas=_native_schemas(db, record.id)
    )
    dependencies = detect_all_dependencies(tables, key_columns=key_columns)

    created = 0
    for candidate in candidates:
        payload = candidate.to_dict()
        stored, is_new = _upsert_relationship(db, record.id, payload)
        created += int(is_new)
        low, high = LLM_EXPLAIN_BAND
        if is_new and low <= payload["score"] <= high and stored.llm_explanation is None:
            stored.llm_explanation = _explain_candidate(db, record.id, payload, claude)

    for dependency in dependencies:
        _, is_new = _upsert_relationship(db, record.id, dependency.to_dict())
        created += int(is_new)

    db.flush()
    payload = relationships_payload(db, record.id)
    payload["created"] = created
    return payload


def relationships_payload(db: Session, session_id: str) -> dict[str, Any]:
    """Everything the relationship stage renders."""

    rows = list(
        db.execute(
            select(RelationshipRecord)
            .where(RelationshipRecord.session_id == session_id)
            .order_by(RelationshipRecord.score.desc())
        )
        .scalars()
        .all()
    )
    tables = analyzable_tables(db, session_id)
    by_status: dict[str, int] = {status.value: 0 for status in RelationshipStatus}
    for row in rows:
        by_status[row.status] = by_status.get(row.status, 0) + 1

    return {
        "relationships": [row.to_dict() for row in rows],
        "counts": {
            "total": len(rows),
            "foreign_keys": sum(
                1 for r in rows if r.rel_type == RelationshipType.FOREIGN_KEY.value
            ),
            "dependencies": sum(
                1 for r in rows if r.rel_type != RelationshipType.FOREIGN_KEY.value
            ),
            **by_status,
        },
        # Named so the UI can explain an empty diagram instead of implying
        # the tables are unrelated.
        "skipped_tables": fk_skipped_tables(tables),
        "graph": relationship_graph(db, session_id),
    }


def relationship_graph(db: Session, session_id: str) -> dict[str, Any]:
    """Node-link payload for the force-directed diagram.

    Rejected relationships are omitted: the diagram is the picture of what the
    data model *is*, and the user has said these are not part of it.  They stay
    in the list so detection does not propose them again.
    """

    tables = analyzable_tables(db, session_id)
    keys = _key_analyses(db, session_id)
    semantics = {
        (c.table_name, c.column_name): c
        for c in db.execute(
            select(ColumnSemantics).where(ColumnSemantics.session_id == session_id)
        )
        .scalars()
        .all()
    }

    rows = [
        row
        for row in db.execute(
            select(RelationshipRecord).where(RelationshipRecord.session_id == session_id)
        )
        .scalars()
        .all()
        if row.status != RelationshipStatus.REJECTED.value
    ]

    foreign_key_columns = {
        (row.from_table, row.from_column)
        for row in rows
        if row.rel_type == RelationshipType.FOREIGN_KEY.value
    }

    nodes = []
    for table, df in sorted(tables.items()):
        primary = list(keys.get(table, KeyAnalysis()).primary_key or [])
        nodes.append(
            {
                "id": table,
                "row_count": int(len(df)),
                "primary_key": primary,
                "columns": [
                    {
                        "name": str(column),
                        "type": (
                            semantics[(table, str(column))].effective_type
                            if (table, str(column)) in semantics
                            else None
                        ),
                        "label": (
                            semantics[(table, str(column))].effective_label
                            if (table, str(column)) in semantics
                            else None
                        ),
                        "is_primary_key": str(column) in primary,
                        "is_foreign_key": (table, str(column)) in foreign_key_columns,
                    }
                    for column in df.columns
                ],
            }
        )

    links = [
        {
            "id": row.id,
            "source": row.from_table,
            "target": row.to_table,
            "from_column": row.from_column,
            "to_column": row.to_column,
            "rel_type": row.rel_type,
            "status": row.status,
            "origin": row.origin,
            "score": round(row.score, 4),
            "uncertain": row.origin == RelationshipOrigin.FUZZY.value,
        }
        for row in rows
        if row.rel_type == RelationshipType.FOREIGN_KEY.value
    ]

    #: Dependencies are drawn inside their table's expanded node, not between
    #: nodes — an arrow from a table to itself says nothing at this zoom level.
    dependencies: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.rel_type == RelationshipType.FOREIGN_KEY.value:
            continue
        dependencies.setdefault(row.from_table, []).append(
            {
                "id": row.id,
                "from_column": row.from_column,
                "to_column": row.to_column,
                "rel_type": row.rel_type,
                "status": row.status,
                "score": round(row.score, 4),
            }
        )

    return {"nodes": nodes, "links": links, "dependencies": dependencies}


def get_relationship(db: Session, session_id: str, relationship_id: str) -> RelationshipRecord:
    row = db.get(RelationshipRecord, relationship_id)
    if row is None or row.session_id != session_id:
        raise KeyError(f"relationship {relationship_id} not found")
    return row


def decide_relationship(
    db: Session,
    session_id: str,
    relationship_id: str,
    confirmed: bool,
) -> RelationshipRecord:
    """Record the user's verdict.  This is what Phase 4 will read."""

    row = get_relationship(db, session_id, relationship_id)
    row.status = (
        RelationshipStatus.CONFIRMED.value if confirmed else RelationshipStatus.REJECTED.value
    )
    row.decided_at = datetime.now(timezone.utc)
    db.flush()
    return row


def add_manual_relationship(
    db: Session,
    record: IngestionSession,
    rel_type: str,
    from_table: str,
    from_column: str,
    to_table: str,
    to_column: str,
) -> RelationshipRecord:
    """An edge the user drew.

    Arrives confirmed — the user asserting a relationship *is* the confirmation
    step, and asking them to confirm what they just drew would be theatre.  It
    is still validated for existence: an edge between columns that are not
    there would break the export rather than the diagram.
    """

    tables = analyzable_tables(db, record.id)
    for table, column in ((from_table, from_column), (to_table, to_column)):
        if table not in tables:
            raise ValueError(f"unknown table {table!r}")
        if column not in {str(c) for c in tables[table].columns}:
            raise ValueError(f"table {table!r} has no column {column!r}")
    if rel_type not in {t.value for t in RelationshipType}:
        raise ValueError(f"unknown relationship type {rel_type!r}")
    if rel_type == RelationshipType.FOREIGN_KEY.value and from_table == to_table:
        # Self-references are real, but a column referencing itself is not.
        if from_column == to_column:
            raise ValueError("a column cannot reference itself")
    if rel_type != RelationshipType.FOREIGN_KEY.value and from_table != to_table:
        raise ValueError("a functional dependency relates two columns of one table")

    existing = db.execute(
        select(RelationshipRecord).where(
            RelationshipRecord.session_id == record.id,
            RelationshipRecord.rel_type == rel_type,
            RelationshipRecord.from_table == from_table,
            RelationshipRecord.from_column == from_column,
            RelationshipRecord.to_table == to_table,
            RelationshipRecord.to_column == to_column,
        )
    ).scalar_one_or_none()
    if existing is not None:
        existing.status = RelationshipStatus.CONFIRMED.value
        existing.origin = RelationshipOrigin.MANUAL.value
        existing.decided_at = datetime.now(timezone.utc)
        db.flush()
        return existing

    row = RelationshipRecord(
        session_id=record.id,
        rel_type=rel_type,
        from_table=from_table,
        from_column=from_column,
        to_table=to_table,
        to_column=to_column,
        score=1.0,
        origin=RelationshipOrigin.MANUAL.value,
        status=RelationshipStatus.CONFIRMED.value,
        explanation="drawn by you in the relationship diagram",
        evidence={},
        decided_at=datetime.now(timezone.utc),
    )
    db.add(row)
    db.flush()
    return row


def relationship_evidence(db: Session, session_id: str, relationship_id: str) -> dict[str, Any]:
    """The rows that support and contradict one relationship.

    Computed on demand rather than at detection time: it reads whole tables,
    and a user confirms a handful of the relationships that get proposed.
    """

    from app.semantics.profiling import _json_safe  # local import: avoids cycle

    row = get_relationship(db, session_id, relationship_id)
    tables = analyzable_tables(db, session_id)
    evidence = build_evidence(
        row.rel_type,
        tables,
        row.from_table,
        row.from_column,
        row.to_table,
        row.to_column,
    )
    payload = evidence.to_dict()
    for sample in ("positive", "negative"):
        payload[sample]["rows"] = [
            {str(k): _json_safe(v) for k, v in item.items()}
            for item in payload[sample]["rows"]
        ]
    return {"relationship": row.to_dict(), "evidence": payload}


def backend_status() -> dict[str, Any]:
    from app.cleaning.checkpointing import checkpointer_backend
    from app.semantics.claude_client import get_claude_client

    settings = get_settings()
    embedder = get_embedder()
    return {
        "claude": get_claude_client().status(),
        "embeddings": {
            "model": embedder.name,
            "dim": embedder.dim,
            "semantic": embedder.is_semantic,
        },
        "checkpointer": checkpointer_backend(),
        "thresholds": {
            "equivalence": settings.equivalence_threshold,
            "numeric_convert": settings.numeric_convert_threshold,
            "date_parse": settings.date_parse_threshold,
            "date_type": settings.date_type_threshold,
        },
    }
