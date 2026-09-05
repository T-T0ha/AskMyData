"""Service layer — the phase orchestration the API routes call into.

Keeps FastAPI handlers thin: a route parses the request, calls one function
here, and serialises the result.
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.cleaning.store import get_store
from app.core.config import get_settings
from app.core.schemas import (
    TABLE_LABELS,
    RelationshipOrigin,
    RelationshipStatus,
    RelationshipType,
    TableType,
    TriageStatus,
)
from app.db.models import (
    ColumnSemantics,
    DashboardCard,
    EquivalenceCandidateRecord,
    IngestionSession,
    QueryHistoryRecord,
    RelationshipRecord,
    SemanticLayerExport,
    SemanticLayerVersion,
    SheetRecord,
)
from app.export.documentation import render_markdown
from app.export.metadata import bundle as semantic_bundle
from app.export.runner import drop_export, explain, read_semantic_layer, run_export, run_readonly
from app.export.schema import build_plan
from app.ingestion.equivalence import detect_equivalences
from app.ingestion.keys import KeyAnalysis, confirm_key, decline_key, is_unique_key
from app.ingestion.loader import load_file_source
from app.ingestion.relational import display_url, load_relational_tables
from app.ingestion.triage import classify_issues
from app.query.guard import QueryRejected, validate_select_only
from app.query.retrieval import (
    expand_by_references,
    rank_tables,
    schema_context,
    select_columns,
    select_tables,
)
from app.query.shape import classify as classify_query_shape
from app.relationships.dependencies import detect_all_dependencies
from app.relationships.evidence import build_evidence
from app.relationships.foreign_keys import (
    detect_foreign_keys,
    skipped_tables as fk_skipped_tables,
)
from app.semantics.embeddings import get_embedder
from app.semantics.pipeline import analyze_tables, statistical_summary
from app.semantics import table_summary
from app.semantics.table_profile import TableProfile, profile_tables as build_table_profiles

logger = logging.getLogger(__name__)

#: Sheets below this row count are ingested but excluded from semantic
#: analysis — statistics over two rows are noise, not signal.
MIN_ANALYZABLE_ROWS = 3

#: Archived semantic layers kept per dataset.  Enough to compare a run against
#: the one before it and to see a trend; bounded because each one is a JSON
#: document of the whole schema, and an unbounded history of them is a slow
#: leak in the control plane.
MAX_LAYER_VERSIONS = 10


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
    # SemTabla runs table profiling as Step 4, on the output of the three
    # detection steps before it.  This is that step.
    payload["profiles"] = profile_tables(db, record)["profiles"]
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


# ---------------------------------------------------------------------------
# Phase 4 — table semantic profiling (SemTabla Step 4)
# ---------------------------------------------------------------------------


def _sheets_by_name(db: Session, session_id: str) -> dict[str, SheetRecord]:
    return {
        sheet.table_name: sheet
        for sheet in db.execute(select(SheetRecord).where(SheetRecord.session_id == session_id))
        .scalars()
        .all()
    }


def profile_tables(db: Session, record: IngestionSession, claude=None) -> dict[str, Any]:
    """Recompute every table's semantic profile and store it on its sheet.

    Cheap enough to run whenever an input changes — it reads statistics the
    earlier phases already computed, and touches the data itself only to
    measure the interval of a time key.  That is what makes the paper's
    behaviour possible: "the profiles of the data tables update accordingly
    when users modify detailed semantic features".

    The user's own corrections are carried across the rebuild, never
    recomputed.
    """

    tables = analyzable_tables(db, record.id)
    sheets = _sheets_by_name(db, record.id)
    columns: dict[str, list[dict[str, Any]]] = {}
    for column in (
        db.execute(select(ColumnSemantics).where(ColumnSemantics.session_id == record.id))
        .scalars()
        .all()
    ):
        columns.setdefault(column.table_name, []).append(column.to_dict())

    relationships = [
        row.to_dict()
        | {"status": row.status}  # to_dict omits it; the profiler reads verdicts
        for row in db.execute(
            select(RelationshipRecord).where(RelationshipRecord.session_id == record.id)
        )
        .scalars()
        .all()
    ]

    previous = {
        name: profile
        for name, sheet in sheets.items()
        if (profile := TableProfile.from_dict(sheet.semantic_profile)) is not None
    }

    profiles = build_table_profiles(
        tables,
        columns,
        key_analyses={name: sheet.key_analysis or {} for name, sheet in sheets.items()},
        relationships=relationships,
        header_info={name: sheet.header_info or {} for name, sheet in sheets.items()},
        previous=previous,
    )

    for name, profile in profiles.items():
        sheet = sheets.get(name)
        if sheet is not None:
            sheet.semantic_profile = profile.to_dict()
    db.flush()

    # Everything downstream of "what is this table" is derived from it, so it
    # is refreshed here rather than left for the export to discover: the key
    # and reference roles the column semantics carry, and the one-line
    # description Phase 5 pre-selects tables on.
    sync_key_roles(db, record.id)
    describe_tables(db, record, claude=claude)
    return profiles_payload(db, record.id)


def profiles_payload(db: Session, session_id: str) -> dict[str, Any]:
    """Every stored profile, plus the counts the stage header shows."""

    # The description is stored on the sheet rather than inside the profile —
    # it is composed from the profile *and* the schema around it — so it is
    # merged in here, where the panel that shows the profile can read it.
    profiles = [
        sheet.semantic_profile | {"description": sheet.effective_description}
        for sheet in _sheets_by_name(db, session_id).values()
        if sheet.semantic_profile
    ]
    profiles.sort(key=lambda payload: str(payload.get("table", "")))
    by_type: dict[str, int] = {}
    for payload in profiles:
        by_type[str(payload.get("effective_type", TableType.UNKNOWN.value))] = (
            by_type.get(str(payload.get("effective_type", TableType.UNKNOWN.value)), 0) + 1
        )
    return {
        "profiles": profiles,
        "counts": {
            "total": len(profiles),
            "unknown": by_type.get(TableType.UNKNOWN.value, 0),
            "by_type": by_type,
            "corrected": sum(1 for p in profiles if p.get("user_table_type")),
        },
    }


def override_profile(
    db: Session,
    record: IngestionSession,
    table: str,
    table_type: str | None = None,
    add_label: str | None = None,
    remove_label: str | None = None,
) -> dict[str, Any]:
    """The paper's Table Profiling Validation: the user corrects the machine.

    Corrections are held beside the detected values rather than over them, so
    the detector's accuracy is still measurable afterwards.  Passing an empty
    ``table_type`` withdraws an override and returns the table to whatever the
    tree currently says.
    """

    sheet = db.execute(
        select(SheetRecord)
        .where(SheetRecord.session_id == record.id)
        .where(SheetRecord.table_name == table)
    ).scalar_one_or_none()
    if sheet is None:
        raise KeyError(f"session {record.id} has no table {table!r}")

    profile = TableProfile.from_dict(sheet.semantic_profile)
    if profile is None:
        raise ValueError(f"{table} has not been profiled yet")

    if table_type is not None:
        if table_type == "":
            profile.user_table_type = None
        elif table_type not in {member.value for member in TableType}:
            raise ValueError(f"{table_type!r} is not a table type")
        else:
            profile.user_table_type = table_type

    if add_label is not None:
        if add_label not in TABLE_LABELS:
            raise ValueError(f"{add_label!r} is not one of the thirteen table labels")
        profile.removed_labels = [n for n in profile.removed_labels if n != add_label]
        if add_label not in {label.name for label in profile.labels}:
            profile.added_labels = [*profile.added_labels, add_label]

    if remove_label is not None:
        if remove_label not in TABLE_LABELS:
            raise ValueError(f"{remove_label!r} is not one of the thirteen table labels")
        profile.added_labels = [n for n in profile.added_labels if n != remove_label]
        if remove_label in {label.name for label in profile.labels}:
            profile.removed_labels = [
                *(n for n in profile.removed_labels if n != remove_label),
                remove_label,
            ]

    sheet.semantic_profile = profile.to_dict()
    db.flush()
    # The description opens with what the table *is*, so a correction to that
    # has to reach it — otherwise the user renames a fact table and Phase 5
    # keeps being told it is a lookup.
    describe_tables(db, record)
    return sheet.semantic_profile


def table_type_of(db: Session, session_id: str, table: str) -> str:
    """The table type as it should read inside a semantic description.

    Falls back to the neutral "data table" when profiling has not run, which
    is what every description said before Phase 4 existed.
    """

    sheet = db.execute(
        select(SheetRecord)
        .where(SheetRecord.session_id == session_id)
        .where(SheetRecord.table_name == table)
    ).scalar_one_or_none()
    profile = TableProfile.from_dict(sheet.semantic_profile) if sheet else None
    if profile is None or profile.effective_type == TableType.UNKNOWN.value:
        return "data table"
    return profile.describe()


def sync_key_roles(db: Session, session_id: str) -> int:
    """Project the confirmed keys and references onto the column semantics.

    ``ColumnSemantics`` is the layer Phase 5 retrieves against, and a retrieval
    that cannot tell an identifier from a measure will join on the wrong thing.
    The facts themselves live where they were decided — the primary key in the
    sheet's ``key_analysis``, the references in ``relationships`` — so this
    copies rather than decides, and can be re-run at any time.

    Every row is cleared before anything is set, because a key the user
    withdrew or a reference they rejected has to *stop* being true here; a
    function that only ever sets flags would leave the retracted ones standing.

    Returns how many columns carry a role afterwards.
    """

    rows = list(
        db.execute(select(ColumnSemantics).where(ColumnSemantics.session_id == session_id))
        .scalars()
        .all()
    )
    if not rows:
        return 0
    by_column = {(row.table_name, row.column_name): row for row in rows}
    for row in rows:
        row.is_primary_key = False
        row.is_foreign_key = False
        row.references_table = None
        row.references_column = None

    for table_name, sheet in _sheets_by_name(db, session_id).items():
        for column_name in (sheet.key_analysis or {}).get("primary_key") or []:
            row = by_column.get((table_name, str(column_name)))
            if row is not None:
                row.is_primary_key = True

    for relation in (
        db.execute(
            select(RelationshipRecord)
            .where(RelationshipRecord.session_id == session_id)
            .where(RelationshipRecord.rel_type == RelationshipType.FOREIGN_KEY.value)
            .where(RelationshipRecord.status == RelationshipStatus.CONFIRMED.value)
        )
        .scalars()
        .all()
    ):
        row = by_column.get((relation.from_table, relation.from_column))
        if row is not None:
            row.is_foreign_key = True
            row.references_table = relation.to_table
            row.references_column = relation.to_column

    db.flush()
    return sum(1 for row in rows if row.is_primary_key or row.is_foreign_key)


def describe_tables(db: Session, record: IngestionSession, claude=None) -> dict[str, str]:
    """Compose (and optionally have Claude rewrite) one sentence per table.

    Called from profiling, so a description exists as soon as the platform has
    an opinion about what a table is, and is rebuilt whenever that opinion
    changes.  When the composed sentence changes, any Claude-written one is
    dropped: it was written about facts that no longer hold, and a stale
    sentence is worse than a plain one because Phase 5 believes it.
    """

    sheets = _sheets_by_name(db, record.id)
    if not sheets:
        return {}

    columns: dict[str, list[dict[str, Any]]] = {}
    for column in (
        db.execute(select(ColumnSemantics).where(ColumnSemantics.session_id == record.id))
        .scalars()
        .all()
    ):
        columns.setdefault(column.table_name, []).append(column.to_dict())

    relationships = [
        row.to_dict() | {"status": row.status}
        for row in db.execute(
            select(RelationshipRecord).where(RelationshipRecord.session_id == record.id)
        )
        .scalars()
        .all()
    ]

    live = {name: sheet for name, sheet in sheets.items() if not sheet.skipped}
    # Row counts come from the store, not from the sheet record: the sheet
    # remembers what was ingested, and a cleaning step that removed duplicate
    # rows would otherwise be described away.
    measured = {meta.name: (meta.row_count, meta.column_count) for meta in get_store(record.id).meta()}
    outgoing, incoming = table_summary.relationship_map(relationships)
    composed = table_summary.describe_tables(
        profiles={
            name: profile
            for name, sheet in live.items()
            if (profile := TableProfile.from_dict(sheet.semantic_profile)) is not None
        },
        columns=columns,
        key_analyses={name: sheet.key_analysis or {} for name, sheet in live.items()},
        relationships=relationships,
        counts={
            name: measured.get(name, (sheet.row_count or 0, sheet.column_count or 0))
            for name, sheet in live.items()
        },
    )

    changed: list[str] = []
    for name, sentence in composed.items():
        sheet = live[name]
        if sheet.table_description != sentence:
            sheet.table_description = sentence
            sheet.llm_table_description = None
            changed.append(name)

    _describe_tables_with_claude(
        live,
        composed,
        changed,
        columns,
        outgoing,
        incoming,
        {name: rows for name, (rows, _) in measured.items()},
        claude=claude,
    )
    _embed_tables(live, changed)
    db.flush()
    return {name: sheet.effective_description for name, sheet in live.items()}


def _describe_tables_with_claude(
    sheets: dict[str, SheetRecord],
    composed: dict[str, str],
    changed: list[str],
    columns: dict[str, list[dict[str, Any]]],
    outgoing: dict[str, list[tuple[str, str]]],
    incoming: dict[str, list[str]],
    counts: dict[str, int],
    claude=None,
) -> None:
    """One call per dataset, and only for tables whose facts moved.

    Profiling re-runs on every relationship decision.  Rewriting forty
    sentences each time a user clicks "confirm" would spend money to produce
    the sentences that are already there, so the model is asked only about the
    tables whose composed description actually changed.
    """

    if not changed or claude is None or not getattr(claude, "available", False):
        return

    payloads = []
    for name in changed:
        sheet = sheets[name]
        profile = TableProfile.from_dict(sheet.semantic_profile)
        key_analysis = sheet.key_analysis or {}
        payloads.append(
            table_summary.payload_for_claude(
                table=name,
                profile=profile,
                columns=columns.get(name) or [],
                primary_key=[str(k) for k in key_analysis.get("primary_key") or []],
                references=outgoing.get(name, []),
                referenced_by=incoming.get(name, []),
                row_count=counts.get(name, sheet.row_count or 0),
            )
        )

    try:
        written = claude.describe_tables(payloads)
    except Exception as exc:  # pragma: no cover - defensive, network dependent
        logger.warning("table descriptions were not rewritten: %s", exc)
        return
    for name, description in (written or {}).items():
        sheet = sheets.get(name)
        if sheet is not None and description:
            sheet.llm_table_description = description


def _embed_tables(sheets: dict[str, SheetRecord], changed: list[str]) -> None:
    """Embed the descriptions that moved, plus any that were never embedded."""

    pending = [
        name
        for name, sheet in sheets.items()
        if name in changed or sheet.table_embedding is None
    ]
    if not pending:
        return
    vectors = table_summary.embed([sheets[name].effective_description for name in pending])
    if vectors is None:
        return
    for name, vector in zip(pending, vectors, strict=False):
        sheets[name].table_embedding = vector


# ---------------------------------------------------------------------------
# Phase 4 — semantic layer construction and export
# ---------------------------------------------------------------------------


def _export_inputs(
    db: Session, record: IngestionSession
) -> tuple[dict, dict, dict, list, dict, dict]:
    """Everything the export plan reads, gathered from the earlier phases.

    The tables here are *every* ingested table, not the analyzable subset the
    other phases work on.  A four-row lookup sheet is below the floor for
    statistics — a distribution over four values says nothing — but it is still
    the user's data, and a database that quietly omits it is wrong in the one
    way this project cannot afford.  What it loses is its detected types, so it
    exports as text, and the plan says so.
    """

    sheets = _sheets_by_name(db, record.id)
    tables = {
        name: df
        for name, df in get_store(record.id).tables().items()
        if not (sheets.get(name) is not None and sheets[name].skipped)
    }

    columns: dict[str, list[dict[str, Any]]] = {}
    for column in (
        db.execute(select(ColumnSemantics).where(ColumnSemantics.session_id == record.id))
        .scalars()
        .all()
    ):
        columns.setdefault(column.table_name, []).append(column.to_dict())

    relationships = [
        row.to_dict() | {"status": row.status}
        for row in db.execute(
            select(RelationshipRecord).where(RelationshipRecord.session_id == record.id)
        )
        .scalars()
        .all()
    ]

    keys = {name: sheet.key_analysis or {} for name, sheet in sheets.items()}
    profiles = {name: sheet.semantic_profile or {} for name, sheet in sheets.items()}
    descriptions = {name: sheet.effective_description for name, sheet in sheets.items()}
    return tables, columns, keys, relationships, profiles, descriptions


def export_semantic_layer(db: Session, record: IngestionSession, claude=None) -> dict[str, Any]:
    """Build the database and its semantic layer.  Returns the report.

    Everything the export asserts comes from a decision the user has already
    made or can see: the primary keys they confirmed, the references they
    approved, the types they corrected.  What it *cannot* assert — a reference
    with orphan rows, a key that cleaning broke — is reported rather than
    forced, which is why the report is part of the return value and not a log
    line.
    """

    tables, columns, keys, relationships, profiles, descriptions = _export_inputs(db, record)
    if not tables:
        raise ValueError("there is nothing to export yet — upload a file first")

    if not columns:
        # The export reads types and labels off Phase 1.  Without it every
        # column would land as TEXT, which is a worse database than the user
        # spent the session building.
        run_field_semantics(db, record, claude=claude)
        _, columns, _, _, _, _ = _export_inputs(db, record)

    # §4.3: the layer is *completed* here.  A user can reach the export without
    # passing through the relationship stage, so the two things derived from
    # the rest of the layer — the key and reference roles on the columns, and
    # the one-line table descriptions — are brought up to date before the plan
    # reads them, rather than exported as whatever they happened to be.
    sync_key_roles(db, record.id)
    descriptions = describe_tables(db, record, claude=claude) or descriptions

    plan = build_plan(tables, columns, keys, relationships, profiles, descriptions)
    plan.skipped.extend(
        {
            "table": sheet.table_name,
            "reason": sheet.skip_reason or "the sheet could not be read during ingestion",
        }
        for sheet in _sheets_by_name(db, record.id).values()
        if sheet.skipped
    )
    for spec in plan.tables:
        if not columns.get(spec.source_name):
            spec.notes.append(
                "too small for semantic analysis, so every column is exported as text — "
                "the rows are all here, their detected types are not"
            )
    report = run_export(record.id, plan, tables)
    payload = report.to_dict()
    payload["cleaning"] = _cleaning_caveat(db, record.id)

    now = datetime.now(timezone.utc)
    # Re-running enrichment produces a new version of the layer rather than an
    # edit to the old one (FR-17).  The counter lives on the dataset because it
    # is a fact about the dataset's meaning, not about this particular export.
    record.semantic_version = (record.semantic_version or 0) + 1
    record.enriched_at = now
    record.db_schema_name = payload["target"]
    payload["semantic_version"] = record.semantic_version

    row = db.execute(
        select(SemanticLayerExport).where(SemanticLayerExport.session_id == record.id)
    ).scalar_one_or_none()
    if row is None:
        row = SemanticLayerExport(session_id=record.id)
        db.add(row)
    row.target = payload["target"]
    row.dialect = payload["dialect"]
    row.status = "ok" if not payload["warnings"] else "ok_with_warnings"
    row.semantic_version = record.semantic_version
    row.table_count = payload["table_count"]
    row.row_count = payload["row_count"]
    row.column_count = payload["metadata_rows"]
    row.embedded_count = payload["embedded_rows"]
    row.vector_index = payload["vector_index"]
    row.warning_count = len(payload["warnings"])
    row.report = payload
    row.bundle = semantic_bundle(plan)
    row.updated_at = now

    _archive_layer(db, record, row)
    record.status = "exported"
    db.flush()
    return {**row.to_dict(), "report": payload}


def _archive_layer(
    db: Session, record: IngestionSession, export: SemanticLayerExport
) -> SemanticLayerVersion:
    """Keep this build of the layer beside the ones before it.

    The archive holds the JSON layer only.  An export replaces the schema it
    writes into, so an earlier version describes tables that no longer exist —
    what is worth keeping is what the platform *believed* about the data at
    that point, which is what a before/after quality comparison reads.
    """

    version = SemanticLayerVersion(
        session_id=record.id,
        version=record.semantic_version,
        target=export.target,
        dialect=export.dialect,
        table_count=export.table_count,
        column_count=export.column_count,
        row_count=export.row_count,
        warning_count=export.warning_count,
        bundle=export.bundle or {},
    )
    db.add(version)
    db.flush()

    stale = list(
        db.execute(
            select(SemanticLayerVersion)
            .where(SemanticLayerVersion.session_id == record.id)
            .order_by(SemanticLayerVersion.version.desc())
            .offset(MAX_LAYER_VERSIONS)
        )
        .scalars()
        .all()
    )
    for old_version in stale:
        db.delete(old_version)
    if stale:
        db.flush()
    return version


def layer_versions(db: Session, session_id: str) -> list[SemanticLayerVersion]:
    """Every archived build of a session's layer, newest first."""

    return list(
        db.execute(
            select(SemanticLayerVersion)
            .where(SemanticLayerVersion.session_id == session_id)
            .order_by(SemanticLayerVersion.version.desc())
        )
        .scalars()
        .all()
    )


def layer_bundle(db: Session, session_id: str, version: int | None = None) -> dict[str, Any] | None:
    """The portable JSON layer — the current one, or an archived version.

    Returns ``None`` when there is no such version, which the route turns into
    a 404: asking for version 7 of a dataset exported twice is a mistake worth
    being told about rather than answered with the newest one.
    """

    if version is None:
        row = export_record(db, session_id)
        return None if row is None else (row.bundle or {})

    archived = db.execute(
        select(SemanticLayerVersion)
        .where(SemanticLayerVersion.session_id == session_id)
        .where(SemanticLayerVersion.version == version)
    ).scalar_one_or_none()
    return None if archived is None else (archived.bundle or {})


def layer_documentation(
    db: Session, record: IngestionSession, version: int | None = None
) -> str | None:
    """The semantic layer as Markdown documentation (FR-18).

    Rendered from the stored bundle, so documenting an earlier version is the
    same operation as documenting the current one — and neither re-analyses
    anything.
    """

    bundle = layer_bundle(db, record.id, version)
    if bundle is None:
        return None
    export = export_record(db, record.id)
    archived = None
    if version is not None:
        archived = next(
            (row for row in layer_versions(db, record.id) if row.version == version), None
        )
    stamp = archived.created_at if archived is not None else (
        export.updated_at if export is not None else None
    )
    return render_markdown(
        bundle,
        dataset=record.name,
        target=(archived.target if archived is not None else (export.target if export else "")),
        dialect=(archived.dialect if archived is not None else (export.dialect if export else "")),
        version=(
            archived.version
            if archived is not None
            else (export.semantic_version if export else 0)
        ),
        exported_at=stamp.strftime("%Y-%m-%d %H:%M UTC") if stamp else "",
    )


def _cleaning_caveat(db: Session, session_id: str) -> dict[str, Any]:
    """Whether the export is of a dataset whose cleaning plan is still open.

    Not an error — a user may export at any point, and the export always
    reflects the tables as they are now.  But "you exported halfway through
    your own cleaning plan" is something they should be told rather than left
    to notice.
    """

    from app.db.models import CleaningRun  # local import: avoids a cycle

    run = db.execute(
        select(CleaningRun).where(CleaningRun.session_id == session_id)
    ).scalar_one_or_none()
    if run is None:
        return {"status": "not_started", "complete": True}
    complete = run.status in {"completed", "cancelled", "aborted", "not_started"}
    return {
        "status": run.status,
        "complete": complete,
        "completed_steps": run.completed_steps,
        "step_count": run.step_count,
    }


def export_record(db: Session, session_id: str) -> SemanticLayerExport | None:
    return db.execute(
        select(SemanticLayerExport).where(SemanticLayerExport.session_id == session_id)
    ).scalar_one_or_none()


def export_payload(db: Session, session_id: str) -> dict[str, Any]:
    """The last export, or an explicit "not yet"."""

    row = export_record(db, session_id)
    if row is None:
        return {"exported": False, "report": None}
    return {"exported": True, **row.to_dict()}


def forget_export(session_id: str) -> None:
    """Drop a session's exported database.  Called when the session is deleted."""

    drop_export(session_id)


# ---------------------------------------------------------------------------
# Phase 5 — natural language query
# ---------------------------------------------------------------------------


def _record_history(db: Session, session_id: str, question: str, result: dict[str, Any]) -> None:
    """Append one question-history row, then trim to the configured limit.

    Written for a failed question too — the sidebar's "last N questions" is a
    log of what was asked, not only of what worked.  Trimming here rather than
    at read time keeps the table itself bounded, since nothing else deletes
    from it.
    """

    db.add(
        QueryHistoryRecord(
            session_id=session_id,
            question=question,
            ok=bool(result.get("ok")),
            sql=result.get("sql"),
            error=result.get("error"),
            chart=(result.get("visualization") or {}).get("chart"),
        )
    )
    db.flush()
    limit = get_settings().query_history_limit
    stale = (
        db.execute(
            select(QueryHistoryRecord.id)
            .where(QueryHistoryRecord.session_id == session_id)
            .order_by(QueryHistoryRecord.created_at.desc())
            .offset(limit)
        )
        .scalars()
        .all()
    )
    if stale:
        db.execute(delete(QueryHistoryRecord).where(QueryHistoryRecord.id.in_(stale)))


def question_history(db: Session, session_id: str) -> list[dict[str, Any]]:
    """The last :attr:`~app.core.config.Settings.query_history_limit` questions, newest first."""

    rows = (
        db.execute(
            select(QueryHistoryRecord)
            .where(QueryHistoryRecord.session_id == session_id)
            .order_by(QueryHistoryRecord.created_at.desc())
            .limit(get_settings().query_history_limit)
        )
        .scalars()
        .all()
    )
    return [row.to_dict() for row in rows]


def answer_question(
    db: Session, record: IngestionSession, question: str, claude=None
) -> dict[str, Any]:
    """Turn one natural-language question into SQL, run it once, and shape the result.

    Retrieval (:mod:`app.query.retrieval`) never touches a data row — it
    compares the question's embedding against table sentences and column
    descriptions Phase 4 already wrote.  Generation asks Claude for SQL over
    that retrieved schema alone, never over the tables themselves.  Only the
    one statement that survives :func:`~app.query.guard.validate_select_only`
    and an ``EXPLAIN`` (§Phase 5's self-correction loop) ever reaches the
    exported database, and it runs exactly once, row-capped, at the very end.

    Unlike taxonomy labelling or plan generation, there is no rule-based
    fallback for "turn English into SQL" — so with no model available this
    reports that plainly (``ok: False``) rather than guessing.  Any other
    failure to produce a usable query (the guard rejects every attempt, the
    database itself rejects the SQL) is reported the same way: a real result
    for the user to read, not a 500.
    """

    question = question.strip()
    if not question:
        raise ValueError("ask a question first")

    export = export_record(db, record.id)
    if export is None:
        raise ValueError("export the semantic layer before asking questions")

    rows = read_semantic_layer(record.id)
    if not rows:
        raise ValueError("the semantic layer has no columns to query yet")

    settings = get_settings()
    question_vector = get_embedder().encode([question])[0]

    sheets = _sheets_by_name(db, record.id)
    exported_tables = sorted({str(row.get("table_name")) for row in rows})
    table_vectors = {
        name: (sheets[name].table_embedding if name in sheets else None)
        for name in exported_tables
    }

    scored = rank_tables(question_vector, table_vectors)
    selected = select_tables(scored, settings.query_max_tables)
    selected = expand_by_references(selected, rows)
    selected_columns = select_columns(
        question_vector, rows, selected, settings.query_max_ranked_columns
    )
    context = schema_context(selected_columns)
    allowed_tables = {name.lower() for name in selected}
    dialect = export.dialect

    attempts: list[dict[str, Any]] = []
    sql: str | None = None
    explanation = ""
    last_error: str | None = None
    prior_sql: str | None = None

    if claude is None or not claude.available:
        result = {
            "ok": False,
            "question": question,
            "tables_considered": selected,
            "attempts": [],
            "error": (
                (claude.last_error if claude is not None else None)
                or "no language model is configured, so this question could not be "
                "turned into SQL"
            ),
        }
        _record_history(db, record.id, question, result)
        return result

    for attempt_no in range(1, settings.query_max_attempts + 1):
        generated = claude.generate_sql(
            question, dialect, context, prior_sql=prior_sql, prior_error=last_error
        )
        if generated is None:
            last_error = claude.last_error or "the language model returned no usable answer"
            attempts.append({"attempt": attempt_no, "sql": None, "error": last_error})
            break

        prior_sql = generated["sql"]
        try:
            validated = validate_select_only(generated["sql"], allowed_tables, dialect)
        except QueryRejected as exc:
            last_error = str(exc)
            attempts.append({"attempt": attempt_no, "sql": generated["sql"], "error": last_error})
            continue

        try:
            explain(record.id, validated)
        except Exception as exc:  # the driver's own exception type, dialect-dependent
            last_error = str(exc)
            attempts.append({"attempt": attempt_no, "sql": validated, "error": last_error})
            continue

        sql = validated
        explanation = generated["explanation"]
        attempts.append({"attempt": attempt_no, "sql": validated, "error": None})
        break

    if sql is None:
        result = {
            "ok": False,
            "question": question,
            "tables_considered": selected,
            "attempts": attempts,
            "error": last_error or "the model could not produce a usable query for this question",
        }
        _record_history(db, record.id, question, result)
        return result

    columns, result_rows, truncated = run_readonly(record.id, sql, settings.query_row_cap)
    visualization = classify_query_shape(columns, result_rows)
    # A dashboard nicety, not part of the answer itself: a fake standing in for
    # Claude in a test need not implement it, and an unavailable/degraded model
    # simply yields no suggestions rather than breaking the question that just
    # succeeded.
    suggest = getattr(claude, "suggest_followups", None)
    suggestions = (suggest(question, sql, columns, selected) if suggest else None) or []

    result = {
        "ok": True,
        "question": question,
        "sql": sql,
        "explanation": explanation,
        "dialect": dialect,
        "columns": columns,
        "rows": result_rows,
        "row_count": len(result_rows),
        "truncated": truncated,
        "tables_used": selected,
        "attempts": attempts,
        "visualization": visualization,
        "suggestions": suggestions,
    }
    _record_history(db, record.id, question, result)
    return result


# ---------------------------------------------------------------------------
# Phase 6 — dashboard
# ---------------------------------------------------------------------------

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def pin_dashboard_card(
    db: Session, record: IngestionSession, payload: dict[str, Any]
) -> DashboardCard:
    """Save one answered question as a dashboard card.

    The SQL is re-validated here, independently of whatever validated it when
    the question was first answered: a pin request is client-supplied input
    like any other, and this is the write path that turns a stored statement
    into something that runs, unattended, on every future dashboard load.
    Re-checked against every exported table rather than only the ones that
    question's retrieval step picked — a card outlives the request that made
    it, so the wider, still-honest bound is the one worth keeping.
    """

    question = str(payload.get("question") or "").strip()
    sql = str(payload.get("sql") or "").strip()
    if not question or not sql:
        raise ValueError("a card needs both a question and its SQL")

    export = export_record(db, record.id)
    if export is None:
        raise ValueError("export the semantic layer before pinning a card")

    rows = read_semantic_layer(record.id)
    exported_tables = {str(row.get("table_name")).lower() for row in rows}
    try:
        validated_sql = validate_select_only(sql, exported_tables, export.dialect)
    except QueryRejected as exc:
        raise ValueError(f"this query can no longer be pinned: {exc}") from exc

    position = db.execute(
        select(func.count())
        .select_from(DashboardCard)
        .where(DashboardCard.session_id == record.id)
    ).scalar_one()

    card = DashboardCard(
        session_id=record.id,
        title=str(payload.get("title") or "").strip() or question,
        question=question,
        sql=validated_sql,
        dialect=export.dialect,
        explanation=str(payload.get("explanation") or ""),
        tables_used=list(payload.get("tables_used") or []),
        visualization=dict(payload.get("visualization") or {}),
        position=position,
    )
    db.add(card)
    db.flush()
    return card


def _quote_identifier(name: str) -> str | None:
    if not _IDENTIFIER_RE.match(name or ""):
        return None
    return f'"{name}"'


def _date_filtered_sql(card: DashboardCard, start: date | None, end: date | None) -> tuple[str, bool]:
    """Wrap a card's SQL in a date-range ``WHERE``, when it unambiguously can be.

    Only applies when the card's own shape names exactly one date column —
    for anything else there is no column a *global* filter could mean, so the
    card runs unfiltered rather than guessing one.  ``start``/``end`` arrive as
    parsed :class:`datetime.date` objects (the route's own parameter type), so
    interpolating their ``isoformat()`` never carries anything but digits and
    hyphens into the statement.
    """

    if start is None and end is None:
        return card.sql, False
    date_columns = (card.visualization or {}).get("date_columns") or []
    if len(date_columns) != 1:
        return card.sql, False
    column = _quote_identifier(date_columns[0])
    if column is None:
        return card.sql, False

    clauses = []
    if start is not None:
        clauses.append(f"{column} >= '{start.isoformat()}'")
    if end is not None:
        clauses.append(f"{column} < '{end.isoformat()}'")
    where = " AND ".join(clauses)
    return f"SELECT * FROM ({card.sql}) AS _dashboard_filtered WHERE {where}", True


def list_dashboard_cards(
    db: Session, record: IngestionSession, start: date | None = None, end: date | None = None
) -> list[dict[str, Any]]:
    """Every pinned card, re-executed against the live export.

    "Results always current, not snapshots" (§Phase 6) means the stored row is
    never trusted for data — even a card whose SQL now fails (a column renamed
    by a later re-export) is reported per-card as ``ok: False``, rather than
    one bad card taking the whole dashboard down.
    """

    cards = (
        db.execute(
            select(DashboardCard)
            .where(DashboardCard.session_id == record.id)
            .order_by(DashboardCard.position)
        )
        .scalars()
        .all()
    )
    settings = get_settings()
    refreshed_at = datetime.now(timezone.utc).isoformat()
    payload: list[dict[str, Any]] = []
    for card in cards:
        item = card.to_dict()
        statement, filtered = _date_filtered_sql(card, start, end)
        try:
            columns, rows, truncated = run_readonly(record.id, statement, settings.query_row_cap)
            item.update(
                ok=True,
                columns=columns,
                rows=rows,
                row_count=len(rows),
                truncated=truncated,
                error=None,
            )
        except Exception as exc:  # the driver's own exception type, dialect-dependent
            item.update(ok=False, columns=[], rows=[], row_count=0, truncated=False, error=str(exc))
        item["date_filtered"] = filtered
        item["refreshed_at"] = refreshed_at
        payload.append(item)
    return payload


def update_dashboard_card(
    db: Session, record: IngestionSession, card_id: str, patch: dict[str, Any]
) -> DashboardCard:
    """Rename a card or move/resize it.  Never touches its question or SQL —
    re-pin to change what a card answers."""

    card = db.execute(
        select(DashboardCard).where(
            DashboardCard.id == card_id, DashboardCard.session_id == record.id
        )
    ).scalar_one_or_none()
    if card is None:
        raise KeyError(f"dashboard card {card_id} not found")
    if patch.get("title") is not None:
        card.title = str(patch["title"]).strip() or card.question
    if patch.get("layout") is not None:
        card.layout = dict(patch["layout"])
    db.flush()
    return card


def delete_dashboard_card(db: Session, record: IngestionSession, card_id: str) -> None:
    db.execute(
        delete(DashboardCard).where(
            DashboardCard.id == card_id, DashboardCard.session_id == record.id
        )
    )


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
