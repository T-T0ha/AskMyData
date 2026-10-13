"""Phase 0 — datasets, ingestion, triage and keys.

Creating and listing datasets, loading a file or a database into a session,
duplicate detection, the triage statistics, primary-key decisions and
cross-sheet equivalences.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

from sqlalchemy import delete, or_, select
from sqlalchemy.orm import Session

from app.cleaning.store import get_store
from app.core.schemas import StepType, TriageStatus
from app.db.models import (
    ColumnSemantics,
    EquivalenceCandidateRecord,
    IngestionSession,
    RelationshipRecord,
    SheetRecord,
)
from app.ingestion.duplicates import duplicate_reason, find_duplicate_tables, is_duplicate_reason
from app.ingestion.equivalence import auto_verdict, detect_equivalences
from app.ingestion.fingerprint import database_fingerprint, file_fingerprint
from app.ingestion.keys import (
    KeyAnalysis,
    adopt_synthetic_key,
    confirm_key,
    decline_key,
    is_unique_key,
    withdraw_synthetic_key,
)
from app.ingestion.loader import load_file_source
from app.ingestion.relational import display_url, load_relational_tables
from app.ingestion.triage import classify_issues, key_issues

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


class DuplicateSourceError(Exception):
    """A source's fingerprint matches one the same account already ingested.

    Carries the existing session so the API can point the caller at it —
    the useful response to "you already have this" is a link to it, not a
    bare refusal.
    """

    def __init__(self, existing_session: IngestionSession) -> None:
        self.existing_session = existing_session
        super().__init__(f"already ingested as session {existing_session.id!r}")


def find_duplicate_session(
    db: Session, record: IngestionSession, fingerprint: str
) -> IngestionSession | None:
    """Another session of the same account that already has this fingerprint.

    Ownerless sessions (predating authentication) are never matched — there is
    no account to have "already" ingested anything into.
    """

    if not record.user_id:
        return None
    for other in list_sessions(db, record.user_id):
        if other.id != record.id and fingerprint in (other.source_fingerprints or []):
            return other
    return None


def ingest_file(db: Session, record: IngestionSession, path: Path) -> list[SheetRecord]:
    """Run Phase 0 over one uploaded file and persist the results."""

    fingerprint = file_fingerprint(path)
    duplicate = find_duplicate_session(db, record, fingerprint)
    if duplicate is not None:
        raise DuplicateSourceError(duplicate)
    sheets = _persist_results(db, record, load_file_source(path), source_label=path.name)
    record.source_fingerprints = [*(record.source_fingerprints or []), fingerprint]
    return sheets


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

    fingerprint = database_fingerprint(url, tables)
    duplicate = find_duplicate_session(db, record, fingerprint)
    if duplicate is not None:
        raise DuplicateSourceError(duplicate)
    results = load_relational_tables(url, tables=tables)
    sheets = _persist_results(db, record, results, source_label=display_url(url))
    record.source_fingerprints = [*(record.source_fingerprints or []), fingerprint]
    return sheets


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

    db.flush()
    # The same data arriving twice — a file uploaded again, two sources that
    # both carry the sheet — is kept as a note, not as a second table to analyse.
    skip_duplicate_tables(db, record)
    store.persist()
    files = list(record.source_files or [])
    if source_label not in files:
        files.append(source_label)
    record.source_files = files
    record.status = "ingested"
    record.stats = {**(record.stats or {}), **triage_stats(db, record)}
    db.flush()
    return sheets


def skip_duplicate_tables(db: Session, record: IngestionSession) -> dict[str, str]:
    """Set aside every table that is an exact copy of an earlier one.

    Returns ``{copy: original}``.  A copy keeps its row in the triage list —
    with the reason, so the user can see what happened to the file they added —
    but leaves the store and everything derived from it: its column semantics,
    its relationships and its equivalence pairs.  Idempotent, and cheap when
    there is nothing to do, so it runs both when a source arrives and when an
    older session is opened (sessions made before this existed still hold
    their ``_2`` copies).
    """

    sheets = {
        sheet.table_name: sheet
        for sheet in db.execute(select(SheetRecord).where(SheetRecord.session_id == record.id))
        .scalars()
        .all()
    }
    if sum(1 for sheet in sheets.values() if not sheet.skipped) < 2:
        return {}

    store = get_store(record.id)
    live = {
        name: store.get(name)
        for name, sheet in sheets.items()
        if not sheet.skipped and store.has(name)
    }
    duplicates = find_duplicate_tables(live)
    if not duplicates:
        return {}

    names = list(duplicates)
    for copy, original in duplicates.items():
        sheet = sheets[copy]
        sheet.skipped = True
        sheet.skip_reason = duplicate_reason(original)
        sheet.triage = TriageStatus.CLEAN.value
        store.drop(copy)
    store.persist()

    db.execute(
        delete(ColumnSemantics).where(
            ColumnSemantics.session_id == record.id, ColumnSemantics.table_name.in_(names)
        )
    )
    db.execute(
        delete(RelationshipRecord).where(
            RelationshipRecord.session_id == record.id,
            or_(RelationshipRecord.from_table.in_(names), RelationshipRecord.to_table.in_(names)),
        )
    )
    db.execute(
        delete(EquivalenceCandidateRecord).where(
            EquivalenceCandidateRecord.session_id == record.id,
            or_(
                EquivalenceCandidateRecord.left_table.in_(names),
                EquivalenceCandidateRecord.right_table.in_(names),
            ),
        )
    )
    record.stats = {**(record.stats or {}), **triage_stats(db, record)}
    db.flush()
    logger.info("session %s: set aside duplicate table(s) %s", record.id, duplicates)
    return duplicates


def triage_stats(db: Session, record: IngestionSession) -> dict[str, Any]:
    counts = {status.value: 0 for status in TriageStatus}
    total_rows = 0
    sheets = [
        sheet
        for sheet in db.execute(select(SheetRecord).where(SheetRecord.session_id == record.id))
        .scalars()
        .all()
        # A set-aside copy is not a table of the dataset; counting it would
        # double its rows in the totals.
        if not is_duplicate_reason(sheet.skip_reason)
    ]
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

    _save_key_analysis(db, record, sheet, analysis)
    return sheet


def _save_key_analysis(
    db: Session, record: IngestionSession, sheet: SheetRecord, analysis: KeyAnalysis
) -> None:
    sheet.key_analysis = analysis.to_dict()
    # The question has been answered, so the issue that asked it goes away and
    # the sheet is re-classified on what is left.
    issues = [
        issue
        for issue in (sheet.issues or [])
        if issue.get("code") not in {"composite_key_candidate", "no_primary_key"}
    ]
    if analysis.needs_confirmation:
        # Asked again (a withdrawn row_id): the same question triage asked.
        issues.extend(
            issue.to_dict()
            for issue in key_issues(analysis)
            if issue.code == "composite_key_candidate"
        )
    elif analysis.needs_synthetic_key:
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


def sync_synthetic_keys(db: Session, session_id: str, plan: Sequence[dict[str, Any]]) -> None:
    """Make a row_id the cleaning plan added the table's key, while it exists.

    ``add_synthetic_key`` runs inside the cleaning graph, which never touches
    this database, so without this the stored analysis would still say "no
    key" after the counter column is in the table — and the export, the
    relationship stage and the planner all read that analysis, not the table.

    Only an *approved* step adopts a key, so a ``row_id`` that arrived with the
    source data is never mistaken for one the plan added.  The key is withdrawn
    once its column is gone from the table (a later step dropped or renamed
    it), which puts the table back to the question it was asking before.
    """

    added = {
        str(step.get("table")): str((step.get("params") or {}).get("column") or "row_id")
        for step in plan
        if step.get("type") == StepType.ADD_SYNTHETIC_KEY.value
        and step.get("status") == "approved"
    }
    store = get_store(session_id)
    record: IngestionSession | None = None
    for name, sheet in _sheets_by_name(db, session_id).items():
        analysis = KeyAnalysis.from_dict(sheet.key_analysis)
        columns = {str(c) for c in store.get(name).columns} if store.has(name) else set()
        if analysis.source == "synthetic":
            if set(analysis.primary_key) <= columns:
                continue
            withdraw_synthetic_key(analysis)
        elif not analysis.primary_key and added.get(name) in columns:
            adopt_synthetic_key(analysis, added[name])
        else:
            continue
        record = record or db.get(IngestionSession, session_id)
        _save_key_analysis(db, record, sheet, analysis)


def detect_and_store_equivalences(
    db: Session, record: IngestionSession
) -> list[EquivalenceCandidateRecord]:
    """Phase 0 cross-sheet equivalence detection.

    Pairs the evidence settles are decided here (``decided_by == "auto"``); only
    a pair it cannot settle is left for the user.  Detection runs again after
    every upload, so an answer the user already gave is carried over rather than
    wiped — and an automatic one is simply recomputed from the new data.
    """

    tables = analyzable_tables(db, record.id)
    earlier = {
        (r.left_table, r.left_column, r.right_table, r.right_column): r.confirmed
        for r in db.execute(
            select(EquivalenceCandidateRecord).where(
                EquivalenceCandidateRecord.session_id == record.id
            )
        )
        .scalars()
        .all()
        # Answered, and not by the platform: a row from before ``decided_by``
        # existed has it unset and can only have been answered by a person.
        if r.confirmed is not None and r.decided_by != "auto"
    }
    db.execute(
        delete(EquivalenceCandidateRecord).where(
            EquivalenceCandidateRecord.session_id == record.id
        )
    )
    candidates = detect_equivalences(tables)
    rows: list[EquivalenceCandidateRecord] = []
    for candidate in candidates:
        payload = candidate.to_dict()
        key = (
            payload["left_table"],
            payload["left_column"],
            payload["right_table"],
            payload["right_column"],
        )
        if key in earlier:
            confirmed, decided_by = earlier[key], "user"
        else:
            confirmed = auto_verdict(candidate)
            decided_by = "auto" if confirmed is not None else None
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
            confirmed=confirmed,
            decided_by=decided_by,
            left_embedding=payload.get("left_embedding"),
            right_embedding=payload.get("right_embedding"),
        )
        db.add(row)
        rows.append(row)
    db.flush()
    return rows


def _sheets_by_name(db: Session, session_id: str) -> dict[str, SheetRecord]:
    return {
        sheet.table_name: sheet
        for sheet in db.execute(select(SheetRecord).where(SheetRecord.session_id == session_id))
        .scalars()
        .all()
    }
