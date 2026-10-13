"""Phase 4 — table semantic profiling (SemTabla Step 4) and table descriptions."""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.services.ingestion import analyzable_tables, _sheets_by_name
from app.cleaning.store import get_store
from app.core.schemas import TABLE_LABELS, RelationshipStatus, RelationshipType, TableType
from app.db.models import ColumnSemantics, IngestionSession, RelationshipRecord, SheetRecord
from app.semantics import table_summary
from app.semantics.table_profile import TableProfile, profile_tables as build_table_profiles

logger = logging.getLogger(__name__)


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
