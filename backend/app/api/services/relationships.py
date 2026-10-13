"""Phase 3 — relationship detection, review and evidence."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.services.ingestion import analyzable_tables, table_keys
from app.api.services.profiling import profile_tables, sync_key_roles
from app.core.schemas import RelationshipOrigin, RelationshipStatus, RelationshipType
from app.db.models import ColumnSemantics, IngestionSession, RelationshipRecord, SheetRecord
from app.ingestion.keys import KeyAnalysis
from app.relationships.auto import IDENTIFIER_LABELS, decide_dependency, settle_foreign_keys
from app.relationships.dependencies import detect_all_dependencies
from app.relationships.evidence import build_evidence
from app.relationships.foreign_keys import detect_foreign_keys, skipped_tables as fk_skipped_tables


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
        # A verdict the platform reached is a default, so it is re-derived from
        # the fresh evidence; one a person reached is theirs and stays.
        if existing.status == RelationshipStatus.PROPOSED.value or existing.decided_by == "auto":
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
    found: set[tuple[str, str, str, str, str]] = set()
    for candidate in candidates:
        payload = candidate.to_dict()
        found.add(_relationship_key(payload))
        stored, is_new = _upsert_relationship(db, record.id, payload)
        created += int(is_new)
        low, high = LLM_EXPLAIN_BAND
        if is_new and low <= payload["score"] <= high and stored.llm_explanation is None:
            stored.llm_explanation = _explain_candidate(db, record.id, payload, claude)

    for dependency in dependencies:
        payload = dependency.to_dict()
        found.add(_relationship_key(payload))
        _, is_new = _upsert_relationship(db, record.id, payload)
        created += int(is_new)

    db.flush()
    _forget_unfound_relationships(db, record.id, found)
    apply_auto_relationship_decisions(db, record.id)
    sync_key_roles(db, record.id)
    payload = relationships_payload(db, record.id)
    payload["created"] = created
    # SemTabla runs table profiling as Step 4, on the output of the three
    # detection steps before it.  This is that step.
    payload["profiles"] = profile_tables(db, record)["profiles"]
    return payload


def _relationship_key(payload: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        payload["rel_type"],
        payload["from_table"],
        payload["from_column"],
        payload["to_table"],
        payload["to_column"],
    )


def _forget_unfound_relationships(
    db: Session, session_id: str, found: set[tuple[str, str, str, str, str]]
) -> int:
    """Drop what detection no longer finds and no person ever ruled on.

    Detection's output is the truth about the data as it is *now*.  A proposal
    or a platform verdict from an earlier run that the cleaned data no longer
    supports (or that named a table since set aside) would otherwise sit in the
    list forever, looking current.  A user's decision, a drawn edge and a
    declared key are never touched.
    """

    removed = 0
    for row in db.execute(
        select(RelationshipRecord).where(RelationshipRecord.session_id == session_id)
    ).scalars():
        if row.origin in (RelationshipOrigin.MANUAL.value, RelationshipOrigin.DECLARED.value):
            continue
        undecided = row.status == RelationshipStatus.PROPOSED.value
        if not (undecided or row.decided_by == "auto"):
            continue
        key = (row.rel_type, row.from_table, row.from_column, row.to_table, row.to_column)
        if key not in found:
            db.delete(row)
            removed += 1
    if removed:
        db.flush()
    return removed


def apply_auto_relationship_decisions(db: Session, session_id: str) -> dict[str, int]:
    """Settle every relationship the evidence settles; leave the rest proposed.

    Touches only rows nobody has ruled on (still proposed, or decided by this
    function on an earlier run).  Returns how many ended up in each state.
    """

    rows = list(
        db.execute(select(RelationshipRecord).where(RelationshipRecord.session_id == session_id))
        .scalars()
        .all()
    )
    labels = {
        (c.table_name, c.column_name): c.effective_label
        for c in db.execute(
            select(ColumnSemantics).where(ColumnSemantics.session_id == session_id)
        )
        .scalars()
        .all()
    }

    def person_decided(row: RelationshipRecord) -> bool:
        return row.status != RelationshipStatus.PROPOSED.value and row.decided_by != "auto"

    eligible = [row for row in rows if not person_decided(row)]
    taken = [
        (row.from_table, row.from_column, row.to_table, row.to_column)
        for row in rows
        if row.rel_type == RelationshipType.FOREIGN_KEY.value
        and row.status == RelationshipStatus.CONFIRMED.value
        and person_decided(row)
    ]

    foreign_keys = [
        {
            "id": row.id,
            "from_table": row.from_table,
            "from_column": row.from_column,
            "to_table": row.to_table,
            "to_column": row.to_column,
            "score": row.score,
            "origin": row.origin,
            "evidence": row.evidence or {},
            "child_is_identifier": labels.get((row.from_table, row.from_column))
            in IDENTIFIER_LABELS,
        }
        for row in eligible
        if row.rel_type == RelationshipType.FOREIGN_KEY.value
    ]
    verdicts = settle_foreign_keys(foreign_keys, taken=taken)

    now = datetime.now(timezone.utc)
    tally = {"confirmed": 0, "rejected": 0, "proposed": 0}
    for row in eligible:
        verdict = (
            verdicts[row.id]
            if row.rel_type == RelationshipType.FOREIGN_KEY.value
            else decide_dependency(row.rel_type)
        )
        row.status = verdict.status
        row.decided_by = "auto" if verdict.decided else None
        row.decided_at = now if verdict.decided else None
        row.evidence = {**(row.evidence or {}), "auto_reason": verdict.reason}
        tally[verdict.status] += 1
    db.flush()
    return tally


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
            # The only thing left for a person: a link that nearly holds.
            # Patterns inside a table describe the data and ask nothing.
            "needs_review": sum(
                1
                for r in rows
                if r.rel_type == RelationshipType.FOREIGN_KEY.value
                and r.status == RelationshipStatus.PROPOSED.value
            ),
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
    row.decided_by = "user"
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
        existing.decided_by = "user"
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
        decided_by="user",
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
