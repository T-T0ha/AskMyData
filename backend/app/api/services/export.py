"""Phase 4 — semantic layer construction and export, versions and documentation."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.services.field_semantics import run_field_semantics
from app.api.services.ingestion import _sheets_by_name
from app.api.services.profiling import describe_tables, sync_key_roles
from app.cleaning.store import get_store
from app.db.models import (
    ColumnSemantics,
    IngestionSession,
    RelationshipRecord,
    SemanticLayerExport,
    SemanticLayerVersion,
)
from app.export.documentation import render_markdown
from app.export.metadata import bundle as semantic_bundle
from app.export.quality import build_quality_report
from app.export.runner import drop_export, read_semantic_layer, run_export
from app.export.schema import build_plan


#: Archived semantic layers kept per dataset.  Enough to compare a run against
#: the one before it and to see a trend; bounded because each one is a JSON
#: document of the whole schema, and an unbounded history of them is a slow
#: leak in the control plane.
MAX_LAYER_VERSIONS = 10


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

    # What the source database called each column's type.  A spreadsheet has no
    # native schema, so it contributes nothing here and its columns are typed by
    # the mapping alone.
    declared_types = {
        name: {
            str(c.get("name")): str(c.get("data_type"))
            for c in (sheet.native_schema or {}).get("columns", [])
        }
        for name, sheet in sheets.items()
    }

    columns: dict[str, list[dict[str, Any]]] = {}
    for column in (
        db.execute(select(ColumnSemantics).where(ColumnSemantics.session_id == record.id))
        .scalars()
        .all()
    ):
        fact = column.to_dict()
        fact["declared_type"] = declared_types.get(column.table_name, {}).get(column.column_name)
        columns.setdefault(column.table_name, []).append(fact)

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
    # Proactive suggestions (§5.4) are cheap to regenerate but not free — clear
    # the cache so the next request builds them from this build of the layer
    # rather than serving suggestions for a schema that no longer exists.
    row.suggested_questions = []
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


def get_quality_report(db: Session, session_id: str) -> dict[str, Any]:
    """The automated data-quality report (§5.4) — a read of live enrichment state.

    Included whenever a prior export exists: a before/after against the build
    immediately before this one (see :mod:`app.export.quality` for exactly
    which dimensions an archived build can honestly support).
    """

    rows = (
        db.execute(select(ColumnSemantics).where(ColumnSemantics.session_id == session_id))
        .scalars()
        .all()
    )
    by_table: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_table.setdefault(row.table_name, []).append(row.to_dict())
    current_tables = [{"name": name, "columns": cols} for name, cols in by_table.items()]

    previous_tables = None
    versions = layer_versions(db, session_id)
    if len(versions) >= 2:
        previous_tables = (versions[1].bundle or {}).get("tables", [])

    return build_quality_report(current_tables, previous_tables)


def get_suggested_questions(
    db: Session, session_id: str, claude=None
) -> dict[str, Any]:
    """Proactive query suggestions (§5.4) — cached on the export, not recomputed per view.

    The cache is cleared on every new export (:func:`export_semantic_layer`)
    and regenerated here on first request after that; ``available=False``
    (never cached) when no Claude client is configured, so the next request
    tries again rather than freezing on an empty list forever.
    """

    export = db.execute(
        select(SemanticLayerExport).where(SemanticLayerExport.session_id == session_id)
    ).scalar_one_or_none()
    if export is None:
        return {"questions": [], "available": False}
    if export.suggested_questions:
        return {"questions": export.suggested_questions, "available": True}
    if claude is None or not claude.available:
        return {"questions": [], "available": False}

    sheets = _sheets_by_name(db, session_id)
    layer_rows = read_semantic_layer(session_id)
    columns_by_table: dict[str, list[dict[str, Any]]] = {}
    for layer_row in layer_rows:
        columns_by_table.setdefault(str(layer_row.get("table_name")), []).append(
            {"name": layer_row.get("column_name"), "taxonomy": layer_row.get("taxonomy_label")}
        )
    tables = [
        {
            "name": name,
            "description": sheets[name].effective_description if name in sheets else "",
            "columns": cols,
        }
        for name, cols in columns_by_table.items()
    ]
    questions = claude.suggest_questions(tables)
    if not questions:
        return {"questions": [], "available": False}
    export.suggested_questions = questions
    db.flush()
    return {"questions": questions, "available": True}


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
