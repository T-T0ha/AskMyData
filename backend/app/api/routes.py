"""HTTP API for Phases 0–3."""

from __future__ import annotations

import logging
import shutil
from datetime import date
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.api import services
from app.auth.deps import current_user, owned_session
from app.cleaning.checkpointing import get_checkpointer
from app.cleaning.graph import (
    build_graph,
    interrupt_payload,
    resume as resume_graph,
    thread_config,
)
from app.cleaning.store import drop_store, get_store
from app.core.config import get_settings
from app.core.schemas import (
    TABLE_LABELS,
    TAXONOMY_LABELS,
    ColumnType,
    RelationshipOrigin,
    RelationshipType,
    StepType,
    TableType,
)
from app.db.base import pending_schema_changes, session_scope
from app.db.models import (
    CleaningRun,
    ColumnSemantics,
    EquivalenceCandidateRecord,
    IngestionSession,
    SheetRecord,
    User,
)
from app.ingestion.loader import (
    CSV_SUFFIXES,
    EXCEL_SUFFIXES,
    FILE_SUFFIXES,
    SQL_DUMP_SUFFIXES,
)
from app.ingestion.relational import (
    ALLOWED_BACKENDS,
    SQLITE_SUFFIXES,
    SourceError,
    inspect_source,
)
from app.semantics.claude_client import get_claude_client

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")

_graph = None


def graph():
    global _graph
    if _graph is None:
        _graph = build_graph(get_checkpointer())
    return _graph


# ---------------------------------------------------------------------------
# request bodies
# ---------------------------------------------------------------------------


class SessionCreate(BaseModel):
    name: str = Field(default="Untitled session", max_length=255)


class EquivalenceDecision(BaseModel):
    confirmed: bool


class ColumnOverride(BaseModel):
    column_type: str | None = None
    taxonomy_label: str | None = None
    validated: bool | None = None


class PlanDecision(BaseModel):
    action: str = Field(default="confirm", pattern="^(confirm|cancel)$")
    plan: list[dict[str, Any]] | None = None


class StepDecision(BaseModel):
    action: str = Field(default="approve", pattern="^(approve|revert|retry|skip|abort)$")
    params: dict[str, Any] | None = None


class KeyDecision(BaseModel):
    """The user's answer to "what identifies a row in this table?".

    An empty list is a real answer — "none of these" — and routes the table to
    the synthetic ``row_id`` branch of the cleaning plan.
    """

    columns: list[str] = Field(default_factory=list, max_length=8)


class RelationshipDecision(BaseModel):
    confirmed: bool


class ProfileOverride(BaseModel):
    """The user's correction to a table's profile (SemTabla §4.2).

    All three fields are optional and independent: an empty ``table_type``
    withdraws a previous override rather than setting one.
    """

    table_type: str | None = Field(default=None, max_length=40)
    add_label: str | None = Field(default=None, max_length=60)
    remove_label: str | None = Field(default=None, max_length=60)


class ManualRelationship(BaseModel):
    """An edge drawn by hand in the relationship diagram."""

    rel_type: str = Field(default="foreign_key")
    from_table: str = Field(min_length=1, max_length=255)
    from_column: str = Field(min_length=1, max_length=255)
    to_table: str = Field(min_length=1, max_length=255)
    to_column: str = Field(min_length=1, max_length=255)


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class DashboardCardLayout(BaseModel):
    x: int = 0
    y: int = 0
    w: int = 4
    h: int = 4


class DashboardCardCreate(BaseModel):
    """A pin request — echoes back what ``/query`` already answered.

    ``sql`` is re-validated server-side (:func:`app.api.services.pin_dashboard_card`);
    everything else here is display metadata the frontend already has in hand
    from the answer it is pinning, so pinning needs no second round trip to
    Claude or to the database.
    """

    question: str = Field(min_length=1, max_length=2000)
    sql: str = Field(min_length=1)
    title: str | None = Field(default=None, max_length=255)
    explanation: str = ""
    tables_used: list[str] = Field(default_factory=list)
    visualization: dict[str, Any] = Field(default_factory=dict)


class DashboardCardUpdate(BaseModel):
    title: str | None = Field(default=None, max_length=255)
    layout: DashboardCardLayout | None = None


class SourceConnection(BaseModel):
    """A database the user wants to read.

    The URL is used for the duration of the request and never persisted: what
    the session stores is the password-free display form.
    """

    url: str = Field(min_length=1, max_length=2048)
    #: Source names of the tables to ingest; empty means every user table.
    tables: list[str] | None = None


# ---------------------------------------------------------------------------
# meta
# ---------------------------------------------------------------------------


@router.get("/status")
def status() -> dict[str, Any]:
    return services.backend_status()


@router.get("/vocabulary")
def vocabulary() -> dict[str, Any]:
    """Everything the UI's dropdowns need."""

    return {
        "column_types": [
            {"value": t.value, "from_paper": t.is_paper_type, "numeric": t.is_numeric}
            for t in ColumnType
        ],
        "taxonomy_labels": list(TAXONOMY_LABELS),
        "step_types": [s.value for s in StepType],
        "relationship_types": [r.value for r in RelationshipType],
        "relationship_origins": [o.value for o in RelationshipOrigin],
        "table_types": [t.value for t in TableType],
        "table_labels": list(TABLE_LABELS),
        "accepted_extensions": sorted(FILE_SUFFIXES),
        "accepted_file_kinds": [
            {"kind": "excel", "label": "Excel workbook", "extensions": sorted(EXCEL_SUFFIXES)},
            {"kind": "csv", "label": "CSV / TSV file", "extensions": sorted(CSV_SUFFIXES)},
            {"kind": "sqlite", "label": "SQLite database", "extensions": sorted(SQLITE_SUFFIXES)},
            {"kind": "sql_dump", "label": "SQL dump", "extensions": sorted(SQL_DUMP_SUFFIXES)},
        ],
        "database_backends": sorted(ALLOWED_BACKENDS),
    }


# ---------------------------------------------------------------------------
# sessions
# ---------------------------------------------------------------------------


def require_session(
    session_id: str,
    db: Session = Depends(session_scope),
    user: User = Depends(current_user),
) -> IngestionSession:
    """Every dataset-scoped route depends on this, directly or in its decorator.

    Authentication and ownership resolve together, before the handler body
    runs, so there is no path into a handler that has not established whose
    dataset it is about (FR-14).  A dataset belonging to somebody else is
    reported as missing, not as forbidden.
    """

    return owned_session(db, session_id, user)


def _our_database_failed(exc: SQLAlchemyError) -> HTTPException:
    """Our own storage broke — which is never the user's upload being unreadable.

    Ingestion reads a file *and* writes what it found, so a failure in the
    second half arrives at the same ``except``.  Reporting it as 422 "could not
    read the file" sends the user off to inspect a perfectly good spreadsheet,
    so a database error keeps its own status and says whose fault it is.
    """

    pending = pending_schema_changes()
    if pending:
        detail = (
            "the backend is running against an out-of-date database (missing "
            + ", ".join(pending)
            + "). Restart the backend — it adds missing columns at startup."
        )
    else:
        detail = f"the file was read, but the platform could not store it: {exc}"
    return HTTPException(status_code=500, detail=detail)


@router.post("/sessions", status_code=201)
def create_session(
    body: SessionCreate,
    db: Session = Depends(session_scope),
    user: User = Depends(current_user),
) -> dict[str, Any]:
    return services.create_session(db, body.name, user.id).to_dict()


@router.get("/sessions")
def list_sessions(
    db: Session = Depends(session_scope),
    user: User = Depends(current_user),
) -> dict[str, Any]:
    return {"sessions": [s.to_dict() for s in services.list_sessions(db, user.id)]}


@router.get("/sessions/{session_id}")
def get_session(
    session_id: str,
    record: IngestionSession = Depends(require_session),
    db: Session = Depends(session_scope),
) -> dict[str, Any]:
    run = db.execute(
        select(CleaningRun).where(CleaningRun.session_id == session_id)
    ).scalar_one_or_none()
    return {
        **record.to_dict(),
        "sheets": [s.to_dict() for s in record.sheets],
        "tables": [t.to_dict() for t in get_store(session_id).meta()],
        "cleaning_run": run.to_dict() if run else None,
    }


@router.delete("/sessions/{session_id}", status_code=204)
def delete_session(
    session_id: str,
    record: IngestionSession = Depends(require_session),
    db: Session = Depends(session_scope),
) -> None:
    db.delete(record)
    drop_store(session_id)
    # The exported database is the user's data too, and it lives outside the
    # cascade — deleting the session has to take it with it.
    services.forget_export(session_id)


# ---------------------------------------------------------------------------
# Phase 0
# ---------------------------------------------------------------------------


@router.post("/sessions/{session_id}/upload")
def upload(
    session_id: str,
    file: UploadFile = File(...),
    record: IngestionSession = Depends(require_session),
    db: Session = Depends(session_scope),
) -> dict[str, Any]:
    """Upload one workbook, CSV, SQLite database or SQL dump and run Phase 0."""

    settings = get_settings()
    filename = (file.filename or "upload.xlsx").replace("/", "_")
    suffix = ("." + filename.rsplit(".", 1)[-1]).lower() if "." in filename else ""
    if suffix not in FILE_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail=f"unsupported file type {suffix!r}; accepted: "
            + ", ".join(sorted(FILE_SUFFIXES)),
        )

    # Keep the original filename: a CSV's table name is derived from it.
    folder = settings.upload_dir / session_id
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / filename
    with destination.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)

    try:
        sheets = services.ingest_file(db, record, destination)
    except SQLAlchemyError as exc:
        logger.exception("storing %s failed", destination)
        raise _our_database_failed(exc) from exc
    except Exception as exc:
        logger.exception("ingestion failed for %s", destination)
        raise HTTPException(status_code=422, detail=f"could not read the file: {exc}") from exc

    equivalences = services.detect_and_store_equivalences(db, record)
    return {
        "session": record.to_dict(),
        "sheets": [s.to_dict() for s in sheets],
        "equivalences": [e.to_dict() for e in equivalences],
    }


@router.post("/sources/inspect")
def inspect_database(
    body: SourceConnection,
    _user: User = Depends(current_user),
) -> dict[str, Any]:
    """List a database's tables, columns and declared keys — reads no rows.

    Deliberately a separate step from ingesting: a production schema has
    hundreds of tables and the user should choose, having seen what is there
    and how large each table is, rather than have the platform pull everything.

    Authenticated despite belonging to no dataset: it opens a connection to a
    host of the caller's choosing and reports what it found, which is a probe
    of the network the server sits in.  That is not something to offer anonymously.
    """

    try:
        return inspect_source(body.url).to_dict()
    except SourceError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - network/driver dependent
        logger.exception("inspecting a source failed")
        raise HTTPException(status_code=422, detail=f"could not read the database: {exc}") from exc


@router.post("/sessions/{session_id}/connect")
def connect_database(
    session_id: str,
    body: SourceConnection,
    record: IngestionSession = Depends(require_session),
    db: Session = Depends(session_scope),
) -> dict[str, Any]:
    """Ingest tables from a live database connection through Phase 0."""

    try:
        sheets = services.ingest_database(db, record, body.url, body.tables or None)
    except SourceError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except SQLAlchemyError as exc:
        # The source is reached through SourceError; this is our side failing.
        logger.exception("storing the ingested tables failed")
        raise _our_database_failed(exc) from exc
    except Exception as exc:  # pragma: no cover - network/driver dependent
        logger.exception("database ingestion failed")
        raise HTTPException(status_code=422, detail=f"could not read the database: {exc}") from exc

    equivalences = services.detect_and_store_equivalences(db, record)
    return {
        "session": record.to_dict(),
        "sheets": [s.to_dict() for s in sheets],
        "equivalences": [e.to_dict() for e in equivalences],
    }


@router.get("/sessions/{session_id}/triage")
def triage(
    record: IngestionSession = Depends(require_session),
    db: Session = Depends(session_scope),
) -> dict[str, Any]:
    return {
        "summary": services.triage_stats(db, record),
        "sheets": [s.to_dict() for s in record.sheets],
    }


@router.get("/sessions/{session_id}/equivalences", dependencies=[Depends(require_session)])
def list_equivalences(session_id: str, db: Session = Depends(session_scope)) -> dict[str, Any]:
    rows = db.execute(
        select(EquivalenceCandidateRecord)
        .where(EquivalenceCandidateRecord.session_id == session_id)
        .order_by(EquivalenceCandidateRecord.score.desc())
    ).scalars().all()
    return {"equivalences": [e.to_dict() for e in rows]}


@router.post(
    "/sessions/{session_id}/equivalences/{candidate_id}",
    dependencies=[Depends(require_session)],
)
def decide_equivalence(
    session_id: str,
    candidate_id: str,
    body: EquivalenceDecision,
    db: Session = Depends(session_scope),
) -> dict[str, Any]:
    candidate = db.get(EquivalenceCandidateRecord, candidate_id)
    if candidate is None or candidate.session_id != session_id:
        raise HTTPException(status_code=404, detail="equivalence candidate not found")
    candidate.confirmed = body.confirmed
    db.flush()
    return candidate.to_dict()


@router.post("/sessions/{session_id}/tables/{table}/key")
def decide_key(
    table: str,
    body: KeyDecision,
    record: IngestionSession = Depends(require_session),
    db: Session = Depends(session_scope),
) -> dict[str, Any]:
    """Confirm (or reject) the key that identifies a row in one table.

    The proposal's no-key handling: a composite candidate is never adopted
    silently, because "these columns together are one row" is a statement about
    the business that only the user can make.
    """

    try:
        sheet = services.set_table_key(db, record, table, body.columns)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return sheet.to_dict()


@router.get(
    "/sessions/{session_id}/tables/{table}/preview",
    dependencies=[Depends(require_session)],
)
def preview(
    session_id: str,
    table: str,
    limit: int = 25,
) -> dict[str, Any]:
    try:
        return services.table_preview(session_id, table, min(limit, 200))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Phase 1
# ---------------------------------------------------------------------------


@router.post("/sessions/{session_id}/semantics")
def run_semantics(
    session_id: str,
    record: IngestionSession = Depends(require_session),
    db: Session = Depends(session_scope),
) -> dict[str, Any]:
    if not get_store(session_id).names():
        raise HTTPException(status_code=409, detail="upload a file before running semantics")
    services.run_field_semantics(db, record, claude=get_claude_client())
    db.flush()
    return services.semantics_payload(db, session_id)


@router.get("/sessions/{session_id}/semantics", dependencies=[Depends(require_session)])
def get_semantics(session_id: str, db: Session = Depends(session_scope)) -> dict[str, Any]:
    return services.semantics_payload(db, session_id)


@router.patch(
    "/sessions/{session_id}/semantics/{column_id}",
    dependencies=[Depends(require_session)],
)
def override_column(
    session_id: str,
    column_id: str,
    body: ColumnOverride,
    db: Session = Depends(session_scope),
) -> dict[str, Any]:
    """Human-in-the-loop correction from the Field Semantic View."""

    column = db.get(ColumnSemantics, column_id)
    if column is None or column.session_id != session_id:
        raise HTTPException(status_code=404, detail="column not found")

    if body.column_type is not None:
        try:
            ColumnType(body.column_type)
        except ValueError as exc:
            raise HTTPException(
                status_code=422, detail=f"unknown column type {body.column_type!r}"
            ) from exc
        column.user_column_type = body.column_type
    if body.taxonomy_label is not None:
        if body.taxonomy_label not in TAXONOMY_LABELS:
            raise HTTPException(
                status_code=422, detail=f"unknown taxonomy label {body.taxonomy_label!r}"
            )
        column.user_taxonomy_label = body.taxonomy_label
    if body.validated is not None:
        column.validated = body.validated
    if body.column_type is not None or body.taxonomy_label is not None:
        column.validated = True
        column.semantic_description = services._semantic_description(column)
        services._embed_columns([column])
    db.flush()
    return column.to_dict()


# ---------------------------------------------------------------------------
# Phase 3
# ---------------------------------------------------------------------------


@router.post("/sessions/{session_id}/relationships")
def run_relationships(
    session_id: str,
    record: IngestionSession = Depends(require_session),
    db: Session = Depends(session_scope),
) -> dict[str, Any]:
    """Detect foreign keys and functional dependencies.

    Safe to run more than once: a relationship the user has already confirmed
    or rejected keeps its verdict, and only genuinely new candidates are added.
    """

    if not get_store(session_id).names():
        raise HTTPException(status_code=409, detail="upload a file before detecting relationships")
    columns = db.execute(
        select(ColumnSemantics).where(ColumnSemantics.session_id == session_id)
    ).scalars().first()
    if columns is None:
        # Detection does not need Phase 1, but the evidence panel and the
        # borderline explainer both read taxonomy labels, and an empty panel
        # is a worse first impression than a slightly longer wait.
        services.run_field_semantics(db, record, claude=get_claude_client())
    return services.detect_relationships(db, record, claude=get_claude_client())


@router.get("/sessions/{session_id}/relationships", dependencies=[Depends(require_session)])
def list_relationships(session_id: str, db: Session = Depends(session_scope)) -> dict[str, Any]:
    return services.relationships_payload(db, session_id)


@router.post("/sessions/{session_id}/relationships/manual", status_code=201)
def draw_relationship(
    body: ManualRelationship,
    record: IngestionSession = Depends(require_session),
    db: Session = Depends(session_scope),
) -> dict[str, Any]:
    """An edge the user drew in the diagram — the last of the proposal's
    fallbacks when scoring finds nothing on messy data."""

    try:
        row = services.add_manual_relationship(
            db,
            record,
            body.rel_type,
            body.from_table,
            body.from_column,
            body.to_table,
            body.to_column,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {**row.to_dict(), "profiles": services.profile_tables(db, record)["profiles"]}


@router.get(
    "/sessions/{session_id}/relationships/{relationship_id}/evidence",
    dependencies=[Depends(require_session)],
)
def relationship_evidence(
    session_id: str,
    relationship_id: str,
    db: Session = Depends(session_scope),
) -> dict[str, Any]:
    """Positive and negative sample rows, with the SQL that produced them."""

    try:
        return services.relationship_evidence(db, session_id, relationship_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post(
    "/sessions/{session_id}/relationships/{relationship_id}",
    dependencies=[Depends(require_session)],
)
def decide_relationship(
    session_id: str,
    relationship_id: str,
    body: RelationshipDecision,
    record: IngestionSession = Depends(require_session),
    db: Session = Depends(session_scope),
) -> dict[str, Any]:
    """Confirm or reject a relationship, having seen its evidence.

    Six of the thirteen table labels are read off the relationship set, so a
    verdict here changes what the tables *are* — the profiles are rebuilt in
    the same request rather than drifting until something else triggers them.
    """

    try:
        row = services.decide_relationship(db, session_id, relationship_id, body.confirmed)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {**row.to_dict(), "profiles": services.profile_tables(db, record)["profiles"]}


# ---------------------------------------------------------------------------
# Phase 4 — table semantic profiling
# ---------------------------------------------------------------------------


@router.post("/sessions/{session_id}/profiles")
def run_profiles(
    record: IngestionSession = Depends(require_session),
    db: Session = Depends(session_scope),
) -> dict[str, Any]:
    """Summarise every table into the thirteen labels and a table type.

    Runs automatically at the end of relationship detection; this endpoint is
    for rebuilding after the user has corrected something upstream, which is
    the update the paper describes.
    """

    if not get_store(record.id).names():
        raise HTTPException(status_code=409, detail="upload a file before profiling tables")
    return services.profile_tables(db, record, claude=get_claude_client())


@router.get("/sessions/{session_id}/profiles", dependencies=[Depends(require_session)])
def list_profiles(session_id: str, db: Session = Depends(session_scope)) -> dict[str, Any]:
    return services.profiles_payload(db, session_id)


@router.patch("/sessions/{session_id}/tables/{table}/profile")
def correct_profile(
    table: str,
    body: ProfileOverride,
    record: IngestionSession = Depends(require_session),
    db: Session = Depends(session_scope),
) -> dict[str, Any]:
    """Override the table type, or add and remove a label by hand."""

    try:
        return services.override_profile(
            db,
            record,
            table,
            table_type=body.table_type,
            add_label=body.add_label,
            remove_label=body.remove_label,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/sessions/{session_id}/export")
def run_semantic_export(
    record: IngestionSession = Depends(require_session),
    db: Session = Depends(session_scope),
) -> dict[str, Any]:
    """Build the clean database and its semantic layer.

    Replaces any previous export of this dataset: the schema is dropped and
    rebuilt, so the result always matches the analysis as it stands now.
    """

    try:
        return services.export_semantic_layer(db, record, claude=get_claude_client())
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except SQLAlchemyError as exc:
        logger.exception("export failed for session %s", record.id)
        raise HTTPException(
            status_code=500, detail=f"the export could not be written: {exc}"
        ) from exc


@router.get("/sessions/{session_id}/export", dependencies=[Depends(require_session)])
def get_semantic_export(session_id: str, db: Session = Depends(session_scope)) -> dict[str, Any]:
    return services.export_payload(db, session_id)


@router.get("/sessions/{session_id}/export/bundle", dependencies=[Depends(require_session)])
def get_semantic_bundle(
    session_id: str,
    version: int | None = None,
    db: Session = Depends(session_scope),
) -> dict[str, Any]:
    """The portable JSON semantic layer.

    The current one by default, or an archived ``version`` — which is what
    makes two builds of the same dataset comparable rather than merely
    countable.
    """

    bundle = services.layer_bundle(db, session_id, version)
    if bundle is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"this dataset has no semantic layer version {version}"
                if version is not None
                else "this dataset has not been exported yet"
            ),
        )
    return bundle


@router.get("/sessions/{session_id}/export/versions", dependencies=[Depends(require_session)])
def list_layer_versions(
    session_id: str, db: Session = Depends(session_scope)
) -> dict[str, Any]:
    """Every build of this dataset's semantic layer, newest first."""

    versions = services.layer_versions(db, session_id)
    current = services.export_record(db, session_id)
    return {
        "current": current.semantic_version if current is not None else 0,
        "versions": [row.to_dict() for row in versions],
    }


@router.get(
    "/sessions/{session_id}/export/documentation",
    dependencies=[Depends(require_session)],
    response_class=PlainTextResponse,
)
def get_semantic_documentation(
    version: int | None = None,
    record: IngestionSession = Depends(require_session),
    db: Session = Depends(session_scope),
) -> PlainTextResponse:
    """The semantic layer as Markdown documentation (FR-18).

    For most datasets this is the first documentation they have ever had, so
    it is served as a file the user can keep rather than as a screen.
    """

    document = services.layer_documentation(db, record, version)
    if document is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"this dataset has no semantic layer version {version}"
                if version is not None
                else "this dataset has not been exported yet"
            ),
        )
    return PlainTextResponse(document, media_type="text/markdown; charset=utf-8")


@router.get(
    "/sessions/{session_id}/export/ddl",
    dependencies=[Depends(require_session)],
    response_class=PlainTextResponse,
)
def get_semantic_ddl(session_id: str, db: Session = Depends(session_scope)) -> str:
    """The ``CREATE TABLE`` script, as PostgreSQL would spell it."""

    row = services.export_record(db, session_id)
    if row is None:
        raise HTTPException(status_code=404, detail="this dataset has not been exported yet")
    return (row.report or {}).get("ddl", "")


# ---------------------------------------------------------------------------
# Phase 5 — natural language query
# ---------------------------------------------------------------------------


@router.post("/sessions/{session_id}/query")
def ask_question(
    body: QuestionRequest,
    record: IngestionSession = Depends(require_session),
    db: Session = Depends(session_scope),
) -> dict[str, Any]:
    """Ask one question of the exported semantic layer.

    A question the model could not turn into a usable query is still a 200:
    ``ok: false`` with the attempts and the last error, so the UI can show the
    user what was tried rather than a generic failure.  Only "there is nothing
    to query yet" is an error status — that is a precondition the user can act
    on (export first), not an outcome of the question itself.
    """

    try:
        return services.answer_question(db, record, body.question, claude=get_claude_client())
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/sessions/{session_id}/query/history", dependencies=[Depends(require_session)])
def query_history(
    session_id: str,
    db: Session = Depends(session_scope),
) -> dict[str, Any]:
    return {"history": services.question_history(db, session_id)}


# ---------------------------------------------------------------------------
# Phase 6 — dashboard
# ---------------------------------------------------------------------------


@router.post("/sessions/{session_id}/dashboard/cards", status_code=201)
def pin_dashboard_card(
    body: DashboardCardCreate,
    record: IngestionSession = Depends(require_session),
    db: Session = Depends(session_scope),
) -> dict[str, Any]:
    try:
        card = services.pin_dashboard_card(db, record, body.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return card.to_dict()


@router.get("/sessions/{session_id}/dashboard/cards")
def list_dashboard_cards(
    start: date | None = None,
    end: date | None = None,
    record: IngestionSession = Depends(require_session),
    db: Session = Depends(session_scope),
) -> dict[str, Any]:
    """Every pinned card, re-executed against the live export just now.

    ``start``/``end`` inject the global date-range picker's ``WHERE`` clause
    into every card whose own shape names exactly one date column; a card
    without one runs unfiltered rather than being silently skipped.
    """

    return {"cards": services.list_dashboard_cards(db, record, start=start, end=end)}


@router.patch("/sessions/{session_id}/dashboard/cards/{card_id}")
def update_dashboard_card(
    card_id: str,
    body: DashboardCardUpdate,
    record: IngestionSession = Depends(require_session),
    db: Session = Depends(session_scope),
) -> dict[str, Any]:
    patch = {
        "title": body.title,
        "layout": body.layout.model_dump() if body.layout is not None else None,
    }
    try:
        card = services.update_dashboard_card(db, record, card_id, patch)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return card.to_dict()


@router.delete("/sessions/{session_id}/dashboard/cards/{card_id}", status_code=204)
def unpin_dashboard_card(
    card_id: str,
    record: IngestionSession = Depends(require_session),
    db: Session = Depends(session_scope),
) -> None:
    services.delete_dashboard_card(db, record, card_id)


# ---------------------------------------------------------------------------
# Phase 2
# ---------------------------------------------------------------------------


def _sync_run(db: Session, session_id: str, state: dict[str, Any]) -> CleaningRun:
    run = db.execute(
        select(CleaningRun).where(CleaningRun.session_id == session_id)
    ).scalar_one_or_none()
    if run is None:
        run = CleaningRun(session_id=session_id, thread_id=f"cleaning-{session_id}")
        db.add(run)
    plan = state.get("plan") or []
    run.status = state.get("status") or "unknown"
    run.plan_source = state.get("plan_source")
    run.step_count = len(plan)
    run.completed_steps = sum(
        1 for s in plan if s.get("status") in {"approved", "skipped", "reverted"}
    )
    db.flush()
    return run


def _graph_state(session_id: str) -> dict[str, Any]:
    snapshot = graph().get_state(thread_config(session_id))
    return dict(snapshot.values or {})


def _pending_interrupt(session_id: str) -> dict[str, Any] | None:
    snapshot = graph().get_state(thread_config(session_id))
    interrupts = getattr(snapshot, "interrupts", None) or ()
    if not interrupts:
        return None
    value = getattr(interrupts[0], "value", None)
    return value if isinstance(value, dict) else None


def _cleaning_response(db: Session, session_id: str, result: Any) -> dict[str, Any]:
    state = _graph_state(session_id)
    pending = interrupt_payload(result) or _pending_interrupt(session_id)
    run = _sync_run(db, session_id, state)
    return {
        "status": state.get("status"),
        "plan": state.get("plan", []),
        "plan_source": state.get("plan_source"),
        "plan_rejections": state.get("plan_rejections", []),
        "cursor": state.get("cursor", 0),
        "history": state.get("history", []),
        "errors": state.get("errors", []),
        "pending": pending,
        "run": run.to_dict(),
        "tables": [t.to_dict() for t in get_store(session_id).meta()],
    }


@router.post("/sessions/{session_id}/cleaning/start")
def start_cleaning(
    session_id: str,
    record: IngestionSession = Depends(require_session),
    db: Session = Depends(session_scope),
) -> dict[str, Any]:
    """Kick off the co-planning graph; stops at the plan review interrupt."""

    if not get_store(session_id).names():
        raise HTTPException(status_code=409, detail="upload a file before cleaning")

    columns = db.execute(
        select(ColumnSemantics).where(ColumnSemantics.session_id == session_id)
    ).scalars().first()
    if columns is None:
        services.run_field_semantics(db, record, claude=get_claude_client())

    equivalences = [
        e.to_dict()
        for e in db.execute(
            select(EquivalenceCandidateRecord).where(
                EquivalenceCandidateRecord.session_id == session_id
            )
        )
        .scalars()
        .all()
    ]
    analyzable = set(services.analyzable_tables(db, session_id))
    result = graph().invoke(
        {
            "session_id": session_id,
            "summary": services.build_summary(db, record),
            "equivalences": equivalences,
            "keys": services.table_keys(db, session_id),
            "excluded_tables": [
                name for name in get_store(session_id).names() if name not in analyzable
            ],
            "history": [],
            "errors": [],
        },
        thread_config(session_id),
    )
    record.status = "cleaning"
    return _cleaning_response(db, session_id, result)


@router.get("/sessions/{session_id}/cleaning/state", dependencies=[Depends(require_session)])
def cleaning_state(session_id: str, db: Session = Depends(session_scope)) -> dict[str, Any]:
    state = _graph_state(session_id)
    if not state:
        return {"status": "not_started", "plan": [], "pending": None}
    run = _sync_run(db, session_id, state)
    return {
        "status": state.get("status"),
        "plan": state.get("plan", []),
        "plan_source": state.get("plan_source"),
        "plan_rejections": state.get("plan_rejections", []),
        "cursor": state.get("cursor", 0),
        "history": state.get("history", []),
        "errors": state.get("errors", []),
        "pending": _pending_interrupt(session_id),
        "run": run.to_dict(),
        "tables": [t.to_dict() for t in get_store(session_id).meta()],
    }


@router.post("/sessions/{session_id}/cleaning/plan", dependencies=[Depends(require_session)])
def submit_plan(
    session_id: str,
    body: PlanDecision,
    db: Session = Depends(session_scope),
) -> dict[str, Any]:
    """Answer the plan-review interrupt with the user's edited plan."""

    pending = _pending_interrupt(session_id)
    if pending is None or pending.get("kind") != "plan_review":
        raise HTTPException(status_code=409, detail="no plan review is pending")
    result = resume_graph(
        graph(), session_id, {"action": body.action, "plan": body.plan}
    )
    return _cleaning_response(db, session_id, result)


@router.post("/sessions/{session_id}/cleaning/step", dependencies=[Depends(require_session)])
def submit_step(
    session_id: str,
    body: StepDecision,
    db: Session = Depends(session_scope),
) -> dict[str, Any]:
    """Answer a step-validation interrupt (approve / revert / retry / skip)."""

    pending = _pending_interrupt(session_id)
    if pending is None or pending.get("kind") not in {"step_validation", "step_failed"}:
        raise HTTPException(status_code=409, detail="no step validation is pending")
    payload: dict[str, Any] = {"action": body.action}
    if body.params is not None:
        payload["params"] = body.params
    result = resume_graph(graph(), session_id, payload)
    return _cleaning_response(db, session_id, result)
