"""Persisted application state.

Only *metadata* lives here — sessions, per-sheet triage, per-column semantics,
cross-sheet equivalence candidates and the cleaning run's status.  The actual
table data lives in :mod:`app.cleaning.store`, and the cleaning graph's
execution state lives in LangGraph's own checkpoint tables.

``ColumnSemantics.embedding`` is a real ``vector(384)`` column on PostgreSQL
(with an HNSW index) and falls back to JSON elsewhere, so the same code runs
in local development.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import TypeDecorator

from app.core.config import get_settings
from app.db.base import Base


def _uuid() -> str:
    return uuid.uuid4().hex


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _aware(value: datetime) -> datetime:
    """Read a stored timestamp back as UTC-aware.

    ``DateTime(timezone=True)`` is honoured by PostgreSQL and ignored by
    SQLite, which hands back a naive value.  Comparing that to ``_now()``
    raises ``TypeError``, so every expiry check would fail on the dialect the
    test suite runs on.
    """

    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


class EmbeddingVector(TypeDecorator):
    """``vector(n)`` on PostgreSQL, JSON list everywhere else."""

    impl = JSON
    cache_ok = True

    def __init__(self, dimensions: int) -> None:
        super().__init__()
        self.dimensions = dimensions

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            try:
                from pgvector.sqlalchemy import Vector  # noqa: PLC0415

                return dialect.type_descriptor(Vector(self.dimensions))
            except Exception:  # pragma: no cover - pgvector not installed
                pass
        return dialect.type_descriptor(JSON())

    def process_bind_param(self, value: Any, dialect) -> Any:
        if value is None:
            return None
        if dialect.name == "postgresql":
            return list(value)
        return [float(v) for v in value]

    def process_result_value(self, value: Any, dialect) -> Any:
        if value is None:
            return None
        return list(value)


class User(Base):
    """An account.  Owns datasets; owns nothing else in the system directly.

    ``password_hash`` is a self-describing scrypt string — algorithm, work
    factors, salt and digest — so the cost can be raised later without a
    migration.  The plaintext password never reaches this module.
    """

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    #: Stored lower-cased and stripped; the uniqueness constraint is therefore
    #: on the normalised form, so "A@b.com" cannot register twice.
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(100), default="")
    password_hash: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    last_login_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    sessions: Mapped[list["IngestionSession"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    tokens: Mapped[list["AuthToken"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )

    def to_dict(self) -> dict[str, Any]:
        """Never includes ``password_hash`` — this is what the API returns."""

        return {
            "id": self.id,
            "email": self.email,
            "name": self.name,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_login_at": self.last_login_at.isoformat() if self.last_login_at else None,
        }


class AuthToken(Base):
    """One live login.

    Only the SHA-256 of the token is stored, so a leaked database dump yields
    no usable credentials — the same reason ``password_hash`` is not the
    password.  Server-side tokens (rather than a self-contained JWT) are what
    make "sign out" mean something: revocation takes effect on the next
    request instead of whenever the token would have expired anyway.
    """

    __tablename__ = "auth_tokens"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    last_used_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    #: Truncated; enough to tell one browser from another when reviewing logins.
    user_agent: Mapped[str] = mapped_column(String(255), default="")

    user: Mapped[User] = relationship(back_populates="tokens")

    def is_live(self, now: datetime | None = None) -> bool:
        moment = now or _now()
        if self.revoked_at is not None:
            return False
        return _aware(self.expires_at) > moment

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None,
            "user_agent": self.user_agent,
        }


class IngestionSession(Base):
    """One upload-to-clean workflow.

    This is the ``Connections`` entity of the design: one dataset, one owner,
    one semantic layer.  ``user_id`` is nullable at the column level only so
    that a database predating authentication can be upgraded in place —
    everything created since has an owner, and an ownerless row is reachable
    by nobody.
    """

    __tablename__ = "ingestion_sessions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    user_id: Mapped[str | None] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=True
    )
    name: Mapped[str] = mapped_column(String(255), default="Untitled session")
    status: Mapped[str] = mapped_column(String(40), default="created")
    source_files: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )
    stats: Mapped[dict] = mapped_column(JSON, default=dict)

    #: Where this dataset's cleaned tables physically live once it has been
    #: exported: the schema name on PostgreSQL, the file name in local
    #: development.  Empty until the first export.  Recorded rather than
    #: recomputed so that a reader of the control plane can find the data
    #: plane without knowing how the name is derived.
    db_schema_name: Mapped[str] = mapped_column(String(100), default="")
    #: How many times the semantic layer has been built.  0 = never.  Every
    #: export increments it and archives that build, so a later layer can be
    #: compared against an earlier one (FR-17).
    semantic_version: Mapped[int] = mapped_column(Integer, default=0)
    #: When the layer was last built.  Distinct from ``updated_at``, which any
    #: edit moves; this one moves only when the semantics were rebuilt.
    enriched_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    user: Mapped[User | None] = relationship(back_populates="sessions")
    sheets: Mapped[list["SheetRecord"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )
    columns: Mapped[list["ColumnSemantics"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )
    equivalences: Mapped[list["EquivalenceCandidateRecord"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )
    relationships: Mapped[list["RelationshipRecord"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )
    #: Declared so that deleting a dataset takes its export record and its
    #: archived layers with it on every dialect.  The database-level cascade
    #: does the same on PostgreSQL, but SQLite only enforces foreign keys when
    #: asked to, and an orphaned semantic layer is still the user's data.
    export: Mapped["SemanticLayerExport | None"] = relationship(
        cascade="all, delete-orphan", uselist=False
    )
    layer_versions: Mapped[list["SemanticLayerVersion"]] = relationship(
        cascade="all, delete-orphan"
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "name": self.name,
            "status": self.status,
            "source_files": self.source_files or [],
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "stats": self.stats or {},
            "db_schema_name": self.db_schema_name or "",
            "semantic_version": self.semantic_version or 0,
            "enriched_at": self.enriched_at.isoformat() if self.enriched_at else None,
        }


class SheetRecord(Base):
    """Phase 0 output for one table: header repair, cleaning log, triage.

    "Sheet" is historical — a record here is one *table*, whether it came from
    a worksheet, a CSV, a SQLite file, a live PostgreSQL connection or a SQL
    dump.  ``source_kind`` says which, and ``native_schema`` carries the keys
    and types a database source declared about itself.
    """

    __tablename__ = "sheet_records"
    __table_args__ = (UniqueConstraint("session_id", "table_name", name="uq_sheet_per_session"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("ingestion_sessions.id", ondelete="CASCADE"), index=True
    )
    table_name: Mapped[str] = mapped_column(String(255))
    source_name: Mapped[str] = mapped_column(String(255))
    source_file: Mapped[str] = mapped_column(String(512), default="")
    source_kind: Mapped[str] = mapped_column(String(40), default="excel")
    #: Declared columns, primary key and foreign keys, when the source had a
    #: schema of its own.  Phase 3 treats these as ground truth rather than
    #: rediscovering them from value overlap.
    native_schema: Mapped[dict] = mapped_column(JSON, default=dict)
    #: What identifies a row: the declared or detected primary key, the
    #: composite candidates awaiting confirmation, and the user's decision.
    #: This is what Phase 4 writes as ``PRIMARY KEY`` and what Phase 3's
    #: foreign-key detection aims at.
    key_analysis: Mapped[dict] = mapped_column(JSON, default=dict)
    #: What the table *is* — SemTabla's thirteen table-level labels, the table
    #: type the decision tree read off them, and the user's corrections to
    #: both.  Recomputed from the other phases rather than measured, which is
    #: why it can be rebuilt whenever any of them changes.
    semantic_profile: Mapped[dict] = mapped_column(JSON, default=dict)
    #: One sentence saying what the table is for, composed from the profile
    #: above and the keys and references around it.  Phase 5's intent step
    #: sends the model table names and these sentences and nothing else, so
    #: this is what decides which tables a question is even read against.
    table_description: Mapped[str] = mapped_column(Text, default="")
    #: Claude's version of the same sentence, when a key is configured.  Held
    #: beside the composed one rather than over it — the same split the rest of
    #: the layer makes between what was derived and what was written — and
    #: dropped whenever the facts underneath it change.
    llm_table_description: Mapped[str | None] = mapped_column(Text, nullable=True)
    #: The description embedded, for intent-based table pre-selection.  The ER
    #: diagram calls this ``TableSemantics.embedding``.
    table_embedding: Mapped[list | None] = mapped_column(
        EmbeddingVector(get_settings().embedding_dim), nullable=True
    )
    triage: Mapped[str] = mapped_column(String(40), default="clean")
    row_count: Mapped[int] = mapped_column(Integer, default=0)
    column_count: Mapped[int] = mapped_column(Integer, default=0)
    skipped: Mapped[bool] = mapped_column(Boolean, default=False)
    skip_reason: Mapped[str | None] = mapped_column(String(255), nullable=True)
    header_info: Mapped[dict] = mapped_column(JSON, default=dict)
    clean_reports: Mapped[list] = mapped_column(JSON, default=list)
    issues: Mapped[list] = mapped_column(JSON, default=list)

    session: Mapped[IngestionSession] = relationship(back_populates="sheets")

    @property
    def effective_description(self) -> str:
        """What a reader — or Phase 5 — should be shown."""

        return self.llm_table_description or self.table_description or ""

    def to_dict(self) -> dict[str, Any]:
        # ``table_embedding`` is deliberately absent: 384 floats per sheet in
        # every session payload, for a value no client can use.
        return {
            "id": self.id,
            "table_name": self.table_name,
            "source_name": self.source_name,
            "source_file": self.source_file,
            "source_kind": self.source_kind,
            "native_schema": self.native_schema or {},
            "key_analysis": self.key_analysis or {},
            "semantic_profile": self.semantic_profile or {},
            "table_description": self.effective_description,
            "description_source": "claude" if self.llm_table_description else "composed",
            "triage": self.triage,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
            "header": self.header_info or {},
            "clean_reports": self.clean_reports or [],
            "issues": self.issues or [],
        }


class ColumnSemantics(Base):
    """Phase 1 output for one column, plus the user's corrections."""

    __tablename__ = "column_semantics"
    __table_args__ = (
        UniqueConstraint("session_id", "table_name", "column_name", name="uq_column_per_session"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("ingestion_sessions.id", ondelete="CASCADE"), index=True
    )
    table_name: Mapped[str] = mapped_column(String(255), index=True)
    column_name: Mapped[str] = mapped_column(String(255))
    original_name: Mapped[str] = mapped_column(String(512), default="")

    column_type: Mapped[str] = mapped_column(String(40))
    type_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    type_evidence: Mapped[list] = mapped_column(JSON, default=list)

    taxonomy_label: Mapped[str] = mapped_column(String(60))
    taxonomy_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    taxonomy_source: Mapped[str] = mapped_column(String(20), default="rule")
    taxonomy_rule: Mapped[str | None] = mapped_column(String(255), nullable=True)

    #: Set when the user overrides a detection in the Field Semantic View.
    user_column_type: Mapped[str | None] = mapped_column(String(40), nullable=True)
    user_taxonomy_label: Mapped[str | None] = mapped_column(String(60), nullable=True)
    validated: Mapped[bool] = mapped_column(Boolean, default=False)

    #: The column's role in the schema, projected here from the key analysis
    #: and the relationships the user confirmed.  It is derived rather than
    #: measured — which is why it is refreshed whenever either changes — but it
    #: is stored, because this table is what Phase 5 retrieves against, and a
    #: retrieval that cannot tell an identifier from a measure joins wrongly.
    is_primary_key: Mapped[bool] = mapped_column(Boolean, default=False)
    is_foreign_key: Mapped[bool] = mapped_column(Boolean, default=False)
    #: The column this one points at, in the source table's own names.  Set
    #: only for a reference a person confirmed; a reference the export could
    #: not enforce still lives here, because it is still true.
    references_table: Mapped[str | None] = mapped_column(String(255), nullable=True)
    references_column: Mapped[str | None] = mapped_column(String(255), nullable=True)

    nullable: Mapped[bool] = mapped_column(Boolean, default=True)
    null_ratio: Mapped[float] = mapped_column(Float, default=0.0)
    unique_count: Mapped[int] = mapped_column(Integer, default=0)
    unique_ratio: Mapped[float] = mapped_column(Float, default=0.0)
    row_count: Mapped[int] = mapped_column(Integer, default=0)
    is_additive: Mapped[bool] = mapped_column(Boolean, default=False)

    sample_values: Mapped[list] = mapped_column(JSON, default=list)
    distribution: Mapped[dict] = mapped_column(JSON, default=dict)
    semantic_description: Mapped[str] = mapped_column(Text, default="")
    embedding: Mapped[list | None] = mapped_column(
        EmbeddingVector(get_settings().embedding_dim), nullable=True
    )

    session: Mapped[IngestionSession] = relationship(back_populates="columns")

    @property
    def effective_type(self) -> str:
        return self.user_column_type or self.column_type

    @property
    def effective_label(self) -> str:
        return self.user_taxonomy_label or self.taxonomy_label

    def to_dict(self, include_embedding: bool = False) -> dict[str, Any]:
        payload = {
            "id": self.id,
            "table": self.table_name,
            "name": self.column_name,
            "original_name": self.original_name,
            "column_type": self.column_type,
            "type_confidence": self.type_confidence,
            "type_evidence": self.type_evidence or [],
            "taxonomy_label": self.taxonomy_label,
            "taxonomy_confidence": self.taxonomy_confidence,
            "taxonomy_source": self.taxonomy_source,
            "taxonomy_rule": self.taxonomy_rule,
            "user_column_type": self.user_column_type,
            "user_taxonomy_label": self.user_taxonomy_label,
            "effective_type": self.effective_type,
            "effective_label": self.effective_label,
            "validated": self.validated,
            "is_primary_key": self.is_primary_key,
            "is_foreign_key": self.is_foreign_key,
            "references_table": self.references_table,
            "references_column": self.references_column,
            "nullable": self.nullable,
            "null_ratio": self.null_ratio,
            "unique_count": self.unique_count,
            "unique_ratio": self.unique_ratio,
            "row_count": self.row_count,
            "is_additive": self.is_additive,
            "sample_values": self.sample_values or [],
            "distribution": self.distribution or {},
            "semantic_description": self.semantic_description,
        }
        if include_embedding:
            payload["embedding"] = self.embedding
        return payload


class EquivalenceCandidateRecord(Base):
    """A cross-sheet "same concept" suggestion awaiting user confirmation."""

    __tablename__ = "equivalence_candidates"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("ingestion_sessions.id", ondelete="CASCADE"), index=True
    )
    left_table: Mapped[str] = mapped_column(String(255))
    left_column: Mapped[str] = mapped_column(String(255))
    right_table: Mapped[str] = mapped_column(String(255))
    right_column: Mapped[str] = mapped_column(String(255))
    score: Mapped[float] = mapped_column(Float, default=0.0)
    embedding_similarity: Mapped[float] = mapped_column(Float, default=0.0)
    lexical_similarity: Mapped[float] = mapped_column(Float, default=0.0)
    value_overlap: Mapped[float | None] = mapped_column(Float, nullable=True)
    type_compatible: Mapped[bool] = mapped_column(Boolean, default=True)
    explanation: Mapped[str] = mapped_column(Text, default="")
    #: ``None`` = undecided, ``True`` = confirmed, ``False`` = rejected.
    confirmed: Mapped[bool | None] = mapped_column(Boolean, nullable=True)

    session: Mapped[IngestionSession] = relationship(back_populates="equivalences")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "left_table": self.left_table,
            "left_column": self.left_column,
            "right_table": self.right_table,
            "right_column": self.right_column,
            "score": self.score,
            "embedding_similarity": self.embedding_similarity,
            "lexical_similarity": self.lexical_similarity,
            "value_overlap": self.value_overlap,
            "type_compatible": self.type_compatible,
            "explanation": self.explanation,
            "confirmed": self.confirmed,
        }


class RelationshipRecord(Base):
    """Phase 3 output: one detected or drawn relationship, and its verdict.

    Covers both kinds with one table.  A foreign key has ``from_table`` and
    ``to_table`` different; a functional dependency has them the same and is a
    statement about two columns of one table.  They share a row because they
    share a lifecycle — proposed with evidence, confirmed or rejected by a
    human, then consumed by Phase 4 — and splitting them would duplicate that
    lifecycle twice over.

    A rejected relationship is kept rather than deleted: re-running detection
    must not re-propose something the user has already turned down.
    """

    __tablename__ = "relationships"
    __table_args__ = (
        UniqueConstraint(
            "session_id",
            "rel_type",
            "from_table",
            "from_column",
            "to_table",
            "to_column",
            name="uq_relationship_per_session",
        ),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("ingestion_sessions.id", ondelete="CASCADE"), index=True
    )
    rel_type: Mapped[str] = mapped_column(String(40), index=True)
    from_table: Mapped[str] = mapped_column(String(255), index=True)
    from_column: Mapped[str] = mapped_column(String(255))
    to_table: Mapped[str] = mapped_column(String(255), index=True)
    to_column: Mapped[str] = mapped_column(String(255))

    score: Mapped[float] = mapped_column(Float, default=0.0)
    #: declared | detected | fuzzy | manual — how it was arrived at, which is
    #: what decides how it is drawn and how far up the list it sits.
    origin: Mapped[str] = mapped_column(String(20), default="detected")
    status: Mapped[str] = mapped_column(String(20), default="proposed", index=True)
    #: The arithmetic behind the score: the two ratios, the name similarity,
    #: the unmatched values.  Kept so the panel can explain a proposal months
    #: later without re-running detection.
    evidence: Mapped[dict] = mapped_column(JSON, default=dict)
    explanation: Mapped[str] = mapped_column(Text, default="")
    #: Claude's plain-English reading of a borderline candidate.  Null when the
    #: score was decisive, or when no API key is configured.
    llm_explanation: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    session: Mapped[IngestionSession] = relationship(back_populates="relationships")

    @property
    def is_self_referencing(self) -> bool:
        return self.from_table == self.to_table

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "rel_type": self.rel_type,
            "from_table": self.from_table,
            "from_column": self.from_column,
            "to_table": self.to_table,
            "to_column": self.to_column,
            "score": round(self.score, 4),
            "origin": self.origin,
            "status": self.status,
            "uncertain": self.origin == "fuzzy" or self.rel_type == "approximate_dependency",
            "evidence": self.evidence or {},
            "explanation": self.explanation,
            "llm_explanation": self.llm_explanation,
            "decided_at": self.decided_at.isoformat() if self.decided_at else None,
        }


class SemanticLayerExport(Base):
    """The last Phase 4 export of one session: where it went and what it did.

    One row per session, replaced on every export, because an export replaces
    the schema it writes into — keeping a history here would describe databases
    that no longer exist.  ``report`` holds the full account the user reads
    (including the DDL); ``bundle`` holds the portable JSON as it was at that
    moment, so downloading it later gives what was exported rather than what
    the analysis has drifted to since.
    """

    __tablename__ = "semantic_layer_exports"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("ingestion_sessions.id", ondelete="CASCADE"), index=True, unique=True
    )
    #: The schema name on PostgreSQL, the file name on SQLite.  Never a URL:
    #: a connection string would carry the database password into every API
    #: response that mentions the export.
    target: Mapped[str] = mapped_column(String(128), default="")
    dialect: Mapped[str] = mapped_column(String(20), default="")
    status: Mapped[str] = mapped_column(String(20), default="ok")
    #: Which build of the layer this is — the session's ``semantic_version`` at
    #: the moment it was written, and the key into ``semantic_layer_versions``.
    semantic_version: Mapped[int] = mapped_column(Integer, default=0)
    table_count: Mapped[int] = mapped_column(Integer, default=0)
    row_count: Mapped[int] = mapped_column(Integer, default=0)
    column_count: Mapped[int] = mapped_column(Integer, default=0)
    embedded_count: Mapped[int] = mapped_column(Integer, default=0)
    vector_index: Mapped[bool] = mapped_column(Boolean, default=False)
    warning_count: Mapped[int] = mapped_column(Integer, default=0)
    report: Mapped[dict] = mapped_column(JSON, default=dict)
    bundle: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )

    def to_dict(self, include_report: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "session_id": self.session_id,
            "target": self.target,
            "dialect": self.dialect,
            "status": self.status,
            "semantic_version": self.semantic_version,
            "table_count": self.table_count,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "embedded_count": self.embedded_count,
            "vector_index": self.vector_index,
            "warning_count": self.warning_count,
            "exported_at": self.updated_at.isoformat() if self.updated_at else None,
        }
        if include_report:
            payload["report"] = self.report or {}
        return payload


class SemanticLayerVersion(Base):
    """An earlier build of one session's semantic layer, kept for comparison.

    ``SemanticLayerExport`` describes the export that *exists*; this describes
    every export that has existed.  Re-running enrichment increments the
    session's ``semantic_version`` and appends a row here, which is what makes
    "the layer is versioned" (FR-17) a fact about the database rather than a
    statement in a document: a before/after quality report can read two of
    these and diff them.

    Only the bundle is archived, never the data.  An export replaces the schema
    it writes into, so an older version describes tables that no longer exist —
    it is a record of what the platform *believed*, which is exactly the thing
    worth comparing, and it costs a few kilobytes instead of a second copy of
    the dataset.
    """

    __tablename__ = "semantic_layer_versions"
    __table_args__ = (
        UniqueConstraint("session_id", "version", name="uq_layer_version_per_session"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("ingestion_sessions.id", ondelete="CASCADE"), index=True
    )
    version: Mapped[int] = mapped_column(Integer, default=1)
    target: Mapped[str] = mapped_column(String(128), default="")
    dialect: Mapped[str] = mapped_column(String(20), default="")
    table_count: Mapped[int] = mapped_column(Integer, default=0)
    column_count: Mapped[int] = mapped_column(Integer, default=0)
    row_count: Mapped[int] = mapped_column(Integer, default=0)
    warning_count: Mapped[int] = mapped_column(Integer, default=0)
    #: The portable JSON layer as it was at this version.  No vectors and no
    #: rows of data — the same document the user can download.
    bundle: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    def to_dict(self, include_bundle: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "version": self.version,
            "target": self.target,
            "dialect": self.dialect,
            "table_count": self.table_count,
            "column_count": self.column_count,
            "row_count": self.row_count,
            "warning_count": self.warning_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
        if include_bundle:
            payload["bundle"] = self.bundle or {}
        return payload


class CleaningRun(Base):
    """Status mirror of the LangGraph thread, for listing sessions cheaply."""

    __tablename__ = "cleaning_runs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("ingestion_sessions.id", ondelete="CASCADE"), index=True, unique=True
    )
    thread_id: Mapped[str] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(40), default="not_started")
    plan_source: Mapped[str | None] = mapped_column(String(60), nullable=True)
    step_count: Mapped[int] = mapped_column(Integer, default=0)
    completed_steps: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "thread_id": self.thread_id,
            "status": self.status,
            "plan_source": self.plan_source,
            "step_count": self.step_count,
            "completed_steps": self.completed_steps,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
