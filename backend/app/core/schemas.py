"""Shared vocabulary for the whole pipeline.

These enums and dataclasses are the contract between Phase 0 (ingestion),
Phase 1 (field semantics) and Phase 2 (co-planned cleaning).  Keeping them in
one module means a taxonomy label or column type can never drift between the
detector that produces it and the UI that renders it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ColumnType(str, Enum):
    """Nine column types.

    The first six are the types defined by SemTabla (Jin et al., CHI '26).
    ``currency``, ``boolean`` and ``identifier`` are extensions contributed by
    this project because business spreadsheets rely on them heavily.
    """

    TEXT = "text"
    INTEGER_ORDINAL = "integer_ordinal"
    INTEGER_NOMINAL = "integer_nominal"
    INTEGER_CONTINUOUS = "integer_continuous"
    FLOAT = "float"
    DATE = "date"
    # project extensions
    CURRENCY = "currency"
    BOOLEAN = "boolean"
    IDENTIFIER = "identifier"

    @property
    def is_paper_type(self) -> bool:
        return self in _PAPER_TYPES

    @property
    def is_numeric(self) -> bool:
        return self in _NUMERIC_TYPES


_PAPER_TYPES = frozenset(
    {
        ColumnType.TEXT,
        ColumnType.INTEGER_ORDINAL,
        ColumnType.INTEGER_NOMINAL,
        ColumnType.INTEGER_CONTINUOUS,
        ColumnType.FLOAT,
        ColumnType.DATE,
    }
)

_NUMERIC_TYPES = frozenset(
    {
        ColumnType.INTEGER_ORDINAL,
        ColumnType.INTEGER_NOMINAL,
        ColumnType.INTEGER_CONTINUOUS,
        ColumnType.FLOAT,
        ColumnType.CURRENCY,
    }
)


class TriageStatus(str, Enum):
    """Data-quality bucket shown to the user before any processing starts."""

    CLEAN = "clean"
    FIXABLE = "fixable"
    NEEDS_ATTENTION = "needs_attention"
    STRUCTURAL_ISSUES = "structural_issues"


class StepType(str, Enum):
    """The ten cleaning operations the co-planning agent may propose.

    ``standardize_casing`` and ``strip_whitespace`` are separate from
    ``standardize_format`` on purpose: collapsing case variants and trimming
    stray whitespace are the two most common fixes, and a user reviewing the
    plan should see exactly which one is proposed rather than a generic
    "standardize" step with a ``format`` parameter.

    **There is deliberately no imputation step.**  Every operation here either
    transforms a value the user already has, removes something, or adds a
    surrogate key — none of them writes a value into a blank cell.  A blank
    cell means "not known", and that is exactly what it has to still mean in
    the exported database: PostgreSQL's ``NULL`` is excluded from ``AVG`` and
    ``SUM``, so a filled-in median would silently corrupt every aggregate the
    Phase 5 query interface generates, and nothing downstream could tell the
    invented value from a measured one.  See :data:`FORBIDDEN_STEP_TYPES`.
    """

    RENAME_COLUMN = "rename_column"
    DROP_COLUMN = "drop_column"
    STANDARDIZE_FORMAT = "standardize_format"
    STANDARDIZE_CASING = "standardize_casing"
    STRIP_WHITESPACE = "strip_whitespace"
    MERGE_SHEETS = "merge_sheets"
    SPLIT_COLUMN = "split_column"
    DEDUPLICATE = "deduplicate"
    TYPE_CAST = "type_cast"
    ADD_SYNTHETIC_KEY = "add_synthetic_key"


#: Step names that used to exist, or that a language model reaches for anyway,
#: and the reason they are refused.  Mapping them explicitly means a rejected
#: plan step tells the user *why* the platform will not do it instead of the
#: unhelpful "unknown step type".
FORBIDDEN_STEP_TYPES: dict[str, str] = {
    "fill_nulls": (
        "missing values are never filled in — a blank cell means 'not known' and "
        "stays blank (NULL) all the way into the database"
    ),
    "impute": (
        "missing values are never imputed — an invented value is indistinguishable "
        "from a measured one once it is in the database"
    ),
    "fill_missing": (
        "missing values are never filled in — a blank cell means 'not known' and "
        "stays blank (NULL) all the way into the database"
    ),
    "interpolate": (
        "missing values are never interpolated — an interpolated point is a guess "
        "the database cannot distinguish from a real reading"
    ),
}


class RelationshipType(str, Enum):
    """What Phase 3 can find between (or inside) tables.

    A foreign key relates two tables; a functional dependency relates two
    columns of one table.  They are separate types rather than one "edge"
    because the evidence that confirms them is different, and because the
    diagram draws them at different levels — between nodes, and inside one.
    """

    FOREIGN_KEY = "foreign_key"
    FUNCTIONAL_DEPENDENCY = "functional_dependency"
    #: Holds for most, not all, values of the left-hand column.  Reported
    #: separately because "almost a rule" is a data-quality finding, not a
    #: constraint anything downstream may rely on.
    APPROXIMATE_DEPENDENCY = "approximate_dependency"


class RelationshipOrigin(str, Enum):
    """Where a relationship came from — which is what decides how much it is
    trusted before a human looks at it."""

    #: The source database declared it.  Ground truth, not a guess.
    DECLARED = "declared"
    #: Value-overlap scoring above the accept threshold.
    DETECTED = "detected"
    #: Below the accept threshold but above the fuzzy floor; shown, not asserted.
    FUZZY = "fuzzy"
    #: The user drew the edge in the diagram.
    MANUAL = "manual"


class RelationshipStatus(str, Enum):
    """Where a relationship is in the validation loop."""

    PROPOSED = "proposed"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"


class TableType(str, Enum):
    """What a whole table *is*, as decided by the table-type decision tree.

    SemTabla (§4.1.5) trains a decision tree over the thirteen table-level
    labels; this project writes the tree down instead, for the same reason the
    column taxonomy uses rules rather than the paper's Random Forest — there is
    no labelled corpus of business workbooks to train on, and a written tree is
    something the user can be shown and can argue with.  A branch that fires
    only weakly returns :attr:`UNKNOWN`, which is what the paper does with a
    low-confidence classification.
    """

    #: Rows are events with measures, hanging off other tables' keys.
    FACT = "fact"
    #: Descriptive attributes of an entity that other tables point at.
    DIMENSION = "dimension"
    #: Little more than two foreign keys: a many-to-many junction.
    BRIDGE = "bridge"
    #: Keyed by time, one row per instant.
    TIME_SERIES = "time_series"
    #: Rows sit inside a tree — a self-reference, or a chain of dependencies.
    HIERARCHY = "hierarchy"
    #: A small closed set of codes other tables borrow from.
    LOOKUP = "lookup"
    #: Denormalised: everything about everything, in one very wide table.
    WIDE = "wide"
    UNKNOWN = "unknown"


#: The thirteen table-level semantic labels of SemTabla, Table 8, verbatim.
#: Each one is a yes/no statement about a table that the decision tree reads,
#: and each is reported with the evidence that produced it so the user can
#: check it — the same contract every other detection in this system honours.
TABLE_LABELS: tuple[str, ...] = (
    "is_primary_key_time",
    "is_primary_key_periodic",
    "is_single_value_column",
    "is_single_enum_column",
    "is_data_discrete",
    "is_enum_containing_most",
    "is_label_having_hierarchy_column",
    "is_mostly_referenced",
    "is_self_reference",
    "is_no_single_value_as_primary_key",
    "is_exactly_two_foreign_keys_existing",
    "is_having_dependency_chain",
    "is_fd_stable_after_null_drop",
)


class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

    @classmethod
    def from_score(cls, score: float) -> "Confidence":
        if score >= 0.85:
            return cls.HIGH
        if score >= 0.60:
            return cls.MEDIUM
        return cls.LOW


TAXONOMY_UNKNOWN = "unknown"

#: Closed set of taxonomy labels produced by the rule engine.  Claude is asked
#: to pick from this list too, so the UI dropdown is always complete.
TAXONOMY_LABELS: tuple[str, ...] = (
    "person_name",
    "organization_name",
    "product_name",
    "email",
    "phone",
    "url",
    "postal_address",
    "city",
    "country",
    "region",
    "postal_code",
    "geo_coordinate",
    "identifier",
    "foreign_identifier",
    "order_number",
    "invoice_number",
    "sku",
    "category",
    "status",
    "boolean_flag",
    "priority",
    "rating",
    "gender",
    "date",
    "datetime",
    "time",
    "duration",
    "year",
    "month",
    "weekday",
    "price",
    "cost",
    "revenue",
    "discount",
    "tax",
    "salary",
    "balance",
    "quantity",
    "percentage",
    "weight",
    "dimension",
    "age",
    "currency_code",
    "payment_method",
    "department",
    "job_title",
    "description",
    "note",
    "language",
    TAXONOMY_UNKNOWN,
)

#: Labels whose values can be meaningfully summed (SemTabla's "additive"
#: notion).  Consumed by Phase 4/5 when building the semantic layer.
ADDITIVE_LABELS: frozenset[str] = frozenset(
    {
        "price",
        "cost",
        "revenue",
        "discount",
        "tax",
        "salary",
        "quantity",
        "weight",
    }
)


@dataclass(slots=True)
class ColumnProfile:
    """Everything Phase 1 knows about a single column."""

    table: str
    name: str
    original_name: str
    column_type: ColumnType
    type_confidence: float
    taxonomy_label: str
    taxonomy_confidence: float
    taxonomy_source: str  # "rule" | "claude" | "fallback"
    taxonomy_rule: str | None = None
    nullable: bool = True
    null_ratio: float = 0.0
    unique_count: int = 0
    unique_ratio: float = 0.0
    row_count: int = 0
    sample_values: list[Any] = field(default_factory=list)
    distribution: dict[str, Any] = field(default_factory=dict)
    type_evidence: list[str] = field(default_factory=list)
    is_additive: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "table": self.table,
            "name": self.name,
            "original_name": self.original_name,
            "column_type": self.column_type.value,
            "type_confidence": round(self.type_confidence, 4),
            "type_confidence_band": Confidence.from_score(self.type_confidence).value,
            "taxonomy_label": self.taxonomy_label,
            "taxonomy_confidence": round(self.taxonomy_confidence, 4),
            "taxonomy_confidence_band": Confidence.from_score(self.taxonomy_confidence).value,
            "taxonomy_source": self.taxonomy_source,
            "taxonomy_rule": self.taxonomy_rule,
            "nullable": self.nullable,
            "null_ratio": round(self.null_ratio, 4),
            "unique_count": self.unique_count,
            "unique_ratio": round(self.unique_ratio, 4),
            "row_count": self.row_count,
            "sample_values": self.sample_values,
            "distribution": self.distribution,
            "type_evidence": self.type_evidence,
            "is_additive": self.is_additive,
        }


@dataclass(slots=True)
class SheetIssue:
    """A single data-quality problem found during triage."""

    code: str
    severity: str  # "info" | "warning" | "error"
    message: str
    columns: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "columns": self.columns,
        }


@dataclass(slots=True)
class CleaningStep:
    """One node of the co-planned cleaning plan (Cocoa co-planning pattern)."""

    id: str
    type: StepType
    table: str
    description: str
    params: dict[str, Any] = field(default_factory=dict)
    origin: str = "agent"  # "agent" | "heuristic" | "user"
    status: str = "pending"  # pending | executed | approved | reverted | skipped
    rationale: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "table": self.table,
            "description": self.description,
            "params": self.params,
            "origin": self.origin,
            "status": self.status,
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "CleaningStep":
        return cls(
            id=raw["id"],
            type=StepType(raw["type"]),
            table=raw["table"],
            description=raw.get("description", ""),
            params=raw.get("params") or {},
            origin=raw.get("origin", "agent"),
            status=raw.get("status", "pending"),
            rationale=raw.get("rationale"),
        )
