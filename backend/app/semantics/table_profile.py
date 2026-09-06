"""Table semantic profiling — SemTabla Step 4 (§4.1.5, Table 8).

The first three levels of the paper's framework describe *columns* (Phase 1),
*references between tables* and *dependencies inside one* (Phase 3).  This is
the fourth: what the table as a whole **is**.  It reads nothing new off the
data that the earlier phases did not already establish — it summarises them
into thirteen yes/no labels, and feeds those labels to a decision tree that
names the table.

Two things make that worth doing rather than skipping to the export:

* Phase 4 writes ``semantic_description`` for every column, and the table type
  is part of that sentence.  "unit_price in order_lines (price, currency) — fact
  table" retrieves differently from the same sentence ending in "data table",
  and Phase 5's whole retrieval step runs on those sentences.
* A user looking at forty sheets needs to know which ones carry the numbers and
  which ones are lookups before they can ask a question about either.

The thirteen labels are the paper's, verbatim, in :data:`TABLE_LABELS`.  Every
one is reported with the evidence that produced it, because a label the user
cannot check is a label they cannot correct — and correcting them is the point
of the paper's Table Profiling Validation step.

The decision tree is written down rather than trained.  See
:class:`~app.core.schemas.TableType` for why.

Nothing here mutates its inputs, and nothing here talks to a database or a
model: profiling is a pure function of the analysis so far, which is what lets
it be recomputed whenever the user changes any of it.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from app.core.config import get_settings
from app.core.schemas import (
    TABLE_LABELS,
    ColumnType,
    RelationshipStatus,
    RelationshipType,
    TableType,
)

logger = logging.getLogger(__name__)

#: A column with at most this many distinct values, and few enough of them
#: relative to the rows, is a closed set of codes rather than free content.
MAX_ENUM_DISTINCT = 25
ENUM_UNIQUE_RATIO = 0.50

#: Taxonomy labels that are enumerations whatever their cardinality says —
#: a status column with 200 distinct values is a messy status column, not a
#: description.
ENUM_LABELS: frozenset[str] = frozenset(
    {
        "status",
        "category",
        "priority",
        "gender",
        "boolean_flag",
        "payment_method",
        "department",
        "job_title",
        "country",
        "region",
        "city",
        "currency_code",
        "language",
        "month",
        "weekday",
        "rating",
    }
)

#: Taxonomy labels that are measurements — the numbers a question is usually
#: *about*.  Kept separate from "numeric type" because a year, an age and a
#: postal code are numeric and are not measures.
MEASURE_LABELS: frozenset[str] = frozenset(
    {
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
    }
)

#: Labels that never count as a measure even when the column is numeric.
NON_MEASURE_LABELS: frozenset[str] = frozenset(
    {
        "identifier",
        "foreign_identifier",
        "order_number",
        "invoice_number",
        "sku",
        "postal_code",
        "year",
        "month",
        "weekday",
        "age",
        "rating",
        "geo_coordinate",
    }
)

#: Column names that say "this row has a parent" or "this is level N of
#: something".  Matched against the flattened name, case-folded.
HIERARCHY_NAME_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern)
    for pattern in (
        r"(^|_)parent(_|$)",
        r"(^|_)child(_|$)",
        r"(^|_)level(_|$)",
        r"(^|_)tier(_|$)",
        r"(^|_)depth(_|$)",
        r"(^|_)path(_|$)",
        r"(^|_)hierarchy(_|$)",
        r"(^|_)ancestor(_|$)",
        r"(^|_)sub_",
        r"_l[1-5]$",
        r"(^|_)group(_|$)",
    )
)

#: Label ladders: holding two rungs of one of these is structural hierarchy,
#: not two unrelated attributes.
LABEL_LADDERS: tuple[frozenset[str], ...] = (
    frozenset({"country", "region", "city", "postal_code"}),
    frozenset({"year", "month", "weekday"}),
    frozenset({"department", "job_title"}),
    frozenset({"category", "sku", "product_name"}),
)

#: A dependency chain has to be this long before it says anything about the
#: table's shape.  The paper's threshold (§ Table 8): three or more columns.
MIN_CHAIN_LENGTH = 3

#: Points needed before an interval is a period rather than two gaps.
MIN_PERIODIC_POINTS = 4
#: Standard deviation over mean interval, below which the spacing is fixed.
MAX_PERIOD_VARIATION = 0.10

#: Columns above which a table is denormalised by definition.
WIDE_COLUMN_COUNT = 25
#: A lookup is small.  Above this it is a table people put data in.
LOOKUP_MAX_ROWS = 60

#: Null ratio above which a column is "high-null" for label 13.
HIGH_NULL_RATIO = 0.30

#: The paper assigns ``unknown`` to a low-confidence classification rather than
#: the class the tree reached.  This is that floor.
MIN_TYPE_CONFIDENCE = 0.55


@dataclass(slots=True)
class TableLabel:
    """One of the thirteen labels, with what made it true.

    ``detail`` carries the numbers a label mentions — the average interval of a
    periodic key, the ratio behind "highly discrete" — so the panel can show
    the arithmetic rather than assert the conclusion.
    """

    name: str
    evidence: str
    columns: list[str] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)
    origin: str = "detected"  # detected | user

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "evidence": self.evidence,
            "columns": self.columns,
            "detail": self.detail,
            "origin": self.origin,
        }


@dataclass(slots=True)
class TableProfile:
    """What one table is, why, and what the user said about it."""

    table: str
    table_type: TableType
    type_confidence: float
    rationale: str
    labels: list[TableLabel] = field(default_factory=list)
    features: dict[str, Any] = field(default_factory=dict)
    #: The user's override of the tree's answer, and their edits to the label
    #: set.  Held beside the detected values, never over them, so that
    #: detector accuracy stays measurable after correction — the same split
    #: ``ColumnSemantics`` makes for types and taxonomy labels.
    user_table_type: str | None = None
    added_labels: list[str] = field(default_factory=list)
    removed_labels: list[str] = field(default_factory=list)

    @property
    def effective_type(self) -> str:
        return self.user_table_type or self.table_type.value

    @property
    def effective_labels(self) -> list[str]:
        removed = set(self.removed_labels)
        names = [label.name for label in self.labels if label.name not in removed]
        names.extend(name for name in self.added_labels if name not in names)
        return names

    def describe(self) -> str:
        """The table type as it reads inside a semantic description."""

        return self.effective_type.replace("_", " ") + " table"

    def to_dict(self) -> dict[str, Any]:
        return {
            "table": self.table,
            "table_type": self.table_type.value,
            "type_confidence": round(self.type_confidence, 4),
            "user_table_type": self.user_table_type,
            "effective_type": self.effective_type,
            "rationale": self.rationale,
            "labels": [label.to_dict() for label in self.labels],
            "effective_labels": self.effective_labels,
            "added_labels": self.added_labels,
            "removed_labels": self.removed_labels,
            "features": self.features,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "TableProfile | None":
        if not payload:
            return None
        try:
            table_type = TableType(payload.get("table_type", TableType.UNKNOWN.value))
        except ValueError:  # a type this build no longer knows
            table_type = TableType.UNKNOWN
        return cls(
            table=str(payload.get("table", "")),
            table_type=table_type,
            type_confidence=float(payload.get("type_confidence", 0.0)),
            rationale=str(payload.get("rationale", "")),
            labels=[
                TableLabel(
                    name=str(raw.get("name", "")),
                    evidence=str(raw.get("evidence", "")),
                    columns=[str(c) for c in raw.get("columns") or []],
                    detail=dict(raw.get("detail") or {}),
                    origin=str(raw.get("origin", "detected")),
                )
                for raw in payload.get("labels") or []
            ],
            features=dict(payload.get("features") or {}),
            user_table_type=payload.get("user_table_type") or None,
            added_labels=[str(x) for x in payload.get("added_labels") or []],
            removed_labels=[str(x) for x in payload.get("removed_labels") or []],
        )


# ---------------------------------------------------------------------------
# column classification helpers
# ---------------------------------------------------------------------------


def _column_type(fact: Mapping[str, Any]) -> ColumnType:
    raw = str(fact.get("effective_type") or fact.get("column_type") or ColumnType.TEXT.value)
    try:
        return ColumnType(raw)
    except ValueError:
        return ColumnType.TEXT


def _label_of(fact: Mapping[str, Any]) -> str:
    return str(fact.get("effective_label") or fact.get("taxonomy_label") or "unknown")


def _name_of(fact: Mapping[str, Any]) -> str:
    return str(fact.get("name") or fact.get("column_name") or "")


def _is_enum(fact: Mapping[str, Any]) -> bool:
    """A closed set of repeated codes, in the sense the paper's labels mean."""

    column_type = _column_type(fact)
    if column_type is ColumnType.BOOLEAN:
        return True
    if column_type in {ColumnType.FLOAT, ColumnType.CURRENCY, ColumnType.DATE}:
        return False
    if _label_of(fact) in ENUM_LABELS:
        return True
    unique_count = int(fact.get("unique_count") or 0)
    unique_ratio = float(fact.get("unique_ratio") or 0.0)
    if unique_count <= 1:
        return False  # a constant is not an enumeration, it is a constant
    return unique_count <= MAX_ENUM_DISTINCT and unique_ratio <= ENUM_UNIQUE_RATIO


def _is_numeric(fact: Mapping[str, Any]) -> bool:
    return _column_type(fact).is_numeric


def _is_measure(fact: Mapping[str, Any], structural: set[str]) -> bool:
    """A number this table is *about*, as opposed to a number that identifies.

    ``structural`` holds the key and foreign-key columns: an ``order_id`` is
    numeric and is not a measurement of anything.
    """

    if _name_of(fact) in structural:
        return False
    label = _label_of(fact)
    if label in MEASURE_LABELS:
        return True
    if label in NON_MEASURE_LABELS:
        return False
    if not _is_numeric(fact):
        return False
    if _column_type(fact) is ColumnType.INTEGER_NOMINAL:
        return False  # nominal integers are codes wearing a number's clothes
    return not _is_enum(fact)


def _relations(
    relationships: Iterable[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Every relationship the user has not rejected.

    Profiling summarises what the analysis currently believes, so a proposal
    counts — but a rejected edge is a statement the user has already refuted,
    and re-reading it here would put it back into the picture through a side
    door.  The profile is recomputed whenever a verdict changes, so a label
    resting on a proposal disappears the moment that proposal does.
    """

    return [
        relation
        for relation in relationships
        if str(relation.get("status", RelationshipStatus.PROPOSED.value))
        != RelationshipStatus.REJECTED.value
    ]


def _hierarchy_columns(
    facts: Sequence[Mapping[str, Any]], header_info: Mapping[str, Any] | None
) -> tuple[list[str], str]:
    """Columns whose *headers* say the table has levels.

    Three sources, in order of how much they mean: a name that says "parent"
    or "level"; two rungs of the same label ladder (city and country, year and
    month); and a header that Phase 0 flattened out of merged cells spanning
    several rows, which is a group hierarchy the spreadsheet author drew by
    hand and which nothing later in the pipeline gets to see again.
    """

    named = [
        _name_of(fact)
        for fact in facts
        if any(pattern.search(_name_of(fact).lower()) for pattern in HIERARCHY_NAME_PATTERNS)
    ]
    if named:
        return named, "column name(s) name a parent or a level"

    labels = {_label_of(fact): _name_of(fact) for fact in facts}
    for ladder in LABEL_LADDERS:
        rungs = sorted(ladder & set(labels))
        if len(rungs) >= 2:
            return [labels[rung] for rung in rungs], (
                f"{' and '.join(rungs)} are rungs of one ladder, so rows sit inside a tree"
            )

    header = header_info or {}
    rows = len(header.get("rows") or ())
    if header.get("is_multi_row") or rows > 1:
        return [_name_of(fact) for fact in facts], (
            f"the header spanned {max(rows, 2)} rows of merged cells, which is a grouping "
            "the spreadsheet author drew by hand"
        )
    return [], ""


def _periodicity(df: pd.DataFrame, columns: Sequence[str]) -> dict[str, Any] | None:
    """Average interval and its variation for a time key, or ``None``.

    The paper asks for "average interval time and standard deviation ratio",
    which is exactly enough to tell a daily reading from a log of whenever
    somebody happened to press the button.
    """

    for column in columns:
        if column not in df.columns:
            continue
        stamps = pd.to_datetime(df[column], errors="coerce").dropna()
        moments = sorted(set(stamps))
        if len(moments) < MIN_PERIODIC_POINTS:
            continue
        # Pinned to seconds explicitly: pandas 2 carries datetimes at whatever
        # resolution the source had — ns, us or s — and reading the raw integers
        # would measure a daily reading as 86 400, 86.4 or 0.0864 depending on
        # which.  The interval is the whole point of the label.
        seconds = np.array(moments, dtype="datetime64[s]").astype("int64")
        deltas = np.diff(seconds).astype("float64")
        mean = float(deltas.mean())
        if mean <= 0:
            continue
        variation = float(deltas.std()) / mean
        return {
            "column": column,
            "points": int(len(moments)),
            "mean_interval_seconds": round(mean, 3),
            "mean_interval_days": round(mean / 86_400, 4),
            "variation_ratio": round(variation, 4),
            "periodic": variation <= MAX_PERIOD_VARIATION,
        }
    return None


def _longest_chain(edges: Sequence[tuple[str, str]]) -> list[str]:
    """Longest simple path through the dependency graph of one table.

    The graph is a handful of nodes — a table's columns, minus the ones the
    dependency detector already filtered out — so an exhaustive depth-first
    walk is both exact and instant.  A cycle cannot lengthen a *simple* path,
    so carrying the visited set is all the protection it needs.
    """

    outgoing: dict[str, list[str]] = {}
    for determinant, dependent in edges:
        outgoing.setdefault(determinant, []).append(dependent)

    best: list[str] = []

    def walk(node: str, path: list[str], seen: set[str]) -> None:
        nonlocal best
        if len(path) > len(best):
            best = list(path)
        for nxt in outgoing.get(node, ()):
            if nxt in seen:
                continue
            seen.add(nxt)
            path.append(nxt)
            walk(nxt, path, seen)
            path.pop()
            seen.discard(nxt)

    for start in outgoing:
        walk(start, [start], {start})
    return best


# ---------------------------------------------------------------------------
# the thirteen labels
# ---------------------------------------------------------------------------


def _detect_labels(
    df: pd.DataFrame,
    facts: Sequence[Mapping[str, Any]],
    features: dict[str, Any],
    header_info: Mapping[str, Any] | None,
) -> list[TableLabel]:
    """Every label that holds, in the order of the paper's Table 8."""

    labels: list[TableLabel] = []
    by_name = {_name_of(fact): fact for fact in facts}
    primary_key: list[str] = features["primary_key"]
    row_count: int = features["row_count"]

    def add(name: str, evidence: str, columns: Sequence[str] = (), **detail: Any) -> None:
        labels.append(
            TableLabel(name=name, evidence=evidence, columns=list(columns), detail=detail)
        )

    # 1 / 2 — a time-typed primary key, and whether it ticks.
    time_key = [
        column
        for column in primary_key
        if column in by_name and _column_type(by_name[column]) is ColumnType.DATE
    ]
    if time_key:
        add(
            "is_primary_key_time",
            f"the primary key is {' + '.join(time_key)}, which is a date/time column",
            time_key,
        )
        period = _periodicity(df, time_key)
        if period and period["periodic"]:
            days = period["mean_interval_days"]
            add(
                "is_primary_key_periodic",
                (
                    f"consecutive {period['column']} values are {days:g} day(s) apart on "
                    f"average, varying by {period['variation_ratio']:.1%} — a fixed period"
                ),
                [period["column"]],
                **period,
            )
        features["periodicity"] = period

    # 3 / 4 — exactly one measure, exactly one enumeration.
    measures: list[str] = features["measure_columns"]
    enums: list[str] = features["enum_columns"]
    if len(measures) == 1:
        add(
            "is_single_value_column",
            f"{measures[0]} is the only measured value in the table",
            measures,
        )
    if len(enums) == 1:
        add(
            "is_single_enum_column",
            f"{enums[0]} is the only enumerated column in the table",
            enums,
        )

    # 5 — how discrete the numbers are.
    numeric = [
        _name_of(fact)
        for fact in facts
        if _is_numeric(fact) and _name_of(fact) not in features["structural_columns"]
    ]
    if numeric:
        threshold = get_settings().nominal_unique_ratio
        ratios = {name: float(by_name[name].get("unique_ratio") or 0.0) for name in numeric}
        discrete = [name for name, ratio in ratios.items() if ratio <= threshold]
        if len(discrete) * 2 >= len(numeric):
            add(
                "is_data_discrete",
                (
                    f"{len(discrete)} of {len(numeric)} numeric column(s) repeat their values "
                    f"often (distinct-to-row ratio at or below {threshold:.0%})"
                ),
                discrete,
                ratios={name: round(ratio, 4) for name, ratio in ratios.items()},
            )

    # 6 — enumerations dominate.
    if enums and len(enums) * 2 > len(facts):
        add(
            "is_enum_containing_most",
            f"{len(enums)} of {len(facts)} columns are enumerations",
            enums,
        )

    # 7 — headers with hierarchy in them.
    hierarchy, why = _hierarchy_columns(facts, header_info)
    if hierarchy:
        add("is_label_having_hierarchy_column", why, hierarchy)
    features["hierarchy_columns"] = hierarchy

    # 8 — other tables lean on this one.
    referenced_by: list[str] = features["referenced_by_tables"]
    peers: int = features["peer_table_count"]
    if len(referenced_by) >= 2 or (referenced_by and peers and len(referenced_by) * 2 >= peers):
        add(
            "is_mostly_referenced",
            (
                f"{len(referenced_by)} of the {peers} other table(s) point at this one: "
                f"{', '.join(referenced_by)}"
            ),
            features["referenced_columns"],
        )

    # 9 — it points at itself.
    if features["self_reference_columns"]:
        add(
            "is_self_reference",
            (
                f"{', '.join(features['self_reference_columns'])} references this same table, "
                "so rows sit above and below one another"
            ),
            features["self_reference_columns"],
        )

    # 10 — no single column identifies a row.
    if len(primary_key) != 1:
        add(
            "is_no_single_value_as_primary_key",
            (
                f"the key is {' + '.join(primary_key)}"
                if primary_key
                else "no column, and no confirmed combination of columns, identifies a row"
            ),
            primary_key,
        )

    # 11 — exactly two outgoing foreign keys: the junction signature.
    outgoing: list[str] = features["foreign_key_columns"]
    if len(outgoing) == 2:
        add(
            "is_exactly_two_foreign_keys_existing",
            f"{' and '.join(outgoing)} are the table's only two foreign keys",
            outgoing,
        )

    # 12 — a chain of dependencies, three columns or longer.
    chain: list[str] = features["dependency_chain"]
    if len(chain) >= MIN_CHAIN_LENGTH:
        add(
            "is_having_dependency_chain",
            "each value fixes the next: " + " → ".join(chain),
            chain,
        )

    # 13 — do the dependencies survive dropping the mostly-empty columns?
    dependencies: list[list[str]] = features["dependencies"]
    if dependencies:
        sparse = {
            _name_of(fact)
            for fact in facts
            if float(fact.get("null_ratio") or 0.0) > HIGH_NULL_RATIO
        }
        fragile = [
            f"{determinant} → {dependent}"
            for determinant, dependent in dependencies
            if determinant in sparse or dependent in sparse
        ]
        if not fragile:
            add(
                "is_fd_stable_after_null_drop",
                (
                    f"all {len(dependencies)} dependency/ies hold between columns that are "
                    f"populated, so dropping the mostly-empty ones changes nothing"
                ),
                sorted({name for pair in dependencies for name in pair}),
                high_null_columns=sorted(sparse),
            )
        else:
            features["fragile_dependencies"] = fragile

    if row_count == 0:  # nothing above can mean anything on an empty table
        return [label for label in labels if label.name == "is_no_single_value_as_primary_key"]
    return labels


# ---------------------------------------------------------------------------
# the decision tree
# ---------------------------------------------------------------------------


def _classify(
    features: Mapping[str, Any], labels: Sequence[TableLabel]
) -> tuple[TableType, float, str]:
    """Name the table from its labels.  Written out, in priority order.

    Each branch returns its own confidence, because the branches are not
    equally sure of themselves: "two foreign keys and nothing else" is a
    junction table beyond argument, while "it has numbers in it" is a guess
    worth showing and worth overriding.  Anything under
    :data:`MIN_TYPE_CONFIDENCE` is reported as ``unknown``.
    """

    held = {label.name for label in labels}
    if not features["row_count"]:
        # Every branch below reads a shape off the rows.  With none, the widest
        # net (a short list of codes) would catch the table and name it.
        return TableType.UNKNOWN, 0.0, "the table has no rows to read a shape from"
    measures: list[str] = features["measure_columns"]
    outgoing: list[str] = features["foreign_key_columns"]
    primary_key: list[str] = features["primary_key"]
    column_count: int = features["column_count"]
    row_count: int = features["row_count"]
    references: list[str] = features["references_tables"]

    if len(outgoing) >= 2 and not measures and column_count <= len(outgoing) + 2:
        return (
            TableType.BRIDGE,
            0.9,
            (
                f"{len(outgoing)} foreign keys and {column_count - len(outgoing)} other "
                "column(s), with nothing measured: the table exists to join "
                f"{' and '.join(references)}"
            ),
        )

    if "is_primary_key_time" in held and ("is_primary_key_periodic" in held or measures):
        periodic = "is_primary_key_periodic" in held
        return (
            TableType.TIME_SERIES,
            0.9 if periodic else 0.8,
            (
                "the primary key is a time column"
                + (" ticking at a fixed interval" if periodic else "")
                + (f", and {', '.join(measures)} are measured against it" if measures else "")
            ),
        )

    if outgoing and measures:
        return (
            TableType.FACT,
            0.9 if len(outgoing) >= 2 else 0.8,
            (
                f"rows measure {', '.join(measures)} and hang off "
                f"{', '.join(references) or 'another table'} through "
                f"{', '.join(outgoing)}"
            ),
        )

    if "is_self_reference" in held:
        return (
            TableType.HIERARCHY,
            0.9,
            "rows reference other rows of this same table, which is a tree",
        )

    if "is_label_having_hierarchy_column" in held and "is_having_dependency_chain" in held:
        return (
            TableType.HIERARCHY,
            0.75,
            (
                "the columns name levels and each level fixes the next: "
                + " → ".join(features["dependency_chain"])
            ),
        )

    if primary_key and not measures:
        referenced: list[str] = features["referenced_by_tables"]
        if referenced:
            return (
                TableType.DIMENSION,
                0.85 if "is_mostly_referenced" in held else 0.75,
                (
                    f"keyed by {' + '.join(primary_key)}, describes rather than measures, "
                    f"and {', '.join(referenced)} point(s) at it"
                ),
            )

    if not outgoing and row_count <= LOOKUP_MAX_ROWS and (
        "is_enum_containing_most" in held or column_count <= 3
    ):
        return (
            TableType.LOOKUP,
            0.7,
            (
                f"{row_count} row(s) of {column_count} mostly enumerated column(s), "
                "referencing nothing: a code list"
            ),
        )

    if column_count >= WIDE_COLUMN_COUNT:
        return (
            TableType.WIDE,
            0.7,
            (
                f"{column_count} columns in one table — descriptive attributes and "
                "measurements are held together rather than split"
            ),
        )

    if measures:
        return (
            TableType.FACT,
            0.6,
            (
                f"rows measure {', '.join(measures)}, though the table stands on its own "
                "with no reference into another"
            ),
        )

    if primary_key:
        return (
            TableType.DIMENSION,
            0.6,
            (
                f"keyed by {' + '.join(primary_key)} and descriptive throughout, though "
                "nothing references it yet"
            ),
        )

    return (
        TableType.UNKNOWN,
        0.0,
        "no branch of the table-type tree fits this table's shape",
    )


# ---------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------


def profile_table(
    table: str,
    df: pd.DataFrame,
    columns: Sequence[Mapping[str, Any]],
    key_analysis: Mapping[str, Any] | None = None,
    relationships: Sequence[Mapping[str, Any]] = (),
    header_info: Mapping[str, Any] | None = None,
    peer_table_count: int = 0,
    previous: TableProfile | None = None,
) -> TableProfile:
    """Profile one table.  ``previous`` carries the user's edits forward.

    ``columns`` are ``ColumnSemantics.to_dict()`` payloads (or anything with
    the same keys), ``relationships`` is the *whole session's* set — this
    function picks out the ones that touch ``table`` — and ``key_analysis`` is
    ``SheetRecord.key_analysis``.
    """

    facts = list(columns)
    names = {_name_of(fact) for fact in facts}
    relations = _relations(relationships)

    primary_key = [
        column for column in (key_analysis or {}).get("primary_key") or [] if column in names
    ]

    foreign_keys = [
        relation
        for relation in relations
        if relation.get("rel_type") == RelationshipType.FOREIGN_KEY.value
    ]
    outgoing = [
        relation
        for relation in foreign_keys
        if relation.get("from_table") == table and relation.get("from_column") in names
    ]
    incoming = [relation for relation in foreign_keys if relation.get("to_table") == table]
    self_references = [
        str(relation["from_column"]) for relation in outgoing if relation.get("to_table") == table
    ]

    dependencies = [
        [str(relation["from_column"]), str(relation["to_column"])]
        for relation in relations
        if relation.get("rel_type") == RelationshipType.FUNCTIONAL_DEPENDENCY.value
        and relation.get("from_table") == table
        and relation.get("to_table") == table
        and str(relation.get("from_column")) in names
        and str(relation.get("to_column")) in names
    ]

    structural = set(primary_key) | {str(r["from_column"]) for r in outgoing}
    measures = [_name_of(fact) for fact in facts if _is_measure(fact, structural)]
    # A measured quantity is not an enumeration even when it repeats: an order
    # line quantity of 1-7 is low-cardinality *and* is the number the table is
    # about.  Counting it twice would make a fact table look like a code list.
    enums = [
        _name_of(fact)
        for fact in facts
        if _name_of(fact) not in set(measures) and _is_enum(fact)
    ]

    features: dict[str, Any] = {
        "row_count": int(len(df)),
        "column_count": len(facts),
        "primary_key": primary_key,
        "key_source": str((key_analysis or {}).get("source", "none")),
        "structural_columns": sorted(structural),
        "measure_columns": measures,
        "enum_columns": enums,
        "foreign_key_columns": sorted({str(r["from_column"]) for r in outgoing}),
        "references_tables": sorted({str(r["to_table"]) for r in outgoing}),
        "referenced_by_tables": sorted(
            {str(r["from_table"]) for r in incoming if r.get("from_table") != table}
        ),
        "referenced_columns": sorted({str(r["to_column"]) for r in incoming}),
        "self_reference_columns": sorted(set(self_references)),
        "dependencies": dependencies,
        "dependency_chain": _longest_chain([(a, b) for a, b in dependencies]),
        "peer_table_count": max(0, peer_table_count),
        "unconfirmed_edges": sum(
            1
            for relation in (*outgoing, *incoming)
            if relation.get("status") != RelationshipStatus.CONFIRMED.value
        ),
    }

    labels = _detect_labels(df, facts, features, header_info)
    table_type, confidence, rationale = _classify(features, labels)
    if confidence < MIN_TYPE_CONFIDENCE:
        table_type, confidence = TableType.UNKNOWN, confidence

    profile = TableProfile(
        table=table,
        table_type=table_type,
        type_confidence=confidence,
        rationale=rationale,
        labels=labels,
        features=features,
    )
    if previous is not None:
        detected = set(TABLE_LABELS)
        profile.user_table_type = previous.user_table_type
        profile.added_labels = [name for name in previous.added_labels if name in detected]
        profile.removed_labels = [name for name in previous.removed_labels if name in detected]
    return profile


def profile_tables(
    tables: Mapping[str, pd.DataFrame],
    columns_by_table: Mapping[str, Sequence[Mapping[str, Any]]],
    key_analyses: Mapping[str, Mapping[str, Any]] | None = None,
    relationships: Sequence[Mapping[str, Any]] = (),
    header_info: Mapping[str, Mapping[str, Any]] | None = None,
    previous: Mapping[str, TableProfile] | None = None,
) -> dict[str, TableProfile]:
    """Profile every table in a session, preserving each one's user edits."""

    keys = key_analyses or {}
    headers = header_info or {}
    prior = previous or {}
    peers = max(0, len(tables) - 1)

    profiles: dict[str, TableProfile] = {}
    for name, df in tables.items():
        profiles[name] = profile_table(
            name,
            df,
            columns_by_table.get(name, ()),
            key_analysis=keys.get(name),
            relationships=relationships,
            header_info=headers.get(name),
            peer_table_count=peers,
            previous=prior.get(name),
        )
    return profiles
