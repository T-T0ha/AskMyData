"""Phase 3 — foreign key detection.

The scoring is SemTabla's (Jin et al., CHI '26, §4.1.3, equations 1–3).  For a
source column *s* and a target table's primary key column *k*:

.. math::

    ratio_{overlap}  &= \\frac{|distinct(s) \\cap distinct(k)|}{|distinct(s)|} \\\\
    ratio_{distinct} &= \\frac{|distinct(s) \\cap distinct(k)|}{|distinct(k)|} \\\\
    Score            &= w_1 \\cdot ratio_{overlap} + w_2 \\cdot ratio_{distinct}

The two ratios answer different questions and both are needed.  Overlap asks
"is every value of *s* a real key?" — a column of order ids that are all
genuine order ids scores 1.0.  Distinct asks "does *s* reach much of the key?"
— without it, a column holding a single repeated id scores a perfect overlap
against a thousand-row table it barely touches.  The paper's guard is the same:
a match is valid only when ``ratio_overlap >= threshold`` **and**
``ratio_distinct > 0.1``.

This project adds two things to the paper's scorer, both from the proposal:

* a **name-embedding boost**, because ``line_items.order_id → orders.order_id``
  and ``line_items.order_id → invoices.invoice_id`` can have identical value
  overlap in a small table, and the column names are the only signal that
  separates them;
* a **fuzzy band** below the accept threshold, so a relationship that messy
  data has degraded (a few orphaned rows) is shown as uncertain rather than
  silently dropped.

What the platform will *not* do is invent confidence it does not have: nothing
here is written as a confirmed relationship.  Every candidate is a proposal
carrying its own arithmetic, and the user confirms or rejects it against the
evidence rows in :mod:`app.relationships.evidence`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import pandas as pd

from app.core.schemas import RelationshipOrigin, RelationshipType
from app.ingestion.keys import KeyAnalysis
from app.semantics.embeddings import cosine_matrix, embed_column_names

logger = logging.getLogger(__name__)

#: SemTabla's w1 and w2.  Overlap is weighted higher because it is the
#: statement that makes a join *valid*; reach is what makes it *useful*.
OVERLAP_WEIGHT = 0.70
DISTINCT_WEIGHT = 0.30

#: The paper's two validity guards.
MIN_OVERLAP = 0.60
MIN_DISTINCT = 0.10

#: How much the column names may move a score.  Small on purpose: names are a
#: tiebreak between value-compatible candidates, never a reason to propose a
#: relationship the values do not support.
EMBEDDING_BOOST = 0.20

#: Above this, the relationship is proposed.  Between the fuzzy floor and this,
#: it is proposed too — but labelled uncertain, and sorted below.
ACCEPT_SCORE = 0.60
FUZZY_FLOOR = 0.40

#: Below this many rows, coincidental overlap is more likely than a real
#: reference: six ids drawn from a ten-id key overlap perfectly by accident.
MIN_ROWS = 20

#: Distinct values above which the value set is sampled.  Set intersection is
#: linear, so this is generous.
MAX_VALUES = 200_000


@dataclass(slots=True)
class ForeignKeyCandidate:
    """A proposed ``source.column → target.column`` reference."""

    from_table: str
    from_column: str
    to_table: str
    to_column: str
    score: float
    ratio_overlap: float
    ratio_distinct: float
    embedding_similarity: float
    origin: RelationshipOrigin
    matched_values: int = 0
    source_distinct: int = 0
    target_distinct: int = 0
    orphan_values: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def is_uncertain(self) -> bool:
        return self.origin is RelationshipOrigin.FUZZY

    @property
    def label(self) -> str:
        return f"{self.from_table}.{self.from_column} → {self.to_table}.{self.to_column}"

    def explanation(self) -> str:
        """Why this was proposed, in the arithmetic that produced it."""

        if self.origin is RelationshipOrigin.DECLARED:
            return "declared as a foreign key by the source database"
        if self.origin is RelationshipOrigin.MANUAL:
            return "drawn by you in the relationship diagram"
        parts = [
            f"{self.ratio_overlap:.0%} of {self.from_column}'s "
            f"{self.source_distinct:,} distinct values are real "
            f"{self.to_table}.{self.to_column} keys",
            f"they reach {self.ratio_distinct:.0%} of that key's "
            f"{self.target_distinct:,} values",
        ]
        if self.embedding_similarity:
            parts.append(f"name similarity {self.embedding_similarity:.2f}")
        if self.orphan_values:
            shown = ", ".join(self.orphan_values[:3])
            parts.append(f"unmatched value(s): {shown}")
        return "; ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rel_type": RelationshipType.FOREIGN_KEY.value,
            "from_table": self.from_table,
            "from_column": self.from_column,
            "to_table": self.to_table,
            "to_column": self.to_column,
            "score": round(self.score, 4),
            "origin": self.origin.value,
            "uncertain": self.is_uncertain,
            "explanation": self.explanation(),
            "evidence": {
                "ratio_overlap": round(self.ratio_overlap, 4),
                "ratio_distinct": round(self.ratio_distinct, 4),
                "embedding_similarity": round(self.embedding_similarity, 4),
                "matched_values": self.matched_values,
                "source_distinct": self.source_distinct,
                "target_distinct": self.target_distinct,
                "orphan_values": self.orphan_values[:10],
                "notes": self.notes,
            },
        }


# ---------------------------------------------------------------------------
# value handling
# ---------------------------------------------------------------------------


def _dtype_family(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    return "text"


def _key_values(series: pd.Series) -> set[str]:
    """Distinct non-null values as strings.

    Compared as text because the two sides routinely disagree about storage
    while agreeing about identity: an id read from a CSV is ``object``, the
    same id read from SQLite is ``Int64``, and ``1 != "1"`` would hide the
    relationship.  Floats that are whole numbers are normalised too, so
    ``1001.0`` from a column pandas widened for a missing value still matches
    ``1001``.
    """

    values = series.dropna()
    if len(values) > MAX_VALUES:
        values = values.head(MAX_VALUES)
    if pd.api.types.is_float_dtype(values):
        whole = values[values == values.round()]
        if len(whole) == len(values):
            values = whole.astype("int64")
    return set(values.astype(str).str.strip())


def _is_referenceable(series: pd.Series) -> bool:
    """Could this column hold a reference at all?

    Excludes what cannot be a key on its own terms: booleans (two values point
    at nothing), constants, and floats.

    Floats are excluded outright, not just fractional ones.  Whole-numbered
    floats are the dangerous case: a column of amounts 1.0 … 40.0 overlaps a
    key of 1 … 40 perfectly and scores 1.00, which is a coincidence of range
    dressed up as a reference.  This is the same rule ``keys._is_measurement``
    applies on the other side — a quantity is not an identity — and applying it
    symmetrically is what keeps the two halves of the model consistent: a
    column that cannot be a primary key has no business being a foreign one.
    """

    if pd.api.types.is_bool_dtype(series):
        return False
    if pd.api.types.is_float_dtype(series):
        return False
    return int(series.nunique(dropna=True)) >= 2


# ---------------------------------------------------------------------------
# targets
# ---------------------------------------------------------------------------


def _single_column_keys(
    tables: Mapping[str, pd.DataFrame],
    keys: Mapping[str, KeyAnalysis],
) -> dict[str, str]:
    """Tables whose key is one column, as ``table -> column``.

    The paper's preprocessing step: *"tables containing composite primary keys
    are excluded"*.  A two-column key cannot be referenced by a single source
    column, and matching one half of it produces exactly the plausible-looking
    nonsense this phase exists to avoid.
    """

    targets: dict[str, str] = {}
    for table, analysis in keys.items():
        if table not in tables:
            continue
        primary = list(analysis.primary_key or [])
        if len(primary) != 1:
            continue
        column = primary[0]
        if column not in tables[table].columns:
            continue
        targets[table] = column
    return targets


# ---------------------------------------------------------------------------
# declared relationships
# ---------------------------------------------------------------------------


def declared_foreign_keys(
    native_schemas: Mapping[str, dict[str, Any]],
    tables: Mapping[str, pd.DataFrame],
) -> list[ForeignKeyCandidate]:
    """Foreign keys a database source stated about itself.

    Not rediscovered statistically and not scored: the source is the authority
    on its own constraints, and a declared reference that scores badly means
    the loaded rows are a subset, not that the constraint is wrong.  They are
    still proposals — a SQL dump can carry a constraint the data no longer
    honours — but they arrive at the top of the list with the reason "declared".
    """

    found: list[ForeignKeyCandidate] = []
    for table, schema in native_schemas.items():
        if table not in tables:
            continue
        for reference in (schema or {}).get("foreign_keys") or []:
            target = str(reference.get("references_table") or "")
            columns = [str(c) for c in reference.get("columns") or []]
            target_columns = [str(c) for c in reference.get("references_columns") or []]
            if not target or len(columns) != 1 or len(target_columns) != 1:
                continue
            if target not in tables:
                continue
            found.append(
                ForeignKeyCandidate(
                    from_table=table,
                    from_column=columns[0],
                    to_table=target,
                    to_column=target_columns[0],
                    score=1.0,
                    ratio_overlap=1.0,
                    ratio_distinct=1.0,
                    embedding_similarity=0.0,
                    origin=RelationshipOrigin.DECLARED,
                )
            )
    return found


# ---------------------------------------------------------------------------
# detection
# ---------------------------------------------------------------------------


def score_pair(source: pd.Series, target: pd.Series) -> tuple[float, float, float, int, int, int, list[str]]:
    """The paper's equations 1–3 for one column pair.

    Returns ``(base_score, ratio_overlap, ratio_distinct, matched,
    source_distinct, target_distinct, orphans)``.
    """

    source_values = _key_values(source)
    target_values = _key_values(target)
    if not source_values or not target_values:
        return 0.0, 0.0, 0.0, 0, len(source_values), len(target_values), []

    matched = source_values & target_values
    ratio_overlap = len(matched) / len(source_values)
    ratio_distinct = len(matched) / len(target_values)
    base = OVERLAP_WEIGHT * ratio_overlap + DISTINCT_WEIGHT * ratio_distinct
    orphans = sorted(source_values - target_values)[:10]
    return (
        base,
        ratio_overlap,
        ratio_distinct,
        len(matched),
        len(source_values),
        len(target_values),
        orphans,
    )


def _name_similarity(pairs: Sequence[tuple[str, str]]) -> list[float]:
    """Cosine similarity of every ``(source column, target column)`` name pair.

    Embedded in one batch: the model is the expensive part, and a schema of a
    few hundred columns produces thousands of pairs.
    """

    if not pairs:
        return []
    left_names = [left for left, _ in pairs]
    right_names = [right for _, right in pairs]
    left_vectors, _ = embed_column_names(left_names)
    right_vectors, _ = embed_column_names(right_names)
    similarity = cosine_matrix(left_vectors, right_vectors)
    return [float(similarity[i, i]) for i in range(len(pairs))]


def detect_foreign_keys(
    tables: Mapping[str, pd.DataFrame],
    keys: Mapping[str, KeyAnalysis],
    native_schemas: Mapping[str, dict[str, Any]] | None = None,
    min_rows: int = MIN_ROWS,
    accept_score: float = ACCEPT_SCORE,
    fuzzy_floor: float = FUZZY_FLOOR,
) -> list[ForeignKeyCandidate]:
    """Every plausible ``source.column → target.key`` reference in one dataset.

    One candidate per source column: the paper retains the best match, because
    a column that appears to reference three tables at once is reporting a
    coincidence in two of them.  Declared references are added on top and are
    never overwritten by a detected one for the same source column.
    """

    declared = declared_foreign_keys(native_schemas or {}, tables)
    claimed = {(c.from_table, c.from_column) for c in declared}

    targets = _single_column_keys(tables, keys)
    if not targets:
        return declared

    # Collect every (source, target) pair worth scoring, then embed the names
    # for all of them in one pass.
    pairs: list[tuple[str, str, str, str]] = []
    for source_table, df in tables.items():
        if len(df) < min_rows:
            continue
        for raw in df.columns:
            column = str(raw)
            if (source_table, column) in claimed:
                continue
            series = df[raw]
            if not _is_referenceable(series):
                continue
            source_family = _dtype_family(series)
            for target_table, target_column in targets.items():
                if target_table == source_table and target_column == column:
                    continue  # a key does not reference itself
                if len(tables[target_table]) < min_rows:
                    continue
                if _dtype_family(tables[target_table][target_column]) != source_family:
                    # Text ids and numeric ids do not reference each other, and
                    # comparing them as strings would occasionally say they do.
                    continue
                pairs.append((source_table, column, target_table, target_column))

    similarities = _name_similarity([(source, target) for _, source, _, target in pairs])

    best: dict[tuple[str, str], ForeignKeyCandidate] = {}
    for (source_table, column, target_table, target_column), similarity in zip(pairs, similarities):
        source_series = tables[source_table][column]
        target_series = tables[target_table][target_column]
        base, overlap, distinct, matched, source_n, target_n, orphans = score_pair(
            source_series, target_series
        )
        if overlap < MIN_OVERLAP or distinct <= MIN_DISTINCT:
            continue

        score = min(1.0, base + EMBEDDING_BOOST * max(0.0, similarity))
        if score < fuzzy_floor:
            continue
        origin = (
            RelationshipOrigin.DETECTED if score >= accept_score else RelationshipOrigin.FUZZY
        )

        candidate = ForeignKeyCandidate(
            from_table=source_table,
            from_column=column,
            to_table=target_table,
            to_column=target_column,
            score=score,
            ratio_overlap=overlap,
            ratio_distinct=distinct,
            embedding_similarity=similarity,
            origin=origin,
            matched_values=matched,
            source_distinct=source_n,
            target_distinct=target_n,
            orphan_values=orphans,
        )
        if orphans:
            candidate.notes.append(
                f"{source_n - matched:,} of {source_n:,} distinct values have no matching "
                f"{target_table}.{target_column} — referential integrity is not complete"
            )

        key = (source_table, column)
        if key not in best or candidate.score > best[key].score:
            best[key] = candidate

    candidates = declared + sorted(best.values(), key=lambda c: c.score, reverse=True)
    return candidates


def skipped_tables(tables: Mapping[str, pd.DataFrame], min_rows: int = MIN_ROWS) -> list[str]:
    """Tables too small to reason about, so the UI can say so rather than
    leave the user wondering why a table has no edges."""

    return sorted(name for name, df in tables.items() if len(df) < min_rows)
