"""Phase 0 — candidate key discovery.

A relational database is its keys.  Without one, a table cannot be referenced,
cannot be joined to and cannot be exported with a ``PRIMARY KEY`` clause — so
the question "what identifies a row here?" has to be answered during ingestion,
not left until the export.

The proposal's no-key handling has two branches and this module is what makes
both possible:

============================  ==================================================
no key, duplicate rows        propose a synthetic ``row_id`` in the cleaning plan
no single key, no duplicates  show the *composite* candidate for confirmation
============================  ==================================================

Discovery is hierarchical and minimal: a declared key wins outright, then
single columns, then pairs, then triples — stopping at the first depth that
yields anything, so a returned candidate never contains a smaller key.

Two cheap facts keep the combinatorics honest on wide tables:

* A set of columns can only be unique if the product of their distinct counts
  is at least the row count.  Combinations that fail this are discarded without
  ever being evaluated.
* Uniqueness over a table implies uniqueness over any subset of its rows.  So a
  combination that is *not* unique in a sample cannot be unique in the whole
  table, and sampling can only ever produce extra candidates to verify — never
  miss one.  Candidates that survive the sample are confirmed against every row.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

#: Deepest composite key searched.  Three columns is already an unusual
#: business key; beyond it the search costs more than the answer is worth.
MAX_KEY_DEPTH = 3

#: Rows above which candidate discovery runs on a sample first.
SAMPLE_ROWS = 10_000

#: Contiguous row-position bands the sample is drawn from, proportionally.  A
#: workbook's shape often changes partway through — a batch appended later
#: with reused ids, a header repeated mid-file — so a sample confined to one
#: region of the file could miss exactly the collision that matters; cutting
#: it into bands and drawing from each guards against that.
SAMPLE_STRATA = 20

#: Columns considered for composite keys, most selective first.  A key needs
#: cardinality, so the least selective columns are the least useful to try.
MAX_COMPOSITE_COLUMNS = 12

#: Enough candidates to choose between; a list of forty is not a choice.
MAX_CANDIDATES = 5

#: Name fragments that make a column read like an identifier.  Used only to
#: order equally-valid candidates — never to accept or reject one.
_KEY_NAME_HINTS = ("id", "key", "code", "no", "num", "number", "ref", "sku", "uuid")

_SAMPLE_SEED = 20240501


@dataclass(slots=True)
class KeyCandidate:
    """A set of columns that uniquely identifies every row."""

    columns: list[str]
    kind: str  # "declared" | "single" | "composite"
    evidence: str
    confidence: float = 1.0

    @property
    def label(self) -> str:
        return " + ".join(self.columns)

    def to_dict(self) -> dict[str, Any]:
        return {
            "columns": self.columns,
            "kind": self.kind,
            "evidence": self.evidence,
            "confidence": round(self.confidence, 4),
            "label": self.label,
        }


@dataclass(slots=True)
class KeyAnalysis:
    """What identifies a row in one table, and how sure we are.

    ``primary_key`` is only ever populated by something the platform can stand
    behind without asking: a key the source database declared, or a single
    column that is unique and never empty.  A composite candidate lands in
    ``candidates`` with ``needs_confirmation`` set, because "these two columns
    together are the key" is a statement about the business, not about the
    data, and only the user can make it.
    """

    primary_key: list[str] = field(default_factory=list)
    source: str = "none"  # declared | detected | confirmed | none
    candidates: list[KeyCandidate] = field(default_factory=list)
    duplicate_rows: int = 0
    row_count: int = 0
    needs_synthetic_key: bool = False
    needs_confirmation: bool = False
    confirmed: bool = False
    sampled: bool = False
    #: ``None`` when the source declared no key; ``False`` when it declared one
    #: that does not hold over the rows actually loaded.
    declared_key_holds: bool | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def has_key(self) -> bool:
        return bool(self.primary_key)

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary_key": self.primary_key,
            "source": self.source,
            "candidates": [c.to_dict() for c in self.candidates],
            "duplicate_rows": self.duplicate_rows,
            "row_count": self.row_count,
            "needs_synthetic_key": self.needs_synthetic_key,
            "needs_confirmation": self.needs_confirmation,
            "confirmed": self.confirmed,
            "sampled": self.sampled,
            "declared_key_holds": self.declared_key_holds,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "KeyAnalysis":
        payload = payload or {}
        return cls(
            primary_key=[str(c) for c in payload.get("primary_key") or []],
            source=str(payload.get("source", "none")),
            candidates=[
                KeyCandidate(
                    columns=[str(c) for c in candidate.get("columns") or []],
                    kind=str(candidate.get("kind", "composite")),
                    evidence=str(candidate.get("evidence", "")),
                    confidence=float(candidate.get("confidence", 1.0)),
                )
                for candidate in payload.get("candidates") or []
            ],
            duplicate_rows=int(payload.get("duplicate_rows", 0)),
            row_count=int(payload.get("row_count", 0)),
            needs_synthetic_key=bool(payload.get("needs_synthetic_key", False)),
            needs_confirmation=bool(payload.get("needs_confirmation", False)),
            confirmed=bool(payload.get("confirmed", False)),
            sampled=bool(payload.get("sampled", False)),
            declared_key_holds=payload.get("declared_key_holds"),
            notes=[str(n) for n in payload.get("notes") or []],
        )


# ---------------------------------------------------------------------------
# uniqueness primitives
# ---------------------------------------------------------------------------


def is_unique_key(df: pd.DataFrame, columns: Sequence[str]) -> bool:
    """True when ``columns`` identify every row: no nulls, no repeats.

    A null in a key column disqualifies it outright — ``NULL`` is not equal to
    itself in SQL, so a "key" containing one cannot be used to reference a row
    no matter how unique pandas thinks it is.
    """

    usable = [c for c in columns if c in df.columns]
    if not usable or len(usable) != len(columns):
        return False
    subset = df.loc[:, usable]
    if subset.isna().any().any():
        return False
    if len(subset) == 0:
        return False
    return not subset.duplicated().any()


def _is_measurement(series: pd.Series) -> bool:
    """True for a column of fractional numbers — a quantity, not an identity.

    An amount that happens to be distinct in every row is unique by accident,
    and accepting it would put ``PRIMARY KEY (amount)`` on the exported table:
    correct arithmetic, nonsense as a schema, and it breaks the moment two
    orders are for the same money.  Identifiers that arrive as numbers are
    whole (Phase 0's cleaner casts them to ``Int64``), so this excludes
    measurements without excluding numeric keys.
    """

    return pd.api.types.is_float_dtype(series)


def _eligible_columns(df: pd.DataFrame) -> tuple[list[tuple[str, int]], list[str]]:
    """Null-free columns with something to distinguish, most selective first.

    Also returns the measurement columns that were passed over, so a table that
    ends up with no key can say why.
    """

    eligible: list[tuple[str, int]] = []
    measures: list[str] = []
    for raw in df.columns:
        column = str(raw)
        series = df[raw]
        if series.isna().any():
            continue
        distinct = int(series.nunique(dropna=False))
        if distinct < 2:
            # A constant column adds nothing to any combination it joins.
            continue
        if _is_measurement(series):
            measures.append(column)
            continue
        eligible.append((column, distinct))
    eligible.sort(key=lambda item: item[1], reverse=True)
    return eligible, measures


def _name_score(columns: Iterable[str]) -> int:
    """How much the names read like a key — a tiebreak, never a filter."""

    score = 0
    for column in columns:
        tokens = set(str(column).lower().replace("-", "_").split("_"))
        if tokens & set(_KEY_NAME_HINTS):
            score += 2
        elif any(hint in str(column).lower() for hint in _KEY_NAME_HINTS):
            score += 1
    return score


def _rank(df: pd.DataFrame, candidate: KeyCandidate) -> tuple[int, int]:
    """Sort order for equally valid keys: key-ish names first, then leftmost.

    Several column sets can be mathematically unique at once
    (``sku + order_id`` as well as ``order_id + line_no``).  Both tiebreaks
    encode the same observation about real tables: the key is written in
    identifier-shaped names, near the left edge.
    """

    positions = {str(c): i for i, c in enumerate(df.columns)}
    return (
        -_name_score(candidate.columns),
        sum(positions.get(c, len(positions)) for c in candidate.columns),
    )


def _sample(df: pd.DataFrame, sample_rows: int) -> tuple[pd.DataFrame, bool]:
    """A stratified sample: the table is cut into equal contiguous row-position
    bands (:data:`SAMPLE_STRATA` of them) and each contributes its
    proportional share, drawn at random within the band.  A table exported in
    key order — or one where a later batch reused ids — cannot dominate or be
    invisible to the sample the way a single unconstrained draw could.

    Final correctness never rests on this: every candidate that survives the
    sample is re-verified against the *whole* table before being reported
    (see :func:`discover_keys`), so this only changes which candidates get
    that expensive check, never which ones are accepted.
    """

    if len(df) <= sample_rows:
        return df, False
    rng = np.random.RandomState(_SAMPLE_SEED)
    bands = np.array_split(np.arange(len(df)), min(SAMPLE_STRATA, len(df), sample_rows))
    per_band = max(1, sample_rows // len(bands))
    chosen = np.concatenate(
        [rng.choice(band, size=min(per_band, len(band)), replace=False) for band in bands if len(band)]
    )
    return df.iloc[np.sort(chosen)], True


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------


def _single_column_candidates(df: pd.DataFrame, eligible: list[tuple[str, int]]) -> list[KeyCandidate]:
    row_count = len(df)
    found: list[KeyCandidate] = []
    for column, distinct in eligible:
        if distinct != row_count:
            continue
        if not is_unique_key(df, [column]):
            continue
        found.append(
            KeyCandidate(
                columns=[column],
                kind="single",
                evidence=f"{distinct:,} distinct values over {row_count:,} rows, none empty",
            )
        )
    found.sort(key=lambda candidate: _rank(df, candidate))
    return found[:MAX_CANDIDATES]


def _composite_candidates(
    df: pd.DataFrame,
    full: pd.DataFrame,
    eligible: list[tuple[str, int]],
    depth: int,
) -> list[KeyCandidate]:
    row_count = len(full)
    pool = eligible[:MAX_COMPOSITE_COLUMNS]
    distinct_by_column = dict(pool)
    found: list[KeyCandidate] = []

    for combination in combinations([column for column, _ in pool], depth):
        # The product of the distinct counts bounds how many distinct
        # combinations can exist: below the row count, uniqueness is
        # arithmetically impossible and the check is skipped entirely.
        capacity = 1
        for column in combination:
            capacity *= distinct_by_column[column]
            if capacity >= len(df):
                break
        if capacity < len(df):
            continue
        if not is_unique_key(df, list(combination)):
            continue
        if df is not full and not is_unique_key(full, list(combination)):
            # Survived the sample but not the whole table.
            continue
        found.append(
            KeyCandidate(
                columns=list(combination),
                kind="composite",
                evidence=(
                    f"together unique across all {row_count:,} rows; "
                    "no single column in this table is"
                ),
            )
        )
        if len(found) >= MAX_CANDIDATES * 2:
            break

    found.sort(key=lambda candidate: _rank(full, candidate))
    return found[:MAX_CANDIDATES]


def discover_keys(
    df: pd.DataFrame,
    declared_primary_key: Sequence[str] | None = None,
    max_depth: int = MAX_KEY_DEPTH,
    sample_rows: int = SAMPLE_ROWS,
) -> KeyAnalysis:
    """Find what identifies a row in ``df``.

    ``declared_primary_key`` is the key a database source stated about itself.
    It is trusted when it still holds over the rows that were loaded, and
    demoted to an ordinary candidate with a note when it does not — a SQL dump
    can lose its constraints, and a declared key that does not hold is a fact
    the user needs rather than one to paper over.

    Discovery runs on the **de-duplicated** frame, because ``deduplicate`` is
    ordered before ``add_synthetic_key`` in the cleaning plan: a table whose
    only key collision is a repeated row does not need a synthetic key once
    that row is gone.
    """

    analysis = KeyAnalysis(row_count=int(len(df)))
    if df.empty or not len(df.columns):
        analysis.notes.append("table has no rows to analyse")
        return analysis

    duplicates = int(df.duplicated().sum())
    analysis.duplicate_rows = duplicates
    working = df.drop_duplicates() if duplicates else df
    if duplicates:
        analysis.notes.append(
            f"{duplicates:,} exact duplicate row(s) ignored while looking for a key — "
            "removing them is a separate, reviewable step"
        )

    declared = [str(c) for c in (declared_primary_key or [])]
    if declared:
        missing = [c for c in declared if c not in working.columns]
        if missing:
            analysis.declared_key_holds = False
            analysis.notes.append(
                "the source declared a primary key on column(s) "
                + ", ".join(missing)
                + " that are not present in the loaded data"
            )
        elif is_unique_key(working, declared):
            analysis.declared_key_holds = True
            analysis.primary_key = declared
            analysis.source = "declared"
            analysis.candidates = [
                KeyCandidate(
                    columns=declared,
                    kind="declared",
                    evidence="declared as the primary key by the source database",
                )
            ]
            return analysis
        else:
            analysis.declared_key_holds = False
            analysis.notes.append(
                "the primary key declared by the source ("
                + ", ".join(declared)
                + ") is not unique over the rows that were loaded — it is offered as a "
                "candidate rather than used"
            )

    eligible, measures = _eligible_columns(working)
    singles = _single_column_candidates(working, eligible)
    if singles:
        analysis.candidates = singles
        analysis.primary_key = singles[0].columns
        analysis.source = "detected"
        return analysis

    sample, sampled = _sample(working, sample_rows)
    analysis.sampled = sampled
    if sampled:
        analysis.notes.append(
            f"composite candidates were searched over a random {sample_rows:,}-row sample "
            "and then verified against every row"
        )

    sample_eligible = _eligible_columns(sample)[0] if sampled else eligible
    for depth in range(2, max(2, max_depth) + 1):
        composites = _composite_candidates(sample, working, sample_eligible, depth)
        if composites:
            analysis.candidates = composites
            # Deliberately not promoted to primary_key: a composite key is a
            # claim about what the business considers one row, and the user
            # confirms it (see the API's key endpoint) before anything relies
            # on it.
            analysis.needs_confirmation = True
            return analysis

    analysis.needs_synthetic_key = True
    analysis.notes.append(
        f"no combination of up to {max_depth} column(s) identifies a row uniquely"
    )
    if measures:
        analysis.notes.append(
            "measurement column(s) "
            + ", ".join(measures)
            + " were not considered — a quantity that happens to be distinct in every "
            "row is unique by accident, not by design"
        )
    return analysis


def decline_key(analysis: KeyAnalysis) -> KeyAnalysis:
    """The user rejected every candidate: fall back to a synthetic ``row_id``."""

    analysis.primary_key = []
    analysis.source = "none"
    analysis.confirmed = True
    analysis.needs_confirmation = False
    analysis.needs_synthetic_key = True
    analysis.notes.append(
        "you rejected the suggested key(s), so the cleaning plan will offer a numbered row_id"
    )
    return analysis


def confirm_key(analysis: KeyAnalysis, columns: Sequence[str]) -> KeyAnalysis:
    """Record the user's decision that ``columns`` are the table's key."""

    chosen = [str(c) for c in columns]
    analysis.primary_key = chosen
    analysis.source = "confirmed"
    analysis.confirmed = True
    analysis.needs_confirmation = False
    analysis.needs_synthetic_key = False
    if not any(candidate.columns == chosen for candidate in analysis.candidates):
        analysis.candidates.insert(
            0,
            KeyCandidate(
                columns=chosen,
                kind="single" if len(chosen) == 1 else "composite",
                evidence="chosen by the user and verified unique over every row",
            ),
        )
    return analysis
