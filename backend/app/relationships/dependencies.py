"""Phase 3 — functional dependency detection (simplified TANE).

A functional dependency ``A → B`` says that fixing A fixes B: every row with
the same ``city`` has the same ``country``.  SemTabla (§4.1.4) uses TANE, whose
bottom-up level-wise search finds dependencies with multi-column left-hand
sides.  This implementation deliberately stops at single-column left sides.

Why that is the right trade here rather than a shortcut: TANE's cost is in the
levels above the first, and multi-column left sides (``A, B → C``) are rare in
business spreadsheets — the ones people actually have express hierarchies
(city → country, sku → category, employee → department), which are
single-column by construction.  The single-column version is a groupby per
ordered pair and finds those; full TANE is future work, and the proposal says
so.

The part that needs more care than the algorithm is deciding which dependencies
are worth *reporting*.  Most true dependencies in a real table are true for
reasons that tell the user nothing:

* a **key determines every column**, by definition of a key;
* a **constant column is determined by everything**, for the same reason;
* a column with almost as many distinct values as rows determines nearly
  anything by accident.

All three are filtered out here.  A dependency that survives is one where a
genuinely repeating value on the left always implies the same value on the
right — which is the hierarchy TANE is being used to reveal.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import pandas as pd

from app.core.schemas import RelationshipOrigin, RelationshipType

logger = logging.getLogger(__name__)

#: A left-hand column this close to unique determines everything by accident.
#: 0.90 keeps ``city`` (30 values over 200 rows) and drops ``email``.
MAX_DETERMINANT_UNIQUE_RATIO = 0.90

#: A determinant must actually repeat — at least this many groups must hold
#: more than one row, or the "dependency" is just a near-key in disguise.
MIN_REPEATED_GROUPS = 2

#: Exact dependency: every group agrees.  Approximate: this share of them does.
APPROXIMATE_THRESHOLD = 0.95

#: Widest table searched exhaustively.  The pair count is quadratic, and past
#: this the answer is worth less than the wait.
MAX_COLUMNS = 30

#: Rows sampled on a large table.  A dependency that fails on a 20k-row sample
#: cannot hold over the whole table, so sampling can only add candidates to
#: verify — never hide one.  Survivors are re-checked against every row.
SAMPLE_ROWS = 20_000
_SAMPLE_SEED = 20240501

#: Below this, "every group agrees" is a statement about almost nothing.
MIN_ROWS = 12


@dataclass(slots=True)
class DependencyCandidate:
    """``table.determinant → table.dependent``."""

    table: str
    determinant: str
    dependent: str
    score: float
    exact: bool
    #: Determinant values occurring on more than one row — the only ones that
    #: can witness the claim, and the denominator of :attr:`score`.
    witness_groups: int
    consistent_witnesses: int
    #: Every distinct determinant value, witnessing or not.
    total_groups: int
    rows_considered: int
    null_rows_dropped: int = 0
    violations: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def rel_type(self) -> RelationshipType:
        return (
            RelationshipType.FUNCTIONAL_DEPENDENCY
            if self.exact
            else RelationshipType.APPROXIMATE_DEPENDENCY
        )

    @property
    def label(self) -> str:
        return f"{self.determinant} → {self.dependent}"

    def explanation(self) -> str:
        if self.exact:
            base = (
                f"{self.witness_groups:,} {self.determinant} values occur on more than one "
                f"row, and every one of them carries the same {self.dependent} every time"
            )
        else:
            failing = self.witness_groups - self.consistent_witnesses
            base = (
                f"of the {self.witness_groups:,} {self.determinant} values that repeat, "
                f"{self.consistent_witnesses:,} always give the same {self.dependent} and "
                f"{failing:,} do not"
            )
        detail = f"{self.total_groups:,} distinct {self.determinant} values in total"
        if self.null_rows_dropped:
            return f"{base}; {detail}; {self.null_rows_dropped:,} row(s) with a blank ignored"
        return f"{base}; {detail}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "rel_type": self.rel_type.value,
            "from_table": self.table,
            "from_column": self.determinant,
            "to_table": self.table,
            "to_column": self.dependent,
            "score": round(self.score, 4),
            "origin": RelationshipOrigin.DETECTED.value,
            "uncertain": not self.exact,
            "explanation": self.explanation(),
            "evidence": {
                "witness_groups": self.witness_groups,
                "consistent_witnesses": self.consistent_witnesses,
                "total_groups": self.total_groups,
                "rows_considered": self.rows_considered,
                "null_rows_dropped": self.null_rows_dropped,
                "violations": self.violations[:10],
                "notes": self.notes,
            },
        }


def _eligible_columns(df: pd.DataFrame) -> list[str]:
    """Columns worth putting on either side of an arrow."""

    usable: list[str] = []
    for raw in df.columns:
        series = df[raw]
        distinct = int(series.nunique(dropna=True))
        if distinct < 2:
            continue  # a constant is determined by everything and determines nothing
        if pd.api.types.is_float_dtype(series):
            # Measurements do not participate in hierarchies: "amount 4500
            # always implies category Appliance" is a coincidence of a small
            # table, not a rule about the business.
            continue
        usable.append(str(raw))
    return usable[:MAX_COLUMNS]


def _check(
    df: pd.DataFrame, determinant: str, dependent: str
) -> tuple[int, int, int, int] | None:
    """``(witnesses, consistent, total_groups, rows)`` for one pair, or None.

    Rows where either side is blank are dropped rather than grouped: SQL's
    ``NULL`` is not equal to itself, so a dependency "holding" because two
    blanks matched would not hold in the exported database.

    The consistency ratio is computed over *witnessing* groups — the
    determinant values that occur on more than one row — not over all of them.
    A value appearing exactly once agrees with itself no matter what the data
    says, so counting it as supporting evidence inflates every score: a column
    with 30 distinct values over 45 rows would score 0.97 against anything at
    all, purely because 15 of its groups are singletons that cannot disagree.
    Only a repeated value can witness the claim, and only witnesses count.
    """

    pair = df.loc[:, [determinant, dependent]].dropna()
    if len(pair) < MIN_ROWS:
        return None
    grouped = pair.groupby(determinant, observed=True, dropna=True)[dependent]
    distinct_per_group = grouped.nunique()
    sizes = grouped.size()
    total_groups = int(len(distinct_per_group))
    if total_groups < 2:
        return None
    witnessing = sizes > 1
    witnesses = int(witnessing.sum())
    if not witnesses:
        return None
    consistent = int((distinct_per_group[witnessing] == 1).sum())
    return witnesses, consistent, total_groups, int(len(pair))


def _violations(df: pd.DataFrame, determinant: str, dependent: str, limit: int = 5) -> list[dict[str, Any]]:
    """The determinant values that map to more than one dependent value."""

    pair = df.loc[:, [determinant, dependent]].dropna()
    grouped = pair.groupby(determinant, observed=True, dropna=True)[dependent]
    offenders = grouped.nunique()
    offenders = offenders[offenders > 1]
    found: list[dict[str, Any]] = []
    for value in list(offenders.index)[:limit]:
        seen = pair.loc[pair[determinant] == value, dependent].unique()
        found.append(
            {
                "value": str(value),
                "maps_to": [str(v) for v in seen[:5]],
                "distinct": int(len(seen)),
            }
        )
    return found


def detect_dependencies(
    df: pd.DataFrame,
    table: str,
    key_columns: Sequence[str] = (),
    approximate_threshold: float = APPROXIMATE_THRESHOLD,
) -> list[DependencyCandidate]:
    """Single-column functional dependencies inside one table.

    ``key_columns`` are excluded as determinants.  A key determines every other
    column in the table — that is what "key" means — so reporting those
    dependencies would bury the ones that say something about the data under a
    row of arrows that say something about relational algebra.
    """

    if df.empty or len(df) < MIN_ROWS:
        return []

    columns = _eligible_columns(df)
    if len(columns) < 2:
        return []

    excluded = {str(c) for c in key_columns}
    row_count = len(df)
    sample = (
        df.sample(n=SAMPLE_ROWS, random_state=_SAMPLE_SEED) if row_count > SAMPLE_ROWS else df
    )

    found: list[DependencyCandidate] = []
    for determinant in columns:
        if determinant in excluded:
            continue
        unique_ratio = float(df[determinant].nunique(dropna=True)) / max(1, row_count)
        if unique_ratio > MAX_DETERMINANT_UNIQUE_RATIO:
            continue  # a near-key determines everything by accident

        for dependent in columns:
            if dependent == determinant:
                continue
            measured = _check(sample, determinant, dependent)
            if measured is None:
                continue
            witnesses, consistent, total_groups, rows = measured
            if witnesses < MIN_REPEATED_GROUPS:
                # Nothing repeats, so "it always implies the same dependent" is
                # a statement about one row at a time.
                continue
            ratio = consistent / witnesses
            if ratio < approximate_threshold:
                continue

            if sample is not df:
                verified = _check(df, determinant, dependent)
                if verified is None:
                    continue
                witnesses, consistent, total_groups, rows = verified
                if witnesses < MIN_REPEATED_GROUPS:
                    continue
                ratio = consistent / witnesses
                if ratio < approximate_threshold:
                    continue

            exact = consistent == witnesses
            candidate = DependencyCandidate(
                table=table,
                determinant=determinant,
                dependent=dependent,
                score=ratio,
                exact=exact,
                witness_groups=witnesses,
                consistent_witnesses=consistent,
                total_groups=total_groups,
                rows_considered=rows,
                null_rows_dropped=int(row_count - rows),
            )
            if not exact:
                candidate.violations = _violations(df, determinant, dependent)
                candidate.notes.append(
                    "holds for most values but not all — a data-quality finding, not a "
                    "constraint the exported schema can enforce"
                )
            found.append(candidate)

    # Exact before approximate, then by how strongly the determinant repeats:
    # a dependency over 30 repeating cities is more informative than one over
    # two.
    found.sort(key=lambda c: (not c.exact, -c.witness_groups, c.determinant, c.dependent))
    return found


def detect_all_dependencies(
    tables: Mapping[str, pd.DataFrame],
    key_columns: Mapping[str, Sequence[str]] | None = None,
) -> list[DependencyCandidate]:
    """Run :func:`detect_dependencies` over every table in a dataset."""

    keys = key_columns or {}
    found: list[DependencyCandidate] = []
    for table, df in tables.items():
        found.extend(detect_dependencies(df, table, key_columns=keys.get(table, ())))
    return found
