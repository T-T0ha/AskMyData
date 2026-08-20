"""Phase 3 — evidence for a detected relationship.

SemTabla's validation framework (Table 1 of the paper) pairs every semantic
feature with two SQL queries: one that retrieves rows supporting the feature,
one that retrieves rows contradicting it.  The user confirms or rejects the
feature by looking at both.  That is the mechanism that makes an ~75 %-accurate
detector produce a semantic layer that is *correct*, and it only works if the
user can see what the claim rests on.

**The SQL shown is the SQL that ran.**  The tables at this stage are pandas
DataFrames, so the queries execute against an in-memory SQLite database built
from the two frames involved.  Filtering the DataFrames directly would be
faster and would let the panel display a query that nothing executed — a
"transparency" feature that shows the user something other than what happened.
Building the database costs a copy of two tables per request; the panel is
opened on demand, for one relationship at a time, and the cost is worth paying
to keep the displayed query honest.

Nothing here writes anything.  The SQLite database is in memory, is thrown away
when the request ends, and is never the source of the tables it was built from.
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import closing
from dataclasses import dataclass, field
from typing import Any, Mapping

import pandas as pd

from app.core.schemas import RelationshipType

logger = logging.getLogger(__name__)

#: Rows returned per sample.  Enough to judge a pattern, few enough to read.
SAMPLE_LIMIT = 10

#: Rows loaded into the scratch database per table.  A negative sample has to
#: be findable wherever it is, so this is a ceiling on very large tables rather
#: than a page size — and when it bites, the payload says so.
MAX_SCRATCH_ROWS = 100_000


@dataclass(slots=True)
class Sample:
    """One side of the evidence: the claim, the query, and the rows it found."""

    kind: str  # "positive" | "negative"
    title: str
    sql: str
    rows: list[dict[str, Any]] = field(default_factory=list)
    row_count: int = 0
    explanation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "title": self.title,
            "sql": self.sql,
            "rows": self.rows,
            "row_count": self.row_count,
            "explanation": self.explanation,
        }


@dataclass(slots=True)
class Evidence:
    positive: Sample
    negative: Sample
    truncated: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def holds(self) -> bool:
        """True when nothing contradicts the relationship."""

        return self.negative.row_count == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "holds": self.holds,
            "positive": self.positive.to_dict(),
            "negative": self.negative.to_dict(),
            "truncated": self.truncated,
            "notes": self.notes,
        }


def _quote(identifier: str) -> str:
    """SQLite identifier quoting — column names here come from spreadsheets."""

    return '"' + str(identifier).replace('"', '""') + '"'


def load_frame(connection: sqlite3.Connection, name: str, df: pd.DataFrame) -> bool:
    """Write one frame into a scratch database.  True if it was truncated.

    Public because the displayed SQL is only meaningful against the database it
    was run on: replaying a query means loading the tables the same way.
    """

    truncated = len(df) > MAX_SCRATCH_ROWS
    frame = (df.head(MAX_SCRATCH_ROWS) if truncated else df).copy()

    # SQLite binds four types and a pandas frame holds many more.  Datetimes
    # become ISO strings — sqlite3 refuses to bind a Timestamp, and ISO text
    # sorts and compares in the order the dates do, so the SQL in the panel
    # still means what it reads like.  Everything else goes through object,
    # where NaT/NA/NaN all become None, which is how NULL stays NULL.
    for column in frame.columns:
        series = frame[column]
        if pd.api.types.is_datetime64_any_dtype(series):
            frame[column] = series.dt.strftime("%Y-%m-%d %H:%M:%S").where(series.notna(), None)
        elif pd.api.types.is_timedelta64_dtype(series):
            frame[column] = series.astype(str).where(series.notna(), None)

    frame = frame.astype(object).where(pd.notna(frame), None)
    frame.to_sql(name, connection, index=False, if_exists="replace")
    return truncated


def _run(connection: sqlite3.Connection, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    cursor = connection.execute(sql, params)
    columns = [description[0] for description in cursor.description or []]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def _count(connection: sqlite3.Connection, base_sql: str) -> int:
    """How many rows there are in total, not how many the sample shows.

    Takes the query *without* its ``LIMIT``: the count is the number the panel
    leads with ("2 rows contradict this"), and deriving it by string-stripping
    the limit off the display query is how it silently becomes "10".
    """

    try:
        return int(connection.execute(f"SELECT COUNT(*) FROM ({base_sql})").fetchone()[0])
    except sqlite3.Error:  # pragma: no cover - defensive
        return 0


# ---------------------------------------------------------------------------
# foreign key
# ---------------------------------------------------------------------------


def foreign_key_evidence(
    source: pd.DataFrame,
    target: pd.DataFrame,
    from_table: str,
    from_column: str,
    to_table: str,
    to_column: str,
    limit: int = SAMPLE_LIMIT,
) -> Evidence:
    """Table 1, "Primary Key and Foreign Key": do the values exist in the key?

    The negative query is the one that decides the relationship.  A foreign key
    means *every* non-null value on the left is a real key on the right, so a
    single row returned by the negative query disproves it — which is why the
    panel leads with the count rather than the sample.
    """

    src, tgt = _quote(from_table), _quote(to_table)
    col, key = _quote(from_column), _quote(to_column)
    # Distinct target keys as a subquery rather than a join: a join multiplies
    # the source row out when the "key" turns out not to be unique, and the
    # user would be reading duplicated evidence for a different defect.
    positive_base = (
        f"SELECT * FROM {src}\n"
        f"WHERE {col} IS NOT NULL\n"
        f"  AND {col} IN (SELECT {key} FROM {tgt} WHERE {key} IS NOT NULL)"
    )
    negative_base = (
        f"SELECT * FROM {src}\n"
        f"WHERE {col} IS NOT NULL\n"
        f"  AND {col} NOT IN (SELECT {key} FROM {tgt} WHERE {key} IS NOT NULL)"
    )
    positive_sql = f"{positive_base}\nLIMIT {limit}"
    negative_sql = f"{negative_base}\nLIMIT {limit}"

    with closing(sqlite3.connect(":memory:")) as connection:
        truncated = load_frame(connection, from_table, source)
        truncated = load_frame(connection, to_table, target) or truncated
        positive = Sample(
            kind="positive",
            title=f"Rows whose {from_column} is a real {to_table}.{to_column}",
            sql=positive_sql,
            rows=_run(connection, positive_sql),
            row_count=_count(connection, positive_base),
            explanation=(
                f"These rows join cleanly: their {from_column} exists in "
                f"{to_table}.{to_column}. A foreign key predicts that all of them do."
            ),
        )
        negative = Sample(
            kind="negative",
            title=f"Rows whose {from_column} matches no {to_table}.{to_column}",
            sql=negative_sql,
            rows=_run(connection, negative_sql),
            row_count=_count(connection, negative_base),
            explanation=(
                f"These rows point at a {to_table} that is not there. Any row here "
                f"means the reference does not hold for every row — either the "
                f"relationship is wrong, or the data has orphans worth knowing about."
            ),
        )

    evidence = Evidence(positive=positive, negative=negative, truncated=truncated)
    if truncated:
        evidence.notes.append(
            f"evidence was gathered over the first {MAX_SCRATCH_ROWS:,} rows of each table"
        )
    if negative.row_count:
        evidence.notes.append(
            f"{negative.row_count:,} row(s) contradict this relationship — confirming it "
            "records a reference the data does not fully honour"
        )
    return evidence


# ---------------------------------------------------------------------------
# functional dependency
# ---------------------------------------------------------------------------


def dependency_evidence(
    df: pd.DataFrame,
    table: str,
    determinant: str,
    dependent: str,
    limit: int = SAMPLE_LIMIT,
) -> Evidence:
    """Table 1, "Functional Dependence": rows that conform, rows that violate.

    Grouped rather than row-by-row, because a functional dependency is a claim
    about groups: the violation is not a row, it is a determinant value that
    reaches two different dependent values.
    """

    name = _quote(table)
    left, right = _quote(determinant), _quote(dependent)
    positive_base = (
        f"SELECT {left}, MIN({right}) AS {right}, COUNT(*) AS rows\n"
        f"FROM {name}\n"
        f"WHERE {left} IS NOT NULL AND {right} IS NOT NULL\n"
        f"GROUP BY {left}\n"
        f"HAVING COUNT(DISTINCT {right}) = 1 AND COUNT(*) > 1\n"
        f"ORDER BY COUNT(*) DESC"
    )
    negative_base = (
        f"SELECT {left}, COUNT(DISTINCT {right}) AS distinct_values, COUNT(*) AS rows\n"
        f"FROM {name}\n"
        f"WHERE {left} IS NOT NULL AND {right} IS NOT NULL\n"
        f"GROUP BY {left}\n"
        f"HAVING COUNT(DISTINCT {right}) > 1\n"
        f"ORDER BY COUNT(DISTINCT {right}) DESC"
    )
    positive_sql = f"{positive_base}\nLIMIT {limit}"
    negative_sql = f"{negative_base}\nLIMIT {limit}"

    with closing(sqlite3.connect(":memory:")) as connection:
        truncated = load_frame(connection, table, df)
        positive = Sample(
            kind="positive",
            title=f"{determinant} values that always give the same {dependent}",
            sql=positive_sql,
            rows=_run(connection, positive_sql),
            row_count=_count(connection, positive_base),
            explanation=(
                f"Each of these {determinant} values appears on several rows and carries "
                f"the same {dependent} every time — which is what the dependency claims."
            ),
        )
        negative = Sample(
            kind="negative",
            title=f"{determinant} values that give more than one {dependent}",
            sql=negative_sql,
            rows=_run(connection, negative_sql),
            row_count=_count(connection, negative_base),
            explanation=(
                f"Each of these reaches two or more different {dependent} values. An exact "
                f"dependency predicts none of these exist; any that do make it approximate."
            ),
        )

    evidence = Evidence(positive=positive, negative=negative, truncated=truncated)
    if truncated:
        evidence.notes.append(
            f"evidence was gathered over the first {MAX_SCRATCH_ROWS:,} rows of the table"
        )
    return evidence


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------


def build_evidence(
    rel_type: str,
    tables: Mapping[str, pd.DataFrame],
    from_table: str,
    from_column: str,
    to_table: str,
    to_column: str,
) -> Evidence:
    """Evidence for whichever kind of relationship this is."""

    if from_table not in tables:
        raise KeyError(f"unknown table {from_table!r}")
    if rel_type == RelationshipType.FOREIGN_KEY.value:
        if to_table not in tables:
            raise KeyError(f"unknown table {to_table!r}")
        return foreign_key_evidence(
            tables[from_table], tables[to_table], from_table, from_column, to_table, to_column
        )
    return dependency_evidence(tables[from_table], from_table, from_column, to_column)
