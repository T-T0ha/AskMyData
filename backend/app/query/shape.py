"""Adaptive result renderer, decided server-side (§Phase 5).

The frontend renders whatever ``chart`` says rather than re-deriving it from
the rows: this is the same judgement call the platform makes everywhere else
— a taxonomy label, a table type, an "uncertain" flag — computed once, close
to the data, and carried to the UI as a fact rather than recomputed per
client. A person can still switch the view by hand; this only decides the
default.

Column kind is inferred from the *values actually returned*, not from the
semantic layer's declared types, because a ``SELECT COUNT(*)`` or a computed
ratio has no column in the schema to look up — its type only exists in the
result. It must also read a date from either a native object or a string: a
raw ``text()`` execution on SQLite (the platform's own development and test
target) hands back every value SQLite's dynamic typing gives it, which for a
``TIMESTAMP``-declared column is the stored string, not a ``datetime`` —
PostgreSQL's driver does return native objects for the same query, so a
classifier that only recognised ``datetime``/``date`` would silently work in
production and silently downgrade every trend question to a table in
development, which is the one place it would go unnoticed the longest.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Mapping, Sequence

#: A bare ``YYYY`` is deliberately excluded — a 4-digit string is too often a
#: code or an id to treat as a date on shape alone.  A grouped period
#: (``YYYY-MM``) or a full date (``YYYY-MM-DD``, optionally with a time) is
#: unambiguous enough to call.
_YEAR_MONTH_RE = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")
_ISO_DATE_PREFIX_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")

#: Row-count band a categorical breakdown renders as a pie chart; outside it,
#: a bar chart. claude.md's spec allows up to 6 slices; this caps at 3.  A
#: pie's slices are effectively an all-pairs comparison (sorted by value, any
#: two can end up adjacent), and the platform's validated categorical palette
#: only clears the color-blindness and normal-vision separation floors for
#: all-pairs comparison on its first three slots — a fourth puts two colors
#: on screen that measure below the "hard to tell apart" floor even for
#: full-color vision. A bar chart needs no such cap: one hue for every bar,
#: categories told apart by axis position rather than color.
PIE_MAX_ROWS = 3
PIE_MIN_ROWS = 2

#: Values sampled per column to decide its kind.  Enough to be sure without
#: walking a thousand-row result for every column of every question.
SAMPLE_SIZE = 20


def _looks_like_date_string(text: str) -> bool:
    if _YEAR_MONTH_RE.match(text):
        return True
    if _ISO_DATE_PREFIX_RE.match(text):
        try:
            date.fromisoformat(text[:10])
            return True
        except ValueError:
            return False
    return False


def _column_kind(values: Sequence[Any]) -> str:
    sample = [v for v in values if v is not None][:SAMPLE_SIZE]
    if not sample:
        return "text"
    if all(isinstance(v, bool) for v in sample):
        return "boolean"
    if all(isinstance(v, (int, float, Decimal)) and not isinstance(v, bool) for v in sample):
        return "numeric"
    if all(isinstance(v, (datetime, date)) for v in sample):
        return "date"
    if all(isinstance(v, str) and _looks_like_date_string(v) for v in sample):
        return "date"
    return "text"


def classify(columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """What the result looks like, and the chart it suggests.

    Follows claude.md's result-shape table: a single number, a category
    breakdown, a trend over time, or — the default whenever the shape does not
    match one of those — a sortable table, which is always a correct rendering
    of any result even when it is not the most illuminating one.
    """

    kinds = {name: _column_kind([row.get(name) for row in rows]) for name in columns}
    numeric_columns = [c for c in columns if kinds[c] == "numeric"]
    date_columns = [c for c in columns if kinds[c] == "date"]
    other_columns = [c for c in columns if kinds[c] in ("text", "boolean")]
    n = len(rows)

    if n == 1 and len(columns) == 1:
        chart = "metric"
    elif len(columns) == 2 and len(date_columns) == 1 and len(numeric_columns) == 1:
        chart = "line"
    elif len(columns) == 2 and len(numeric_columns) == 1 and len(other_columns) == 1:
        chart = "pie" if PIE_MIN_ROWS <= n <= PIE_MAX_ROWS else "bar"
    elif len(columns) >= 3 and len(date_columns) == 1 and len(numeric_columns) >= 1:
        chart = "multi_line" if len(numeric_columns) > 1 else "line"
    else:
        chart = "table"

    return {
        "chart": chart,
        "column_kinds": kinds,
        "numeric_columns": numeric_columns,
        "date_columns": date_columns,
    }
