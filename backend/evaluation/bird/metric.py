"""Scoring a predicted query against BIRD's gold query.

Three nested verdicts are computed for every prediction, so a run can be
re-read under whichever one is wanted without re-running anything:

``strict``
    BIRD's own execution accuracy — the predicted and gold *result sets* are
    equal as sets of rows (order- and duplicate-insensitive), matching the
    official script's ``set(predicted) == set(gold)``.  Floats are rounded to
    six places, a deliberate deviation so harmless rounding noise is not a miss.

``normalized``
    ``strict`` after the *representation* of a value is neutralised: a number
    equals the same number written as text (``103`` / ``'103'`` / ``103.0``) and
    a midnight timestamp equals its date.  The semantic arm queries the
    product's exported database, whose type fitting stores some numeric columns
    as text and dates as timestamps; without this a correct query would be
    marked wrong for what the *export* did to a value, which is a data-fidelity
    question (reported separately, see ``session_build.column_drift``) and not
    a question about the SQL.  The raw arm queries BIRD's own file, where this
    changes nothing but the occasional ``'1'`` / ``1`` pair.

``relaxed``
    ``normalized``, except the prediction may return *extra columns*.  A query
    that answers "which schools…?" with the school name **and** its id has
    still found the answer; BIRD's strict comparison calls that a miss.  The
    prediction passes when some choice of its columns, in any order, reproduces
    the gold result exactly — same set of rows, no extra rows — and the number
    of surplus columns is at most ``max_extra_columns``.  The row condition is
    what keeps this honest: ``SELECT *`` over an unfiltered table does not
    contain a *filtered* gold answer.

``strict`` ⊆ ``normalized`` ⊆ ``relaxed`` always holds.
"""

from __future__ import annotations

import itertools
import re
import threading
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Sequence

from . import sqlite_exec

#: More surplus columns than this is no longer "an answer with a little extra"
#: — it is a different, much wider result (``SELECT *``).  A per-run setting;
#: the report can re-apply a different cap because ``extra_columns`` is kept.
DEFAULT_MAX_EXTRA_COLUMNS = 3

#: Column-assignment search is exponential in the worst case (many identical
#: columns); this bounds the number of full row-set comparisons per question.
_MAX_ASSIGNMENTS = 2000

_NUMERIC = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")
_MIDNIGHT = re.compile(r"(\d{4}-\d{2}-\d{2})[ T]00:00:00(?:\.0+)?")


def strict_value(value: Any) -> Any:
    return round(value, 6) if isinstance(value, float) else value


def normalized_value(value: Any) -> Any:
    """A value reduced to what it *says*, not how it happens to be stored."""

    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float, Decimal)):
        return round(float(value), 6)
    if isinstance(value, str):
        text = value.strip()
        if _NUMERIC.fullmatch(text):
            return round(float(text), 6)
        midnight = _MIDNIGHT.fullmatch(text)
        if midnight:
            return midnight.group(1)
        return value
    return value


def _row_set(rows: Sequence[Sequence[Any]], convert) -> set[tuple]:
    return {tuple(convert(v) for v in row) for row in rows}


def _column_sets(rows: Sequence[Sequence[Any]], width: int) -> list[set]:
    return [{normalized_value(row[i]) for row in rows} for i in range(width)]


@dataclass(slots=True)
class Comparison:
    strict: bool
    normalized: bool
    #: A matching choice of columns exists (before applying ``max_extra_columns``).
    relaxed_match: bool
    #: Predicted columns not needed to reproduce the gold result; ``None`` when
    #: no choice of columns reproduces it.
    extra_columns: int | None
    relaxed: bool


def compare(
    pred_rows: Sequence[Sequence[Any]],
    gold_rows: Sequence[Sequence[Any]],
    pred_width: int,
    gold_width: int,
    max_extra_columns: int = DEFAULT_MAX_EXTRA_COLUMNS,
) -> Comparison:
    strict = _row_set(pred_rows, strict_value) == _row_set(gold_rows, strict_value)
    gold_set = _row_set(gold_rows, normalized_value)
    pred_set = _row_set(pred_rows, normalized_value)
    normalized = pred_width == gold_width and pred_set == gold_set

    matched = False
    if normalized:
        matched = True
    elif pred_width >= gold_width:
        # Same width in another order, or wider: look for the gold columns
        # among the predicted ones.
        if not gold_rows:
            matched = not pred_rows
        else:
            matched = _columns_reproduce(pred_rows, gold_set, gold_rows, pred_width, gold_width)
    extra = pred_width - gold_width if matched else None
    relaxed = matched and extra <= max_extra_columns
    return Comparison(
        strict=strict,
        normalized=normalized,
        relaxed_match=matched,
        extra_columns=extra,
        relaxed=relaxed,
    )


def _columns_reproduce(
    pred_rows: Sequence[Sequence[Any]],
    gold_set: set[tuple],
    gold_rows: Sequence[Sequence[Any]],
    pred_width: int,
    gold_width: int,
) -> bool:
    if pred_width < gold_width:
        return False
    pred_columns = _column_sets(pred_rows, pred_width)
    gold_columns = _column_sets(gold_rows, gold_width)
    candidates = [
        [i for i, values in enumerate(pred_columns) if values == wanted] for wanted in gold_columns
    ]
    if any(not options for options in candidates):
        return False

    budget = _MAX_ASSIGNMENTS
    for assignment in itertools.product(*candidates):
        if len(set(assignment)) != len(assignment):
            continue
        budget -= 1
        if budget < 0:
            return False
        projected = {tuple(normalized_value(row[i]) for i in assignment) for row in pred_rows}
        if projected == gold_set:
            return True
    return False


@dataclass(slots=True)
class ScoredQuery:
    strict: bool = False
    normalized: bool = False
    relaxed: bool = False
    relaxed_match: bool = False
    extra_columns: int | None = None
    #: Set when the *predicted* query failed to execute at all (distinct from
    #: "executed but gave a different answer").
    predicted_error: str | None = None
    #: Set when the *gold* query itself failed to execute — a dataset issue,
    #: never counted against either arm.
    gold_error: str | None = None
    predicted_rows: int | None = None
    gold_rows: int | None = None
    predicted_columns: int | None = None
    gold_columns: int | None = None
    #: First few rows of each side, for reading a failure without re-running it.
    predicted_preview: list[list[Any]] = field(default_factory=list)
    gold_preview: list[list[Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__slots__}


_gold_cache: dict[tuple[str, str], tuple[int, list[tuple]] | Exception] = {}
_gold_lock = threading.Lock()
#: Results bigger than this are not cached — a handful of such questions would
#: otherwise pin most of the process's memory for the whole run.
_GOLD_CACHE_ROW_LIMIT = 20_000


def _gold(db_path: Path, gold_sql: str, timeout: float) -> tuple[int, list[tuple]]:
    key = (str(db_path), gold_sql)
    with _gold_lock:
        cached = _gold_cache.get(key)
    if isinstance(cached, Exception):
        raise cached
    if cached is not None:
        return cached
    try:
        columns, rows, _ = sqlite_exec.run_readonly(db_path, gold_sql, None, timeout_seconds=timeout)
        value = (len(columns), rows)
    except Exception as exc:
        with _gold_lock:
            _gold_cache[key] = exc
        raise
    if len(rows) <= _GOLD_CACHE_ROW_LIMIT:
        with _gold_lock:
            _gold_cache[key] = value
    return value


def _preview(rows: Sequence[Sequence[Any]], limit: int = 5) -> list[list[Any]]:
    return [[v if isinstance(v, (int, float, str, type(None))) else str(v) for v in row] for row in rows[:limit]]


def score(
    pred_sql: str | None,
    gold_sql: str,
    gold_db: Path,
    pred_db: Path | None = None,
    max_extra_columns: int = DEFAULT_MAX_EXTRA_COLUMNS,
    timeout: float = sqlite_exec.DEFAULT_TIMEOUT_SECONDS,
) -> ScoredQuery:
    """Run both queries and compare.

    ``pred_db`` is where the prediction runs — the same file as ``gold_db`` for
    the raw arm, the product's exported database for the semantic arm.  The gold
    query always runs against BIRD's own file.
    """

    if not pred_sql:
        return ScoredQuery(predicted_error="no SQL produced")

    try:
        gold_width, gold_rows = _gold(gold_db, gold_sql, timeout)
    except Exception as exc:
        return ScoredQuery(gold_error=str(exc))

    try:
        columns, pred_rows, _ = sqlite_exec.run_readonly(
            pred_db or gold_db, pred_sql, None, timeout_seconds=timeout
        )
    except Exception as exc:
        return ScoredQuery(
            predicted_error=str(exc),
            gold_rows=len(gold_rows),
            gold_columns=gold_width,
            gold_preview=_preview(gold_rows),
        )

    verdict = compare(pred_rows, gold_rows, len(columns), gold_width, max_extra_columns)
    return ScoredQuery(
        strict=verdict.strict,
        normalized=verdict.normalized,
        relaxed=verdict.relaxed,
        relaxed_match=verdict.relaxed_match,
        extra_columns=verdict.extra_columns,
        predicted_rows=len(pred_rows),
        gold_rows=len(gold_rows),
        predicted_columns=len(columns),
        gold_columns=gold_width,
        predicted_preview=_preview(pred_rows),
        gold_preview=_preview(gold_rows),
    )


def execution_accuracy(pred_sql: str | None, gold_sql: str, db_path: Path) -> ScoredQuery:
    """Back-compatible entry point: both queries against one database file."""

    return score(pred_sql, gold_sql, db_path)
