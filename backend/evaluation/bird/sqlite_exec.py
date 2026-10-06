"""Read-only SQLite execution for one BIRD database file.

Mirrors the shape of ``app/export/runner.py``'s ``explain``/``run_readonly``
pair (validate cheaply first, only then run) but talks to a bare BIRD
``.sqlite`` file directly instead of a session's exported Postgres schema —
there is no session here, just the database BIRD ships.

Every connection is opened ``mode=ro`` (SQLite refuses any write at the OS
level, not just by convention) and every execution is wrapped in a step-count
progress handler so a pathological join can't hang the whole eval run.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

#: Aborts a query that runs longer than this many seconds — a bad join or an
#: accidental cross product must not stall an eval run of a thousand questions.
DEFAULT_TIMEOUT_SECONDS = 15.0

#: SQLite VM instructions between timeout checks. Small enough to cut off a
#: runaway query within a fraction of a second of the deadline, large enough
#: that the check itself is not the bottleneck.
_PROGRESS_STEPS = 50_000


class QueryTimeout(Exception):
    pass


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only = ON")
    return conn


def _with_timeout(conn: sqlite3.Connection, timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds

    def handler() -> int:
        return 1 if time.monotonic() > deadline else 0

    conn.set_progress_handler(handler, _PROGRESS_STEPS)


def explain(db_path: Path, sql: str, timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS) -> None:
    """Validate ``sql`` against the schema without materialising its result.

    Raises ``sqlite3.Error`` (unknown table/column, syntax error, ...) or
    :class:`QueryTimeout` — either way the caller's retry loop treats it as
    "this attempt failed" and feeds the message back to the model, exactly
    like the product's own generate -> guard -> explain loop.
    """

    conn = _connect(db_path)
    try:
        _with_timeout(conn, timeout_seconds)
        conn.execute(f"EXPLAIN QUERY PLAN {sql}")
    finally:
        conn.close()


def run_readonly(
    db_path: Path,
    sql: str,
    row_cap: int | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> tuple[list[str], list[tuple], bool]:
    """Execute ``sql`` and return ``(columns, rows, truncated)``.

    ``truncated`` is true when the result had more than ``row_cap`` rows and
    was cut off — same contract as ``app.export.runner.run_readonly``.
    """

    conn = _connect(db_path)
    try:
        _with_timeout(conn, timeout_seconds)
        cursor = conn.execute(sql)
        columns = [d[0] for d in (cursor.description or [])]
        if row_cap is None:
            rows = cursor.fetchall()
            truncated = False
        else:
            rows = cursor.fetchmany(row_cap + 1)
            truncated = len(rows) > row_cap
            rows = rows[:row_cap]
        return columns, rows, truncated
    except sqlite3.OperationalError as exc:
        if "interrupted" in str(exc).lower():
            raise QueryTimeout(f"query exceeded {timeout_seconds:.0f}s") from exc
        raise
    finally:
        conn.close()


def execute_all(
    db_path: Path, sql: str, timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
) -> list[tuple]:
    """Every row of ``sql`` — used only for scoring (comparing full result sets)."""

    _, rows, _ = run_readonly(db_path, sql, row_cap=None, timeout_seconds=timeout_seconds)
    return rows
