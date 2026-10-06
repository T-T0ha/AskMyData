"""The "raw LLM" arm's view of a BIRD database: table and column names, nothing else.

The semantic arm no longer lives here — it is the product's own pipeline run on
the database (see :mod:`evaluation.bird.session_build`), not a reconstruction of
what that pipeline would have sent.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


def _connect(db_path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)


def _table_names(conn: sqlite3.Connection) -> list[str]:
    return [
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
    ]


def allowed_tables(db_path: Path) -> set[str]:
    conn = _connect(db_path)
    try:
        return {name.lower() for name in _table_names(conn)}
    finally:
        conn.close()


def bare_schema(db_path: Path) -> list[dict[str, Any]]:
    """Table and column names only — the raw arm's entire view of the database.

    No types, no keys, no descriptions, no sample values.
    """

    conn = _connect(db_path)
    try:
        out = []
        for table in _table_names(conn):
            columns = [r[1] for r in conn.execute(f'PRAGMA table_info("{table}")').fetchall()]
            out.append({"table": table, "columns": [{"name": c} for c in columns]})
        return out
    finally:
        conn.close()
