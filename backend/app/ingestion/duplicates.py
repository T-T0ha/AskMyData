"""Phase 0 — tables that are exact copies of one another.

Uploading the same file twice, or two files that both carry the same sheet,
leaves a session holding ``frpm`` and ``frpm_2``.  Nothing is gained by
analysing both, and a good deal is lost: every column is described twice, every
key matches its own copy, and the relationship stage proposes a link between a
table and its twin for each of them.

Only a **byte-for-byte equal** table counts: same columns in the same order and
the same value in every cell.  Two tables that merely look alike — a Q1 and a Q2
sheet, say — differ somewhere and are left alone; that is a different question
(one table split in two) and it belongs to the cleaning plan.
"""

from __future__ import annotations

import hashlib
import re
from typing import Mapping

import pandas as pd

#: The reason recorded on a skipped copy.  Public because the API and the
#: triage screen recognise a duplicate by it rather than by a separate column.
DUPLICATE_PREFIX = "Identical copy of"

_SUFFIX_RE = re.compile(r"^(?P<base>.*?)(?:_(?P<n>\d+))?$")


def table_fingerprint(df: pd.DataFrame) -> str:
    """A digest of a table's column names and every cell, in row order."""

    digest = hashlib.sha256()
    digest.update(repr([str(c) for c in df.columns]).encode())
    digest.update(str(len(df)).encode())
    try:
        row_hashes = pd.util.hash_pandas_object(df, index=False).to_numpy()
    except TypeError:
        # A cell holding a list or dict cannot be hashed; its text can.
        row_hashes = pd.util.hash_pandas_object(df.astype(str), index=False).to_numpy()
    digest.update(row_hashes.tobytes())
    return digest.hexdigest()


def _original_first(name: str) -> tuple[int, str]:
    """Sort key putting ``frpm`` before ``frpm_2`` before ``frpm_3``.

    ``_persist_results`` names a colliding table ``<name>_<n>``, so the number
    is the order the tables arrived in.
    """

    match = _SUFFIX_RE.match(name)
    number = int(match.group("n")) if match and match.group("n") else 1
    return number, name


def find_duplicate_tables(tables: Mapping[str, pd.DataFrame]) -> dict[str, str]:
    """``{copy: original}`` for every table that repeats an earlier one.

    The original is the one that arrived first (no suffix, then the lowest
    ``_n``).  Tables are only fingerprinted when another table has the same
    shape, so a session with no copies costs a dictionary lookup.
    """

    by_shape: dict[tuple[tuple[str, ...], int], list[str]] = {}
    for name, df in tables.items():
        shape = (tuple(str(c) for c in df.columns), int(len(df)))
        by_shape.setdefault(shape, []).append(name)

    duplicates: dict[str, str] = {}
    for names in by_shape.values():
        if len(names) < 2:
            continue
        seen: dict[str, str] = {}
        for name in sorted(names, key=_original_first):
            fingerprint = table_fingerprint(tables[name])
            if fingerprint in seen:
                duplicates[name] = seen[fingerprint]
            else:
                seen[fingerprint] = name
    return duplicates


def duplicate_reason(original: str) -> str:
    return (
        f"{DUPLICATE_PREFIX} '{original}' — left out, so the same data is not analysed twice."
    )


def is_duplicate_reason(reason: str | None) -> bool:
    return bool(reason) and str(reason).startswith(DUPLICATE_PREFIX)
