"""Legal, stable, safe SQL identifiers from spreadsheet headers.

A column called ``Total Amount (৳)`` cannot be written into a ``CREATE TABLE``
as it stands, and a sheet called ``"; DROP TABLE users; --`` must not be
written into one at all.  This module answers both, and it is worth being
explicit about which risk each half addresses, because they are different:

* **Legality and stability** are what the sanitiser is for.  PostgreSQL folds
  unquoted identifiers to lower case, stops at 63 bytes, and cannot hold a
  space or a currency sign without quoting.  A user who then types
  ``SELECT total_amount FROM sales`` by hand should find the column where they
  expect it, which means the name has to be predictable, not merely valid.
* **Safety** is *not* delegated to the sanitiser.  Every identifier this
  package emits goes into a SQLAlchemy ``Table``/``Column`` object, which
  quotes it, rather than into a formatted SQL string.  The sanitiser is a
  second line: even if some future caller does format a name into text, what
  it gets has no quote, semicolon, backslash or space in it to break out with.

Nothing is lost by renaming — the original header is kept in ``sem_metadata``
beside the new name, and the export report lists every rename it made.
"""

from __future__ import annotations

import re
import unicodedata

#: PostgreSQL truncates identifiers at 63 *bytes* (NAMEDATALEN - 1).  Since
#: every character that survives sanitisation is ASCII, bytes and characters
#: are the same thing here.
MAX_IDENTIFIER_LENGTH = 63

#: Everything that is not an unaccented letter, a digit or an underscore.
_ILLEGAL = re.compile(r"[^a-z0-9_]+")
_UNDERSCORES = re.compile(r"_{2,}")

#: Reserved words that cannot be a bare identifier in PostgreSQL.  Not the
#: complete list — the complete list is 400 entries and most of them are legal
#: as column names anyway.  These are the ones a business spreadsheet actually
#: produces: a column headed "Order", "Group", "Table", "Check", "Default".
RESERVED_WORDS: frozenset[str] = frozenset(
    {
        "all", "analyse", "analyze", "and", "any", "array", "as", "asc",
        "authorization", "between", "binary", "both", "case", "cast", "check",
        "collate", "column", "constraint", "create", "cross", "current_date",
        "current_role", "current_time", "current_timestamp", "current_user",
        "default", "deferrable", "desc", "distinct", "do", "else", "end",
        "except", "false", "for", "foreign", "freeze", "from", "full", "grant",
        "group", "having", "ilike", "in", "initially", "inner", "intersect",
        "into", "is", "isnull", "join", "leading", "left", "like", "limit",
        "localtime", "localtimestamp", "natural", "not", "notnull", "null",
        "offset", "on", "only", "or", "order", "outer", "overlaps", "placing",
        "primary", "references", "returning", "right", "select", "session_user",
        "similar", "some", "symmetric", "table", "then", "to", "trailing",
        "true", "union", "unique", "user", "using", "verbose", "when", "where",
        "with",
    }
)

#: Appended to a reserved word rather than quoting it.  ``order`` becomes
#: ``order_col``, which needs no quotes anywhere and reads the same.
_RESERVED_SUFFIX = "_col"

#: Prefix for a name that would otherwise start with a digit.  ``2024_total``
#: is not an identifier; ``c_2024_total`` is.
_DIGIT_PREFIX = "c_"


def sanitize_identifier(raw: object, fallback: str = "column") -> str:
    """One header to one legal PostgreSQL identifier.

    Deterministic and idempotent: sanitising an already-sanitised name returns
    it unchanged, so re-exporting a session does not rename its columns a
    second time.
    """

    text = "" if raw is None else str(raw)
    # Decompose accents and drop the combining marks: "Prénom" → "prenom"
    # rather than "prnom".  Characters with no ASCII form at all (Bengali,
    # CJK) have none to keep, and fall through to the fallback below.
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = _ILLEGAL.sub("_", text.strip().lower())
    text = _UNDERSCORES.sub("_", text).strip("_")

    if not text:
        return fallback
    if text[0].isdigit():
        text = _DIGIT_PREFIX + text
    if text in RESERVED_WORDS:
        text += _RESERVED_SUFFIX
    return text[:MAX_IDENTIFIER_LENGTH].rstrip("_") or fallback


def unique_identifiers(
    names: list[object] | tuple[object, ...],
    fallback: str = "column",
    reserved: frozenset[str] | set[str] | None = None,
) -> dict[str, str]:
    """Map every original name to a distinct identifier, in input order.

    Two different headers can sanitise to the same thing — ``Total (BDT)`` and
    ``Total $`` both reduce to ``total`` — and two columns of one table cannot
    share a name.  Later collisions get a numeric suffix, with the base
    shortened as needed so the result still fits in 63 characters.

    ``reserved`` names are treated as already taken; it exists so a table's
    columns cannot collide with a name the exporter adds itself, such as a
    synthetic ``row_id``.

    Returns ``{original: identifier}``.  Duplicate originals collapse to one
    entry, because they are the same column asked about twice.
    """

    taken: set[str] = set(reserved or ())
    mapping: dict[str, str] = {}

    for raw in names:
        original = "" if raw is None else str(raw)
        if original in mapping:
            continue
        candidate = sanitize_identifier(original, fallback=fallback)
        if candidate in taken:
            candidate = _disambiguate(candidate, taken)
        taken.add(candidate)
        mapping[original] = candidate
    return mapping


def _disambiguate(base: str, taken: set[str]) -> str:
    for suffix in range(2, 1000):
        tail = f"_{suffix}"
        candidate = base[: MAX_IDENTIFIER_LENGTH - len(tail)].rstrip("_") + tail
        if candidate not in taken:
            return candidate
    # A thousand columns sanitising to one name is not a data set, but the
    # loop above must not be able to fall off the end into a duplicate name.
    raise ValueError(f"cannot find a distinct identifier for {base!r}")


#: A dataset schema is ``ds_`` plus the session's 32-character hex id, and this
#: is the only shape the exporter will ever create or drop.  Checking the name
#: against this before a ``DROP SCHEMA`` is what keeps a bug — or a crafted
#: session id — from reaching ``public``.
SCHEMA_PREFIX = "ds_"
_SCHEMA_PATTERN = re.compile(r"^ds_[0-9a-f]{6,64}$")


def schema_name(session_id: str) -> str:
    """The PostgreSQL schema one session's exported tables live in."""

    cleaned = re.sub(r"[^0-9a-f]", "", str(session_id).lower())
    if len(cleaned) < 6:
        raise ValueError(f"{session_id!r} is not a usable session id")
    name = f"{SCHEMA_PREFIX}{cleaned[: MAX_IDENTIFIER_LENGTH - len(SCHEMA_PREFIX)]}"
    if not is_managed_schema(name):
        raise ValueError(f"refusing to use schema name {name!r}")
    return name


def is_managed_schema(name: str) -> bool:
    """True only for a schema this package created and may therefore drop."""

    return bool(_SCHEMA_PATTERN.fullmatch(name or ""))
