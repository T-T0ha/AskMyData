"""One sentence per table — the first thing Phase 5 reads about a dataset.

Phase 1 describes columns and Phase 3's profiler decides what a table *is*.
Neither produces the thing the query pipeline actually needs first: a sentence
short enough to send for every table at once, specific enough to pick the two
tables a question is about out of forty.  The intent-classification prompt
(§5.3) sends table names and these sentences and nothing else, and the same
sentence is embedded so that a question can reach a table by meaning rather
than by name.

The sentence is **composed, not measured**.  Every clause restates something an
earlier phase established — the table type, the confirmed key, the confirmed
references, which columns can be summed — so it can be rebuilt from stored
state whenever any of that changes, and it never asserts anything the user has
not already seen asserted somewhere else.  That is also why a description going
stale is impossible by construction: it is derived, so it is recomputed.

Claude may rewrite it into better prose when a key is configured
(:func:`~app.semantics.claude_client.ClaudeClient.describe_tables`), which is
stored separately.  The composed sentence stays as the fallback and as the
thing to compare against when deciding whether the model's version is still
about the same table.

Nothing here touches a database, a data row, or a model.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from app.core.schemas import RelationshipStatus, RelationshipType, TableType
from app.semantics.embeddings import get_embedder, humanize
from app.semantics.table_profile import TableProfile

logger = logging.getLogger(__name__)

#: Column names listed in one clause before it stops naming them.  Past this a
#: reader has stopped reading and an embedding has stopped discriminating.
MAX_LISTED = 6

#: Tables named in the "references" and "referenced by" clauses.
MAX_RELATED = 4


def _join(names: Sequence[str], limit: int) -> str:
    """``"a, b and c"``, truncated with a count rather than an ellipsis."""

    shown = [str(name) for name in names[:limit]]
    extra = len(names) - len(shown)
    if extra > 0:
        shown.append(f"{extra} more")
    if len(shown) == 1:
        return shown[0]
    return ", ".join(shown[:-1]) + " and " + shown[-1]


def compose(
    table: str,
    profile: TableProfile | None = None,
    columns: Sequence[Mapping[str, Any]] = (),
    primary_key: Sequence[str] = (),
    references: Sequence[tuple[str, str]] = (),
    referenced_by: Sequence[str] = (),
    row_count: int = 0,
    column_count: int = 0,
) -> str:
    """The sentence for one table.

    ``references`` is ``(column, target table)`` for each confirmed outgoing
    reference; ``referenced_by`` is the tables that point back.  Both are given
    rather than looked up, because the caller has already gathered them for
    every table and doing it per table would be quadratic.
    """

    kind = profile.effective_type if profile is not None else TableType.UNKNOWN.value
    if kind == TableType.UNKNOWN.value:
        # "unknown table" describes the analysis, not the table.  A reader — and
        # an embedding — is better served by the neutral word.
        kind = "data"
    column_count = column_count or len(columns)

    parts = [
        f"{table} is a {kind.replace('_', ' ')} table with "
        f"{row_count:,} rows and {column_count} columns."
    ]

    if primary_key:
        parts.append(
            "Each row is identified by "
            f"{_join([humanize(name) for name in primary_key], MAX_LISTED)}."
        )
    else:
        parts.append("It has no primary key, so no other table can point at its rows.")

    if references:
        phrases = [
            f"{target} through {humanize(column)}" for column, target in references
        ]
        parts.append(f"It references {_join(phrases, MAX_RELATED)}.")
    if referenced_by:
        parts.append(f"It is referenced by {_join(list(referenced_by), MAX_RELATED)}.")

    key_names = {str(name) for name in primary_key}
    reference_names = {str(column) for column, _ in references}
    measures = [
        humanize(str(column.get("name", "")))
        for column in columns
        if column.get("is_additive") and str(column.get("name", "")) not in key_names
    ]
    if measures:
        parts.append(f"It measures {_join(measures, MAX_LISTED)}.")

    described = [
        humanize(str(column.get("name", "")))
        for column in columns
        if not column.get("is_additive")
        and str(column.get("name", "")) not in key_names
        and str(column.get("name", "")) not in reference_names
    ]
    if described:
        parts.append(f"It also records {_join(described, MAX_LISTED)}.")

    return " ".join(parts)


def relationship_map(
    relationships: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, list[tuple[str, str]]], dict[str, list[str]]]:
    """Confirmed foreign keys, bucketed by the table each end belongs to.

    Only confirmed ones: a description is read as a statement of fact, and a
    proposal the user has not looked at yet is not one.
    """

    outgoing: dict[str, list[tuple[str, str]]] = {}
    incoming: dict[str, list[str]] = {}
    for relation in relationships:
        if relation.get("rel_type") != RelationshipType.FOREIGN_KEY.value:
            continue
        if relation.get("status") != RelationshipStatus.CONFIRMED.value:
            continue
        child = str(relation.get("from_table", ""))
        parent = str(relation.get("to_table", ""))
        column = str(relation.get("from_column", ""))
        if not child or not parent:
            continue
        outgoing.setdefault(child, []).append((column, parent))
        if parent != child and child not in incoming.setdefault(parent, []):
            incoming[parent].append(child)
    return outgoing, incoming


def describe_tables(
    profiles: Mapping[str, TableProfile],
    columns: Mapping[str, Sequence[Mapping[str, Any]]],
    key_analyses: Mapping[str, Mapping[str, Any]],
    relationships: Iterable[Mapping[str, Any]],
    counts: Mapping[str, tuple[int, int]],
) -> dict[str, str]:
    """A sentence for every table named in ``counts``.

    ``counts`` decides the membership rather than ``profiles`` so that a table
    too small to profile still gets described — it is still a table the user
    can ask about.
    """

    outgoing, incoming = relationship_map(relationships)
    descriptions: dict[str, str] = {}
    for table, (row_count, column_count) in counts.items():
        key_analysis = key_analyses.get(table) or {}
        descriptions[table] = compose(
            table,
            profile=profiles.get(table),
            columns=columns.get(table) or (),
            primary_key=[str(name) for name in key_analysis.get("primary_key") or []],
            references=outgoing.get(table, []),
            referenced_by=incoming.get(table, []),
            row_count=row_count,
            column_count=column_count,
        )
    return descriptions


def embed(descriptions: Sequence[str]) -> list[list[float]] | None:
    """Embed table descriptions, or ``None`` when no model could be loaded.

    A missing vector costs Phase 5 its table pre-selection, which then falls
    back to sending every table's sentence — slower and dearer, still correct.
    Failing the caller over that would be the wrong trade.
    """

    if not descriptions:
        return []
    try:
        vectors = np.asarray(get_embedder().encode(list(descriptions)), dtype="float32")
    except Exception as exc:  # pragma: no cover - depends on the environment
        logger.warning("table descriptions were not embedded: %s", exc)
        return None
    if vectors.ndim != 2 or len(vectors) != len(descriptions):
        logger.warning(
            "embedder returned %s vectors for %s descriptions", len(vectors), len(descriptions)
        )
        return None
    return [[float(value) for value in vector] for vector in vectors]


def payload_for_claude(
    table: str,
    profile: TableProfile | None,
    columns: Sequence[Mapping[str, Any]],
    primary_key: Sequence[str],
    references: Sequence[tuple[str, str]],
    referenced_by: Sequence[str],
    row_count: int,
) -> dict[str, Any]:
    """What the model is allowed to see about one table.

    Schema and structure only: names, labels, counts, keys.  No sample values
    and no rows — a table description does not need them, and the privacy
    boundary in §5.3 is easier to keep when each prompt asks for the minimum
    rather than the maximum it could use.
    """

    return {
        "name": table,
        "table_type": profile.effective_type if profile is not None else "unknown",
        "labels": profile.effective_labels if profile is not None else [],
        "row_count": row_count,
        "primary_key": list(primary_key),
        "references": [
            {"column": column, "table": target} for column, target in references
        ],
        "referenced_by": list(referenced_by),
        "columns": [
            {
                "name": str(column.get("name", "")),
                "taxonomy": str(column.get("effective_label", "unknown")),
                "is_additive": bool(column.get("is_additive")),
            }
            for column in columns
        ],
    }
