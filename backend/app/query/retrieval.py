"""Table and column retrieval for one natural-language question (§Phase 5).

Two embeddings already exist for exactly this purpose. ``SheetRecord.table_embedding``
(one sentence per table, composed in :mod:`app.semantics.table_summary`) narrows a
question to the handful of tables it is actually about — "intent-based
pre-selection". ``sem_metadata.embedding`` (built in :mod:`app.export.metadata`) then
ranks the columns of those tables against the question's wording, mirroring
claude.md's retrieval step: ``ORDER BY embedding <-> $1 LIMIT 15``.

Two refinements go beyond sending the literal top-K as-is, because top-K by
sentence similarity silently drops what a *query* needs rather than what a
*sentence* needs:

* **A confirmed reference pulls its other table in.** "How much did Acme
  order?" scores ``customers`` — it names one — but may not score ``orders``
  highly by wording alone; a query answering it needs both. One hop of
  confirmed foreign keys is added after ranking, in either direction.
* **A key column is never dropped for scoring low.** ``customer_id`` rarely
  resembles the words of a business question, but every join depends on it
  being in the schema handed to the model. Primary and foreign key columns of
  a selected table are kept regardless of similarity score, and only the
  remaining columns compete for the top-K slots.

Nothing here opens a connection: every function takes rows and vectors the
caller already fetched, so it is exercisable with plain Python objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

#: Ceiling on how many tables are sent, however many the dataset has. See
#: ``Settings.query_max_tables``; passed in explicitly rather than read here so
#: this module has no dependency on app.core.config and stays a pure function
#: of its arguments.
DEFAULT_MAX_TABLES = 8

#: Ceiling on ranked (non-key) columns sent. See ``Settings.query_max_ranked_columns``.
DEFAULT_MAX_RANKED_COLUMNS = 15


def _cosine(a: Sequence[float] | None, b: Sequence[float] | None) -> float:
    """Cosine similarity, or 0.0 for anything that is not two real vectors.

    Takes ``a``/``b`` as arrays or plain sequences — ``question_vector`` is
    whatever the embedder returned (a numpy array), ``b`` is whatever a JSON or
    ``vector(n)`` column deserialised to (a list) or ``None`` for an unranked
    table.  A bare ``if not a`` here would raise on a numpy array with more
    than one element, since its truthiness is ambiguous rather than falsy.
    """

    if a is None or b is None:
        return 0.0
    va = np.asarray(a, dtype="float32")
    vb = np.asarray(b, dtype="float32")
    if va.size == 0 or vb.size == 0:
        return 0.0
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


@dataclass(slots=True)
class TableScore:
    table: str
    score: float
    #: False when the table has no embedding to score with — an export that
    #: predates table embeddings, or one built with the embedder unavailable.
    ranked: bool


def rank_tables(
    question_vector: Sequence[float], tables: Mapping[str, Sequence[float] | None]
) -> list[TableScore]:
    """Every candidate table, ranked by how closely its sentence reads to the question."""

    scored = [
        TableScore(table=name, score=_cosine(question_vector, vector), ranked=vector is not None)
        for name, vector in tables.items()
    ]
    scored.sort(key=lambda item: item.score, reverse=True)
    return scored


def select_tables(
    scored: Sequence[TableScore], max_tables: int = DEFAULT_MAX_TABLES
) -> list[str]:
    """The tables to send, in relevance order.

    Falls back to *every* table, unranked, the moment even one of them cannot
    be ranked. An arbitrary, possibly-wrong subset is a worse failure than a
    longer prompt: a business dataset that hits this path is small enough that
    "everything" costs little, and staying degradable is worth more here than
    staying inside ``max_tables``.
    """

    if any(not item.ranked for item in scored):
        return [item.table for item in scored]
    return [item.table for item in scored[:max_tables]]


def expand_by_references(
    selected: Sequence[str], rows: Sequence[Mapping[str, Any]]
) -> list[str]:
    """Add the table on the other end of any confirmed reference, one hop only."""

    names = list(dict.fromkeys(selected))
    known = set(names)
    for row in rows:
        if not row.get("is_foreign_key"):
            continue
        target = row.get("references_table")
        if not target:
            continue
        table = str(row.get("table_name"))
        target = str(target)
        if table in known and target not in known:
            names.append(target)
            known.add(target)
        elif target in known and table not in known:
            names.append(table)
            known.add(table)
    return names


def select_columns(
    question_vector: Sequence[float],
    rows: Sequence[Mapping[str, Any]],
    selected_tables: Sequence[str],
    max_ranked: int = DEFAULT_MAX_RANKED_COLUMNS,
) -> list[dict[str, Any]]:
    """``sem_metadata`` rows for the selected tables: every key column, plus the
    ``max_ranked`` best-scoring remaining columns across all of them combined.

    Grouped by table in ``selected_tables`` order on the way out, so the schema
    reads as one block per table rather than columns interleaved by score —
    a model reasons about a schema, not a leaderboard.
    """

    by_table: dict[str, list[dict[str, Any]]] = {name: [] for name in selected_tables}
    for row in rows:
        table = str(row.get("table_name"))
        if table in by_table:
            by_table[table].append(dict(row))

    kept: list[dict[str, Any]] = []
    candidates: list[tuple[float, dict[str, Any]]] = []
    for table in selected_tables:
        for row in by_table[table]:
            if row.get("is_primary_key") or row.get("is_foreign_key"):
                kept.append(row)
            else:
                candidates.append((_cosine(question_vector, row.get("embedding")), row))

    candidates.sort(key=lambda item: item[0], reverse=True)
    kept.extend(row for _, row in candidates[:max_ranked])

    order = {name: index for index, name in enumerate(selected_tables)}
    kept.sort(key=lambda row: order.get(str(row.get("table_name")), len(order)))
    return kept


def schema_context(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The retrieved columns, nested under their table — the shape the SQL prompt reads.

    Never carries ``embedding``: 384 floats per column is pure token cost to a
    prompt that only needs the eleven fields below.
    """

    tables: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for row in rows:
        table = str(row.get("table_name"))
        if table not in tables:
            tables[table] = {
                "table": table,
                "table_type": row.get("table_type") or "unknown",
                "columns": [],
            }
            order.append(table)
        references = None
        if row.get("is_foreign_key") and row.get("references_table"):
            references = f"{row.get('references_table')}.{row.get('references_column')}"
        tables[table]["columns"].append(
            {
                "name": row.get("column_name"),
                "type": row.get("sql_type"),
                "taxonomy": row.get("taxonomy_label"),
                "is_primary_key": bool(row.get("is_primary_key")),
                "is_foreign_key": bool(row.get("is_foreign_key")),
                "references": references,
                "is_additive": bool(row.get("is_additive")),
                "null_ratio": round(float(row.get("null_ratio") or 0.0), 4),
                "sample_values": row.get("sample_values") or [],
            }
        )
    return [tables[name] for name in order]


#: Ceiling on retrieved business rules. See ``Settings.query_max_business_rules``.
DEFAULT_MAX_BUSINESS_RULES = 5

#: Ceiling on retrieved verified examples. See ``Settings.query_max_examples``.
DEFAULT_MAX_EXAMPLES = 3


def retrieve_business_rules(
    question_vector: Sequence[float],
    rules: Sequence[Mapping[str, Any]],
    max_rules: int = DEFAULT_MAX_BUSINESS_RULES,
) -> list[str]:
    """The user's own recorded rules, ranked by relevance to the question.

    The third retrieval index alongside table and column embeddings (§Phase 5)
    — a rule with no embedding yet (written before the embedder was available)
    scores 0.0 rather than being dropped, since a business rule is short enough
    that sending it costs little even when it cannot be ranked.
    """

    scored = sorted(
        rules,
        key=lambda row: _cosine(question_vector, row.get("embedding")),
        reverse=True,
    )
    return [str(row.get("rule_text")) for row in scored[:max_rules] if row.get("rule_text")]


def retrieve_examples(
    question_vector: Sequence[float],
    examples: Sequence[Mapping[str, Any]],
    max_examples: int = DEFAULT_MAX_EXAMPLES,
) -> list[dict[str, str]]:
    """Verified question/SQL pairs for this dataset, ranked by similarity.

    Only ever called with already-``verified`` rows — an unverified guess has
    no business sitting beside real evidence in the same prompt.
    """

    scored = sorted(
        examples,
        key=lambda row: _cosine(question_vector, row.get("embedding")),
        reverse=True,
    )
    return [
        {"question": str(row.get("question")), "sql": str(row.get("sql"))}
        for row in scored[:max_examples]
        if row.get("question") and row.get("sql")
    ]
