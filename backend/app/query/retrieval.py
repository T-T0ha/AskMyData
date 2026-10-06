"""Table and column retrieval for one natural-language question (§Phase 5).

Two embeddings already exist for exactly this purpose. ``SheetRecord.table_embedding``
(one sentence per table, composed in :mod:`app.semantics.table_summary`) narrows a
question to the handful of tables it is actually about — "intent-based
pre-selection". ``sem_metadata.embedding`` (built in :mod:`app.export.metadata`) then
ranks the columns of those tables against the question's wording, mirroring
claude.md's retrieval step: ``ORDER BY embedding <-> $1 LIMIT 15``.

Three refinements go beyond sending the literal top-K as-is, because top-K by
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
* **A column a rule names is never dropped either.** "Followed up at the
  outpatient clinic refers to ``Admission = '-'``" is a rule about a column
  whose description says nothing of clinics.  The business rules retrieved for
  a question are read for the columns they call by name
  (:func:`columns_named_in`), and those are kept the way a key is — the model
  is told to use only the columns it was given, so a column left out is a rule
  it cannot follow.

Nothing here opens a connection: every function takes rows and vectors the
caller already fetched, so it is exercisable with plain Python objects.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Collection, Mapping, Sequence

import numpy as np

#: Ceiling on how many tables are sent, however many the dataset has. See
#: ``Settings.query_max_tables``; passed in explicitly rather than read here so
#: this module has no dependency on app.core.config and stays a pure function
#: of its arguments.
DEFAULT_MAX_TABLES = 8

#: Ceiling on ranked (non-key) columns sent. See ``Settings.query_max_ranked_columns``.
DEFAULT_MAX_RANKED_COLUMNS = 100

#: How much a question's own words, found in a column's name or description, add
#: to its embedding similarity.  Embeddings blur exact wording — "writing score"
#: against a column called ``avgscrwrite`` — and a literal match is the cheapest
#: and most trustworthy signal there is.  Large enough to lift a literal match
#: over a vaguely similar column, small enough that it never outvotes meaning.
LEXICAL_WEIGHT = 0.30

_STOPWORDS = frozenset(
    """a an the of in on at to for from by with and or but not is are was were be been being
    what which who whom whose when where why how many much most least more less than then
    that this these those there their they them his her its our your you we i me my do does did
    have has had can could would should will shall may might per each every all any some
    list show give find tell please between among over under about""".split()
)
_WORD_RE = re.compile(r"[a-z0-9]+")


def _stem(word: str) -> str:
    for suffix in ("ing", "ed", "es", "s", "ly"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


def question_terms(question: str) -> list[str]:
    """The words of a question worth looking for: no filler, lightly stemmed."""

    terms: list[str] = []
    for word in _WORD_RE.findall(question.lower()):
        if len(word) < 3 or word in _STOPWORDS:
            continue
        stem = _stem(word)
        if stem not in terms:
            terms.append(stem)
    return terms


def _row_text(row: Mapping[str, Any]) -> str:
    return f"{row.get('column_name', '')} {row.get('semantic_description', '')}".lower()


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


def _lexical_scores(terms: Sequence[str], texts: Sequence[str]) -> list[float]:
    """Share of the question's terms each text contains, rarer terms counting more.

    A term that appears in most columns ("score" in a table of scores) says
    little about which one is meant; the one that appears in a few ("write")
    says a great deal.  Weighting by inverse document frequency over the
    candidates themselves keeps the signal on the distinguishing word.
    """

    if not terms or not texts:
        return [0.0] * len(texts)
    total = len(texts)
    weights = {
        term: math.log(1.0 + total / (1 + sum(1 for text in texts if term in text)))
        for term in terms
    }
    denominator = sum(weights.values()) or 1.0
    return [sum(weights[t] for t in terms if t in text) / denominator for text in texts]


#: Shortest column name a rule can be said to call by name.  ``id`` and ``no`` are
#: in every sentence; a name this short is kept by being a key, or not at all.
MIN_NAMED_COLUMN_LENGTH = 3

#: Longest run of words read as one name ("Free Meal Count (K-12)" is five).  Source
#: headers run long; a run is only ever compared with a column's own name, so a
#: generous limit costs a few set entries and no false matches.
_MAX_NAME_WORDS = 10

#: Letters and digits of any script — ``\w`` without the underscore, so that
#: ``total_amount``, ``Total Amount`` and ``T-BIL`` all split into their words.
_NAME_WORD = re.compile(r"[^\W_]+")


def columns_named_in(
    texts: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
    tables: Collection[str],
) -> set[str]:
    """Lower-cased names of the columns, in ``tables``, that any of ``texts`` calls by name.

    A business rule is written in terms of the data — "only orders with status
    'shipped' count as revenue", "followed up at the outpatient clinic refers to
    ``Admission = '-'``" — so a column it names is a column the question needs,
    however little that column's description reads like the question.  Ranking by
    similarity to the question alone can leave it out of the top-K, and the model
    is told to use only the columns it is given: a column that was not sent
    cannot be used, whatever the rule says about it.

    A column is named when its name, as exported (``total_amount``) or as the
    source spelled it (``Total Amount``, ``T-BIL``), appears in the text, ignoring
    case and punctuation and whether the words are run together or apart
    (``fastestLapSpeed`` / "fastest lap speed").  Names shorter than
    :data:`MIN_NAMED_COLUMN_LENGTH` are never matched, and a name inside a longer
    word is not a mention of it ("sexual" does not name ``sex``).
    """

    runs: set[str] = set()
    for text in texts:
        words = _NAME_WORD.findall(str(text).casefold())
        for start in range(len(words)):
            run = ""
            for word in words[start : start + _MAX_NAME_WORDS]:
                run += word
                runs.add(run)
    if not runs:
        return set()

    named: set[str] = set()
    for row in rows:
        if str(row.get("table_name")) not in tables:
            continue
        for spelling in (row.get("column_name"), row.get("source_column")):
            key = "".join(_NAME_WORD.findall(str(spelling or "").casefold()))
            if len(key) >= MIN_NAMED_COLUMN_LENGTH and key in runs:
                named.add(str(row.get("column_name")).lower())
                break
    return named


def select_columns(
    question_vector: Sequence[float],
    rows: Sequence[Mapping[str, Any]],
    selected_tables: Sequence[str],
    max_ranked: int = DEFAULT_MAX_RANKED_COLUMNS,
    question: str | None = None,
    include_columns: Collection[str] = (),
) -> list[dict[str, Any]]:
    """``sem_metadata`` rows for the selected tables: every key column, plus the
    ``max_ranked`` best-scoring remaining columns across all of them combined.

    A column's score is its embedding similarity to the question, plus a bonus
    for the question's own words appearing in its name or description when the
    ``question`` text is given.

    ``include_columns`` names (lower-cased) are kept regardless of score: the
    columns a verified example's SQL already uses, which would otherwise be
    left out of the very schema the model is asked to follow that example in,
    and the columns a business rule names (:func:`columns_named_in`).

    Grouped by table in ``selected_tables`` order on the way out, so the schema
    reads as one block per table rather than columns interleaved by score —
    a model reasons about a schema, not a leaderboard.
    """

    by_table: dict[str, list[dict[str, Any]]] = {name: [] for name in selected_tables}
    for row in rows:
        table = str(row.get("table_name"))
        if table in by_table:
            by_table[table].append(dict(row))

    forced = {name.lower() for name in include_columns}
    kept: list[dict[str, Any]] = []
    unranked: list[dict[str, Any]] = []
    for table in selected_tables:
        for row in by_table[table]:
            if (
                row.get("is_primary_key")
                or row.get("is_foreign_key")
                or str(row.get("column_name", "")).lower() in forced
            ):
                kept.append(row)
            else:
                unranked.append(row)

    lexical = (
        _lexical_scores(question_terms(question), [_row_text(row) for row in unranked])
        if question
        else [0.0] * len(unranked)
    )
    candidates = [
        (_cosine(question_vector, row.get("embedding")) + LEXICAL_WEIGHT * bonus, row)
        for row, bonus in zip(unranked, lexical)
    ]
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


def normalize_question(question: str) -> str:
    """A question reduced to its words — case, punctuation and spacing dropped.

    Two spellings of the same question compare equal; two different questions
    do not (digits are kept, so "top 5" and "top 10" stay apart).
    """

    return " ".join(_WORD_RE.findall((question or "").lower()))


def rank_examples(
    question_vector: Sequence[float], examples: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Every verified example with its similarity to the question, best first."""

    ranked = [
        {
            **{key: value for key, value in row.items() if key != "embedding"},
            "similarity": _cosine(question_vector, row.get("embedding")),
        }
        for row in examples
        if row.get("question") and row.get("sql")
    ]
    ranked.sort(key=lambda row: row["similarity"], reverse=True)
    return ranked


def examples_worth_showing(
    ranked: Sequence[Mapping[str, Any]],
    max_examples: int = DEFAULT_MAX_EXAMPLES,
    min_similarity: float = 0.0,
) -> list[dict[str, Any]]:
    """The best-ranked examples that are also *similar* to the question.

    Ranking only puts the examples in order.  With three saved, the three
    "nearest" are three questions about something else, and a model told to
    follow an example's pattern when the question resembles one will weigh them
    anyway — at a cost in tokens and in distraction, for no pattern to follow.
    An example below ``min_similarity`` is left out however few others there are.
    ``ranked`` is :func:`rank_examples`' output, best first.
    """

    return [dict(row) for row in ranked[:max_examples] if row["similarity"] >= min_similarity]


def retrieve_examples(
    question_vector: Sequence[float],
    examples: Sequence[Mapping[str, Any]],
    max_examples: int = DEFAULT_MAX_EXAMPLES,
    min_similarity: float = 0.0,
) -> list[dict[str, str]]:
    """Verified question/SQL pairs for this dataset, ranked by similarity.

    Only ever called with already-``verified`` rows — an unverified guess has
    no business sitting beside real evidence in the same prompt.  Those less
    similar to the question than ``min_similarity`` are not returned
    (:func:`examples_worth_showing`).
    """

    return [
        {"question": str(row["question"]), "sql": str(row["sql"])}
        for row in examples_worth_showing(
            rank_examples(question_vector, examples), max_examples, min_similarity
        )
    ]
