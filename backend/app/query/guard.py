"""The SELECT-only guard for Claude-generated SQL (§Phase 5).

Claude proposes SQL against the exported semantic layer; nothing it writes
reaches a live connection until this module has re-parsed it and confirmed it
is exactly one read against tables the retrieval step actually offered it.
This is independent of the system prompt asking nicely — a model that hedges,
misreads an instruction, or is handed an adversarial question is still bound
by a parser, not by its own claim to have behaved.

Three checks, each closing a different door:

* **Exactly one statement, and it is a SELECT** (a bare query, a set operation
  over two SELECTs, or a ``WITH`` block ending in one). Not a semicolon away
  from an ``INSERT``, a ``DROP``, or a second, unreviewed statement riding
  along with the first.
* **No table outside ``allowed_tables``.** The retrieval step decided which
  tables the question needs; a name the model invents, or a name it read
  correctly but was never offered, is refused rather than guessed at. A CTE's
  own alias is not a table and is excluded from this check — except where the
  CTE's *own body* references that same name: ordinary (non-``RECURSIVE``)
  SQL cannot see a CTE's name inside its own definition, so that particular
  occurrence can only resolve to a real table if one exists, and is checked
  accordingly. ``WITH RECURSIVE`` is refused outright rather than reasoned
  about, since a legitimate self-reference would otherwise need a real
  recursion check this platform has no use for.
* **No schema or database qualification at all** (``public.orders``,
  ``other_db.orders``). The exported tables are reached by bare name once
  ``search_path`` is restricted to the export's own schema
  (:func:`app.export.runner._set_search_path`); a qualified reference is the
  one way a generated query could still reach outside it, so it is refused
  outright rather than checked against a second allowlist.
"""

from __future__ import annotations

import sqlglot
from sqlglot import exp

#: sqlglot's dialect name for the two backends this platform exports to.
_READ_DIALECT = {"postgresql": "postgres", "sqlite": "sqlite"}

#: Expression types that mean "this touches something other than reading rows".
#: Checked node-by-node, not just at the statement root, so a DDL/DML statement
#: hidden inside a subquery or a CTE is caught the same way a top-level one is.
_FORBIDDEN: tuple[type, ...] = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Drop,
    exp.Create,
    exp.Alter,
    exp.TruncateTable,
    exp.Grant,
    exp.Merge,
    exp.Command,
    exp.Pragma,
)


class QueryRejected(ValueError):
    """The model's SQL was refused before it ever reached a connection."""


def _read_dialect(dialect: str) -> str:
    return _READ_DIALECT.get(dialect, "postgres")


def validate_select_only(
    sql: str, allowed_tables: set[str] | frozenset[str], dialect: str = "postgresql"
) -> str:
    """Parse ``sql``; return it re-rendered, or raise :class:`QueryRejected`.

    ``allowed_tables`` must already be lower-cased — every exported identifier
    is, by construction (:mod:`app.export.naming`), so callers pass the names
    straight from the semantic layer.
    """

    read = _read_dialect(dialect)
    text = (sql or "").strip()
    if text.endswith(";"):
        text = text[:-1].rstrip()
    if not text:
        raise QueryRejected("the model returned no SQL")

    try:
        statements = [s for s in sqlglot.parse(text, read=read) if s is not None]
    except Exception as exc:  # sqlglot raises a family of parser errors
        raise QueryRejected(f"the SQL could not be parsed: {exc}") from exc

    if len(statements) != 1:
        raise QueryRejected(
            f"expected exactly one statement, found {len(statements)} "
            "(no semicolon-separated statements are allowed)"
        )
    statement = statements[0]

    if not isinstance(statement, (exp.Select, exp.Union, exp.Intersect, exp.Except)):
        raise QueryRejected(
            f"only SELECT is allowed; this statement is {type(statement).__name__.upper()}"
        )

    for node in statement.walk():
        candidate = node[0] if isinstance(node, tuple) else node
        if isinstance(candidate, _FORBIDDEN):
            raise QueryRejected(
                f"{type(candidate).__name__} is not allowed inside a query"
            )

    for table in statement.find_all(exp.Table):
        if table.db or table.catalog:
            raise QueryRejected(
                f"{table.sql(dialect=read)!r} may not name a schema or database — "
                "reference tables by their bare name"
            )

    for with_clause in statement.find_all(exp.With):
        if with_clause.args.get("recursive"):
            raise QueryRejected("WITH RECURSIVE is not allowed")

    ctes = list(statement.find_all(exp.CTE))
    cte_names = {cte.alias_or_name.lower() for cte in ctes}
    # A CTE cannot see its own name inside its own (non-recursive) body — a
    # reference there can only resolve to a real table, so those specific
    # occurrences are checked against ``allowed_tables`` rather than waved
    # through as "just the CTE" the way every other occurrence of the name is.
    self_referencing = {
        id(table)
        for cte in ctes
        for table in cte.this.find_all(exp.Table)
        if table.name.lower() == cte.alias_or_name.lower()
    }

    unknown: set[str] = set()
    for table in statement.find_all(exp.Table):
        name = table.name.lower()
        if id(table) in self_referencing:
            if name not in allowed_tables:
                unknown.add(name)
            continue
        if name in cte_names or name in allowed_tables:
            continue
        unknown.add(name)

    if unknown:
        raise QueryRejected(
            "references table(s) outside the tables offered for this question: "
            + ", ".join(sorted(unknown))
        )

    return statement.sql(dialect=read)
