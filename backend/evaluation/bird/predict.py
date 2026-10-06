"""The raw arm's generate -> guard -> explain -> execute retry loop.

The semantic arm does not use this: it runs ``app.api.services.answer_question``,
which has the same loop (plus retrieval, examples and the rest of Phase 5).  This
is that loop with every Phase 5 addition removed, so the two arms differ in
what the model is *shown*, not in how forgiving the harness is about its output:

1. one call to the shared :class:`~evaluation.bird.llm.EvalModel` — the same
   system prompt, the same client, the same parsing as the semantic arm;
2. the product's own :func:`app.query.guard.validate_select_only`;
3. ``EXPLAIN QUERY PLAN``, then a real, row-capped execution — a failure at
   either step goes back to the model as ``previous_attempt`` for another try,
   exactly as in ``answer_question``.

No ambiguity check, no confidence gate, no retrieval (the raw arm is handed the
whole bare schema), no examples: there is nothing for it to retrieve them from.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.query.guard import QueryRejected, validate_select_only

from . import sqlite_exec

#: The product returns at most this many rows to a caller (``query_row_cap``);
#: execution here only has to prove the query runs.
ROW_CAP = 1000


def generate_with_retry(
    model: Any,
    question: str,
    schema: list[dict[str, Any]],
    db_path: Path,
    allowed_tables: set[str],
    max_attempts: int = 3,
    sql_timeout: float = sqlite_exec.DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Up to ``max_attempts`` SQL attempts for ``question`` against ``schema``.

    Returns ``{"sql", "explanation", "confidence", "attempts", "error"}`` —
    ``sql`` is ``None`` iff every attempt was rejected, failed to run, or the
    model produced nothing usable.
    """

    attempts: list[dict[str, Any]] = []
    prior_sql: str | None = None
    last_error: str | None = None
    sql: str | None = None
    explanation = ""
    confidence: float | None = None

    for attempt_no in range(1, max_attempts + 1):
        generated = model.generate_sql(
            question, "sqlite", schema, prior_sql=prior_sql, prior_error=last_error
        )
        if generated is None:
            last_error = model.last_error or "the model returned no usable answer"
            attempts.append({"attempt": attempt_no, "sql": None, "error": last_error})
            break

        prior_sql = generated["sql"]
        explanation = generated["explanation"]
        confidence = generated["confidence"]
        try:
            validated = validate_select_only(generated["sql"], allowed_tables, "sqlite")
        except QueryRejected as exc:
            last_error = str(exc)
            attempts.append({"attempt": attempt_no, "sql": generated["sql"], "error": last_error})
            continue

        try:
            sqlite_exec.explain(db_path, validated, sql_timeout)
            sqlite_exec.run_readonly(db_path, validated, ROW_CAP, sql_timeout)
        except Exception as exc:  # sqlite3.Error or sqlite_exec.QueryTimeout
            last_error = str(exc)
            attempts.append({"attempt": attempt_no, "sql": validated, "error": last_error})
            continue

        sql = validated
        attempts.append({"attempt": attempt_no, "sql": validated, "error": None})
        break

    return {
        "sql": sql,
        "explanation": explanation if sql else "",
        "confidence": confidence if sql else None,
        "attempts": attempts,
        "error": None if sql else last_error,
    }
