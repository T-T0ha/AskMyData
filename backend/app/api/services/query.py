"""Phase 5 — natural language query.

Conversation history, the business rules and verified examples that form the
third retrieval index, and :func:`answer_question` itself.
"""

from __future__ import annotations

from typing import Any, Sequence

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.api.services.export import export_record
from app.api.services.ingestion import _sheets_by_name
from app.core.config import get_settings
from app.db.models import BusinessRule, IngestionSession, QueryExample, QueryHistoryRecord
from app.export.runner import explain, read_semantic_layer, run_readonly
from app.query.guard import QueryRejected, referenced_names, validate_select_only
from app.query.retrieval import (
    columns_named_in,
    examples_worth_showing,
    expand_by_references,
    normalize_question,
    rank_examples,
    rank_tables,
    retrieve_business_rules,
    schema_context,
    select_columns,
    select_tables,
)
from app.query.shape import classify as classify_query_shape
from app.semantics.embeddings import get_embedder


def _record_history(db: Session, session_id: str, question: str, result: dict[str, Any]) -> None:
    """Append one question-history row, then trim to the configured limit.

    Written for a failed question too — the sidebar's "last N questions" is a
    log of what was asked, not only of what worked.  Trimming here rather than
    at read time keeps the table itself bounded, since nothing else deletes
    from it.
    """

    db.add(
        QueryHistoryRecord(
            session_id=session_id,
            question=question,
            ok=bool(result.get("ok")),
            sql=result.get("sql"),
            error=result.get("error"),
            chart=(result.get("visualization") or {}).get("chart"),
            confidence=result.get("confidence"),
            outcome=result.get("outcome") or ("answered" if result.get("ok") else "rejected"),
            resolved_question=(
                result.get("resolved_question")
                if result.get("resolved_question") not in (None, question)
                else None
            ),
            explanation=result.get("explanation"),
            tables_used=result.get("tables_used") or result.get("tables_considered"),
            clarification_options=result.get("clarification_options") or None,
            clarifies=result.get("clarifies"),
            source=result.get("source"),
        )
    )
    db.flush()
    limit = get_settings().query_history_limit
    stale = (
        db.execute(
            select(QueryHistoryRecord.id)
            .where(QueryHistoryRecord.session_id == session_id)
            .order_by(QueryHistoryRecord.created_at.desc())
            .offset(limit)
        )
        .scalars()
        .all()
    )
    if stale:
        db.execute(delete(QueryHistoryRecord).where(QueryHistoryRecord.id.in_(stale)))


def recent_conversation(db: Session, session_id: str, limit: int) -> list[dict[str, Any]]:
    """The last ``limit`` turns of this session, oldest first, as the server reads them.

    Kept server-side (``query_history``) rather than echoed back by the browser
    so a follow-up is understood the same way after a reload, and so what a
    turn "meant" is whatever the server decided it meant — never client input.
    """

    rows = (
        db.execute(
            select(QueryHistoryRecord)
            .where(QueryHistoryRecord.session_id == session_id)
            .order_by(QueryHistoryRecord.created_at.desc())
            .limit(limit)
        )
        .scalars()
        .all()
    )
    return [
        {
            "question": row.resolved_question or row.question,
            "outcome": row.outcome,
            "sql": row.sql,
            "explanation": row.explanation or "",
            "tables_used": list(row.tables_used or []),
            "clarification_options": list(row.clarification_options or []),
            "reason": row.error,
        }
        for row in reversed(rows)
    ]


def _awaiting_clarification(turn: dict[str, Any] | None) -> bool:
    """Whether this turn ended by asking the user which reading they meant."""

    return bool(turn and turn["outcome"] == "abstained" and turn["clarification_options"])


def _conversation_for_model(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """What a model is shown of earlier turns: the question, how it ended, what it computed.

    Never a result row — the same boundary every other call holds.  A turn that
    failed outright says nothing a later question can build on, so it is left
    out.
    """

    shown: list[dict[str, Any]] = []
    for turn in turns:
        if turn["outcome"] == "answered":
            shown.append(
                {
                    "question": turn["question"],
                    "outcome": "answered",
                    "computed": turn["explanation"],
                    "sql": turn["sql"],
                }
            )
        elif _awaiting_clarification(turn):
            shown.append(
                {
                    "question": turn["question"],
                    "outcome": "clarification requested",
                    "asked": turn["reason"],
                    "options_offered": turn["clarification_options"],
                }
            )
    return shown


def question_history(db: Session, session_id: str) -> list[dict[str, Any]]:
    """The last :attr:`~app.core.config.Settings.query_history_limit` questions, newest first."""

    rows = (
        db.execute(
            select(QueryHistoryRecord)
            .where(QueryHistoryRecord.session_id == session_id)
            .order_by(QueryHistoryRecord.created_at.desc())
            .limit(get_settings().query_history_limit)
        )
        .scalars()
        .all()
    )
    return [row.to_dict() for row in rows]


# ---------------------------------------------------------------------------
# Phase 5 — the third retrieval index: business rules + verified examples
# ---------------------------------------------------------------------------


def list_business_rules(db: Session, session_id: str) -> list[BusinessRule]:
    """Every business rule recorded for this dataset, newest first."""

    return list(
        db.execute(
            select(BusinessRule)
            .where(BusinessRule.session_id == session_id)
            .order_by(BusinessRule.created_at.desc())
        )
        .scalars()
        .all()
    )


def add_business_rule(db: Session, record: IngestionSession, rule_text: str) -> BusinessRule:
    """Record one plain-English domain rule, embedded for retrieval at query time."""

    rule_text = (rule_text or "").strip()
    if not rule_text:
        raise ValueError("a business rule needs some text")
    vector = get_embedder().encode([rule_text])[0]
    rule = BusinessRule(session_id=record.id, rule_text=rule_text, embedding=vector)
    db.add(rule)
    db.flush()
    return rule


def delete_business_rule(db: Session, record: IngestionSession, rule_id: str) -> None:
    rule = db.execute(
        select(BusinessRule).where(
            BusinessRule.session_id == record.id, BusinessRule.id == rule_id
        )
    ).scalar_one_or_none()
    if rule is None:
        raise KeyError(f"no business rule {rule_id!r}")
    db.delete(rule)
    db.flush()


def list_query_examples(
    db: Session, session_id: str, verified_only: bool = False
) -> list[QueryExample]:
    """Verified question/SQL pairs kept as few-shot examples for this dataset."""

    query = select(QueryExample).where(QueryExample.session_id == session_id)
    if verified_only:
        query = query.where(QueryExample.verified.is_(True))
    return list(db.execute(query.order_by(QueryExample.created_at.desc())).scalars().all())


def save_query_example(
    db: Session,
    record: IngestionSession,
    question: str,
    sql: str,
    explanation: str = "",
    also_asked_as: list[str] | None = None,
) -> QueryExample:
    """Save one successful answer as a verified example — the click itself is the verification.

    The SQL is re-validated here, independently of whatever validated it when
    the question was first answered, for the same reason
    :func:`pin_dashboard_card` re-validates: a save request is client-supplied
    input, and this is the write path that turns a stored statement into
    context every future question is judged against.

    ``question`` is the complete, standalone form.  ``also_asked_as`` carries
    the vaguer wording that led to it — an ambiguous question the user settled
    by picking a reading — so typing that wording again lands on this SQL
    instead of being asked about all over again.  Saving a question that is
    already an example replaces its SQL: the latest verification wins, rather
    than two rows disagreeing about one question.
    """

    question = (question or "").strip()
    sql = (sql or "").strip()
    if not question or not sql:
        raise ValueError("an example needs both a question and its SQL")

    export = export_record(db, record.id)
    if export is None:
        raise ValueError("export the semantic layer before saving an example")

    rows = read_semantic_layer(record.id)
    exported_tables = {str(row.get("table_name")).lower() for row in rows}
    try:
        validated_sql = validate_select_only(sql, exported_tables, export.dialect)
    except QueryRejected as exc:
        raise ValueError(f"this query can no longer be saved as an example: {exc}") from exc

    wordings: dict[str, str] = {normalize_question(question): question}
    for alias in also_asked_as or []:
        alias = (alias or "").strip()
        if alias:
            wordings.setdefault(normalize_question(alias), alias)
    wordings.pop("", None)
    if not wordings:
        raise ValueError("an example needs both a question and its SQL")

    existing = {
        normalize_question(example.question): example
        for example in list_query_examples(db, record.id)
    }
    texts = list(wordings.values())
    vectors = get_embedder().encode(texts)
    saved: list[QueryExample] = []
    for key, text, vector in zip(wordings, texts, vectors):
        example = existing.get(key)
        if example is None:
            example = QueryExample(session_id=record.id, question=text)
            db.add(example)
        example.sql = validated_sql
        example.explanation = (explanation or "").strip() or None
        example.embedding = vector
        example.verified = True
        saved.append(example)
    db.flush()
    return saved[0]


def delete_query_example(db: Session, record: IngestionSession, example_id: str) -> None:
    example = db.execute(
        select(QueryExample).where(
            QueryExample.session_id == record.id, QueryExample.id == example_id
        )
    ).scalar_one_or_none()
    if example is None:
        raise KeyError(f"no query example {example_id!r}")
    db.delete(example)
    db.flush()


def _run_and_package(
    db: Session,
    record: IngestionSession,
    claude,
    *,
    question: str,
    sql: str,
    explanation: str,
    confidence: float,
    dialect: str,
    tables: list[str],
    columns: list[str],
    result_rows: list[dict[str, Any]],
    truncated: bool,
    attempts: list[dict[str, Any]],
    meta: dict[str, Any],
) -> dict[str, Any]:
    """Shape an already-executed statement's result into an answer — shared by every path that gets here."""

    visualization = classify_query_shape(columns, result_rows)
    # A dashboard nicety, not part of the answer itself: a fake standing in for
    # Claude in a test need not implement it, and an unavailable/degraded model
    # simply yields no suggestions rather than breaking the question that just
    # succeeded.
    suggest = getattr(claude, "suggest_followups", None)
    suggestions = (suggest(question, sql, columns, tables) if suggest else None) or []

    result = {
        "ok": True,
        "outcome": "answered",
        "question": question,
        "sql": sql,
        "explanation": explanation,
        "confidence": confidence,
        "dialect": dialect,
        "columns": columns,
        "rows": result_rows,
        "row_count": len(result_rows),
        "truncated": truncated,
        "tables_used": tables,
        "attempts": attempts,
        "visualization": visualization,
        "suggestions": suggestions,
        **meta,
    }
    _record_history(db, record.id, question, result)
    return result


def answer_question(
    db: Session,
    record: IngestionSession,
    question: str,
    claude=None,
    extra_rules: Sequence[str] = (),
) -> dict[str, Any]:
    """Turn one natural-language question into SQL, run it once, and shape the result.

    Retrieval (:mod:`app.query.retrieval`) never touches a data row — it
    compares the question's embedding against table sentences and column
    descriptions Phase 4 already wrote.  Generation asks Claude for SQL over
    that retrieved schema alone, never over the tables themselves.  Only the
    one statement that survives :func:`~app.query.guard.validate_select_only`
    and an ``EXPLAIN`` (§Phase 5's self-correction loop) ever reaches the
    exported database, and it runs exactly once, row-capped, at the very end.

    A question is read **in the context of the session's last few turns**: a
    follow-up ("and by month?") or the reply to a clarification is first made
    a complete, standalone question (``resolved_question``), and everything
    after — retrieval, the ambiguity check, generation, saving an example —
    works from that.  What the user has already settled is not asked again: a
    question that answers a clarification, refines an answer, or matches a
    verified example skips the ambiguity check, and one that *is* a verified
    example is answered from its saved SQL without a model call at all.

    ``extra_rules`` are rules for this one question — what a person types beside
    it to say how to read it ("revenue means shipped orders") — as opposed to the
    ones recorded for the whole dataset.  They are read exactly like recorded
    rules: shown to the model ahead of them, and any column they name by name is
    kept in the schema the model is given however the retrieval scored it.

    Unlike taxonomy labelling or plan generation, there is no rule-based
    fallback for "turn English into SQL" — so with no model available this
    reports that plainly (``ok: False``) rather than guessing.  Any other
    failure to produce a usable query (the guard rejects every attempt, the
    database itself rejects the SQL) is reported the same way: a real result
    for the user to read, not a 500.
    """

    question = question.strip()
    if not question:
        raise ValueError("ask a question first")

    export = export_record(db, record.id)
    if export is None:
        raise ValueError("export the semantic layer before asking questions")

    rows = read_semantic_layer(record.id)
    if not rows:
        raise ValueError("the semantic layer has no columns to query yet")

    settings = get_settings()
    embedder = get_embedder()
    dialect = export.dialect

    sheets = _sheets_by_name(db, record.id)
    exported_tables = sorted({str(row.get("table_name")) for row in rows})
    table_vectors = {
        name: (sheets[name].table_embedding if name in sheets else None)
        for name in exported_tables
    }

    def pick_tables(vector) -> list[str]:
        scored = rank_tables(vector, table_vectors)
        return expand_by_references(select_tables(scored, settings.query_max_tables), rows)

    if claude is None or not claude.available:
        selected = pick_tables(embedder.encode([question])[0])
        result = {
            "ok": False,
            "outcome": "rejected",
            "question": question,
            "tables_considered": selected,
            "attempts": [],
            "error": (
                (claude.last_error if claude is not None else None)
                or "no language model is configured, so this question could not be "
                "turned into SQL"
            ),
        }
        _record_history(db, record.id, question, result)
        return result

    # -- What is actually being asked, given what came before ----------------
    conversation = recent_conversation(db, record.id, settings.query_conversation_turns)
    previous = conversation[-1] if conversation else None
    pending = previous if _awaiting_clarification(previous) else None

    resolved = question
    follow_up = False
    clarifies: str | None = None
    if pending and normalize_question(question) in {
        normalize_question(option) for option in pending["clarification_options"]
    }:
        # One of the rephrasings the user was just offered — already complete.
        follow_up = True
        clarifies = pending["question"]
    elif conversation:
        rewrite = getattr(claude, "rewrite_question", None)
        rewritten = rewrite(question, _conversation_for_model(conversation)) if rewrite else None
        if rewritten and rewritten["follow_up"]:
            resolved = rewritten["question"]
            follow_up = True
            if pending:
                clarifies = pending["question"]

    meta: dict[str, Any] = {
        "resolved_question": resolved,
        "follow_up": follow_up,
        "clarifies": clarifies,
    }

    question_vector = embedder.encode([resolved])[0]
    selected = pick_tables(question_vector)
    if follow_up and previous:
        # A refinement lives in the tables the answer it refines already read.
        known = {name.lower() for name in selected}
        selected += [
            name
            for name in previous["tables_used"]
            if name.lower() not in known and name in exported_tables
        ]

    # The third retrieval index (§4.2): the user's own recorded rules and
    # previously verified question/SQL pairs for this dataset, ranked by
    # relevance to this question exactly like tables and columns are.
    rule_records = list_business_rules(db, record.id)
    retrieved_rules = [
        *(str(rule).strip() for rule in extra_rules if str(rule).strip()),
        *retrieve_business_rules(
            question_vector,
            [{"rule_text": r.rule_text, "embedding": r.embedding} for r in rule_records],
            settings.query_max_business_rules,
        ),
    ]
    ranked_examples = rank_examples(
        question_vector,
        [
            {
                "id": e.id,
                "question": e.question,
                "sql": e.sql,
                "explanation": e.explanation,
                "embedding": e.embedding,
            }
            for e in list_query_examples(db, record.id, verified_only=True)
        ],
    )

    # -- A question the user already verified is answered from what they verified.
    # Only a *same question* qualifies (not merely a similar one — "top 5" and
    # "top 10" read alike to an embedder): reusing SQL for a different question
    # would be a wrong answer shown as a trusted one.  If the saved SQL no
    # longer holds (a re-export renamed something) it falls through to ordinary
    # generation, with the example still offered as a pattern.
    asked = {normalize_question(resolved), normalize_question(question)}
    exact = next((e for e in ranked_examples if normalize_question(e["question"]) in asked), None)
    if exact is not None:
        saved_columns: list[str] = []
        saved_rows: list[dict[str, Any]] = []
        saved_truncated = False
        try:
            saved_sql = validate_select_only(
                exact["sql"], {name.lower() for name in exported_tables}, dialect
            )
            explain(record.id, saved_sql)
            saved_columns, saved_rows, saved_truncated = run_readonly(
                record.id, saved_sql, settings.query_row_cap
            )
        except Exception:  # QueryRejected, or the driver's own error
            saved_sql = None
        if saved_sql is not None:
            used, _ = referenced_names(saved_sql, dialect)
            return _run_and_package(
                db,
                record,
                claude,
                question=question,
                sql=saved_sql,
                explanation=exact["explanation"] or "Answered from the query you saved for this question.",
                confidence=1.0,
                dialect=dialect,
                tables=[name for name in exported_tables if name.lower() in used],
                columns=saved_columns,
                result_rows=saved_rows,
                truncated=saved_truncated,
                attempts=[],
                meta={**meta, "source": "saved_example"},
            )

    # An example is shown only if it resembles the question: ranking orders the
    # examples, it does not make the nearest of a few unrelated ones relevant.  A
    # *close* one is more — what the user already settled.  The floor never sits
    # above the close threshold, so a close example is always shown.
    shown_examples = examples_worth_showing(
        ranked_examples,
        settings.query_max_examples,
        min(settings.query_example_min_similarity, settings.query_example_match_threshold),
    )
    close_examples = [
        e for e in shown_examples if e["similarity"] >= settings.query_example_match_threshold
    ]
    example_columns: set[str] = set()
    for example in shown_examples:
        example_tables, columns_named = referenced_names(example["sql"], dialect)
        example_columns |= columns_named
        known = {name.lower() for name in selected}
        selected += [
            name for name in exported_tables if name.lower() in example_tables - known
        ]

    selected_columns = select_columns(
        question_vector,
        rows,
        selected,
        settings.query_max_ranked_columns,
        question=resolved,
        include_columns=example_columns | columns_named_in(retrieved_rules, rows, set(selected)),
    )
    context = schema_context(selected_columns)
    allowed_tables = {name.lower() for name in selected}
    retrieved_examples = [
        {"question": str(e["question"]), "sql": str(e["sql"])} for e in shown_examples
    ]

    # -- Ambiguity resolution (§3.1.6) ---------------------------------------
    # A discrete call on the pre-selected tables and their columns, before any
    # SQL is written.  Skipped whenever the user has already told us what they
    # mean — by picking an offered reading, by refining an answer they got, or
    # by saving a close example — so a clarification is asked at most once and
    # never about something already settled.  Degrades to "not ambiguous" with
    # no client, same as every other optional Claude call.
    check_ambiguity = getattr(claude, "check_ambiguity", None)
    already_settled = follow_up or bool(close_examples)
    if check_ambiguity and not already_settled:
        columns_by_table: dict[str, list[str]] = {}
        for row in selected_columns:
            description = str(row.get("semantic_description") or "")[:80]
            entry = f"{row.get('column_name')}: {description}" if description else str(row.get("column_name"))
            columns_by_table.setdefault(str(row.get("table_name")), []).append(entry)
        ambiguity_tables = [
            {
                "name": name,
                "description": sheets[name].effective_description if name in sheets else "",
                "columns": columns_by_table.get(name, []),
            }
            for name in selected
        ]
        ambiguity_args: dict[str, Any] = {}
        if retrieved_rules:
            ambiguity_args["business_rules"] = retrieved_rules
        ambiguity = check_ambiguity(resolved, ambiguity_tables, **ambiguity_args)
        if ambiguity and ambiguity.get("ambiguous"):
            result = {
                "ok": False,
                "outcome": "abstained",
                "ambiguous": True,
                "question": question,
                "tables_considered": selected,
                "attempts": [],
                "clarification_options": ambiguity.get("options", []),
                "error": ambiguity.get("reason")
                or "this question is ambiguous — pick a specific interpretation",
                **meta,
            }
            _record_history(db, record.id, question, result)
            return result

    attempts: list[dict[str, Any]] = []
    sql: str | None = None
    explanation = ""
    confidence = 0.0
    last_error: str | None = None
    prior_sql: str | None = None
    result_columns: list[str] = []
    result_rows: list[dict[str, Any]] = []
    result_truncated = False

    # A model fake that predates conversation support is simply not offered it.
    generation_extras: dict[str, Any] = {}
    if follow_up and conversation:
        shown = _conversation_for_model(conversation)
        if shown:
            generation_extras["conversation"] = shown

    for attempt_no in range(1, settings.query_max_attempts + 1):
        generated = claude.generate_sql(
            resolved,
            dialect,
            context,
            prior_sql=prior_sql,
            prior_error=last_error,
            business_rules=retrieved_rules,
            examples=retrieved_examples,
            **generation_extras,
        )
        if generated is None:
            last_error = claude.last_error or "the language model returned no usable answer"
            attempts.append({"attempt": attempt_no, "sql": None, "error": last_error})
            break

        prior_sql = generated["sql"]
        try:
            validated = validate_select_only(generated["sql"], allowed_tables, dialect)
        except QueryRejected as exc:
            last_error = str(exc)
            attempts.append({"attempt": attempt_no, "sql": generated["sql"], "error": last_error})
            continue

        try:
            explain(record.id, validated)
        except Exception as exc:  # the driver's own exception type, dialect-dependent
            last_error = str(exc)
            attempts.append({"attempt": attempt_no, "sql": validated, "error": last_error})
            continue

        sql = validated
        explanation = generated["explanation"]
        # A fake standing in for Claude in a test need not implement
        # confidence — treated as fully confident, so existing "this succeeds"
        # tests keep succeeding rather than starting to abstain.
        confidence = generated.get("confidence", 1.0)
        attempts.append({"attempt": attempt_no, "sql": validated, "error": None})

        # A candidate below the confidence gate is withheld below — never
        # executed, never rendered — so it is not run here either; the loop
        # ends with ``sql`` set and the gate reports the abstention.
        if confidence < settings.query_confidence_threshold:
            break

        # ``EXPLAIN`` plans the query without evaluating it, so it never sees
        # a value-dependent failure — a CAST that only breaks on an empty
        # string, a date parse that only breaks on a "TOTAL" footer row.
        # Running it for real, still inside the retry loop, turns that class
        # of error into a retryable one too: the driver's message goes back
        # to the model as ``prior_error`` so the next attempt can write
        # defensive SQL instead of the question failing with a 500.
        try:
            result_columns, result_rows, result_truncated = run_readonly(
                record.id, validated, settings.query_row_cap
            )
        except Exception as exc:  # the driver's own exception type, dialect-dependent
            last_error = str(exc)
            sql = None
            attempts[-1]["error"] = last_error
            continue

        break

    if sql is None:
        result = {
            "ok": False,
            "outcome": "rejected",
            "question": question,
            "tables_considered": selected,
            "attempts": attempts,
            "error": last_error or "the model could not produce a usable query for this question",
            **meta,
        }
        _record_history(db, record.id, question, result)
        return result

    # Confidence-based abstention gate (§5.5): a candidate query exists, but
    # is withheld — never executed, never rendered — when the model itself
    # was not confident it answers the question correctly.
    if confidence < settings.query_confidence_threshold:
        result = {
            "ok": False,
            "outcome": "abstained",
            "question": question,
            "sql": sql,
            "confidence": confidence,
            "tables_considered": selected,
            "attempts": attempts,
            "error": (
                f"low confidence ({confidence:.2f}) that this query answers the question "
                f"correctly — {explanation}" if explanation else
                f"low confidence ({confidence:.2f}) that this query answers the question correctly"
            ),
            **meta,
        }
        _record_history(db, record.id, question, result)
        return result

    return _run_and_package(
        db,
        record,
        claude,
        question=question,
        sql=sql,
        explanation=explanation,
        confidence=confidence,
        dialect=dialect,
        tables=selected,
        columns=result_columns,
        result_rows=result_rows,
        truncated=result_truncated,
        attempts=attempts,
        meta=meta,
    )
