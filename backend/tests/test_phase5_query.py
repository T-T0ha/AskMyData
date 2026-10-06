"""Phase 5 — natural language questions answered against the semantic layer.

Three layers, tested at the level each one actually decides something:

* :mod:`app.query.guard` — pure, no fixture needed: does a string of SQL
  survive, and does exactly the right thing about it fail when it should not.
* :mod:`app.query.retrieval` and :mod:`app.query.shape` — pure functions over
  plain dicts, standing in for what ``sem_metadata`` and an embedder would
  hand them.
* :func:`app.api.services.answer_question` — the orchestration, exercised
  against the real relational fixture with a scripted fake model standing in
  for Claude, the same pattern ``test_phase4_alignment.py`` uses.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.db.base import get_session_factory
from app.db.models import IngestionSession
from app.export.runner import read_back
from app.query.guard import QueryRejected, validate_select_only
from app.query.retrieval import (
    TableScore,
    columns_named_in,
    examples_worth_showing,
    expand_by_references,
    rank_tables,
    retrieve_business_rules,
    retrieve_examples,
    schema_context,
    select_columns,
    select_tables,
)
from app.query.shape import classify


# ---------------------------------------------------------------------------
# guard
# ---------------------------------------------------------------------------


ALLOWED = {"orders", "customers"}


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT order_id, total_amount FROM orders WHERE total_amount > 100",
        "SELECT c.city, COUNT(*) FROM customers c JOIN orders o "
        "ON o.customer_id = c.customer_id GROUP BY c.city",
        "WITH big AS (SELECT * FROM orders WHERE total_amount > 100) SELECT * FROM big",
        "SELECT * FROM orders UNION SELECT * FROM orders",
    ],
)
def test_an_ordinary_select_is_accepted(sql):
    validate_select_only(sql, ALLOWED, "postgresql")


def test_a_second_statement_is_refused():
    with pytest.raises(QueryRejected, match="one statement"):
        validate_select_only("SELECT * FROM orders; DROP TABLE orders;", ALLOWED, "postgresql")


@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO orders (order_id) VALUES ('X')",
        "UPDATE orders SET total_amount = 0",
        "DELETE FROM orders",
        "DROP TABLE orders",
        "ALTER TABLE orders ADD COLUMN x INT",
        "CREATE TABLE evil (x INT)",
    ],
)
def test_anything_that_is_not_a_read_is_refused(sql):
    with pytest.raises(QueryRejected, match="SELECT"):
        validate_select_only(sql, ALLOWED, "postgresql")


def test_a_table_outside_the_offered_schema_is_refused():
    with pytest.raises(QueryRejected, match="outside the tables offered"):
        validate_select_only("SELECT * FROM users", ALLOWED, "postgresql")


def test_a_schema_qualified_name_is_refused_even_when_the_table_is_allowed():
    with pytest.raises(QueryRejected, match="schema or database"):
        validate_select_only("SELECT * FROM public.orders", ALLOWED, "postgresql")


def test_blank_sql_is_refused():
    with pytest.raises(QueryRejected, match="no SQL"):
        validate_select_only("   ", ALLOWED, "postgresql")


def test_unparseable_sql_is_refused():
    with pytest.raises(QueryRejected, match="could not be parsed"):
        validate_select_only("SELECT FROM WHERE ]]]", ALLOWED, "postgresql")


def test_a_cte_alias_is_not_mistaken_for_an_unknown_table():
    """``recent`` names a CTE, not a table — it must not need to be in ``allowed_tables``."""

    sql = "WITH recent AS (SELECT * FROM orders) SELECT * FROM recent"
    validate_select_only(sql, {"orders"}, "postgresql")


def test_a_cte_cannot_smuggle_in_a_same_named_real_table():
    """A non-RECURSIVE CTE cannot see its own name inside its own body — that
    occurrence can only resolve to a real table, so it must still be offered."""

    sql = (
        "WITH employees_salary AS (SELECT * FROM employees_salary) "
        "SELECT * FROM employees_salary"
    )
    with pytest.raises(QueryRejected, match="outside the tables offered"):
        validate_select_only(sql, ALLOWED, "postgresql")

    # The same query succeeds when the colliding real table genuinely was offered.
    validate_select_only(sql, {"employees_salary"}, "postgresql")


def test_with_recursive_is_refused_outright():
    sql = (
        "WITH RECURSIVE t AS (SELECT 1 AS n UNION ALL SELECT n + 1 FROM t WHERE n < 5) "
        "SELECT * FROM t"
    )
    with pytest.raises(QueryRejected, match="RECURSIVE"):
        validate_select_only(sql, ALLOWED, "postgresql")


def test_the_validated_sql_is_returned_re_rendered():
    out = validate_select_only("select * from orders", ALLOWED, "postgresql")
    assert out.strip().upper().startswith("SELECT")


# ---------------------------------------------------------------------------
# retrieval
# ---------------------------------------------------------------------------


def test_tables_are_ranked_by_cosine_similarity_to_the_question():
    question = [1.0, 0.0]
    tables = {"orders": [1.0, 0.0], "customers": [0.0, 1.0]}
    scored = rank_tables(question, tables)
    assert [s.table for s in scored] == ["orders", "customers"]
    assert scored[0].score == pytest.approx(1.0)
    assert scored[1].score == pytest.approx(0.0)


def test_selection_caps_at_max_tables_when_everything_is_ranked():
    scored = [TableScore(table=f"t{i}", score=1.0 - i * 0.1, ranked=True) for i in range(5)]
    assert select_tables(scored, max_tables=2) == ["t0", "t1"]


def test_selection_falls_back_to_everything_when_one_table_cannot_be_ranked():
    """An unranked table cannot be told apart from a genuinely irrelevant one by
    score alone, so losing the ranking for even one table means the ranking as
    a whole cannot be trusted to decide what gets left out."""

    scored = [
        TableScore(table="a", score=0.9, ranked=True),
        TableScore(table="b", score=0.1, ranked=False),
    ]
    assert set(select_tables(scored, max_tables=1)) == {"a", "b"}


def test_a_confirmed_reference_pulls_the_other_table_in():
    rows = [
        {"table_name": "orders", "is_foreign_key": True, "references_table": "customers"},
        {"table_name": "orders", "is_foreign_key": False, "references_table": None},
        {"table_name": "customers", "is_foreign_key": False, "references_table": None},
    ]
    assert expand_by_references(["orders"], rows) == ["orders", "customers"]


def test_reference_expansion_works_in_either_direction():
    rows = [{"table_name": "orders", "is_foreign_key": True, "references_table": "customers"}]
    assert expand_by_references(["customers"], rows) == ["customers", "orders"]


def test_key_columns_are_kept_regardless_of_score_but_ranked_ones_are_capped():
    rows = [
        {
            "table_name": "orders",
            "column_name": "order_id",
            "is_primary_key": True,
            "is_foreign_key": False,
            "embedding": [0.0, 0.0],  # would score last if it competed
        },
        {
            "table_name": "orders",
            "column_name": "customer_id",
            "is_primary_key": False,
            "is_foreign_key": True,
            "embedding": [0.0, 0.0],
        },
        {
            "table_name": "orders",
            "column_name": "status",
            "is_primary_key": False,
            "is_foreign_key": False,
            "embedding": [1.0, 0.0],
        },
        {
            "table_name": "orders",
            "column_name": "notes",
            "is_primary_key": False,
            "is_foreign_key": False,
            "embedding": [0.0, 1.0],
        },
    ]
    kept = select_columns([1.0, 0.0], rows, ["orders"], max_ranked=1)
    names = {row["column_name"] for row in kept}
    assert {"order_id", "customer_id", "status"} <= names
    assert "notes" not in names  # outranked, and the cap is 1 non-key column


def test_schema_context_groups_columns_under_their_table_and_hides_the_vector():
    rows = [
        {
            "table_name": "orders",
            "column_name": "order_id",
            "sql_type": "TEXT",
            "taxonomy_label": "order_number",
            "is_primary_key": True,
            "is_foreign_key": False,
            "references_table": None,
            "references_column": None,
            "is_additive": False,
            "null_ratio": 0.0,
            "sample_values": ["ORD-1"],
            "embedding": [0.1, 0.2],
        }
    ]
    context = schema_context(rows)
    assert context == [
        {
            "table": "orders",
            "table_type": "unknown",
            "columns": [
                {
                    "name": "order_id",
                    "type": "TEXT",
                    "taxonomy": "order_number",
                    "is_primary_key": True,
                    "is_foreign_key": False,
                    "references": None,
                    "is_additive": False,
                    "null_ratio": 0.0,
                    "sample_values": ["ORD-1"],
                }
            ],
        }
    ]


# ---------------------------------------------------------------------------
# shape
# ---------------------------------------------------------------------------


def test_one_row_one_column_is_a_metric():
    assert classify(["n"], [{"n": 42}])["chart"] == "metric"


def test_text_and_numeric_is_a_bar_when_there_are_many_categories():
    rows = [{"city": f"city{i}", "total": i} for i in range(10)]
    assert classify(["city", "total"], rows)["chart"] == "bar"


def test_text_and_numeric_is_a_pie_within_the_small_category_band():
    rows = [{"status": "Shipped", "n": 5}, {"status": "Pending", "n": 2}]
    assert classify(["status", "n"], rows)["chart"] == "pie"


def test_date_and_numeric_is_a_line():
    import datetime

    rows = [{"day": datetime.date(2024, 1, i + 1), "total": i} for i in range(5)]
    result = classify(["day", "total"], rows)
    assert result["chart"] == "line"
    assert result["date_columns"] == ["day"]


def test_date_with_two_numerics_is_a_multi_line():
    import datetime

    rows = [
        {"day": datetime.date(2024, 1, i + 1), "revenue": i, "cost": i / 2} for i in range(5)
    ]
    assert classify(["day", "revenue", "cost"], rows)["chart"] == "multi_line"


def test_general_shapes_fall_back_to_a_table():
    rows = [{"a": "x", "b": "y", "c": "z"}]
    assert classify(["a", "b", "c"], rows)["chart"] == "table"


def test_no_rows_is_a_table_not_a_crash():
    assert classify(["a", "b"], [])["chart"] == "table"


def test_a_date_column_is_recognised_when_the_driver_hands_back_a_string():
    """SQLite's raw ``text()`` execution returns a TIMESTAMP column's stored
    string, not a ``datetime`` — this is the shape a real query result on the
    platform's own dev/test target actually has, not a hypothetical."""

    rows = [
        {"day": "2024-01-01 00:00:00.000000", "total": 10},
        {"day": "2024-01-02 00:00:00.000000", "total": 20},
    ]
    result = classify(["day", "total"], rows)
    assert result["chart"] == "line"
    assert result["date_columns"] == ["day"]


def test_a_year_month_string_from_a_group_by_is_recognised_as_a_date():
    rows = [{"month": "2024-01", "revenue": 100}, {"month": "2024-02", "revenue": 150}]
    assert classify(["month", "revenue"], rows)["chart"] == "line"


def test_a_bare_four_digit_string_is_not_assumed_to_be_a_year():
    """Too easily a code or an id to call it a date from shape alone."""

    rows = [{"code": "2024", "n": 1}, {"code": "2025", "n": 2}]
    assert classify(["code", "n"], rows)["chart"] != "line"


# ---------------------------------------------------------------------------
# the third retrieval index — business rules + verified examples
# ---------------------------------------------------------------------------


def test_business_rules_are_ranked_by_relevance_to_the_question():
    rules = [
        {"rule_text": "irrelevant rule", "embedding": [0.0, 1.0]},
        {"rule_text": "relevant rule", "embedding": [1.0, 0.0]},
    ]
    assert retrieve_business_rules([1.0, 0.0], rules) == ["relevant rule", "irrelevant rule"]


def test_business_rules_are_capped_at_max_rules():
    rules = [{"rule_text": f"rule {i}", "embedding": [1.0, 0.0]} for i in range(10)]
    assert len(retrieve_business_rules([1.0, 0.0], rules, max_rules=3)) == 3


def test_a_rule_with_no_embedding_yet_still_comes_back_scored_last():
    rules = [
        {"rule_text": "no embedding yet", "embedding": None},
        {"rule_text": "scored rule", "embedding": [1.0, 0.0]},
    ]
    assert retrieve_business_rules([1.0, 0.0], rules) == ["scored rule", "no embedding yet"]


def test_examples_are_ranked_by_relevance_and_shaped_as_question_sql_pairs():
    examples = [
        {"question": "irrelevant", "sql": "SELECT 1", "embedding": [0.0, 1.0]},
        {"question": "relevant", "sql": "SELECT 2", "embedding": [1.0, 0.0]},
    ]
    result = retrieve_examples([1.0, 0.0], examples)
    assert result == [{"question": "relevant", "sql": "SELECT 2"}, {"question": "irrelevant", "sql": "SELECT 1"}]


def test_examples_are_capped_at_max_examples():
    examples = [
        {"question": f"q{i}", "sql": "SELECT 1", "embedding": [1.0, 0.0]} for i in range(10)
    ]
    assert len(retrieve_examples([1.0, 0.0], examples, max_examples=2)) == 2


def test_an_example_less_similar_than_the_floor_is_not_returned_however_few_there_are():
    examples = [
        {"question": "same kind of question", "sql": "SELECT 1", "embedding": [1.0, 0.1]},
        {"question": "about something else", "sql": "SELECT 2", "embedding": [0.1, 1.0]},
    ]
    # Without a floor the two nearest are returned, the unrelated one included.
    assert [e["question"] for e in retrieve_examples([1.0, 0.0], examples)] == [
        "same kind of question",
        "about something else",
    ]
    # With one, only what resembles the question is.
    assert [e["question"] for e in retrieve_examples([1.0, 0.0], examples, min_similarity=0.6)] == [
        "same kind of question"
    ]
    assert retrieve_examples([0.0, 1.0], examples[:1], min_similarity=0.6) == []


def test_examples_worth_showing_keeps_the_ranking_the_cap_and_the_similarity():
    ranked = [
        {"question": "a", "similarity": 0.95},
        {"question": "b", "similarity": 0.7},
        {"question": "c", "similarity": 0.65},
        {"question": "d", "similarity": 0.2},
    ]
    assert [e["question"] for e in examples_worth_showing(ranked, 3, 0.6)] == ["a", "b", "c"]
    assert [e["question"] for e in examples_worth_showing(ranked, 2, 0.6)] == ["a", "b"]
    assert [e["question"] for e in examples_worth_showing(ranked, 4, 0.6)] == ["a", "b", "c"]
    assert examples_worth_showing(ranked, 3, 0.99) == []
    # The floor is inclusive: an example exactly that similar is shown.
    assert [e["question"] for e in examples_worth_showing(ranked, 4, 0.7)] == ["a", "b"]


def test_the_default_cap_sends_every_column_of_an_ordinary_sized_schema():
    """Ranking exists to protect the prompt budget; below the budget nothing is cut."""

    from app.core.config import get_settings

    columns = 90
    rows = [
        {
            "table_name": "wide",
            "column_name": f"c{i}",
            "is_primary_key": i == 0,
            "is_foreign_key": False,
            "embedding": [0.0, 1.0],
        }
        for i in range(columns)
    ]
    kept = select_columns([1.0, 0.0], rows, ["wide"], get_settings().query_max_ranked_columns)
    assert len(kept) == columns
    # ...and a table wider than the budget is still cut down to it.
    wide = [dict(rows[0])] + [
        {**rows[1], "column_name": f"w{i}"} for i in range(get_settings().query_max_ranked_columns + 50)
    ]
    assert len(select_columns([1.0, 0.0], wide, ["wide"], get_settings().query_max_ranked_columns)) == (
        1 + get_settings().query_max_ranked_columns
    )


# ---------------------------------------------------------------------------
# service orchestration, against the real relational fixture
# ---------------------------------------------------------------------------


def _db():
    return get_session_factory()()


def _uploaded(client, workbook) -> str:
    session_id = client.post("/api/sessions", json={"name": "phase 5"}).json()["id"]
    with workbook.open("rb") as handle:
        response = client.post(
            f"/api/sessions/{session_id}/upload", files={"file": (workbook.name, handle)}
        )
    assert response.status_code == 200, response.text
    return session_id


def _exported(client, workbook) -> str:
    session_id = _uploaded(client, workbook)
    client.post(f"/api/sessions/{session_id}/relationships")
    response = client.post(f"/api/sessions/{session_id}/export")
    assert response.status_code == 200, response.text
    return session_id


class _ScriptedClaude:
    """Stands in for Claude: returns queued answers, one per call.

    Recording every call is what lets a test assert the retry loop actually
    saw the previous attempt's error rather than merely tried again.
    """

    available = True

    def __init__(self, answers: list[dict[str, str] | None]) -> None:
        self._answers = list(answers)
        self.calls: list[dict] = []
        self.last_error = "the model declined"

    def generate_sql(
        self,
        question,
        dialect,
        schema,
        prior_sql=None,
        prior_error=None,
        business_rules=None,
        examples=None,
        conversation=None,
    ):
        self.calls.append(
            {
                "question": question,
                "dialect": dialect,
                "schema": schema,
                "prior_sql": prior_sql,
                "prior_error": prior_error,
                "business_rules": business_rules,
                "examples": examples,
                "conversation": conversation,
            }
        )
        if not self._answers:
            return None
        return self._answers.pop(0)


class _UnavailableClaude:
    available = False
    last_error = "ANTHROPIC_API_KEY is not set"


def test_a_good_query_is_validated_explained_and_executed(client, relational_workbook):
    from app.api import services

    session_id = _exported(client, relational_workbook)
    fake = _ScriptedClaude(
        [
            {
                "sql": "SELECT city, COUNT(*) AS n FROM customers GROUP BY city",
                "explanation": "counts customers per city",
            }
        ]
    )

    with _db() as db:
        record = db.get(IngestionSession, session_id)
        result = services.answer_question(db, record, "how many customers per city?", claude=fake)

    assert result["ok"] is True
    assert result["sql"].upper().startswith("SELECT")
    assert result["row_count"] > 0
    assert set(result["columns"]) == {"city", "n"}
    assert result["visualization"]["chart"] in {"bar", "pie", "table"}
    assert len(fake.calls) == 1


def test_a_rejected_first_attempt_is_retried_with_the_error(client, relational_workbook):
    from app.api import services

    session_id = _exported(client, relational_workbook)
    fake = _ScriptedClaude(
        [
            {"sql": "DELETE FROM orders", "explanation": "wrong"},
            {"sql": "SELECT COUNT(*) AS n FROM orders", "explanation": "counts orders"},
        ]
    )

    with _db() as db:
        record = db.get(IngestionSession, session_id)
        result = services.answer_question(db, record, "how many orders are there?", claude=fake)

    assert result["ok"] is True
    assert len(result["attempts"]) == 2
    assert result["attempts"][0]["error"] is not None
    assert result["attempts"][1]["error"] is None
    # The retry saw what went wrong, not just a second blind guess.
    assert fake.calls[1]["prior_sql"] == "DELETE FROM orders"
    assert fake.calls[1]["prior_error"]


def test_a_query_that_fails_only_at_execution_is_retried_with_the_error(
    client, relational_workbook, monkeypatch
):
    """``EXPLAIN`` plans a query without evaluating it, so a value-dependent
    failure — a CAST that only breaks on one row's data, a date parse that
    only breaks on a "TOTAL" footer row — only ever surfaces once the
    statement actually runs.  That has to be retryable too, exactly like a
    guard rejection or an ``EXPLAIN`` failure, not an unhandled 500."""

    from app.api import services

    session_id = _exported(client, relational_workbook)
    fake = _ScriptedClaude(
        [
            {"sql": "SELECT COUNT(*) AS n FROM orders", "explanation": "first try"},
            {"sql": "SELECT COUNT(*) AS n FROM orders", "explanation": "second try"},
        ]
    )

    real_run_readonly = services.run_readonly
    calls = {"n": 0}

    def flaky_run_readonly(session_id_, sql, row_cap):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError('invalid input syntax for type numeric: ""')
        return real_run_readonly(session_id_, sql, row_cap)

    monkeypatch.setattr(services, "run_readonly", flaky_run_readonly)

    with _db() as db:
        record = db.get(IngestionSession, session_id)
        result = services.answer_question(db, record, "how many orders are there?", claude=fake)

    assert result["ok"] is True
    assert len(result["attempts"]) == 2
    assert result["attempts"][0]["error"] and "numeric" in result["attempts"][0]["error"]
    assert result["attempts"][1]["error"] is None
    assert fake.calls[1]["prior_sql"] == "SELECT COUNT(*) AS n FROM orders"
    assert fake.calls[1]["prior_error"]


def test_a_query_that_always_fails_at_execution_still_returns_a_normal_result(
    client, relational_workbook, monkeypatch
):
    from app.api import services

    session_id = _exported(client, relational_workbook)
    fake = _ScriptedClaude(
        [{"sql": "SELECT COUNT(*) AS n FROM orders", "explanation": "x"}] * 5
    )

    def always_fails(session_id_, sql, row_cap):
        raise RuntimeError('invalid value "TOTAL" for "Mon"')

    monkeypatch.setattr(services, "run_readonly", always_fails)

    with _db() as db:
        record = db.get(IngestionSession, session_id)
        result = services.answer_question(db, record, "how many orders?", claude=fake)

    assert result["ok"] is False
    assert "TOTAL" in result["error"]
    assert len(result["attempts"]) == db_settings_max_attempts()


def test_giving_up_after_every_attempt_still_returns_a_normal_result(client, relational_workbook):
    from app.api import services

    session_id = _exported(client, relational_workbook)
    fake = _ScriptedClaude([{"sql": "DROP TABLE orders", "explanation": "wrong"}] * 5)

    with _db() as db:
        record = db.get(IngestionSession, session_id)
        result = services.answer_question(db, record, "destroy everything", claude=fake)

    assert result["ok"] is False
    assert result["error"]
    assert len(result["attempts"]) == db_settings_max_attempts()


def db_settings_max_attempts() -> int:
    from app.core.config import get_settings

    return get_settings().query_max_attempts


def test_a_query_outside_the_offered_tables_is_caught_by_the_guard_not_the_database(
    client, relational_workbook
):
    from app.api import services

    session_id = _exported(client, relational_workbook)
    fake = _ScriptedClaude([{"sql": "SELECT * FROM users", "explanation": "wrong table"}])

    with _db() as db:
        record = db.get(IngestionSession, session_id)
        result = services.answer_question(db, record, "list every user", claude=fake)

    assert result["ok"] is False
    # The guard's own message is what explains the *first* attempt failing;
    # the model then has nothing left queued, which is a separate, later
    # failure and is what `result["error"]` reports last.
    assert "outside the tables offered" in result["attempts"][0]["error"]


def test_with_no_model_available_the_question_is_answered_honestly(
    client, relational_workbook
):
    from app.api import services

    session_id = _exported(client, relational_workbook)

    with _db() as db:
        record = db.get(IngestionSession, session_id)
        result = services.answer_question(
            db, record, "how many orders?", claude=_UnavailableClaude()
        )

    assert result["ok"] is False
    assert "language model" in result["error"] or "ANTHROPIC_API_KEY" in result["error"]


def test_asking_before_exporting_is_refused(client, relational_workbook):
    from app.api import services

    session_id = _uploaded(client, relational_workbook)

    with _db() as db:
        record = db.get(IngestionSession, session_id)
        with pytest.raises(ValueError, match="export"):
            services.answer_question(db, record, "anything", claude=_ScriptedClaude([]))


def test_a_blank_question_is_refused(client, relational_workbook):
    from app.api import services

    session_id = _exported(client, relational_workbook)

    with _db() as db:
        record = db.get(IngestionSession, session_id)
        with pytest.raises(ValueError, match="question"):
            services.answer_question(db, record, "   ", claude=_ScriptedClaude([]))


def test_a_date_and_a_measure_from_the_real_export_chart_as_a_line(
    client, relational_workbook
):
    """Exercises the actual round trip through the exported database, which
    unit tests against hand-built ``datetime`` objects cannot: SQLite hands a
    ``TIMESTAMP`` column back as a string over a raw connection, and the shape
    classifier has to recognise it as a date anyway."""

    from app.api import services

    session_id = _exported(client, relational_workbook)
    fake = _ScriptedClaude(
        [
            {
                "sql": "SELECT order_date, total_amount FROM orders ORDER BY order_date",
                "explanation": "orders over time",
            }
        ]
    )

    with _db() as db:
        record = db.get(IngestionSession, session_id)
        result = services.answer_question(db, record, "how did order value trend?", claude=fake)

    assert result["ok"] is True
    assert result["visualization"]["chart"] == "line"
    assert result["visualization"]["date_columns"] == ["order_date"]


def test_the_result_only_ever_ran_the_final_validated_statement(
    client, relational_workbook, monkeypatch
):
    """The rejected first attempt in the retry test must never reach the
    database — this asserts it directly against the exported rows rather than
    trusting the guard alone."""

    from app.api import services

    session_id = _exported(client, relational_workbook)
    before = read_back(session_id, "SELECT COUNT(*) AS n FROM orders")[0]["n"]
    fake = _ScriptedClaude(
        [
            {"sql": "DELETE FROM orders", "explanation": "wrong"},
            {"sql": "SELECT COUNT(*) AS n FROM orders", "explanation": "right"},
        ]
    )

    with _db() as db:
        record = db.get(IngestionSession, session_id)
        services.answer_question(db, record, "how many orders?", claude=fake)

    after = read_back(session_id, "SELECT COUNT(*) AS n FROM orders")[0]["n"]
    assert after == before


# ---------------------------------------------------------------------------
# the route
# ---------------------------------------------------------------------------


def test_the_route_refuses_a_question_before_export(client, relational_workbook):
    session_id = _uploaded(client, relational_workbook)

    response = client.post(f"/api/sessions/{session_id}/query", json={"question": "anything"})

    assert response.status_code == 409


def test_the_route_belongs_to_its_owner(client, other_client, relational_workbook):
    session_id = _exported(client, relational_workbook)

    response = other_client.post(
        f"/api/sessions/{session_id}/query", json={"question": "anything"}
    )

    assert response.status_code == 404


def test_the_route_answers_ok_false_with_no_api_key_configured(client, relational_workbook):
    """The suite runs with ANTHROPIC_API_KEY unset (conftest), so the route's
    own ``get_claude_client()`` is genuinely unavailable — a real end-to-end
    check of the degraded path, not a mock."""

    session_id = _exported(client, relational_workbook)

    response = client.post(
        f"/api/sessions/{session_id}/query", json={"question": "how many orders?"}
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["ok"] is False
    assert payload["error"]


def test_a_blank_question_is_rejected_by_validation(client, relational_workbook):
    session_id = _exported(client, relational_workbook)

    response = client.post(f"/api/sessions/{session_id}/query", json={"question": ""})

    assert response.status_code == 422


# ---------------------------------------------------------------------------
# confidence-based abstention and ambiguity resolution (C-2)
# ---------------------------------------------------------------------------


class _AmbiguousClaude(_ScriptedClaude):
    """Also flags every question as ambiguous — proves the short-circuit never
    reaches ``generate_sql`` at all."""

    def __init__(self, answers, options):
        super().__init__(answers)
        self._options = options
        self.ambiguity_checks: list[str] = []

    def check_ambiguity(self, question, tables, business_rules=None):
        self.ambiguity_checks.append(question)
        return {"ambiguous": True, "reason": "could mean several things", "options": self._options}


def test_a_low_confidence_answer_is_abstained_not_executed(client, relational_workbook):
    from app.api import services

    session_id = _exported(client, relational_workbook)
    fake = _ScriptedClaude(
        [
            {
                "sql": "SELECT COUNT(*) AS n FROM orders",
                "explanation": "a guess",
                "confidence": 0.1,
            }
        ]
    )

    with _db() as db:
        record = db.get(IngestionSession, session_id)
        result = services.answer_question(db, record, "how many orders, roughly?", claude=fake)

    assert result["ok"] is False
    assert result["outcome"] == "abstained"
    assert result["confidence"] == pytest.approx(0.1)
    assert "rows" not in result


def test_a_high_confidence_answer_is_executed_and_recorded_as_answered(
    client, relational_workbook
):
    from app.api import services

    session_id = _exported(client, relational_workbook)
    fake = _ScriptedClaude(
        [
            {
                "sql": "SELECT COUNT(*) AS n FROM orders",
                "explanation": "counts orders",
                "confidence": 0.9,
            }
        ]
    )

    with _db() as db:
        record = db.get(IngestionSession, session_id)
        result = services.answer_question(db, record, "how many orders?", claude=fake)

    assert result["ok"] is True
    assert result["outcome"] == "answered"
    assert result["confidence"] == pytest.approx(0.9)


def test_an_ambiguous_question_is_abstained_before_any_sql_is_attempted(
    client, relational_workbook
):
    from app.api import services

    session_id = _exported(client, relational_workbook)
    fake = _AmbiguousClaude([], ["how many orders by status?", "how many orders by city?"])

    with _db() as db:
        record = db.get(IngestionSession, session_id)
        result = services.answer_question(db, record, "how many orders?", claude=fake)

    assert result["ok"] is False
    assert result["outcome"] == "abstained"
    assert result["ambiguous"] is True
    assert result["clarification_options"] == [
        "how many orders by status?",
        "how many orders by city?",
    ]
    # The ambiguity check ran instead of, not before, generation.
    assert fake.calls == []


def test_a_fake_with_no_confidence_or_ambiguity_support_still_answers(
    client, relational_workbook
):
    """A test double that predates C-2 (no ``confidence`` key, no
    ``check_ambiguity`` method) must keep answering exactly as it always did —
    the same graceful-degradation contract ``suggest_followups`` already has."""

    from app.api import services

    session_id = _exported(client, relational_workbook)
    fake = _ScriptedClaude(
        [{"sql": "SELECT COUNT(*) AS n FROM orders", "explanation": "counts orders"}]
    )
    assert not hasattr(fake, "check_ambiguity")

    with _db() as db:
        record = db.get(IngestionSession, session_id)
        result = services.answer_question(db, record, "how many orders?", claude=fake)

    assert result["ok"] is True
    assert result["outcome"] == "answered"


# ---------------------------------------------------------------------------
# the third retrieval index, wired end to end (C-1)
# ---------------------------------------------------------------------------


def test_a_business_rule_is_retrieved_into_generation(client, relational_workbook):
    from app.api import services

    session_id = _exported(client, relational_workbook)
    with _db() as db:
        record = db.get(IngestionSession, session_id)
        services.add_business_rule(db, record, "only orders with status 'shipped' count as revenue")
        db.commit()

    fake = _ScriptedClaude(
        [{"sql": "SELECT COUNT(*) AS n FROM orders", "explanation": "counts orders"}]
    )
    with _db() as db:
        record = db.get(IngestionSession, session_id)
        services.answer_question(db, record, "how many orders count as revenue?", claude=fake)

    assert fake.calls[0]["business_rules"] == [
        "only orders with status 'shipped' count as revenue"
    ]


def test_a_saved_example_is_retrieved_into_a_later_generation(
    client, relational_workbook, monkeypatch
):
    from app.api import services
    from app.core.config import get_settings

    # Which embedder the suite runs on decides how alike two wordings score; what is
    # under test is that an example that clears the floor reaches the model.
    monkeypatch.setattr(get_settings(), "query_example_min_similarity", 0.0)
    session_id = _exported(client, relational_workbook)
    with _db() as db:
        record = db.get(IngestionSession, session_id)
        services.save_query_example(
            db, record, "how many orders are there?", "SELECT COUNT(*) AS n FROM orders"
        )
        db.commit()

    fake = _ScriptedClaude(
        [{"sql": "SELECT COUNT(*) AS n FROM orders", "explanation": "counts orders"}]
    )
    with _db() as db:
        record = db.get(IngestionSession, session_id)
        services.answer_question(db, record, "how many orders total?", claude=fake)

    assert fake.calls[0]["examples"] == [
        {"question": "how many orders are there?", "sql": "SELECT COUNT(*) AS n FROM orders"}
    ]


def test_an_example_below_the_similarity_floor_is_not_sent_to_the_model(
    client, relational_workbook, monkeypatch
):
    from app.api import services
    from app.core.config import get_settings

    session_id = _exported(client, relational_workbook)
    with _db() as db:
        record = db.get(IngestionSession, session_id)
        services.save_query_example(
            db, record, "how many orders are there?", "SELECT COUNT(*) AS n FROM orders"
        )
        db.commit()

    # Nothing can be more similar than 1.0: the one saved example is, by construction,
    # never similar enough, so the prompt carries no examples at all.
    monkeypatch.setattr(get_settings(), "query_example_min_similarity", 1.01)
    monkeypatch.setattr(get_settings(), "query_example_match_threshold", 1.01)
    fake = _ScriptedClaude(
        [{"sql": "SELECT COUNT(*) AS n FROM orders", "explanation": "counts orders"}]
    )
    with _db() as db:
        record = db.get(IngestionSession, session_id)
        result = services.answer_question(db, record, "how many orders total?", claude=fake)

    assert result["ok"] is True
    assert fake.calls[0]["examples"] == []


def test_a_close_example_is_shown_even_when_the_floor_is_set_above_it(
    client, relational_workbook, monkeypatch
):
    from app.api import services
    from app.core.config import get_settings

    session_id = _exported(client, relational_workbook)
    with _db() as db:
        record = db.get(IngestionSession, session_id)
        services.save_query_example(
            db, record, "how many orders are there?", "SELECT COUNT(*) AS n FROM orders"
        )
        db.commit()

    monkeypatch.setattr(get_settings(), "query_example_min_similarity", 1.01)
    monkeypatch.setattr(get_settings(), "query_example_match_threshold", 0.0)
    fake = _ScriptedClaude(
        [{"sql": "SELECT COUNT(*) AS n FROM orders", "explanation": "counts orders"}]
    )
    with _db() as db:
        record = db.get(IngestionSession, session_id)
        services.answer_question(db, record, "how many orders total?", claude=fake)

    assert [e["question"] for e in fake.calls[0]["examples"]] == ["how many orders are there?"]


def test_business_rule_and_example_routes_round_trip(client, relational_workbook):
    session_id = _exported(client, relational_workbook)

    created = client.post(
        f"/api/sessions/{session_id}/business-rules",
        json={"rule_text": "returns are never counted as revenue"},
    )
    assert created.status_code == 201, created.text
    rule_id = created.json()["id"]

    listed = client.get(f"/api/sessions/{session_id}/business-rules")
    assert [r["rule_text"] for r in listed.json()["rules"]] == [
        "returns are never counted as revenue"
    ]

    deleted = client.delete(f"/api/sessions/{session_id}/business-rules/{rule_id}")
    assert deleted.status_code == 204
    assert client.get(f"/api/sessions/{session_id}/business-rules").json()["rules"] == []

    saved = client.post(
        f"/api/sessions/{session_id}/examples",
        json={"question": "how many orders?", "sql": "SELECT COUNT(*) AS n FROM orders"},
    )
    assert saved.status_code == 201, saved.text
    example_id = saved.json()["id"]
    assert saved.json()["verified"] is True

    examples = client.get(f"/api/sessions/{session_id}/examples")
    assert len(examples.json()["examples"]) == 1

    deleted_example = client.delete(f"/api/sessions/{session_id}/examples/{example_id}")
    assert deleted_example.status_code == 204
    assert client.get(f"/api/sessions/{session_id}/examples").json()["examples"] == []


def test_an_example_with_sql_outside_the_export_is_rejected(client, relational_workbook):
    session_id = _exported(client, relational_workbook)

    response = client.post(
        f"/api/sessions/{session_id}/examples",
        json={"question": "list every user", "sql": "SELECT * FROM users"},
    )

    assert response.status_code == 409


# ---------------------------------------------------------------------------
# conversation memory, one clarification at most, and verified examples that
# actually reproduce their SQL
# ---------------------------------------------------------------------------

ORDERS_SQL = "SELECT COUNT(*) AS n FROM orders"


class _ConversationalClaude(_ScriptedClaude):
    """Reads a follow-up the way a model would: rewrites it, and records what it was shown."""

    def __init__(self, answers, rewrites):
        super().__init__(answers)
        self._rewrites = list(rewrites)
        self.rewrite_calls: list[dict] = []

    def rewrite_question(self, question, conversation):
        self.rewrite_calls.append({"question": question, "conversation": conversation})
        return self._rewrites.pop(0) if self._rewrites else None


def _ask(session_id, question, claude):
    from app.api import services

    with _db() as db:
        record = db.get(IngestionSession, session_id)
        result = services.answer_question(db, record, question, claude=claude)
        db.commit()
    return result


def _save(session_id, question, sql=ORDERS_SQL, **kwargs):
    from app.api import services

    with _db() as db:
        record = db.get(IngestionSession, session_id)
        example = services.save_query_example(db, record, question, sql, **kwargs)
        db.commit()
        return example.id


def test_a_saved_question_is_answered_from_its_saved_sql_without_asking_the_model(
    client, relational_workbook
):
    session_id = _exported(client, relational_workbook)
    _save(session_id, "how many orders are there?", explanation="counts every order")

    # A model that would both flag ambiguity and write different SQL — neither
    # may be consulted for a question the user already verified.
    fake = _AmbiguousClaude(
        [{"sql": "SELECT 1 AS n", "explanation": "wrong"}], ["a?", "b?"]
    )
    result = _ask(session_id, "How many orders are there", fake)

    assert result["ok"] is True
    assert result["source"] == "saved_example"
    assert result["sql"].upper().startswith("SELECT COUNT(*)")
    assert result["explanation"] == "counts every order"
    assert result["confidence"] == 1.0
    assert result["row_count"] == 1 and result["columns"] == ["n"]
    assert fake.calls == []
    assert fake.ambiguity_checks == []


def test_only_the_same_question_reuses_saved_sql_not_a_similar_one(client, relational_workbook):
    session_id = _exported(client, relational_workbook)
    _save(session_id, "top 5 customers by orders", "SELECT customer_id FROM orders LIMIT 5")

    fake = _ScriptedClaude(
        [{"sql": "SELECT customer_id FROM orders LIMIT 10", "explanation": "ten"}]
    )
    result = _ask(session_id, "top 10 customers by orders", fake)

    assert result["ok"] is True
    assert result.get("source") != "saved_example"
    assert len(fake.calls) == 1


def test_saved_sql_that_no_longer_runs_falls_back_to_generation(client, relational_workbook):
    session_id = _exported(client, relational_workbook)
    example_id = _save(session_id, "how many orders are there?")
    from app.db.models import QueryExample

    with _db() as db:
        db.get(QueryExample, example_id).sql = "SELECT COUNT(*) AS n FROM no_such_table"
        db.commit()

    fake = _ScriptedClaude([{"sql": ORDERS_SQL, "explanation": "counts orders"}])
    result = _ask(session_id, "how many orders are there?", fake)

    assert result["ok"] is True
    assert result.get("source") != "saved_example"
    assert len(fake.calls) == 1


def test_saving_the_same_question_again_replaces_the_sql_instead_of_duplicating(
    client, relational_workbook
):
    from app.api import services

    session_id = _exported(client, relational_workbook)
    first = _save(session_id, "how many orders are there?")
    second = _save(
        session_id, "How many orders are there", "SELECT COUNT(order_id) AS n FROM orders"
    )

    assert first == second
    with _db() as db:
        examples = services.list_query_examples(db, session_id)
    assert len(examples) == 1
    assert "order_id" in examples[0].sql


def test_an_ambiguous_question_the_user_settled_is_remembered_under_both_wordings(
    client, relational_workbook
):
    session_id = _exported(client, relational_workbook)
    vague, settled = "how many orders?", "how many orders are there in total?"

    ambiguous = _AmbiguousClaude([{"sql": ORDERS_SQL, "explanation": "all"}], [settled])
    asked = _ask(session_id, vague, ambiguous)
    assert asked["outcome"] == "abstained" and asked["ambiguous"] is True

    # Picking the offered reading answers it — and is not itself second-guessed.
    answered = _ask(session_id, settled, ambiguous)
    assert answered["ok"] is True
    assert answered["clarifies"] == vague
    assert ambiguous.ambiguity_checks == [vague]

    _save(
        session_id,
        answered["resolved_question"],
        answered["sql"],
        also_asked_as=[answered["clarifies"]],
    )

    # The vague wording alone now lands on the verified SQL, with no question back.
    again = _ask(session_id, vague, _AmbiguousClaude([], [settled]))
    assert again["ok"] is True
    assert again["source"] == "saved_example"


def test_a_free_text_reply_to_a_clarification_is_not_asked_about_again(
    client, relational_workbook
):
    session_id = _exported(client, relational_workbook)

    class _Claude(_AmbiguousClaude):
        rewrite_calls: list = []

        def rewrite_question(self, question, conversation):
            self.rewrite_calls.append({"question": question, "conversation": conversation})
            return {"question": "how many orders are there in total?", "follow_up": True}

    fake = _Claude(
        [{"sql": ORDERS_SQL, "explanation": "all orders"}], ["by status", "by city"]
    )
    fake.rewrite_calls = []
    first = _ask(session_id, "how many orders?", fake)
    assert first["ambiguous"] is True

    reply = _ask(session_id, "just the total", fake)

    assert reply["ok"] is True
    assert reply["resolved_question"] == "how many orders are there in total?"
    assert reply["clarifies"] == "how many orders?"
    assert fake.ambiguity_checks == ["how many orders?"]  # never asked a second time
    assert fake.calls[0]["question"] == "how many orders are there in total?"
    # The model was shown the clarification it had asked, to make sense of the reply.
    shown = fake.rewrite_calls[0]["conversation"]
    assert shown[-1]["outcome"] == "clarification requested"
    assert shown[-1]["options_offered"] == ["by status", "by city"]


def test_a_follow_up_is_made_whole_from_the_previous_answer(client, relational_workbook):
    session_id = _exported(client, relational_workbook)
    fake = _ConversationalClaude(
        [
            {"sql": ORDERS_SQL, "explanation": "counts orders"},
            {
                "sql": "SELECT status, COUNT(*) AS n FROM orders GROUP BY status",
                "explanation": "orders per status",
            },
        ],
        rewrites=[{"question": "how many orders are there by status?", "follow_up": True}],
    )

    first = _ask(session_id, "how many orders are there?", fake)
    assert first["ok"] is True
    assert fake.rewrite_calls == []  # nothing came before the first question

    second = _ask(session_id, "and by status?", fake)

    assert second["ok"] is True
    assert second["follow_up"] is True
    assert second["resolved_question"] == "how many orders are there by status?"
    # The model is shown the previous question, what it computed and its SQL — no rows.
    shown = fake.rewrite_calls[0]["conversation"]
    assert shown == [
        {
            "question": "how many orders are there?",
            "outcome": "answered",
            "computed": "counts orders",
            "sql": first["sql"],
        }
    ]
    assert fake.calls[1]["question"] == "how many orders are there by status?"
    assert fake.calls[1]["conversation"] == shown
    # The follow-up reads the table its predecessor read.
    assert "orders" in second["tables_used"]


def test_a_question_that_stands_alone_is_not_treated_as_a_follow_up(client, relational_workbook):
    session_id = _exported(client, relational_workbook)
    fake = _ConversationalClaude(
        [
            {"sql": ORDERS_SQL, "explanation": "counts orders"},
            {"sql": "SELECT city, COUNT(*) AS n FROM customers GROUP BY city", "explanation": "x"},
        ],
        rewrites=[{"question": "something else entirely", "follow_up": False}],
    )
    _ask(session_id, "how many orders are there?", fake)
    second = _ask(session_id, "how many customers per city?", fake)

    assert second["follow_up"] is False
    # A rewrite the model did not call for never replaces what the user typed.
    assert second["resolved_question"] == "how many customers per city?"
    assert fake.calls[1]["question"] == "how many customers per city?"
    assert fake.calls[1]["conversation"] is None


def test_a_close_saved_example_skips_the_ambiguity_check(client, relational_workbook, monkeypatch):
    from app.core.config import get_settings

    session_id = _exported(client, relational_workbook)
    _save(session_id, "how many orders are there?")
    monkeypatch.setattr(get_settings(), "query_example_match_threshold", 0.0)

    fake = _AmbiguousClaude([{"sql": ORDERS_SQL, "explanation": "counts orders"}], ["a", "b"])
    result = _ask(session_id, "number of orders overall", fake)

    assert result["ok"] is True
    assert fake.ambiguity_checks == []
    assert len(fake.calls) == 1


def test_the_ambiguity_check_is_shown_the_columns_not_only_the_table_names(
    client, relational_workbook
):
    session_id = _exported(client, relational_workbook)
    seen: dict = {}

    class _Spy(_ScriptedClaude):
        def check_ambiguity(self, question, tables, business_rules=None):
            seen["tables"] = tables
            return {"ambiguous": False, "reason": "", "options": []}

    _ask(session_id, "how many orders are there?", _Spy([{"sql": ORDERS_SQL, "explanation": "x"}]))

    orders = next(t for t in seen["tables"] if t["name"] == "orders")
    assert any(entry.startswith("total_amount") for entry in orders["columns"])


def test_example_routes_carry_the_explanation_and_the_settled_wording(client, relational_workbook):
    session_id = _exported(client, relational_workbook)

    saved = client.post(
        f"/api/sessions/{session_id}/examples",
        json={
            "question": "how many orders are there in total?",
            "sql": ORDERS_SQL,
            "explanation": "counts every order",
            "also_asked_as": ["how many orders?"],
        },
    )
    assert saved.status_code == 201, saved.text
    assert saved.json()["explanation"] == "counts every order"

    questions = {
        e["question"] for e in client.get(f"/api/sessions/{session_id}/examples").json()["examples"]
    }
    assert questions == {"how many orders are there in total?", "how many orders?"}


# -- pure helpers ----------------------------------------------------------


def test_normalize_question_ignores_case_and_punctuation_but_not_numbers():
    from app.query.retrieval import normalize_question

    assert normalize_question("  How many Orders?! ") == normalize_question("how many orders")
    assert normalize_question("top 5 customers") != normalize_question("top 10 customers")


def test_rank_examples_reports_similarity_and_drops_the_embedding():
    from app.query.retrieval import rank_examples

    ranked = rank_examples(
        [1.0, 0.0],
        [
            {"question": "far", "sql": "SELECT 1", "embedding": [0.0, 1.0]},
            {"question": "near", "sql": "SELECT 2", "embedding": [1.0, 0.0]},
        ],
    )
    assert [row["question"] for row in ranked] == ["near", "far"]
    assert ranked[0]["similarity"] == pytest.approx(1.0)
    assert "embedding" not in ranked[0]


def test_select_columns_keeps_the_columns_an_example_uses_however_they_score():
    rows = [
        {"table_name": "t", "column_name": f"c{i}", "embedding": [0.0, 1.0]} for i in range(5)
    ] + [{"table_name": "t", "column_name": "rare_metric", "embedding": [0.0, 1.0]}]

    without = select_columns([1.0, 0.0], rows, ["t"], max_ranked=2)
    with_it = select_columns(
        [1.0, 0.0], rows, ["t"], max_ranked=2, include_columns={"rare_metric"}
    )

    assert "rare_metric" not in {r["column_name"] for r in without}
    assert "rare_metric" in {r["column_name"] for r in with_it}


def test_referenced_names_lists_tables_and_columns_but_not_ctes():
    from app.query.guard import referenced_names

    tables, columns = referenced_names(
        "WITH big AS (SELECT order_id, total_amount FROM orders) "
        "SELECT c.city, b.total_amount FROM customers c JOIN big b ON b.order_id = c.customer_id",
        "postgresql",
    )
    assert tables == {"orders", "customers"}
    assert {"city", "total_amount", "order_id"} <= columns
    assert referenced_names("not sql at all (", "postgresql") == (set(), set())


# ---------------------------------------------------------------------------
# a column a rule names is a column the question needs
# ---------------------------------------------------------------------------


def _columns(*pairs):
    """``sem_metadata`` rows: (exported name, the source's spelling)."""

    return [
        {"table_name": "t", "column_name": name, "source_column": source} for name, source in pairs
    ]


def test_a_rule_names_a_column_by_its_exported_or_its_original_spelling():
    rows = _columns(
        ("t_bil", "T-BIL"),
        ("admission", "Admission"),
        ("examination_date", "Examination Date"),
        ("notes", "Notes"),
    )

    named = columns_named_in(
        ["followed up at the outpatient clinic refers to Admission = '-'; T-BIL < 2.0"],
        rows,
        {"t"},
    )
    assert named == {"admission", "t_bil"}
    assert columns_named_in(["`Examination Date` > '1997-01-01'"], rows, {"t"}) == {"examination_date"}
    assert columns_named_in(["order by examination_date"], rows, {"t"}) == {"examination_date"}


def test_a_name_may_be_written_run_together_or_apart():
    rows = _columns(("fastestlapspeed", "fastestLapSpeed"))

    assert columns_named_in(["refers to (MAX) fastestLapSpeed"], rows, {"t"}) == {"fastestlapspeed"}
    assert columns_named_in(["the fastest lap speed is what counts"], rows, {"t"}) == {"fastestlapspeed"}


def test_a_name_inside_a_longer_word_is_not_a_mention_of_it():
    rows = _columns(("sex", "Sex"), ("date", "Date"))

    assert columns_named_in(["sexual health and data quality"], rows, {"t"}) == set()
    assert columns_named_in(["split by sex and date"], rows, {"t"}) == {"sex", "date"}


def test_a_name_too_short_to_mean_anything_is_never_matched():
    rows = _columns(("id", "ID"), ("no", "No"), ("age", "Age"))

    assert columns_named_in(["the id is no good for the age"], rows, {"t"}) == {"age"}


def test_only_the_columns_of_the_given_tables_are_named():
    rows = [
        {"table_name": "a", "column_name": "status", "source_column": "Status"},
        {"table_name": "b", "column_name": "status", "source_column": "Status"},
        {"table_name": "b", "column_name": "region", "source_column": "Region"},
    ]

    assert columns_named_in(["status and region"], rows, {"a"}) == {"status"}
    assert columns_named_in([], rows, {"a", "b"}) == set()
    assert columns_named_in(["status"], [], {"a"}) == set()


def test_a_non_latin_name_is_matched_too():
    rows = _columns(("prenom", "Prénom"), ("মূল্য", "মূল্য"))

    assert columns_named_in(["le prénom est requis"], rows, {"t"}) == {"prenom"}


def test_a_column_a_stored_rule_names_is_sent_to_the_model_however_it_scores(
    client, relational_workbook, monkeypatch
):
    from app.api import services
    from app.core.config import get_settings

    session_id = _exported(client, relational_workbook)
    monkeypatch.setattr(get_settings(), "query_max_ranked_columns", 0)  # keys, and what a rule names

    def sent(fake):
        return {column["name"] for table in fake.calls[0]["schema"] for column in table["columns"]}

    plain = _ScriptedClaude([{"sql": ORDERS_SQL, "explanation": "counts orders"}])
    _ask(session_id, "how many orders are there?", plain)
    assert "total_amount" not in sent(plain), "ranked columns were capped at none"

    with _db() as db:
        record = db.get(IngestionSession, session_id)
        services.add_business_rule(db, record, "revenue is the total_amount of the orders")
        db.commit()
    ruled = _ScriptedClaude([{"sql": ORDERS_SQL, "explanation": "counts orders"}])
    _ask(session_id, "how many orders are there?", ruled)

    assert "total_amount" in sent(ruled)
    assert ruled.calls[0]["business_rules"] == ["revenue is the total_amount of the orders"]


def test_a_rule_given_with_one_question_is_read_like_a_recorded_one_and_not_kept(
    client, relational_workbook, monkeypatch
):
    from app.api import services
    from app.core.config import get_settings

    session_id = _exported(client, relational_workbook)
    monkeypatch.setattr(get_settings(), "query_max_ranked_columns", 0)
    fake = _ScriptedClaude([{"sql": ORDERS_SQL, "explanation": "counts orders"}])

    with _db() as db:
        record = db.get(IngestionSession, session_id)
        services.answer_question(
            db,
            record,
            "how many orders are there?",
            claude=fake,
            extra_rules=["the amount to use is total_amount"],
        )
        db.commit()

    call = fake.calls[0]
    assert call["business_rules"] == ["the amount to use is total_amount"]
    assert "total_amount" in {c["name"] for t in call["schema"] for c in t["columns"]}
    with _db() as db:
        assert services.list_business_rules(db, session_id) == []
