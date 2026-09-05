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
    expand_by_references,
    rank_tables,
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

    def generate_sql(self, question, dialect, schema, prior_sql=None, prior_error=None):
        self.calls.append(
            {
                "question": question,
                "dialect": dialect,
                "schema": schema,
                "prior_sql": prior_sql,
                "prior_error": prior_error,
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
