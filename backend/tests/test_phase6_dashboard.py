"""Phase 6 — the pinned dashboard.

Two layers, tested at the level each one actually decides something:

* :func:`app.api.services.pin_dashboard_card` /
  :func:`~app.api.services.list_dashboard_cards` — orchestration against the
  real relational fixture, the same pattern ``test_phase5_query.py`` uses for
  ``answer_question``.  The one thing worth re-proving here that Phase 5
  already proved for the ask-time guard is that pinning re-validates too: a
  pin request is client-supplied input like any other, and it is the one path
  that turns a stored statement into something that runs unattended on every
  future dashboard load.
* the routes — ownership, status codes, and that the history sidebar and the
  suggested-follow-ups field are actually wired into ``/query``.
"""

from __future__ import annotations

import pytest

from app.db.base import get_session_factory
from app.db.models import IngestionSession


def _db():
    return get_session_factory()()


def _uploaded(client, workbook) -> str:
    session_id = client.post("/api/sessions", json={"name": "phase 6"}).json()["id"]
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


class _UnavailableClaude:
    available = False
    last_error = "ANTHROPIC_API_KEY is not set"


class _ScriptedClaude:
    """Same fake ``test_phase5_query.py`` uses — deliberately has no
    ``suggest_followups``, so it also proves that omission degrades cleanly."""

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
    ):
        self.calls.append({"question": question})
        if not self._answers:
            return None
        return self._answers.pop(0)


class _ClaudeWithSuggestions(_ScriptedClaude):
    def __init__(self, answers, suggestions):
        super().__init__(answers)
        self._suggestions = suggestions

    def suggest_followups(self, question, sql, columns, tables):
        return list(self._suggestions)


ORDER_COUNT_SQL = "SELECT COUNT(*) AS n FROM orders"
ORDER_DATE_SQL = "SELECT order_date, total_amount FROM orders"

METRIC_VIZ = {"chart": "metric", "numeric_columns": [], "date_columns": []}
LINE_VIZ = {"chart": "line", "numeric_columns": ["total_amount"], "date_columns": ["order_date"]}


# ---------------------------------------------------------------------------
# pin_dashboard_card
# ---------------------------------------------------------------------------


def test_pinning_stores_the_validated_sql_and_defaults_the_title(client, relational_workbook):
    from app.api import services

    session_id = _exported(client, relational_workbook)
    with _db() as db:
        record = db.get(IngestionSession, session_id)
        card = services.pin_dashboard_card(
            db,
            record,
            {"question": "how many orders?", "sql": ORDER_COUNT_SQL, "visualization": METRIC_VIZ},
        )

    assert card.title == "how many orders?"
    assert card.sql.upper().startswith("SELECT")
    assert card.position == 0


def test_pinning_a_second_card_increments_position(client, relational_workbook):
    from app.api import services

    session_id = _exported(client, relational_workbook)
    with _db() as db:
        record = db.get(IngestionSession, session_id)
        services.pin_dashboard_card(
            db, record, {"question": "q1", "sql": ORDER_COUNT_SQL, "visualization": METRIC_VIZ}
        )
        second = services.pin_dashboard_card(
            db, record, {"question": "q2", "sql": ORDER_COUNT_SQL, "visualization": METRIC_VIZ}
        )

    assert second.position == 1


def test_a_custom_title_overrides_the_question(client, relational_workbook):
    from app.api import services

    session_id = _exported(client, relational_workbook)
    with _db() as db:
        record = db.get(IngestionSession, session_id)
        card = services.pin_dashboard_card(
            db,
            record,
            {
                "question": "how many orders?",
                "sql": ORDER_COUNT_SQL,
                "title": "Order volume",
                "visualization": METRIC_VIZ,
            },
        )

    assert card.title == "Order volume"


def test_pinning_rejects_a_table_outside_the_export(client, relational_workbook):
    from app.api import services

    session_id = _exported(client, relational_workbook)
    with _db() as db:
        record = db.get(IngestionSession, session_id)
        with pytest.raises(ValueError, match="no longer be pinned"):
            services.pin_dashboard_card(
                db,
                record,
                {"question": "list users", "sql": "SELECT * FROM users", "visualization": {}},
            )


def test_pinning_rejects_a_write_statement(client, relational_workbook):
    from app.api import services

    session_id = _exported(client, relational_workbook)
    with _db() as db:
        record = db.get(IngestionSession, session_id)
        with pytest.raises(ValueError, match="no longer be pinned"):
            services.pin_dashboard_card(
                db,
                record,
                {"question": "wipe orders", "sql": "DELETE FROM orders", "visualization": {}},
            )


def test_pinning_before_export_is_refused(client, relational_workbook):
    from app.api import services

    session_id = _uploaded(client, relational_workbook)
    with _db() as db:
        record = db.get(IngestionSession, session_id)
        with pytest.raises(ValueError, match="export"):
            services.pin_dashboard_card(
                db, record, {"question": "q", "sql": ORDER_COUNT_SQL, "visualization": {}}
            )


# ---------------------------------------------------------------------------
# list_dashboard_cards
# ---------------------------------------------------------------------------


def test_listing_re_executes_the_card_and_returns_fresh_rows(client, relational_workbook):
    from app.api import services

    session_id = _exported(client, relational_workbook)
    with _db() as db:
        record = db.get(IngestionSession, session_id)
        services.pin_dashboard_card(
            db,
            record,
            {"question": "how many orders?", "sql": ORDER_COUNT_SQL, "visualization": METRIC_VIZ},
        )
        db.commit()

    with _db() as db:
        record = db.get(IngestionSession, session_id)
        cards = services.list_dashboard_cards(db, record)

    assert len(cards) == 1
    assert cards[0]["ok"] is True
    assert cards[0]["rows"][0]["n"] == 45  # the fixture's order count
    assert cards[0]["refreshed_at"]


def test_a_card_that_no_longer_executes_is_reported_without_breaking_the_others(
    client, relational_workbook
):
    """Simulates schema drift after pinning (e.g. a later re-export renamed a
    column): the stored SQL is mutated directly, bypassing the pin-time guard,
    exactly as a card that has simply outlived its schema would look."""

    from app.api import services

    session_id = _exported(client, relational_workbook)
    with _db() as db:
        record = db.get(IngestionSession, session_id)
        good = services.pin_dashboard_card(
            db, record, {"question": "good", "sql": ORDER_COUNT_SQL, "visualization": METRIC_VIZ}
        )
        stale = services.pin_dashboard_card(
            db, record, {"question": "stale", "sql": ORDER_COUNT_SQL, "visualization": METRIC_VIZ}
        )
        stale.sql = "SELECT nonexistent_column FROM orders"
        good_id, stale_id = good.id, stale.id
        db.commit()

    with _db() as db:
        record = db.get(IngestionSession, session_id)
        cards = {c["id"]: c for c in services.list_dashboard_cards(db, record)}

    assert cards[good_id]["ok"] is True
    assert cards[stale_id]["ok"] is False
    assert cards[stale_id]["error"]


def test_date_range_filters_a_card_with_exactly_one_date_column(client, relational_workbook):
    from datetime import date

    from app.api import services

    session_id = _exported(client, relational_workbook)
    with _db() as db:
        record = db.get(IngestionSession, session_id)
        services.pin_dashboard_card(
            db, record, {"question": "orders over time", "sql": ORDER_DATE_SQL, "visualization": LINE_VIZ}
        )
        db.commit()

    with _db() as db:
        record = db.get(IngestionSession, session_id)
        unfiltered = services.list_dashboard_cards(db, record)[0]
        filtered = services.list_dashboard_cards(
            db, record, start=date(2024, 1, 1), end=date(2024, 7, 1)
        )[0]

    assert unfiltered["date_filtered"] is False
    assert filtered["date_filtered"] is True
    assert 0 < filtered["row_count"] < unfiltered["row_count"]


def test_date_range_leaves_a_card_with_no_date_column_unfiltered(client, relational_workbook):
    from datetime import date

    from app.api import services

    session_id = _exported(client, relational_workbook)
    with _db() as db:
        record = db.get(IngestionSession, session_id)
        services.pin_dashboard_card(
            db, record, {"question": "how many orders?", "sql": ORDER_COUNT_SQL, "visualization": METRIC_VIZ}
        )
        db.commit()

    with _db() as db:
        record = db.get(IngestionSession, session_id)
        result = services.list_dashboard_cards(db, record, start=date(2024, 1, 1), end=date(2024, 7, 1))[0]

    assert result["date_filtered"] is False
    assert result["rows"][0]["n"] == 45


# ---------------------------------------------------------------------------
# update / delete
# ---------------------------------------------------------------------------


def test_updating_a_cards_title_and_layout(client, relational_workbook):
    from app.api import services

    session_id = _exported(client, relational_workbook)
    with _db() as db:
        record = db.get(IngestionSession, session_id)
        card = services.pin_dashboard_card(
            db, record, {"question": "q", "sql": ORDER_COUNT_SQL, "visualization": METRIC_VIZ}
        )
        card_id = card.id
        db.commit()

    with _db() as db:
        record = db.get(IngestionSession, session_id)
        updated = services.update_dashboard_card(
            db, record, card_id, {"title": "Renamed", "layout": {"x": 1, "y": 2, "w": 3, "h": 4}}
        )

    assert updated.title == "Renamed"
    assert updated.layout == {"x": 1, "y": 2, "w": 3, "h": 4}


def test_updating_an_unknown_card_raises(client, relational_workbook):
    from app.api import services

    session_id = _exported(client, relational_workbook)
    with _db() as db:
        record = db.get(IngestionSession, session_id)
        with pytest.raises(KeyError):
            services.update_dashboard_card(db, record, "does-not-exist", {"title": "x"})


def test_deleting_a_card_removes_it_from_the_list(client, relational_workbook):
    from app.api import services

    session_id = _exported(client, relational_workbook)
    with _db() as db:
        record = db.get(IngestionSession, session_id)
        card = services.pin_dashboard_card(
            db, record, {"question": "q", "sql": ORDER_COUNT_SQL, "visualization": METRIC_VIZ}
        )
        card_id = card.id
        db.commit()

    with _db() as db:
        record = db.get(IngestionSession, session_id)
        services.delete_dashboard_card(db, record, card_id)
        db.commit()

    with _db() as db:
        record = db.get(IngestionSession, session_id)
        assert services.list_dashboard_cards(db, record) == []


# ---------------------------------------------------------------------------
# question history + suggestions, via answer_question
# ---------------------------------------------------------------------------


def test_every_question_is_recorded_ok_or_not(client, relational_workbook):
    from app.api import services

    session_id = _exported(client, relational_workbook)
    with _db() as db:
        record = db.get(IngestionSession, session_id)
        services.answer_question(db, record, "a question with no model", claude=_UnavailableClaude())
        db.commit()

    with _db() as db:
        history = services.question_history(db, session_id)

    assert len(history) == 1
    assert history[0]["question"] == "a question with no model"
    assert history[0]["ok"] is False


def test_history_trims_to_the_configured_limit(client, relational_workbook):
    from app.api import services
    from app.core.config import get_settings

    session_id = _exported(client, relational_workbook)
    limit = get_settings().query_history_limit
    with _db() as db:
        record = db.get(IngestionSession, session_id)
        for i in range(limit + 5):
            services.answer_question(db, record, f"question {i}", claude=_UnavailableClaude())
        db.commit()

    with _db() as db:
        history = services.question_history(db, session_id)

    assert len(history) == limit
    # Newest first, oldest ones evicted.
    assert history[0]["question"] == f"question {limit + 4}"
    assert all(row["question"] != "question 0" for row in history)


def test_a_successful_answer_carries_claudes_suggested_followups(client, relational_workbook):
    from app.api import services

    session_id = _exported(client, relational_workbook)
    fake = _ClaudeWithSuggestions(
        [{"sql": ORDER_COUNT_SQL, "explanation": "counts orders"}],
        ["how many orders shipped?", "which city orders most?", "orders by month?"],
    )
    with _db() as db:
        record = db.get(IngestionSession, session_id)
        result = services.answer_question(db, record, "how many orders?", claude=fake)

    assert result["ok"] is True
    assert len(result["suggestions"]) == 3


def test_suggestions_default_to_empty_without_the_method(client, relational_workbook):
    """``_ScriptedClaude`` (same fake Phase 5's tests use) has no
    ``suggest_followups`` — the service must not blow up on that, and every
    Phase 5 test already relies on this implicitly by continuing to pass."""

    from app.api import services

    session_id = _exported(client, relational_workbook)
    fake = _ScriptedClaude([{"sql": ORDER_COUNT_SQL, "explanation": "counts orders"}])
    with _db() as db:
        record = db.get(IngestionSession, session_id)
        result = services.answer_question(db, record, "how many orders?", claude=fake)

    assert result["ok"] is True
    assert result["suggestions"] == []


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------


def test_the_pin_route_stores_and_the_list_route_re_executes(client, relational_workbook):
    session_id = _exported(client, relational_workbook)

    pin = client.post(
        f"/api/sessions/{session_id}/dashboard/cards",
        json={"question": "how many orders?", "sql": ORDER_COUNT_SQL, "visualization": METRIC_VIZ},
    )
    assert pin.status_code == 201, pin.text
    card_id = pin.json()["id"]

    listed = client.get(f"/api/sessions/{session_id}/dashboard/cards")
    assert listed.status_code == 200
    cards = listed.json()["cards"]
    assert len(cards) == 1
    assert cards[0]["id"] == card_id
    assert cards[0]["ok"] is True
    assert cards[0]["rows"][0]["n"] == 45


def test_the_pin_route_rejects_a_query_outside_the_export(client, relational_workbook):
    session_id = _exported(client, relational_workbook)

    response = client.post(
        f"/api/sessions/{session_id}/dashboard/cards",
        json={"question": "list users", "sql": "SELECT * FROM users"},
    )

    assert response.status_code == 409


def test_the_pin_route_requires_export_first(client, relational_workbook):
    session_id = _uploaded(client, relational_workbook)

    response = client.post(
        f"/api/sessions/{session_id}/dashboard/cards",
        json={"question": "q", "sql": ORDER_COUNT_SQL},
    )

    assert response.status_code == 409


def test_dashboard_routes_belong_to_their_owner(client, other_client, relational_workbook):
    session_id = _exported(client, relational_workbook)
    pin = client.post(
        f"/api/sessions/{session_id}/dashboard/cards",
        json={"question": "q", "sql": ORDER_COUNT_SQL, "visualization": METRIC_VIZ},
    )
    card_id = pin.json()["id"]

    assert other_client.get(f"/api/sessions/{session_id}/dashboard/cards").status_code == 404
    assert (
        other_client.post(
            f"/api/sessions/{session_id}/dashboard/cards",
            json={"question": "q", "sql": ORDER_COUNT_SQL},
        ).status_code
        == 404
    )
    assert (
        other_client.patch(
            f"/api/sessions/{session_id}/dashboard/cards/{card_id}", json={"title": "mine now"}
        ).status_code
        == 404
    )
    assert (
        other_client.delete(f"/api/sessions/{session_id}/dashboard/cards/{card_id}").status_code
        == 404
    )


def test_the_patch_route_updates_title_and_layout(client, relational_workbook):
    session_id = _exported(client, relational_workbook)
    pin = client.post(
        f"/api/sessions/{session_id}/dashboard/cards",
        json={"question": "q", "sql": ORDER_COUNT_SQL, "visualization": METRIC_VIZ},
    )
    card_id = pin.json()["id"]

    response = client.patch(
        f"/api/sessions/{session_id}/dashboard/cards/{card_id}",
        json={"title": "Renamed", "layout": {"x": 0, "y": 0, "w": 6, "h": 3}},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["title"] == "Renamed"
    assert payload["layout"]["w"] == 6


def test_the_patch_route_404s_for_an_unknown_card(client, relational_workbook):
    session_id = _exported(client, relational_workbook)

    response = client.patch(
        f"/api/sessions/{session_id}/dashboard/cards/does-not-exist", json={"title": "x"}
    )

    assert response.status_code == 404


def test_the_delete_route_unpins_a_card(client, relational_workbook):
    session_id = _exported(client, relational_workbook)
    pin = client.post(
        f"/api/sessions/{session_id}/dashboard/cards",
        json={"question": "q", "sql": ORDER_COUNT_SQL, "visualization": METRIC_VIZ},
    )
    card_id = pin.json()["id"]

    response = client.delete(f"/api/sessions/{session_id}/dashboard/cards/{card_id}")
    assert response.status_code == 204

    listed = client.get(f"/api/sessions/{session_id}/dashboard/cards")
    assert listed.json()["cards"] == []


def test_the_history_route_lists_recent_questions_newest_first(client, relational_workbook):
    session_id = _exported(client, relational_workbook)
    client.post(f"/api/sessions/{session_id}/query", json={"question": "first question"})
    client.post(f"/api/sessions/{session_id}/query", json={"question": "second question"})

    response = client.get(f"/api/sessions/{session_id}/query/history")

    assert response.status_code == 200
    history = response.json()["history"]
    assert len(history) == 2
    assert history[0]["question"] == "second question"
    assert history[1]["question"] == "first question"


def test_the_history_route_belongs_to_its_owner(client, other_client, relational_workbook):
    session_id = _exported(client, relational_workbook)
    client.post(f"/api/sessions/{session_id}/query", json={"question": "q"})

    response = other_client.get(f"/api/sessions/{session_id}/query/history")

    assert response.status_code == 404


def test_the_query_route_reports_no_suggestions_with_no_model_configured(
    client, relational_workbook
):
    """The suite runs with ANTHROPIC_API_KEY unset (conftest) — a genuine
    end-to-end check that a degraded ask still answers a normal ``ok: false``
    with nothing under ``suggestions`` for the UI to render."""

    session_id = _exported(client, relational_workbook)

    response = client.post(
        f"/api/sessions/{session_id}/query", json={"question": "how many orders?"}
    )

    assert response.status_code == 200
    assert response.json().get("suggestions") in (None, [])


# ---------------------------------------------------------------------------
# the data-quality report (C-3)
# ---------------------------------------------------------------------------


def test_the_quality_report_route_scores_every_exported_table(client, relational_workbook):
    session_id = _exported(client, relational_workbook)

    response = client.get(f"/api/sessions/{session_id}/quality-report")

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["tables"]
    for table in payload["tables"]:
        assert 0.0 <= table["overall"] <= 100.0
        assert "previous" not in table  # only one export has ever run


def test_the_quality_report_route_belongs_to_its_owner(client, other_client, relational_workbook):
    session_id = _exported(client, relational_workbook)

    response = other_client.get(f"/api/sessions/{session_id}/quality-report")

    assert response.status_code == 404


def test_a_second_export_attaches_a_before_after_comparison(client, relational_workbook):
    session_id = _exported(client, relational_workbook)
    second = client.post(f"/api/sessions/{session_id}/export")
    assert second.status_code == 200, second.text

    response = client.get(f"/api/sessions/{session_id}/quality-report")

    payload = response.json()
    assert any("previous" in table for table in payload["tables"])


# ---------------------------------------------------------------------------
# proactive query suggestions (C-3)
# ---------------------------------------------------------------------------


class _SuggestingClaude:
    available = True

    def __init__(self, questions):
        self._questions = questions
        self.calls = 0

    def suggest_questions(self, tables):
        self.calls += 1
        return list(self._questions)


def test_suggested_questions_are_generated_and_then_cached(client, relational_workbook):
    from app.api import services

    session_id = _exported(client, relational_workbook)
    fake = _SuggestingClaude(["which city has the most customers?", "total revenue by month"])

    with _db() as db:
        first = services.get_suggested_questions(db, session_id, claude=fake)
        db.commit()
    assert first["available"] is True
    assert first["questions"] == [
        "which city has the most customers?",
        "total revenue by month",
    ]
    assert fake.calls == 1

    # A second call is served from the cache on the export row, not a second
    # model call — proactive suggestions are computed once per export.
    with _db() as db:
        second = services.get_suggested_questions(db, session_id, claude=fake)
    assert second["questions"] == first["questions"]
    assert fake.calls == 1


def test_re_exporting_clears_the_suggested_questions_cache(client, relational_workbook):
    from app.api import services

    session_id = _exported(client, relational_workbook)
    fake = _SuggestingClaude(["first build's question"])
    with _db() as db:
        services.get_suggested_questions(db, session_id, claude=fake)
        db.commit()

    second_export = client.post(f"/api/sessions/{session_id}/export")
    assert second_export.status_code == 200, second_export.text

    response = client.get(f"/api/sessions/{session_id}/suggested-questions")
    assert response.status_code == 200
    # No API key in this test process's own client (conftest) — regeneration
    # degrades to an empty, not-available list rather than serving stale
    # questions from a schema that export just replaced.
    assert response.json() == {"questions": [], "available": False}


def test_the_suggested_questions_route_degrades_cleanly_with_no_model_configured(
    client, relational_workbook
):
    session_id = _exported(client, relational_workbook)

    response = client.get(f"/api/sessions/{session_id}/suggested-questions")

    assert response.status_code == 200
    assert response.json() == {"questions": [], "available": False}
