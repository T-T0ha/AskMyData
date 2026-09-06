"""End-to-end API walkthrough: upload → triage → semantics → co-planned cleaning."""

from __future__ import annotations

from pathlib import Path

import pytest


def _create_session(client, name="Test session") -> str:
    response = client.post("/api/sessions", json={"name": name})
    assert response.status_code == 201
    return response.json()["id"]


def _upload(client, session_id: str, path: Path):
    with path.open("rb") as handle:
        return client.post(
            f"/api/sessions/{session_id}/upload",
            files={"file": (path.name, handle, "application/vnd.ms-excel")},
        )


def test_health_and_status(client):
    assert client.get("/health").json() == {"status": "ok"}

    status = client.get("/api/status").json()
    assert "claude" in status and "embeddings" in status
    assert status["embeddings"]["dim"] == 384
    assert status["checkpointer"]
    assert status["durable"] is True  # the suite runs on a real SQLite checkpointer


def test_vocabulary_lists_nine_types_and_ten_steps(client):
    vocabulary = client.get("/api/vocabulary").json()
    assert len(vocabulary["column_types"]) == 9
    assert len(vocabulary["step_types"]) == 10
    assert {"standardize_casing", "strip_whitespace"} <= set(vocabulary["step_types"])
    # No step the UI can offer is capable of writing into an empty cell.
    assert not [s for s in vocabulary["step_types"] if "fill" in s]
    assert "unknown" in vocabulary["taxonomy_labels"]
    assert ".xlsx" in vocabulary["accepted_extensions"]


def test_unknown_session_returns_404(client):
    assert client.get("/api/sessions/does-not-exist").status_code == 404


def test_rejects_unsupported_file_type(client):
    session_id = _create_session(client)
    response = client.post(
        f"/api/sessions/{session_id}/upload",
        files={"file": ("notes.pdf", b"%PDF-1.4", "application/pdf")},
    )
    assert response.status_code == 415


def test_upload_runs_phase0(client, messy_workbook):
    session_id = _create_session(client)
    payload = _upload(client, session_id, messy_workbook).json()

    sheets = {s["table_name"]: s for s in payload["sheets"]}
    assert set(sheets) == {"sales_data", "customers", "line_items", "notes"}
    assert sheets["sales_data"]["header"]["is_multi_row"]
    assert sheets["sales_data"]["header"]["banner_rows"] == [1]
    assert sheets["notes"]["triage"] == "structural_issues"

    pairs = {
        (e["left_column"], e["right_column"]) for e in payload["equivalences"]
    }
    assert ("order_info_cust_id", "customer_id") in pairs

    # The column-name embeddings behind the score are stored, not discarded
    # the moment a scalar score is derived from them (M-4).
    from app.core.config import get_settings
    from app.db.base import get_session_factory
    from app.db.models import EquivalenceCandidateRecord

    db = get_session_factory()()
    try:
        row = (
            db.query(EquivalenceCandidateRecord)
            .filter_by(session_id=session_id, left_column="order_info_cust_id")
            .one()
        )
        dim = get_settings().embedding_dim
        assert row.left_embedding is not None and len(row.left_embedding) == dim
        assert row.right_embedding is not None and len(row.right_embedding) == dim
    finally:
        db.close()


def test_uploading_a_sqlite_database_ingests_its_tables(client, tmp_path):
    import sqlite3

    path = tmp_path / "shop.sqlite"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE customers (customer_id TEXT PRIMARY KEY, city TEXT);
        CREATE TABLE orders (
            order_id INTEGER PRIMARY KEY,
            customer_id TEXT REFERENCES customers(customer_id),
            amount REAL
        );
        INSERT INTO customers VALUES ('C-1','Dhaka'),('C-2',NULL),('C-3','Sylhet');
        INSERT INTO orders VALUES (1,'C-1',100.0),(2,'C-2',NULL),(3,'C-3',300.0);
        """
    )
    connection.commit()
    connection.close()

    session_id = _create_session(client)
    with path.open("rb") as handle:
        response = client.post(
            f"/api/sessions/{session_id}/upload",
            files={"file": (path.name, handle, "application/octet-stream")},
        )
    assert response.status_code == 200

    sheets = {s["table_name"]: s for s in response.json()["sheets"]}
    assert set(sheets) == {"customers", "orders"}
    assert sheets["orders"]["source_kind"] == "sqlite"
    assert sheets["orders"]["native_schema"]["foreign_keys"][0]["references_table"] == "customers"


def test_inspect_and_connect_read_a_live_database(client, tmp_path):
    """A SQLite URL exercises the same code path a PostgreSQL URL takes."""

    import sqlite3

    path = tmp_path / "live.sqlite"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE staff (staff_id INTEGER PRIMARY KEY, name TEXT, email TEXT);
        INSERT INTO staff VALUES (1,'Rahim','r@example.com'),(2,'Karim',NULL),(3,'Nusrat','n@example.com');
        """
    )
    connection.commit()
    connection.close()
    url = f"sqlite:///{path}"

    inspected = client.post("/api/sources/inspect", json={"url": url})
    assert inspected.status_code == 200
    body = inspected.json()
    assert body["table_count"] == 1
    assert body["tables"][0]["name"] == "staff"
    assert body["tables"][0]["row_count"] == 3

    connected = client.post(f"/api/sessions/{_create_session(client)}/connect", json={"url": url})
    assert connected.status_code == 200
    sheets = connected.json()["sheets"]
    assert sheets[0]["table_name"] == "staff"
    assert sheets[0]["row_count"] == 3


def test_connecting_to_an_unsupported_backend_is_refused(client):
    response = client.post("/api/sources/inspect", json={"url": "redis://localhost:6379/0"})

    assert response.status_code == 422
    assert "unsupported database type" in response.json()["detail"]


def test_a_connection_is_recorded_by_its_credential_free_label(client, tmp_path):
    """The session keeps a label for the source, never the connection string.

    ``display_url`` is what redacts the password (unit-tested separately); this
    checks that it is what actually reaches the database, so a stored session
    can never be a place credentials leak from.
    """

    import sqlite3

    from app.ingestion.relational import display_url

    path = tmp_path / "src.sqlite"
    connection = sqlite3.connect(path)
    connection.executescript(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT);"
        "INSERT INTO t VALUES (1,'a'),(2,'b'),(3,'c');"
    )
    connection.commit()
    connection.close()

    url = f"sqlite:///{path}"
    session_id = _create_session(client)
    client.post(f"/api/sessions/{session_id}/connect", json={"url": url})
    session = client.get(f"/api/sessions/{session_id}").json()

    assert session["source_files"] == [display_url(url)]


def test_triage_endpoint_summarises_every_sheet(client, messy_workbook):
    session_id = _create_session(client)
    _upload(client, session_id, messy_workbook)

    triage = client.get(f"/api/sessions/{session_id}/triage").json()
    assert triage["summary"]["sheet_count"] == 4
    assert sum(triage["summary"]["triage_counts"].values()) == 4
    assert all("issues" in sheet for sheet in triage["sheets"])


def test_table_preview(client, messy_workbook):
    session_id = _create_session(client)
    _upload(client, session_id, messy_workbook)

    preview = client.get(f"/api/sessions/{session_id}/tables/customers/preview?limit=3").json()
    assert preview["row_count"] == 8
    assert len(preview["rows"]) == 3
    assert "customer_id" in preview["columns"]

    assert client.get(f"/api/sessions/{session_id}/tables/ghost/preview").status_code == 404


def test_semantics_endpoint_and_user_override(client, messy_workbook):
    session_id = _create_session(client)
    _upload(client, session_id, messy_workbook)

    payload = client.post(f"/api/sessions/{session_id}/semantics").json()
    assert payload["total_columns"] == 18
    assert payload["unknown_count"] == 0

    columns = {
        c["name"]: c
        for table in payload["tables"]
        if table["table"] == "customers"
        for c in table["columns"]
    }
    assert columns["email_address"]["taxonomy_label"] == "email"
    assert columns["phone"]["column_type"] == "identifier"

    target = columns["city"]
    response = client.patch(
        f"/api/sessions/{session_id}/semantics/{target['id']}",
        json={"taxonomy_label": "region"},
    )
    assert response.status_code == 200
    assert response.json()["effective_label"] == "region"
    assert response.json()["validated"] is True
    # The detected value is preserved alongside the correction.
    assert response.json()["taxonomy_label"] == "city"


def test_override_rejects_values_outside_the_vocabulary(client, messy_workbook):
    session_id = _create_session(client)
    _upload(client, session_id, messy_workbook)
    payload = client.post(f"/api/sessions/{session_id}/semantics").json()
    column_id = payload["tables"][0]["columns"][0]["id"]

    assert (
        client.patch(
            f"/api/sessions/{session_id}/semantics/{column_id}",
            json={"taxonomy_label": "not_a_label"},
        ).status_code
        == 422
    )
    assert (
        client.patch(
            f"/api/sessions/{session_id}/semantics/{column_id}",
            json={"column_type": "not_a_type"},
        ).status_code
        == 422
    )


def test_a_keyless_sheet_asks_for_its_key_and_accepts_the_answer(client, messy_workbook):
    """The proposal's no-key handling, end to end.

    ``line_items`` has no unique column — order_id repeats across its lines —
    so ingestion proposes the composite that does identify a line and turns the
    sheet orange until the user answers.  Answering resolves it, and the answer
    is what the cleaning plan then respects.
    """

    session_id = _create_session(client)
    _upload(client, session_id, messy_workbook)

    sheets = {s["table_name"]: s for s in client.get(f"/api/sessions/{session_id}/triage").json()["sheets"]}
    line_items = sheets["line_items"]
    keys = line_items["key_analysis"]

    assert line_items["triage"] == "needs_attention"
    assert keys["needs_confirmation"] is True
    assert keys["primary_key"] == []
    assert keys["candidates"][0]["columns"] == ["order_id", "product_sku"]

    updated = client.post(
        f"/api/sessions/{session_id}/tables/line_items/key",
        json={"columns": ["order_id", "product_sku"]},
    )
    assert updated.status_code == 200
    confirmed = updated.json()
    assert confirmed["key_analysis"]["primary_key"] == ["order_id", "product_sku"]
    assert confirmed["key_analysis"]["source"] == "confirmed"
    # The question is gone; the sheet is re-classified on whatever is left of
    # it (here, a mostly-empty note column — nothing to do with keys).
    assert not [i for i in confirmed["issues"] if i["code"] == "composite_key_candidate"]
    assert {i["code"] for i in confirmed["issues"]} == {"high_null_ratio", "retyped_columns"}

    # And the plan leaves the confirmed key alone rather than burying it.
    client.post(f"/api/sessions/{session_id}/semantics")
    plan = client.post(f"/api/sessions/{session_id}/cleaning/start").json()["pending"]["plan"]
    assert not [
        s for s in plan if s["type"] == "add_synthetic_key" and s["table"] == "line_items"
    ]


def test_a_key_that_does_not_hold_is_refused_with_the_reason(client, messy_workbook):
    """Confirming a false key would put a PRIMARY KEY on data that breaks it."""

    session_id = _create_session(client)
    _upload(client, session_id, messy_workbook)

    response = client.post(
        f"/api/sessions/{session_id}/tables/line_items/key",
        json={"columns": ["order_id"]},
    )
    assert response.status_code == 422
    assert "cannot be the key" in response.json()["detail"]

    missing = client.post(
        f"/api/sessions/{session_id}/tables/line_items/key",
        json={"columns": ["nope"]},
    )
    assert missing.status_code == 422


def test_rejecting_every_candidate_falls_back_to_a_synthetic_key(client, messy_workbook):
    session_id = _create_session(client)
    _upload(client, session_id, messy_workbook)

    rejected = client.post(
        f"/api/sessions/{session_id}/tables/line_items/key",
        json={"columns": []},
    ).json()
    assert rejected["key_analysis"]["needs_synthetic_key"] is True
    assert rejected["key_analysis"]["confirmed"] is True

    client.post(f"/api/sessions/{session_id}/semantics")
    plan = client.post(f"/api/sessions/{session_id}/cleaning/start").json()["pending"]["plan"]
    synthetic = [
        s for s in plan if s["type"] == "add_synthetic_key" and s["table"] == "line_items"
    ]
    assert synthetic, "rejecting the candidates must leave the table with a row_id offer"
    assert synthetic[0]["params"]["column"] == "row_id"


def test_a_confirmed_equivalence_is_a_relationship_not_a_merge(client, messy_workbook):
    """orders.cust_id ~ customers.customer_id is a foreign key, not a join.

    Folding the two tables together would flatten exactly the structure the
    platform exists to recover, and would repeat every customer's details once
    per order.  The confirmation is kept as evidence for relationship
    detection; a user who really wants the join can still add the step.
    """

    session_id = _create_session(client)
    upload = _upload(client, session_id, messy_workbook).json()

    candidate = next(
        e
        for e in upload["equivalences"]
        if (e["left_column"], e["right_column"]) == ("order_info_cust_id", "customer_id")
    )
    confirmed = client.post(
        f"/api/sessions/{session_id}/equivalences/{candidate['id']}",
        json={"confirmed": True},
    ).json()
    assert confirmed["confirmed"] is True

    started = client.post(f"/api/sessions/{session_id}/cleaning/start").json()
    merges = [s for s in started["pending"]["plan"] if s["type"] == "merge_sheets"]
    assert not merges, "confirming a shared key must not denormalise the two tables"


def test_two_sheets_of_the_same_table_are_offered_as_a_merge(tmp_path):
    """The one case where a join really is the answer: one table, two sheets."""

    import pandas as pd

    from app.cleaning.planner import heuristic_plan
    from app.core.schemas import StepType

    q1 = pd.DataFrame({"order_id": ["A1", "A2"], "amount": [10.0, 20.0]})
    q2 = pd.DataFrame({"order_id": ["B1", "B2"], "amount": [30.0, 40.0]})
    plan = heuristic_plan(
        {"orders_q1": q1, "orders_q2": q2},
        equivalences=[
            {
                "left_table": "orders_q1",
                "right_table": "orders_q2",
                "left_column": "order_id",
                "right_column": "order_id",
                "confirmed": True,
            }
        ],
    )
    merges = [s for s in plan if s.type is StepType.MERGE_SHEETS]
    assert len(merges) == 1
    assert merges[0].params["left_key"] == "order_id"


def test_full_cleaning_workflow(client, messy_workbook):
    session_id = _create_session(client)
    _upload(client, session_id, messy_workbook)
    client.post(f"/api/sessions/{session_id}/semantics")

    started = client.post(f"/api/sessions/{session_id}/cleaning/start").json()
    assert started["status"] == "awaiting_plan_review"
    assert started["pending"]["kind"] == "plan_review"
    plan = started["pending"]["plan"]
    assert plan

    # The excluded "notes" sheet must not appear in the plan.
    assert {s["table"] for s in plan}.isdisjoint({"notes"})

    state = client.post(
        f"/api/sessions/{session_id}/cleaning/plan",
        json={"action": "confirm", "plan": plan},
    ).json()

    approved = 0
    while state["pending"] is not None:
        action = "skip" if state["pending"]["kind"] == "step_failed" else "approve"
        approved += 1
        state = client.post(
            f"/api/sessions/{session_id}/cleaning/step", json={"action": action}
        ).json()
        assert approved <= len(plan) + 1

    assert state["status"] == "completed"
    assert state["run"]["completed_steps"] == len(plan)

    tables = {t["name"]: t for t in state["tables"]}
    assert tables["sales_data"]["row_count"] == 11  # duplicate removed


def test_step_endpoint_rejects_out_of_order_calls(client, messy_workbook):
    session_id = _create_session(client)
    _upload(client, session_id, messy_workbook)

    assert (
        client.post(
            f"/api/sessions/{session_id}/cleaning/step", json={"action": "approve"}
        ).status_code
        == 409
    )

    client.post(f"/api/sessions/{session_id}/cleaning/start")
    assert (
        client.post(
            f"/api/sessions/{session_id}/cleaning/step", json={"action": "approve"}
        ).status_code
        == 409
    )


def test_cleaning_requires_an_upload(client):
    session_id = _create_session(client)
    assert client.post(f"/api/sessions/{session_id}/cleaning/start").status_code == 409
    assert client.post(f"/api/sessions/{session_id}/semantics").status_code == 409


def test_session_deletion_removes_everything(client, messy_workbook):
    session_id = _create_session(client)
    _upload(client, session_id, messy_workbook)

    assert client.delete(f"/api/sessions/{session_id}").status_code == 204
    assert client.get(f"/api/sessions/{session_id}").status_code == 404


def test_csv_upload_is_supported(client, tmp_path):
    csv = tmp_path / "products.csv"
    csv.write_text(
        "Product ID,Product Name,Unit Price\nP-1,Fan,4500\nP-2,Bulb,320\nP-3,Cord,250\n",
        encoding="utf-8",
    )
    session_id = _create_session(client)
    payload = _upload(client, session_id, csv).json()

    assert payload["sheets"][0]["table_name"] == "products"
    assert payload["sheets"][0]["row_count"] == 3


# ---------------------------------------------------------------------------
# schema drift
# ---------------------------------------------------------------------------


def test_a_database_missing_a_column_is_found_healed_and_reported(client):
    """A server started before a model changed answers every request with an error.

    ``init_db`` adds new columns at startup, so this only bites a process that
    has been up since before the change — which is exactly the case where the
    message needs to say so, because nothing about the request is wrong.
    """

    from sqlalchemy import text

    from app.db.base import get_engine, init_db, pending_schema_changes

    with get_engine().begin() as connection:
        connection.execute(text("ALTER TABLE sheet_records DROP COLUMN key_analysis"))

    assert pending_schema_changes() == ["sheet_records.key_analysis"]
    health = client.get("/health").json()
    assert health["status"] == "degraded"
    assert health["pending_schema_changes"] == ["sheet_records.key_analysis"]

    assert init_db()["added_columns"] == ["sheet_records.key_analysis"]
    assert pending_schema_changes() == []
    assert client.get("/health").json() == {"status": "ok"}


def test_a_storage_failure_is_not_blamed_on_the_uploaded_file(client, tmp_path, monkeypatch):
    """The CSV parsed fine; saying "could not read the file" sends the user hunting."""

    from sqlalchemy.exc import OperationalError

    from app.api import services

    def explode(*args, **kwargs):
        raise OperationalError("SELECT sheet_records.key_analysis", {}, Exception("no column"))

    monkeypatch.setattr(services, "ingest_file", explode)

    csv = tmp_path / "sales.csv"
    csv.write_text("id,city\n1,Dhaka\n2,Sylhet\n", encoding="utf-8")
    response = _upload(client, _create_session(client), csv)

    assert response.status_code == 500
    detail = response.json()["detail"]
    assert "could not read the file" not in detail
    assert "could not store it" in detail
