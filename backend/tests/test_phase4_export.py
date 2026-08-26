"""Phase 4 — schema construction and the physical export.

The plan is tested for the decisions it makes (which key, which constraint,
which order) and the runner for the one thing that matters once rows are
moving: what comes out of the database has to be what went into the
spreadsheet, including the parts that were not there.
"""

from __future__ import annotations

import uuid

import pandas as pd
import pytest

from app.export.naming import schema_name
from app.export.runner import (
    ExportTarget,
    drop_export,
    prepare_frame,
    read_back,
    read_semantic_layer,
    reset_target,
    resolve_target,
    run_export,
)
from app.export.schema import build_plan, render_ddl


def _session_id() -> str:
    return uuid.uuid4().hex


def _column(name: str, column_type: str, label: str = "unknown", **extra) -> dict:
    return {
        "name": name,
        "effective_type": column_type,
        "effective_label": label,
        "null_ratio": extra.pop("null_ratio", 0.0),
        "is_additive": extra.pop("is_additive", False),
        "sample_values": extra.pop("sample_values", []),
        "semantic_description": extra.pop("semantic_description", ""),
        **extra,
    }


def _fk(from_table, from_column, to_table, to_column, status="confirmed") -> dict:
    return {
        "rel_type": "foreign_key",
        "from_table": from_table,
        "from_column": from_column,
        "to_table": to_table,
        "to_column": to_column,
        "status": status,
    }


@pytest.fixture
def shop():
    """Two tables, one clean reference between them."""

    tables = {
        "customers": pd.DataFrame(
            {"customer_id": ["C-1", "C-2", "C-3"], "city": ["Dhaka", "Sylhet", None]}
        ),
        "orders": pd.DataFrame(
            {
                "order_id": ["O-1", "O-2"],
                "customer_id": ["C-1", "C-2"],
                "total": [1250.00, 37.50],
            }
        ),
    }
    columns = {
        "customers": [
            _column("customer_id", "identifier", "identifier"),
            _column("city", "text", "city", null_ratio=1 / 3),
        ],
        "orders": [
            _column("order_id", "identifier", "order_number"),
            _column("customer_id", "identifier", "foreign_identifier"),
            _column("total", "currency", "revenue", is_additive=True),
        ],
    }
    keys = {
        "customers": {"primary_key": ["customer_id"], "source": "detected"},
        "orders": {"primary_key": ["order_id"], "source": "detected"},
    }
    return tables, columns, keys


# ---------------------------------------------------------------------------
# the plan
# ---------------------------------------------------------------------------


def test_the_confirmed_key_becomes_the_primary_key(shop):
    tables, columns, keys = shop

    plan = build_plan(tables, columns, key_analyses=keys)

    customers = plan.table("customers")
    assert customers.primary_key == ["customer_id"]
    assert customers.column("customer_id").nullable is False
    assert customers.column("city").nullable is True


def test_a_key_that_no_longer_holds_is_dropped_rather_than_declared(shop):
    """Cleaning runs between key discovery and the export.  A key the data no
    longer satisfies would fail at ``CREATE TABLE``, or worse, at insert."""

    tables, columns, keys = shop
    tables["customers"] = pd.DataFrame(
        {"customer_id": ["C-1", "C-1", "C-2"], "city": ["Dhaka", "Dhaka", "Sylhet"]}
    )

    plan = build_plan(tables, columns, key_analyses=keys)

    assert plan.table("customers").primary_key == []
    assert any("repeat the same value" in note for note in plan.table("customers").notes)
    assert any("primary key dropped" in warning for warning in plan.warnings)


def test_a_key_column_containing_a_blank_is_refused(shop):
    tables, columns, keys = shop
    tables["customers"].loc[0, "customer_id"] = None

    plan = build_plan(tables, columns, key_analyses=keys)

    assert plan.table("customers").primary_key == []
    assert any("NULL cannot identify a row" in note for note in plan.table("customers").notes)


def test_only_a_confirmed_reference_becomes_a_constraint(shop):
    """The human-in-the-loop contract, at the point it finally means something."""

    tables, columns, keys = shop
    edge = _fk("orders", "customer_id", "customers", "customer_id")

    confirmed = build_plan(tables, columns, keys, relationships=[edge])
    proposed = build_plan(tables, columns, keys, relationships=[{**edge, "status": "proposed"}])
    rejected = build_plan(tables, columns, keys, relationships=[{**edge, "status": "rejected"}])

    assert len(confirmed.enforced_keys) == 1
    assert proposed.enforced_keys == [] and proposed.unenforced_keys == []
    assert rejected.enforced_keys == [] and rejected.unenforced_keys == []


def test_a_reference_with_orphans_is_kept_as_a_hint_and_not_enforced(shop):
    """The fixture case: two orders point at a customer nobody has.

    Creating the constraint anyway would abort the export at ``INSERT`` — on
    data the user was shown and accepted.  The reference stays in the semantic
    layer, where Phase 5 can still join on it, and the report says why it is
    not enforced.
    """

    tables, columns, keys = shop
    tables["orders"].loc[1, "customer_id"] = "C-999"

    plan = build_plan(
        tables, columns, keys, relationships=[_fk("orders", "customer_id", "customers", "customer_id")]
    )

    assert plan.enforced_keys == []
    (unenforced,) = plan.unenforced_keys
    assert "C-999" in unenforced.reason
    assert plan.table("orders").column("customer_id").references_table == "customers"


def test_a_reference_to_something_that_is_not_a_key_is_not_enforced(shop):
    tables, columns, keys = shop

    plan = build_plan(
        tables, columns, keys, relationships=[_fk("orders", "customer_id", "customers", "city")]
    )

    (unenforced,) = plan.unenforced_keys
    assert "not the primary key" in unenforced.reason


def test_a_blank_child_value_is_not_an_orphan(shop):
    """An order with no customer recorded is unknown, not wrong — the same
    reading of a blank the rest of the platform gives."""

    tables, columns, keys = shop
    tables["orders"].loc[1, "customer_id"] = None

    plan = build_plan(
        tables, columns, keys, relationships=[_fk("orders", "customer_id", "customers", "customer_id")]
    )

    assert len(plan.enforced_keys) == 1


def test_a_key_read_as_a_number_on_one_side_still_matches(shop):
    """``1001`` from one sheet and ``1001.0`` from another are one key."""

    tables = {
        "parent": pd.DataFrame({"code": [1001, 1002]}),
        "child": pd.DataFrame({"code": [1001.0, 1002.0]}),
    }
    columns = {
        "parent": [_column("code", "integer_nominal")],
        "child": [_column("code", "integer_nominal")],
    }
    keys = {"parent": {"primary_key": ["code"]}}

    plan = build_plan(
        tables, columns, keys, relationships=[_fk("child", "code", "parent", "code")]
    )

    assert len(plan.enforced_keys) == 1


def test_parents_are_created_before_the_tables_that_reference_them(shop):
    tables, columns, keys = shop

    plan = build_plan(
        tables, columns, keys, relationships=[_fk("orders", "customer_id", "customers", "customer_id")]
    )

    assert [table.name for table in plan.tables] == ["customers", "orders"]


def test_a_cycle_is_reported_and_deferred_rather_than_failing(shop):
    """No insert order satisfies a cycle, so its constraints move to the commit."""

    tables = {
        "a": pd.DataFrame({"id": ["a1", "a2"], "b_id": ["b1", "b2"]}),
        "b": pd.DataFrame({"id": ["b1", "b2"], "a_id": ["a1", "a2"]}),
    }
    columns = {
        "a": [_column("id", "identifier"), _column("b_id", "identifier")],
        "b": [_column("id", "identifier"), _column("a_id", "identifier")],
    }
    keys = {"a": {"primary_key": ["id"]}, "b": {"primary_key": ["id"]}}

    plan = build_plan(
        tables,
        columns,
        keys,
        relationships=[_fk("a", "b_id", "b", "id"), _fk("b", "a_id", "a", "id")],
    )

    assert len(plan.tables) == 2
    assert all(fk.deferrable for fk in plan.enforced_keys)
    assert any("circular references" in warning for warning in plan.warnings)


def test_a_self_reference_is_deferred_so_a_manager_may_come_last():
    tables = {
        "employees": pd.DataFrame(
            {"id": ["e1", "e2", "e3"], "manager_id": [None, "e1", "e1"]}
        )
    }
    columns = {
        "employees": [_column("id", "identifier"), _column("manager_id", "identifier")]
    }

    plan = build_plan(
        tables,
        columns,
        {"employees": {"primary_key": ["id"]}},
        relationships=[_fk("employees", "manager_id", "employees", "id")],
    )

    (fk,) = plan.enforced_keys
    assert fk.deferrable is True


# ---------------------------------------------------------------------------
# names and DDL
# ---------------------------------------------------------------------------


def test_headers_are_sanitised_and_the_renames_are_reported():
    tables = {"Sales Data": pd.DataFrame({"Total Amount (৳)": [1.0], "Order": ["x"]})}
    columns = {
        "Sales Data": [
            _column("Total Amount (৳)", "currency", "revenue"),
            _column("Order", "text", "order_number"),
        ]
    }

    plan = build_plan(tables, columns)
    spec = plan.tables[0]

    assert spec.name == "sales_data"
    assert [column.name for column in spec.columns] == ["total_amount", "order_col"]
    assert spec.column("total_amount").source_name == "Total Amount (৳)"


def test_a_sheet_named_like_an_attack_produces_an_ordinary_table():
    """The sanitiser is the second line; SQLAlchemy quoting is the first.  This
    asserts the second one, because the first is not visible in the DDL."""

    name = '"; DROP TABLE users; --'
    plan = build_plan(
        {name: pd.DataFrame({"a": [1]})}, {name: [_column("a", "integer_ordinal")]}
    )
    ddl = render_ddl(plan, schema="ds_deadbeef")

    assert plan.tables[0].name == "drop_table_users"
    assert "DROP TABLE users" not in ddl
    # One CREATE SCHEMA and one CREATE TABLE: two statements, two semicolons.
    # A third would mean the header had ended a statement of its own.
    assert ddl.count(";") == 2, "one semicolon per statement, none injected"


def test_the_ddl_says_what_was_built(shop):
    tables, columns, keys = shop

    ddl = render_ddl(
        build_plan(
            tables,
            columns,
            keys,
            relationships=[_fk("orders", "customer_id", "customers", "customer_id")],
        ),
        schema="ds_deadbeef",
    )

    assert "CREATE SCHEMA IF NOT EXISTS ds_deadbeef;" in ddl
    assert "PRIMARY KEY (customer_id)" in ddl
    assert "FOREIGN KEY(customer_id) REFERENCES ds_deadbeef.customers (customer_id)" in ddl
    assert "NUMERIC(15, 2)" in ddl and "VARCHAR(100)" in ddl


# ---------------------------------------------------------------------------
# the physical export
# ---------------------------------------------------------------------------


def test_the_export_creates_tables_that_hold_the_rows(shop):
    tables, columns, keys = shop
    session_id = _session_id()
    plan = build_plan(
        tables, columns, keys, relationships=[_fk("orders", "customer_id", "customers", "customer_id")]
    )

    report = run_export(session_id, plan, tables).to_dict()

    assert report["table_count"] == 2
    assert report["row_count"] == 5
    assert report["enforced_foreign_keys"] == 1
    assert read_back(session_id, "SELECT COUNT(*) AS n FROM orders") == [{"n": 2}]
    assert read_back(session_id, "SELECT total FROM orders ORDER BY order_id")[0]["total"] == 1250.0
    drop_export(session_id)


def test_a_blank_cell_arrives_in_the_database_as_null(shop):
    """The no-imputation rule, at the end of the pipeline.  A blank that became
    a zero or an empty string here would undo every guarantee before it."""

    tables, columns, keys = shop
    session_id = _session_id()

    run_export(session_id, build_plan(tables, columns, keys), tables)
    rows = read_back(session_id, "SELECT city FROM customers WHERE city IS NULL")

    assert rows == [{"city": None}]
    assert read_back(session_id, "SELECT COUNT(city) AS n FROM customers") == [{"n": 2}]
    drop_export(session_id)


def test_re_exporting_replaces_the_previous_export(shop):
    """Twice must not mean double the rows, and must not fail on the tables the
    first run left behind."""

    tables, columns, keys = shop
    session_id = _session_id()
    plan = build_plan(tables, columns, keys)

    run_export(session_id, plan, tables)
    second = run_export(session_id, plan, tables).to_dict()

    assert second["row_count"] == 5
    assert read_back(session_id, "SELECT COUNT(*) AS n FROM customers") == [{"n": 3}]
    drop_export(session_id)


def test_an_export_can_be_dropped_completely(shop):
    tables, columns, keys = shop
    session_id = _session_id()

    run_export(session_id, build_plan(tables, columns, keys), tables)
    drop_export(session_id)

    with pytest.raises(Exception):
        read_back(session_id, "SELECT 1 FROM customers")


def test_two_sessions_export_into_separate_namespaces(shop):
    tables, columns, keys = shop
    first, second = _session_id(), _session_id()

    run_export(first, build_plan(tables, columns, keys), tables)
    smaller = {name: frame.head(1) for name, frame in tables.items()}
    run_export(second, build_plan(smaller, columns, keys), smaller)

    assert read_back(first, "SELECT COUNT(*) AS n FROM customers") == [{"n": 3}]
    assert read_back(second, "SELECT COUNT(*) AS n FROM customers") == [{"n": 1}]
    drop_export(first)
    drop_export(second)


def test_the_target_is_named_after_the_session_and_carries_no_credentials():
    session_id = _session_id()
    target = resolve_target(session_id)

    try:
        assert schema_name(session_id) in target.label
        assert "@" not in target.label and "password" not in target.label.lower()
    finally:
        target.dispose()


def test_the_runner_refuses_to_reset_anything_it_does_not_own():
    """The guard immediately before ``DROP``.  If this ever passes something
    like ``public``, the statement must not run."""

    from sqlalchemy import create_engine

    engine = create_engine("sqlite://", future=True)
    hostile = ExportTarget(engine=engine, schema="public", label="public", dialect="postgresql")

    with pytest.raises(ValueError, match="refusing to drop"):
        reset_target(hostile)
    engine.dispose()


def test_values_are_prepared_under_their_exported_names(shop):
    tables, columns, keys = shop
    plan = build_plan(tables, columns, keys)

    frame = prepare_frame(plan.table("orders"), tables["orders"])

    assert list(frame.columns) == ["order_id", "customer_id", "total"]
    assert frame["total"].tolist() == [1250.0, 37.5]


def test_the_whole_relational_fixture_exports_and_answers_a_join(relational_tables):
    """End to end on real ingested data, finishing with the question the whole
    platform exists to make answerable."""

    from app.ingestion.keys import discover_keys
    from app.relationships.foreign_keys import detect_foreign_keys
    from app.semantics.pipeline import analyze_tables

    session_id = _session_id()
    keys = {name: discover_keys(df) for name, df in relational_tables.items()}
    semantics = analyze_tables(relational_tables)
    relationships = [
        {**candidate.to_dict(), "status": "confirmed"}
        for candidate in detect_foreign_keys(relational_tables, keys)
    ]

    plan = build_plan(
        relational_tables,
        {name: [c.to_dict() for c in sem.columns] for name, sem in semantics.items()},
        key_analyses={name: analysis.to_dict() for name, analysis in keys.items()},
        relationships=relationships,
    )
    report = run_export(session_id, plan, relational_tables).to_dict()

    assert report["row_count"] == sum(len(df) for df in relational_tables.values())
    joined = read_back(
        session_id,
        "SELECT c.region AS region, COUNT(*) AS orders "
        "FROM orders o JOIN customers c ON o.customer_id = c.customer_id "
        "GROUP BY c.region ORDER BY orders DESC",
    )
    assert joined and sum(row["orders"] for row in joined) == 43, "two orphan orders drop out"
    drop_export(session_id)


# ---------------------------------------------------------------------------
# sem_metadata — the semantic layer inside the exported database
# ---------------------------------------------------------------------------


def test_the_exported_database_describes_its_own_columns(shop):
    """The claim the whole project rests on: the database carries its meaning."""

    tables, columns, keys = shop
    session_id = _session_id()
    plan = build_plan(
        tables,
        columns,
        keys,
        relationships=[_fk("orders", "customer_id", "customers", "customer_id")],
        profiles={"orders": {"effective_type": "fact"}},
    )

    report = run_export(session_id, plan, tables).to_dict()
    rows = {
        (row["table_name"], row["column_name"]): row
        for row in read_back(session_id, "SELECT * FROM sem_metadata")
    }

    assert report["metadata_rows"] == 5 == len(rows)
    total = rows[("orders", "total")]
    assert total["taxonomy_label"] == "revenue"
    assert total["data_type"] == "currency" and total["sql_type"] == "NUMERIC(15, 2)"
    assert total["is_additive"] and total["table_type"] == "fact"

    reference = rows[("orders", "customer_id")]
    assert reference["is_foreign_key"] and reference["references_table"] == "customers"
    assert reference["reference_enforced"]
    assert rows[("customers", "customer_id")]["is_primary_key"]
    drop_export(session_id)


def test_the_description_says_what_the_column_is_for(shop):
    from app.export.metadata import describe

    tables, columns, keys = shop
    plan = build_plan(
        tables,
        columns,
        keys,
        relationships=[_fk("orders", "customer_id", "customers", "customer_id")],
        profiles={"orders": {"effective_type": "fact"}},
    )
    orders = plan.table("orders")

    sentence = describe(orders.column("customer_id"), orders)

    assert "customer id in orders" in sentence
    assert "foreign identifier" in sentence
    assert "fact table" in sentence
    assert "references customers.customer_id" in sentence


def test_an_unnamed_table_type_reads_as_a_data_table_not_as_unknown(shop):
    from app.export.metadata import describe

    tables, columns, keys = shop
    plan = build_plan(tables, columns, keys)
    spec = plan.table("customers")

    assert "data table" in describe(spec.column("city"), spec)
    assert "unknown" not in describe(spec.column("city"), spec)


def test_every_described_column_carries_a_vector(shop):
    """384 dimensions, from the same local model the rest of the platform uses.

    Phase 5 retrieves on these; a missing vector is a column the user's question
    can never reach.
    """

    tables, columns, keys = shop
    session_id = _session_id()

    report = run_export(session_id, build_plan(tables, columns, keys), tables).to_dict()
    rows = read_semantic_layer(session_id)

    assert report["embedded_rows"] == report["metadata_rows"]
    assert all(len(row["embedding"]) == 384 for row in rows)
    assert all(isinstance(row["sample_values"], list) for row in rows)
    drop_export(session_id)


def test_an_export_still_succeeds_when_nothing_can_be_embedded(shop, monkeypatch):
    """Degradability: no model means no ranking, not no database."""

    from app.export import metadata as semantic_metadata

    def explode(_texts):
        raise RuntimeError("no model on this machine")

    monkeypatch.setattr(
        semantic_metadata, "get_embedder", lambda: type("E", (), {"encode": staticmethod(explode)})
    )
    tables, columns, keys = shop
    session_id = _session_id()

    report = run_export(session_id, build_plan(tables, columns, keys), tables).to_dict()

    assert report["row_count"] == 5
    assert report["metadata_rows"] == 5 and report["embedded_rows"] == 0
    assert report["vector_index"] is False
    assert read_back(session_id, "SELECT COUNT(*) AS n FROM sem_metadata") == [{"n": 5}]
    drop_export(session_id)


def test_a_reference_that_is_not_enforced_is_still_offered_as_a_join(shop):
    """Phase 5 has to be able to join on it, and to know it is not policed."""

    tables, columns, keys = shop
    tables["orders"].loc[1, "customer_id"] = "C-999"
    session_id = _session_id()
    plan = build_plan(
        tables, columns, keys, relationships=[_fk("orders", "customer_id", "customers", "customer_id")]
    )

    run_export(session_id, plan, tables)
    (row,) = read_back(
        session_id,
        "SELECT references_table, reference_enforced FROM sem_metadata "
        "WHERE table_name = 'orders' AND column_name = 'customer_id'",
    )

    assert row["references_table"] == "customers"
    assert not row["reference_enforced"]
    drop_export(session_id)


def test_the_json_bundle_is_portable_and_carries_no_vectors(shop):
    """Tool-agnostic by design: names, types, keys, labels and sentences.

    The vectors are left out because they are reproducible from the sentences
    and would multiply the file for no information.
    """

    import json

    from app.export.metadata import bundle

    tables, columns, keys = shop
    plan = build_plan(
        tables, columns, keys, relationships=[_fk("orders", "customer_id", "customers", "customer_id")]
    )

    payload = bundle(plan)
    text = json.dumps(payload)

    assert payload["format"] == "semanticlayer/1"
    assert [table["name"] for table in payload["tables"]] == ["customers", "orders"]
    assert "embedding" not in text
    orders = next(table for table in payload["tables"] if table["name"] == "orders")
    assert orders["primary_key"] == ["order_id"]
    assert orders["foreign_keys"][0]["target_table"] == "customers"
    assert {column["column_name"] for column in orders["columns"]} == {
        "order_id",
        "customer_id",
        "total",
    }


def test_the_bundle_names_the_original_columns_as_well_as_the_new_ones():
    """A user has to be able to find their own column after it was renamed."""

    from app.export.metadata import bundle

    tables = {"Sales Data": pd.DataFrame({"Total Amount (৳)": [1.0]})}
    columns = {"Sales Data": [_column("Total Amount (৳)", "currency", "revenue")]}

    payload = bundle(build_plan(tables, columns))
    column = payload["tables"][0]["columns"][0]

    assert payload["tables"][0]["source_name"] == "Sales Data"
    assert column["source_column"] == "Total Amount (৳)"
    assert column["column_name"] == "total_amount"


# ---------------------------------------------------------------------------
# the API
# ---------------------------------------------------------------------------


def _uploaded(client, workbook) -> str:
    session_id = client.post("/api/sessions", json={"name": "phase 4"}).json()["id"]
    with workbook.open("rb") as handle:
        response = client.post(
            f"/api/sessions/{session_id}/upload", files={"file": (workbook.name, handle)}
        )
    assert response.status_code == 200, response.text
    return session_id


def test_the_export_endpoint_builds_a_database_and_reports_it(client, relational_workbook):
    session_id = _uploaded(client, relational_workbook)

    response = client.post(f"/api/sessions/{session_id}/export")

    assert response.status_code == 200, response.text
    payload = response.json()
    report = payload["report"]
    assert payload["table_count"] == 4
    assert payload["row_count"] == report["row_count"] > 0
    assert payload["column_count"] == report["metadata_rows"]
    assert report["ddl"].startswith("CREATE")
    assert client.get(f"/api/sessions/{session_id}/export").json()["exported"] is True


def test_a_confirmed_reference_reaches_the_exported_schema(client, relational_workbook):
    """The whole pipeline, ending in a constraint the database enforces."""

    session_id = _uploaded(client, relational_workbook)
    detected = client.post(f"/api/sessions/{session_id}/relationships").json()
    edge = next(
        r
        for r in detected["relationships"]
        if r["rel_type"] == "foreign_key" and r["to_table"] == "orders"
    )
    client.post(f"/api/sessions/{session_id}/relationships/{edge['id']}", json={"confirmed": True})

    report = client.post(f"/api/sessions/{session_id}/export").json()["report"]

    enforced = [fk for fk in report["foreign_keys"] if fk["enforced"]]
    assert any(fk["target_table"] == "orders" for fk in enforced)
    assert "FOREIGN KEY" in report["ddl"]


def test_an_unconfirmed_reference_does_not_reach_it(client, relational_workbook):
    session_id = _uploaded(client, relational_workbook)
    client.post(f"/api/sessions/{session_id}/relationships")

    report = client.post(f"/api/sessions/{session_id}/export").json()["report"]

    assert report["foreign_keys"] == []
    assert "FOREIGN KEY" not in report["ddl"]


def test_the_exported_database_answers_a_question_about_the_workbook(
    client, relational_workbook
):
    session_id = _uploaded(client, relational_workbook)
    client.post(f"/api/sessions/{session_id}/export")

    rows = read_back(
        session_id, "SELECT COUNT(*) AS n FROM orders WHERE status = 'Shipped'"
    )

    assert rows[0]["n"] > 0


def test_the_table_types_from_profiling_reach_the_semantic_layer(
    client, relational_workbook
):
    session_id = _uploaded(client, relational_workbook)
    client.post(f"/api/sessions/{session_id}/relationships")  # ends with profiling

    client.post(f"/api/sessions/{session_id}/export")
    rows = {row["table_name"]: row for row in read_semantic_layer(session_id)}

    assert rows["orders"]["table_type"] == "fact"
    assert "fact table" in rows["orders"]["semantic_description"]


def test_the_ddl_and_the_bundle_can_be_downloaded(client, relational_workbook):
    session_id = _uploaded(client, relational_workbook)
    client.post(f"/api/sessions/{session_id}/export")

    ddl = client.get(f"/api/sessions/{session_id}/export/ddl")
    bundle = client.get(f"/api/sessions/{session_id}/export/bundle")

    assert ddl.status_code == 200 and "CREATE TABLE" in ddl.text
    assert bundle.json()["format"] == "semanticlayer/1"
    assert len(bundle.json()["tables"]) == 4


def test_nothing_can_be_downloaded_before_an_export(client, relational_workbook):
    session_id = _uploaded(client, relational_workbook)

    assert client.get(f"/api/sessions/{session_id}/export").json()["exported"] is False
    assert client.get(f"/api/sessions/{session_id}/export/ddl").status_code == 404
    assert client.get(f"/api/sessions/{session_id}/export/bundle").status_code == 404


def test_exporting_an_empty_dataset_says_so(client):
    session_id = client.post("/api/sessions", json={"name": "empty"}).json()["id"]

    response = client.post(f"/api/sessions/{session_id}/export")

    assert response.status_code == 409
    assert "upload a file" in response.json()["detail"]


def test_the_export_belongs_to_its_owner(client, other_client, relational_workbook):
    session_id = _uploaded(client, relational_workbook)
    client.post(f"/api/sessions/{session_id}/export")

    for call in (
        lambda: other_client.post(f"/api/sessions/{session_id}/export"),
        lambda: other_client.get(f"/api/sessions/{session_id}/export"),
        lambda: other_client.get(f"/api/sessions/{session_id}/export/ddl"),
        lambda: other_client.get(f"/api/sessions/{session_id}/export/bundle"),
    ):
        assert call().status_code == 404


def test_deleting_the_dataset_deletes_the_database_it_exported(
    client, relational_workbook
):
    """The exported schema lives outside the metadata cascade, so it has to be
    dropped explicitly — otherwise a deleted user's rows stay readable."""

    session_id = _uploaded(client, relational_workbook)
    client.post(f"/api/sessions/{session_id}/export")
    assert read_back(session_id, "SELECT COUNT(*) AS n FROM customers")[0]["n"] == 30

    assert client.delete(f"/api/sessions/{session_id}").status_code == 204

    with pytest.raises(Exception):
        read_back(session_id, "SELECT 1 FROM customers")


def test_re_exporting_through_the_api_replaces_rather_than_appends(
    client, relational_workbook
):
    session_id = _uploaded(client, relational_workbook)

    first = client.post(f"/api/sessions/{session_id}/export").json()
    second = client.post(f"/api/sessions/{session_id}/export").json()

    assert first["row_count"] == second["row_count"]
    assert read_back(session_id, "SELECT COUNT(*) AS n FROM customers") == [{"n": 30}]


def test_a_table_too_small_to_analyse_is_still_exported(client):
    """Statistics over three rows say nothing; the three rows are still data.

    A database that silently omits a sheet the user uploaded is wrong in the
    one way this platform cannot afford, so the small table is exported as text
    and the omission of its types is stated instead.
    """

    csv = b"code,label\nA,Active\nB,Closed\n"
    session_id = client.post("/api/sessions", json={"name": "tiny"}).json()["id"]
    upload = client.post(
        f"/api/sessions/{session_id}/upload", files={"file": ("statuses.csv", csv, "text/csv")}
    )
    assert upload.status_code == 200, upload.text

    report = client.post(f"/api/sessions/{session_id}/export").json()["report"]

    assert [table["name"] for table in report["tables"]] == ["statuses"]
    assert report["row_count"] == 2
    assert any("too small for semantic analysis" in note for note in report["tables"][0]["notes"])
    assert read_back(session_id, "SELECT code FROM statuses ORDER BY code") == [
        {"code": "A"},
        {"code": "B"},
    ]


def test_deleting_the_account_drops_every_schema_it_materialized(client, relational_workbook):
    """FR-15, at the account level.

    Deleting the dataset was already covered; an account holds several, and a
    ``ds_…`` schema left behind by a deleted account holds the rows themselves
    in a database that keeps running.
    """

    first = _uploaded(client, relational_workbook)
    second = _uploaded(client, relational_workbook)
    client.post(f"/api/sessions/{first}/export")
    client.post(f"/api/sessions/{second}/export")

    response = client.post(
        "/api/auth/delete", json={"password": "correct-horse-battery"}
    )

    assert response.status_code == 200, response.text
    assert response.json()["deleted_datasets"] == 2
    for session_id in (first, second):
        with pytest.raises(Exception):
            read_back(session_id, "SELECT 1 FROM customers")
