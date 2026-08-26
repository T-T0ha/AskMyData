"""The control plane as the ER model describes it.

Phase 4 built the exported database and wrote its semantic layer into it.
These tests cover the other half of §4.3: that the *application's own* record
of the layer is complete enough to be the thing Phase 5 reads — key and
reference roles on the column semantics, a one-line description and an
embedding per table, a version on the dataset — and that the layer renders as
documentation a person can keep (FR-18).

The distinction under test throughout is between what was *decided* and what is
*derived from* it.  A confirmed foreign key is decided once and lives in
``relationships``; the ``is_foreign_key`` flag beside the column is a copy, and
the interesting cases are all about the copy staying honest when the decision
changes underneath it.
"""

from __future__ import annotations

import uuid

import pandas as pd
import pytest
from sqlalchemy import select

from app.db.base import get_session_factory
from app.db.models import ColumnSemantics, IngestionSession, SemanticLayerVersion, SheetRecord
from app.export.documentation import render_markdown
from app.semantics.table_profile import TableProfile
from app.semantics.table_summary import compose, describe_tables, relationship_map


def _uploaded(client, workbook) -> str:
    session_id = client.post("/api/sessions", json={"name": "alignment"}).json()["id"]
    with workbook.open("rb") as handle:
        response = client.post(
            f"/api/sessions/{session_id}/upload", files={"file": (workbook.name, handle)}
        )
    assert response.status_code == 200, response.text
    return session_id


def _db():
    return get_session_factory()()


def _columns(session_id: str, table: str) -> dict[str, ColumnSemantics]:
    with _db() as db:
        rows = (
            db.execute(
                select(ColumnSemantics)
                .where(ColumnSemantics.session_id == session_id)
                .where(ColumnSemantics.table_name == table)
            )
            .scalars()
            .all()
        )
        return {row.column_name: row for row in rows}


def _sheet(session_id: str, table: str) -> SheetRecord:
    with _db() as db:
        return db.execute(
            select(SheetRecord)
            .where(SheetRecord.session_id == session_id)
            .where(SheetRecord.table_name == table)
        ).scalar_one()


def _confirm_a_reference(client, session_id: str) -> dict:
    """Detect relationships and confirm the first foreign key.  Returns it."""

    detected = client.post(f"/api/sessions/{session_id}/relationships").json()
    edge = next(r for r in detected["relationships"] if r["rel_type"] == "foreign_key")
    client.post(
        f"/api/sessions/{session_id}/relationships/{edge['id']}", json={"confirmed": True}
    )
    return edge


# ---------------------------------------------------------------------------
# key and reference roles on the column semantics
# ---------------------------------------------------------------------------


def test_a_confirmed_key_is_projected_onto_the_column_semantics(client, relational_workbook):
    session_id = _uploaded(client, relational_workbook)
    client.post(f"/api/sessions/{session_id}/semantics")
    client.post(f"/api/sessions/{session_id}/relationships")  # ends with profiling

    customers = _columns(session_id, "customers")

    assert customers["customer_id"].is_primary_key is True
    assert customers["city"].is_primary_key is False


def test_a_confirmed_reference_is_projected_onto_the_column_semantics(
    client, relational_workbook
):
    session_id = _uploaded(client, relational_workbook)
    client.post(f"/api/sessions/{session_id}/semantics")
    edge = _confirm_a_reference(client, session_id)

    column = _columns(session_id, edge["from_table"])[edge["from_column"]]

    assert column.is_foreign_key is True
    assert column.references_table == edge["to_table"]
    assert column.references_column == edge["to_column"]


def test_the_role_is_visible_through_the_semantics_endpoint(client, relational_workbook):
    """Phase 5 reads this table; so, therefore, can the API."""

    session_id = _uploaded(client, relational_workbook)
    client.post(f"/api/sessions/{session_id}/semantics")
    edge = _confirm_a_reference(client, session_id)

    payload = client.get(f"/api/sessions/{session_id}/semantics").json()
    table = next(t for t in payload["tables"] if t["table"] == edge["from_table"])
    column = next(c for c in table["columns"] if c["name"] == edge["from_column"])

    assert column["is_foreign_key"] is True
    assert column["references_table"] == edge["to_table"]


def test_withdrawing_a_reference_withdraws_its_role(client, relational_workbook):
    """The copy has to be able to stop being true, not only to start."""

    session_id = _uploaded(client, relational_workbook)
    client.post(f"/api/sessions/{session_id}/semantics")
    edge = _confirm_a_reference(client, session_id)
    assert _columns(session_id, edge["from_table"])[edge["from_column"]].is_foreign_key

    client.post(
        f"/api/sessions/{session_id}/relationships/{edge['id']}", json={"confirmed": False}
    )

    column = _columns(session_id, edge["from_table"])[edge["from_column"]]
    assert column.is_foreign_key is False
    assert column.references_table is None


def test_rerunning_field_semantics_keeps_the_roles(client, relational_workbook):
    """Phase 1 replaces every column row; the roles outlive the replacement."""

    session_id = _uploaded(client, relational_workbook)
    client.post(f"/api/sessions/{session_id}/semantics")
    edge = _confirm_a_reference(client, session_id)

    client.post(f"/api/sessions/{session_id}/semantics")  # re-runs from scratch

    column = _columns(session_id, edge["from_table"])[edge["from_column"]]
    assert column.is_foreign_key is True
    assert column.references_table == edge["to_table"]


def test_the_export_completes_the_roles_even_without_a_profiling_pass(
    client, relational_workbook
):
    """A user who goes straight to the export still gets a complete layer."""

    session_id = _uploaded(client, relational_workbook)
    assert client.post(f"/api/sessions/{session_id}/export").status_code == 200

    customers = _columns(session_id, "customers")
    assert customers["customer_id"].is_primary_key is True


# ---------------------------------------------------------------------------
# one sentence per table
# ---------------------------------------------------------------------------


def test_the_sentence_names_the_type_the_key_and_the_measures():
    sentence = compose(
        "orders",
        profile=TableProfile.from_dict({"table_type": "fact", "table": "orders"}),
        columns=[
            {"name": "order_id", "is_additive": False},
            {"name": "customer_id", "is_additive": False},
            {"name": "total_amount", "is_additive": True},
            {"name": "order_date", "is_additive": False},
        ],
        primary_key=["order_id"],
        references=[("customer_id", "customers")],
        referenced_by=["order_lines"],
        row_count=1240,
        column_count=4,
    )

    assert "orders is a fact table with 1,240 rows and 4 columns." in sentence
    assert "Each row is identified by order id." in sentence
    assert "It references customers through customer id." in sentence
    assert "It is referenced by order_lines." in sentence
    assert "It measures total amount." in sentence
    assert "It also records order date." in sentence
    # The reference column is named once, as a reference, not again as content.
    assert sentence.count("customer id") == 1


def test_a_table_without_a_key_says_so():
    sentence = compose("notes", columns=[{"name": "body"}], row_count=4, column_count=1)

    assert "notes is a data table with 4 rows and 1 columns." in sentence
    assert "It has no primary key, so no other table can point at its rows." in sentence


def test_a_long_column_list_is_counted_rather_than_recited():
    sentence = compose(
        "wide",
        columns=[{"name": f"field_{i}"} for i in range(20)],
        row_count=3,
        column_count=20,
    )

    assert "and 14 more." in sentence
    assert "field_19" not in sentence


def test_only_confirmed_relationships_reach_the_sentence():
    outgoing, incoming = relationship_map(
        [
            {
                "rel_type": "foreign_key",
                "from_table": "orders",
                "from_column": "customer_id",
                "to_table": "customers",
                "to_column": "customer_id",
                "status": "proposed",
            }
        ]
    )

    assert outgoing == {}
    assert incoming == {}


def test_every_counted_table_is_described_even_without_a_profile():
    descriptions = describe_tables(
        profiles={},
        columns={"tiny": [{"name": "code"}]},
        key_analyses={},
        relationships=[],
        counts={"tiny": (2, 1)},
    )

    assert descriptions["tiny"].startswith("tiny is a data table with 2 rows")


def test_profiling_stores_a_description_and_an_embedding(client, relational_workbook):
    session_id = _uploaded(client, relational_workbook)
    client.post(f"/api/sessions/{session_id}/semantics")
    client.post(f"/api/sessions/{session_id}/relationships")

    sheet = _sheet(session_id, "customers")

    assert sheet.table_description.startswith("customers is a")
    assert sheet.table_embedding is not None
    assert len(sheet.table_embedding) == 384


def test_the_sentence_counts_the_rows_that_are_there_now(client, relational_workbook):
    """Cleaning changes the row count; the sheet record remembers the upload."""

    session_id = _uploaded(client, relational_workbook)
    with _db() as db:
        sheet = db.execute(
            select(SheetRecord)
            .where(SheetRecord.session_id == session_id)
            .where(SheetRecord.table_name == "customers")
        ).scalar_one()
        stale = sheet.row_count
        sheet.row_count = stale + 999
        db.commit()

    client.post(f"/api/sessions/{session_id}/profiles")

    assert f"{stale:,} rows" in _sheet(session_id, "customers").table_description


def test_the_description_is_visible_on_the_session(client, relational_workbook):
    session_id = _uploaded(client, relational_workbook)
    client.post(f"/api/sessions/{session_id}/profiles")

    sheets = client.get(f"/api/sessions/{session_id}").json()["sheets"]
    customers = next(s for s in sheets if s["table_name"] == "customers")

    assert customers["table_description"]
    assert customers["description_source"] == "composed"


def test_correcting_the_table_type_rewrites_the_description(client, relational_workbook):
    session_id = _uploaded(client, relational_workbook)
    client.post(f"/api/sessions/{session_id}/semantics")
    client.post(f"/api/sessions/{session_id}/relationships")
    before = _sheet(session_id, "customers").table_description

    client.patch(
        f"/api/sessions/{session_id}/tables/customers/profile",
        json={"table_type": "bridge"},
    )

    after = _sheet(session_id, "customers").table_description
    assert after != before
    assert "customers is a bridge table" in after


class _FakeClaude:
    """A model that always answers, so the two paths can be told apart."""

    available = True

    def __init__(self) -> None:
        self.calls: list[list[dict]] = []

    def describe_tables(self, tables):
        self.calls.append(tables)
        return {table["name"]: f"claude on {table['name']}" for table in tables}


def test_claude_rewrites_the_sentence_and_the_composed_one_survives(
    client, relational_workbook
):
    from app.api import services

    session_id = _uploaded(client, relational_workbook)
    client.post(f"/api/sessions/{session_id}/semantics")
    fake = _FakeClaude()

    with _db() as db:
        record = db.get(IngestionSession, session_id)
        services.profile_tables(db, record, claude=fake)
        db.commit()

    sheet = _sheet(session_id, "customers")
    assert sheet.llm_table_description == "claude on customers"
    assert sheet.table_description.startswith("customers is a")
    assert sheet.effective_description == "claude on customers"
    # The model is told what the table is, and never shown a row of it.
    payload = next(t for t in fake.calls[0] if t["name"] == "customers")
    assert payload["columns"]
    assert "sample_values" not in str(payload)


def test_a_stale_model_sentence_is_dropped_when_the_facts_change(
    client, relational_workbook
):
    from app.api import services

    session_id = _uploaded(client, relational_workbook)
    client.post(f"/api/sessions/{session_id}/semantics")
    with _db() as db:
        record = db.get(IngestionSession, session_id)
        services.profile_tables(db, record, claude=_FakeClaude())
        db.commit()
    assert _sheet(session_id, "customers").llm_table_description is not None

    client.patch(
        f"/api/sessions/{session_id}/tables/customers/profile",
        json={"table_type": "bridge"},
    )

    sheet = _sheet(session_id, "customers")
    assert sheet.llm_table_description is None
    assert sheet.effective_description == sheet.table_description


def test_the_description_reaches_the_exported_layer(client, relational_workbook):
    session_id = _uploaded(client, relational_workbook)
    client.post(f"/api/sessions/{session_id}/export")

    bundle = client.get(f"/api/sessions/{session_id}/export/bundle").json()
    customers = next(t for t in bundle["tables"] if t["name"] == "customers")

    assert customers["description"].startswith("customers is a")


# ---------------------------------------------------------------------------
# versioning
# ---------------------------------------------------------------------------


def test_the_first_export_records_where_the_data_went(client, relational_workbook):
    session_id = _uploaded(client, relational_workbook)
    export = client.post(f"/api/sessions/{session_id}/export").json()

    session = client.get(f"/api/sessions/{session_id}").json()

    assert session["semantic_version"] == 1
    assert session["db_schema_name"] == export["target"]
    assert session["enriched_at"] is not None
    assert export["semantic_version"] == 1


def test_re_exporting_increments_the_version_and_keeps_the_old_one(
    client, relational_workbook
):
    session_id = _uploaded(client, relational_workbook)
    client.post(f"/api/sessions/{session_id}/export")
    second = client.post(f"/api/sessions/{session_id}/export").json()

    versions = client.get(f"/api/sessions/{session_id}/export/versions").json()

    assert second["semantic_version"] == 2
    assert versions["current"] == 2
    assert [row["version"] for row in versions["versions"]] == [2, 1]
    # The archived layer is the document, not a summary of it.
    first = client.get(f"/api/sessions/{session_id}/export/bundle?version=1").json()
    assert first["format"] == "semanticlayer/1"
    assert first["tables"]


def test_asking_for_a_version_that_was_never_built_is_a_404(client, relational_workbook):
    session_id = _uploaded(client, relational_workbook)
    client.post(f"/api/sessions/{session_id}/export")

    assert client.get(f"/api/sessions/{session_id}/export/bundle?version=7").status_code == 404


def test_the_archive_is_bounded(client, relational_workbook, monkeypatch):
    """A history of whole schemas cannot be allowed to grow without limit."""

    from app.api import services

    monkeypatch.setattr(services, "MAX_LAYER_VERSIONS", 2)
    session_id = _uploaded(client, relational_workbook)
    for _ in range(3):
        client.post(f"/api/sessions/{session_id}/export")

    versions = client.get(f"/api/sessions/{session_id}/export/versions").json()["versions"]

    assert [row["version"] for row in versions] == [3, 2]


def test_deleting_the_dataset_deletes_its_archived_layers(client, relational_workbook):
    session_id = _uploaded(client, relational_workbook)
    client.post(f"/api/sessions/{session_id}/export")

    client.delete(f"/api/sessions/{session_id}")

    with _db() as db:
        remaining = (
            db.execute(
                select(SemanticLayerVersion).where(
                    SemanticLayerVersion.session_id == session_id
                )
            )
            .scalars()
            .all()
        )
    assert remaining == []


def test_the_embedding_column_reaches_a_database_that_predates_it(client):
    """The new columns carry a dialect-dependent type, so the healer must too.

    ``table_embedding`` is a ``vector(384)`` on PostgreSQL and JSON elsewhere.
    Adding it by hand means spelling that choice a second time, which is the
    part worth a test: a JSON column on PostgreSQL is an embedding no index can
    reach.
    """

    from sqlalchemy import text

    from app.db.base import get_engine, init_db, pending_schema_changes

    with get_engine().begin() as connection:
        connection.execute(text("ALTER TABLE sheet_records DROP COLUMN table_embedding"))

    assert pending_schema_changes() == ["sheet_records.table_embedding"]
    assert init_db()["added_columns"] == ["sheet_records.table_embedding"]
    assert pending_schema_changes() == []


# ---------------------------------------------------------------------------
# documentation (FR-18)
# ---------------------------------------------------------------------------


def test_the_documentation_describes_every_table_and_its_columns(
    client, relational_workbook
):
    session_id = _uploaded(client, relational_workbook)
    client.post(f"/api/sessions/{session_id}/export")

    response = client.get(f"/api/sessions/{session_id}/export/documentation")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/markdown")
    document = response.text
    assert document.startswith("# alignment — schema documentation")
    assert "## `customers`" in document
    assert "| Column | Type | Meaning | Role | Blank | Examples |" in document
    # A contents entry whose link does not land is worse than none at all.
    assert "- [`order_lines`](#order_lines)" in document
    assert "`customer_id`" in document
    assert "semantic layer version 1" in document


def test_a_confirmed_reference_is_documented_as_enforced(client, relational_workbook):
    session_id = _uploaded(client, relational_workbook)
    client.post(f"/api/sessions/{session_id}/semantics")
    edge = _confirm_a_reference(client, session_id)
    client.post(f"/api/sessions/{session_id}/export")

    document = client.get(f"/api/sessions/{session_id}/export/documentation").text

    assert "## Relationships" in document
    assert f"`{edge['from_table']}.{edge['from_column']}`" in document


def test_documentation_can_be_produced_for_an_earlier_version(client, relational_workbook):
    session_id = _uploaded(client, relational_workbook)
    client.post(f"/api/sessions/{session_id}/export")
    client.post(f"/api/sessions/{session_id}/export")

    document = client.get(
        f"/api/sessions/{session_id}/export/documentation?version=1"
    ).text

    assert "semantic layer version 1" in document


def test_documentation_before_an_export_is_a_404(client, relational_workbook):
    session_id = _uploaded(client, relational_workbook)

    assert client.get(f"/api/sessions/{session_id}/export/documentation").status_code == 404


def test_a_spreadsheet_cannot_break_the_table_it_is_printed_in():
    """Column names and values are user input; a pipe is not a column border."""

    document = render_markdown(
        {
            "tables": [
                {
                    "name": "prices",
                    "source_name": "prices",
                    "table_type": "fact",
                    "description": "",
                    "row_count": 1,
                    "primary_key": [],
                    "foreign_keys": [],
                    "notes": [],
                    "columns": [
                        {
                            "column_name": "label",
                            "source_column": "a | b",
                            "sql_type": "TEXT",
                            "taxonomy_label": "unknown",
                            "null_ratio": 0.0,
                            "sample_values": ["one | two", "three\nfour"],
                            "semantic_description": "",
                        }
                    ],
                }
            ]
        },
        dataset="risky",
    )

    body = [line for line in document.splitlines() if line.startswith("| `label`")]
    assert len(body) == 1
    # Six declared columns means seven pipes; an unescaped one would add more.
    assert body[0].count("|") - body[0].count("\\|") == 7
    assert "three four" in body[0]  # the newline was folded, not printed
