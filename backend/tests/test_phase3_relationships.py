"""Phase 3 — relationship detection and interactive validation.

The detectors are tested against arithmetic that can be checked by hand, and
the API is tested for the property that matters more than any score: a
relationship the user has decided on stays decided.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from app.ingestion.keys import KeyAnalysis, discover_keys
from app.relationships.dependencies import detect_dependencies
from app.relationships.evidence import build_evidence, foreign_key_evidence
from app.relationships.foreign_keys import (
    DISTINCT_WEIGHT,
    OVERLAP_WEIGHT,
    declared_foreign_keys,
    detect_foreign_keys,
    score_pair,
    skipped_tables,
)


def _keys(tables: dict[str, pd.DataFrame]) -> dict[str, KeyAnalysis]:
    return {name: discover_keys(df) for name, df in tables.items()}


# ---------------------------------------------------------------------------
# foreign keys — SemTabla equations 1-3
# ---------------------------------------------------------------------------


def test_the_score_is_the_papers_two_ratios():
    """Score = w1·ratio_overlap + w2·ratio_distinct, checked by hand.

    Source holds {1,2,3,4}; target key holds {1,2,3,4,5,6,7,8}.  Every source
    value is a real key (overlap 4/4 = 1.0) but they reach only half of it
    (distinct 4/8 = 0.5).
    """

    source = pd.Series([1, 2, 3, 4] * 5)
    target = pd.Series(list(range(1, 9)))

    base, overlap, distinct, matched, source_n, target_n, orphans = score_pair(source, target)

    assert (overlap, distinct) == (1.0, 0.5)
    assert matched == 4 and source_n == 4 and target_n == 8
    assert orphans == []
    assert base == pytest.approx(OVERLAP_WEIGHT * 1.0 + DISTINCT_WEIGHT * 0.5)


def test_overlap_and_distinct_answer_different_questions():
    """A column of one repeated id covers the key perfectly and reaches almost
    none of it — which is why the paper needs both ratios, not just the first."""

    source = pd.Series(["C-001"] * 40)
    target = pd.Series([f"C-{i:03d}" for i in range(1, 101)])

    _, overlap, distinct, *_ = score_pair(source, target)

    assert overlap == 1.0, "every value is a real key"
    assert distinct == pytest.approx(0.01), "but it touches one key in a hundred"


def test_finds_the_references_in_a_real_workbook(relational_tables):
    keys = _keys(relational_tables)

    found = {
        (c.from_table, c.from_column, c.to_table, c.to_column): c
        for c in detect_foreign_keys(relational_tables, keys)
    }

    assert ("orders", "customer_id", "customers", "customer_id") in found
    assert ("order_lines", "order_id", "orders", "order_id") in found

    orphaned = found[("orders", "customer_id", "customers", "customer_id")]
    # Two of the 45 orders name a customer that does not exist, so the
    # reference is proposed but its overlap is not 1.0.
    assert orphaned.ratio_overlap < 1.0
    assert "C-999" in orphaned.orphan_values
    assert orphaned.notes, "incomplete referential integrity has to be said out loud"


def test_a_column_gets_at_most_one_proposed_target():
    """The paper retains the best match.  A column that appears to reference
    three tables is reporting a coincidence in two of them."""

    shared = [f"K-{i:03d}" for i in range(1, 41)]
    tables = {
        "facts": pd.DataFrame({"ref": shared, "amount": range(40)}),
        "dim_a": pd.DataFrame({"a_id": shared, "label": ["x"] * 40}),
        "dim_b": pd.DataFrame({"b_id": shared, "label": ["y"] * 40}),
    }

    candidates = detect_foreign_keys(tables, _keys(tables))
    sources = [(c.from_table, c.from_column) for c in candidates]

    assert sources.count(("facts", "ref")) == 1


def test_a_composite_key_is_never_a_target():
    """The paper's preprocessing step.  Half of a two-column key is not a key,
    and matching it produces exactly the plausible nonsense this phase avoids."""

    composite = pd.DataFrame(
        {
            "order_id": [f"ORD-{i // 2}" for i in range(40)],
            "line_no": [i % 2 for i in range(40)],
            "sku": [f"SKU-{i % 8}" for i in range(40)],
        }
    )
    tables = {
        "lines": composite,
        "other": pd.DataFrame({"order_id": [f"ORD-{i // 2}" for i in range(40)]}),
    }
    keys = _keys(tables)
    assert len(keys["lines"].candidates[0].columns) == 2, "fixture must have a composite key"

    candidates = detect_foreign_keys(tables, keys)

    assert not [c for c in candidates if c.to_table == "lines"]


def test_small_tables_are_left_alone(relational_tables):
    """With four rows, values land inside another column's by coincidence often
    enough that any verdict would be noise."""

    keys = _keys(relational_tables)
    candidates = detect_foreign_keys(relational_tables, keys)

    assert "regions" in skipped_tables(relational_tables)
    assert not [c for c in candidates if "regions" in (c.from_table, c.to_table)]


def test_measurements_and_flags_are_not_references():
    """A price that happens to fall inside a key's value set is not a
    reference, and a boolean cannot point at anything."""

    tables = {
        "sales": pd.DataFrame(
            {
                "sale_id": [f"S-{i:03d}" for i in range(1, 41)],
                "amount": [float(i) for i in range(1, 41)],
                "is_paid": [i % 2 == 0 for i in range(1, 41)],
            }
        ),
        "codes": pd.DataFrame({"code": list(range(1, 41))}),
    }

    candidates = detect_foreign_keys(tables, _keys(tables))

    assert not [c for c in candidates if c.from_column in {"amount", "is_paid"}]


def test_a_declared_foreign_key_is_adopted_not_rescored():
    """A database is the authority on its own constraints.  A declared
    reference that scores badly means the loaded rows are a subset."""

    tables = {
        "orders": pd.DataFrame({"customer_id": ["C-1", "C-2"]}),
        "customers": pd.DataFrame({"customer_id": ["C-1", "C-2", "C-3"]}),
    }
    schemas = {
        "orders": {
            "foreign_keys": [
                {
                    "columns": ["customer_id"],
                    "references_table": "customers",
                    "references_columns": ["customer_id"],
                }
            ]
        }
    }

    declared = declared_foreign_keys(schemas, tables)

    assert len(declared) == 1
    assert declared[0].origin.value == "declared"
    assert "declared" in declared[0].explanation()
    # Both tables are far below the row floor, so detection alone finds nothing.
    assert detect_foreign_keys(tables, _keys(tables), native_schemas=schemas) == declared


# ---------------------------------------------------------------------------
# functional dependencies
# ---------------------------------------------------------------------------


def test_finds_a_hierarchy(relational_tables):
    found = detect_dependencies(
        relational_tables["customers"], "customers", key_columns=["customer_id"]
    )
    pairs = {(d.determinant, d.dependent): d for d in found}

    assert ("city", "region") in pairs
    dependency = pairs[("city", "region")]
    assert dependency.exact
    assert dependency.score == 1.0
    assert dependency.witness_groups == 6


def test_a_key_is_not_reported_as_a_determinant(relational_tables):
    """A key determines every column in its table by definition.  Reporting
    those arrows would bury the ones that say something about the data."""

    found = detect_dependencies(
        relational_tables["customers"], "customers", key_columns=["customer_id"]
    )

    assert not [d for d in found if d.determinant == "customer_id"]


def test_values_that_occur_once_do_not_count_as_evidence():
    """A determinant value appearing on a single row agrees with itself no
    matter what the data says.  Counting those inflates every score."""

    # 30 ids over 45 rows: 15 repeat, 15 are singletons.  The repeating ones
    # disagree about status, so the dependency is false — but scored over all
    # 30 groups it would read 0.97 and be reported as almost-a-rule.
    ids = [f"C-{i:03d}" for i in range(1, 16)] * 2 + [f"C-{i:03d}" for i in range(16, 31)]
    status = ["Shipped", "Pending"] * 15 + ["Shipped"] * 15
    df = pd.DataFrame({"customer_id": ids, "status": status, "note": ["n"] * 45})

    found = detect_dependencies(df, "orders")

    assert not [d for d in found if (d.determinant, d.dependent) == ("customer_id", "status")]


def test_an_almost_rule_is_reported_as_approximate_with_its_violations():
    city = ["Dhaka"] * 10 + ["Sylhet"] * 10 + ["Khulna"] * 10
    region = ["Central"] * 10 + ["North-East"] * 10 + ["South"] * 9 + ["Coastal"]
    df = pd.DataFrame({"city": city, "region": region, "filler": list(range(30))})

    found = detect_dependencies(df, "places", approximate_threshold=0.60)
    dependency = next(d for d in found if (d.determinant, d.dependent) == ("city", "region"))

    assert not dependency.exact
    assert dependency.rel_type.value == "approximate_dependency"
    assert dependency.score == pytest.approx(2 / 3)
    assert dependency.violations[0]["value"] == "Khulna"
    assert set(dependency.violations[0]["maps_to"]) == {"South", "Coastal"}


def test_blank_cells_are_dropped_rather_than_grouped():
    """NULL is not equal to itself in SQL, so a dependency that holds only
    because two blanks matched would not hold in the exported database."""

    df = pd.DataFrame(
        {
            "city": ["Dhaka"] * 10 + [None] * 10 + ["Sylhet"] * 10,
            "region": ["Central"] * 10 + ["A"] * 5 + ["B"] * 5 + ["North-East"] * 10,
        }
    )

    found = detect_dependencies(df, "places")
    dependency = next(d for d in found if d.determinant == "city")

    assert dependency.exact, "the blank rows must not be grouped together and counted as a violation"
    assert dependency.null_rows_dropped == 10


# ---------------------------------------------------------------------------
# evidence — SemTabla Table 1
# ---------------------------------------------------------------------------


def test_the_negative_sample_finds_the_orphans(relational_tables):
    evidence = foreign_key_evidence(
        relational_tables["orders"],
        relational_tables["customers"],
        "orders",
        "customer_id",
        "customers",
        "customer_id",
    )

    assert not evidence.holds
    assert evidence.negative.row_count == 2
    assert {row["customer_id"] for row in evidence.negative.rows} == {"C-999"}
    assert evidence.positive.row_count == 43
    assert evidence.notes


def test_a_clean_reference_has_nothing_to_contradict_it(relational_tables):
    evidence = foreign_key_evidence(
        relational_tables["order_lines"],
        relational_tables["orders"],
        "order_lines",
        "order_id",
        "orders",
        "order_id",
    )

    assert evidence.holds
    assert evidence.negative.row_count == 0
    assert evidence.positive.row_count == 90


def test_the_sql_shown_is_the_sql_that_ran(relational_tables):
    """The panel's transparency claim is only worth anything if the query is
    the one that produced the rows below it."""

    import sqlite3
    from contextlib import closing

    from app.relationships.evidence import load_frame

    evidence = foreign_key_evidence(
        relational_tables["orders"],
        relational_tables["customers"],
        "orders",
        "customer_id",
        "customers",
        "customer_id",
    )

    with closing(sqlite3.connect(":memory:")) as connection:
        for name in ("orders", "customers"):
            load_frame(connection, name, relational_tables[name])
        replayed = connection.execute(evidence.negative.sql).fetchall()

    assert len(replayed) == len(evidence.negative.rows)
    assert {row[1] for row in replayed} == {"C-999"}


def test_dependency_evidence_groups_rather_than_lists_rows(relational_tables):
    evidence = build_evidence(
        "functional_dependency", relational_tables, "customers", "city", "customers", "region"
    )

    assert evidence.holds
    assert evidence.negative.row_count == 0
    assert evidence.positive.row_count == 6, "one row per repeating city, not one per customer"
    assert {row["city"] for row in evidence.positive.rows} <= set(
        relational_tables["customers"]["city"]
    )


def test_evidence_survives_date_columns(relational_tables):
    """sqlite3 cannot bind a pandas Timestamp; the orders table has one."""

    evidence = build_evidence(
        "foreign_key", relational_tables, "orders", "customer_id", "customers", "customer_id"
    )

    assert isinstance(evidence.negative.rows[0]["order_date"], str)


# ---------------------------------------------------------------------------
# the API
# ---------------------------------------------------------------------------


def _session_with(client, workbook: Path) -> str:
    session_id = client.post("/api/sessions", json={"name": "phase 3"}).json()["id"]
    with workbook.open("rb") as handle:
        response = client.post(
            f"/api/sessions/{session_id}/upload", files={"file": (workbook.name, handle)}
        )
    assert response.status_code == 200, response.text
    return session_id


@pytest.fixture
def detected(client, relational_workbook):
    session_id = _session_with(client, relational_workbook)
    payload = client.post(f"/api/sessions/{session_id}/relationships").json()
    return session_id, payload


def test_detection_endpoint_returns_relationships_and_a_graph(detected):
    _, payload = detected

    assert payload["counts"]["foreign_keys"] >= 2
    assert payload["counts"]["proposed"] == payload["counts"]["total"]
    assert {node["id"] for node in payload["graph"]["nodes"]} >= {
        "customers",
        "orders",
        "order_lines",
    }
    assert payload["skipped_tables"] == ["regions"]

    # Key and reference flags are what the diagram colours columns by.
    customers = next(n for n in payload["graph"]["nodes"] if n["id"] == "customers")
    assert customers["primary_key"] == ["customer_id"]
    orders = next(n for n in payload["graph"]["nodes"] if n["id"] == "orders")
    flagged = {c["name"] for c in orders["columns"] if c["is_foreign_key"]}
    assert "customer_id" in flagged


def test_confirming_and_rejecting_are_remembered_across_a_re_run(client, detected):
    session_id, payload = detected
    relationships = payload["relationships"]
    confirmed = next(r for r in relationships if r["rel_type"] == "foreign_key")
    rejected = next(
        r for r in relationships if r["rel_type"] == "foreign_key" and r["id"] != confirmed["id"]
    )

    assert client.post(
        f"/api/sessions/{session_id}/relationships/{confirmed['id']}", json={"confirmed": True}
    ).json()["status"] == "confirmed"
    assert client.post(
        f"/api/sessions/{session_id}/relationships/{rejected['id']}", json={"confirmed": False}
    ).json()["status"] == "rejected"

    # Re-running detection is something a user does after cleaning changed the
    # data.  It must not resurrect a rejected edge or downgrade a confirmed one.
    again = client.post(f"/api/sessions/{session_id}/relationships").json()
    by_id = {r["id"]: r for r in again["relationships"]}

    assert by_id[confirmed["id"]]["status"] == "confirmed"
    assert by_id[rejected["id"]]["status"] == "rejected"
    assert again["created"] == 0


def test_a_rejected_edge_leaves_the_diagram_but_not_the_record(client, detected):
    session_id, payload = detected
    edge = next(r for r in payload["relationships"] if r["rel_type"] == "foreign_key")

    client.post(
        f"/api/sessions/{session_id}/relationships/{edge['id']}", json={"confirmed": False}
    )
    after = client.get(f"/api/sessions/{session_id}/relationships").json()

    assert edge["id"] in {r["id"] for r in after["relationships"]}
    assert edge["id"] not in {link["id"] for link in after["graph"]["links"]}


def test_the_evidence_endpoint_shows_both_sides(client, detected):
    session_id, payload = detected
    edge = next(
        r
        for r in payload["relationships"]
        if r["from_table"] == "orders" and r["to_table"] == "customers"
    )

    evidence = client.get(
        f"/api/sessions/{session_id}/relationships/{edge['id']}/evidence"
    ).json()["evidence"]

    assert evidence["holds"] is False
    assert evidence["negative"]["row_count"] == 2
    assert "NOT IN" in evidence["negative"]["sql"]
    assert evidence["positive"]["rows"], "a supporting sample as well as a refuting one"


def test_a_user_can_draw_an_edge_the_scores_missed(client, detected):
    """The proposal's last fallback for messy data: manual relationship drawing."""

    session_id, _ = detected

    drawn = client.post(
        f"/api/sessions/{session_id}/relationships/manual",
        json={
            "rel_type": "foreign_key",
            "from_table": "customers",
            "from_column": "region",
            "to_table": "regions",
            "to_column": "region",
        },
    )

    assert drawn.status_code == 201
    body = drawn.json()
    assert body["origin"] == "manual"
    assert body["status"] == "confirmed", "drawing it is the confirmation"

    links = client.get(f"/api/sessions/{session_id}/relationships").json()["graph"]["links"]
    assert body["id"] in {link["id"] for link in links}


def test_a_drawn_edge_must_point_at_columns_that_exist(client, detected):
    session_id, _ = detected

    response = client.post(
        f"/api/sessions/{session_id}/relationships/manual",
        json={
            "rel_type": "foreign_key",
            "from_table": "orders",
            "from_column": "customer_id",
            "to_table": "customers",
            "to_column": "does_not_exist",
        },
    )

    assert response.status_code == 422
    assert "does_not_exist" in response.json()["detail"]


def test_detection_needs_data_first(client):
    session_id = client.post("/api/sessions", json={"name": "empty"}).json()["id"]

    assert client.post(f"/api/sessions/{session_id}/relationships").status_code == 409
