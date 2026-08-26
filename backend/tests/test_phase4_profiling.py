"""Table semantic profiling — SemTabla Step 4 (§4.1.5, Table 8).

Two things are under test.  The thirteen labels are checked one at a time
against a frame built so that exactly one of them can fire, because a label
that fires for the wrong reason is worse than one that does not fire at all —
the user is being asked to trust the evidence line beside it.  The decision
tree is checked branch by branch, including the branch that refuses to answer.
"""

from __future__ import annotations

import pandas as pd

from app.core.schemas import TABLE_LABELS, RelationshipType, TableType
from app.semantics.table_profile import (
    MIN_TYPE_CONFIDENCE,
    TableProfile,
    _longest_chain,
    profile_table,
    profile_tables,
)


def _column(name: str, column_type: str, label: str = "unknown", **extra) -> dict:
    """A ColumnSemantics-shaped payload, with the statistics profiling reads."""

    payload = {
        "name": name,
        "effective_type": column_type,
        "effective_label": label,
        "null_ratio": 0.0,
        "unique_count": extra.pop("unique_count", 10),
        "unique_ratio": extra.pop("unique_ratio", 1.0),
        "row_count": extra.pop("row_count", 10),
        "is_additive": extra.pop("is_additive", False),
    }
    payload.update(extra)
    return payload


def _fk(from_table: str, from_column: str, to_table: str, to_column: str, **extra) -> dict:
    return {
        "rel_type": RelationshipType.FOREIGN_KEY.value,
        "from_table": from_table,
        "from_column": from_column,
        "to_table": to_table,
        "to_column": to_column,
        "status": extra.pop("status", "confirmed"),
        **extra,
    }


def _fd(table: str, determinant: str, dependent: str, **extra) -> dict:
    return {
        "rel_type": RelationshipType.FUNCTIONAL_DEPENDENCY.value,
        "from_table": table,
        "from_column": determinant,
        "to_table": table,
        "to_column": dependent,
        "status": extra.pop("status", "confirmed"),
        **extra,
    }


def _labels(profile: TableProfile) -> set[str]:
    return {label.name for label in profile.labels}


# ---------------------------------------------------------------------------
# the vocabulary itself
# ---------------------------------------------------------------------------


def test_the_thirteen_labels_are_the_papers_thirteen():
    """Table 8 of SemTabla has thirteen rows, and every one is `is_`-prefixed."""

    assert len(TABLE_LABELS) == 13
    assert len(set(TABLE_LABELS)) == 13
    assert all(name.startswith("is_") for name in TABLE_LABELS)


def test_every_label_the_profiler_can_emit_is_in_the_vocabulary():
    """A label the UI has no name for cannot be shown, explained or removed."""

    df = pd.DataFrame(
        {
            "reading_at": pd.date_range("2024-01-01", periods=12, freq="D"),
            "sensor": ["a"] * 12,
            "celsius": [20 + i for i in range(12)],
        }
    )
    profile = profile_table(
        "readings",
        df,
        [
            _column("reading_at", "date", "datetime", unique_count=12, unique_ratio=1.0),
            _column("sensor", "text", "category", unique_count=1, unique_ratio=0.08),
            _column("celsius", "float", "dimension", unique_count=12, unique_ratio=1.0),
        ],
        key_analysis={"primary_key": ["reading_at"], "source": "detected"},
    )

    assert _labels(profile) <= set(TABLE_LABELS)


# ---------------------------------------------------------------------------
# the thirteen labels, one at a time
# ---------------------------------------------------------------------------


def test_a_time_typed_primary_key_is_labelled_and_its_period_measured():
    """Labels 1 and 2: a daily reading is a periodic time key.

    The paper asks for the average interval and the standard deviation ratio,
    so both are asserted rather than just the boolean.
    """

    df = pd.DataFrame(
        {
            "reading_at": pd.date_range("2024-01-01", periods=30, freq="D"),
            "celsius": [20.0 + i * 0.5 for i in range(30)],
        }
    )
    profile = profile_table(
        "readings",
        df,
        [
            _column("reading_at", "date", "datetime", unique_count=30, unique_ratio=1.0),
            _column("celsius", "float", "dimension", unique_count=30, unique_ratio=1.0),
        ],
        key_analysis={"primary_key": ["reading_at"], "source": "detected"},
    )

    assert {"is_primary_key_time", "is_primary_key_periodic"} <= _labels(profile)
    period = next(l for l in profile.labels if l.name == "is_primary_key_periodic")
    assert period.detail["mean_interval_days"] == 1.0
    assert period.detail["variation_ratio"] == 0.0


def test_an_irregular_time_key_is_a_time_key_but_not_a_periodic_one():
    """Whenever somebody pressed the button is not a period."""

    df = pd.DataFrame(
        {
            "logged_at": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-02",
                    "2024-03-15",
                    "2024-03-16",
                    "2024-09-30",
                    "2024-12-31",
                ]
            ),
            "note": list("abcdef"),
        }
    )
    profile = profile_table(
        "events",
        df,
        [
            _column("logged_at", "date", "datetime", unique_count=6, unique_ratio=1.0),
            _column("note", "text", "note", unique_count=6, unique_ratio=1.0),
        ],
        key_analysis={"primary_key": ["logged_at"], "source": "detected"},
    )

    assert "is_primary_key_time" in _labels(profile)
    assert "is_primary_key_periodic" not in _labels(profile)
    assert profile.features["periodicity"]["periodic"] is False


def test_one_measure_and_one_enum_are_labelled_separately():
    """Labels 3 and 4, and the rule that keeps them from double-counting.

    ``quantity`` repeats (7 distinct values over 40 rows) which would make it
    an enumeration by cardinality alone.  It is the number the table is about,
    so it counts once, as the measure.
    """

    df = pd.DataFrame(
        {
            "line_id": range(40),
            "status": ["open", "closed"] * 20,
            "quantity": [(i % 7) + 1 for i in range(40)],
        }
    )
    profile = profile_table(
        "lines",
        df,
        [
            _column("line_id", "identifier", "identifier", unique_count=40, unique_ratio=1.0),
            _column("status", "text", "status", unique_count=2, unique_ratio=0.05),
            _column("quantity", "integer_continuous", "quantity", unique_count=7, unique_ratio=0.18),
        ],
        key_analysis={"primary_key": ["line_id"], "source": "detected"},
    )

    assert {"is_single_value_column", "is_single_enum_column"} <= _labels(profile)
    assert profile.features["measure_columns"] == ["quantity"]
    assert profile.features["enum_columns"] == ["status"]


def test_repeating_numbers_are_labelled_discrete_and_unique_ones_are_not():
    """Label 5, in both directions, on the same table shape."""

    discrete = profile_table(
        "t",
        pd.DataFrame({"score": [(i % 5) for i in range(100)]}),
        [_column("score", "integer_continuous", "rating", unique_count=5, unique_ratio=0.05)],
    )
    continuous = profile_table(
        "t",
        pd.DataFrame({"amount": [float(i) for i in range(100)]}),
        [_column("amount", "currency", "price", unique_count=100, unique_ratio=1.0)],
    )

    assert "is_data_discrete" in _labels(discrete)
    assert "is_data_discrete" not in _labels(continuous)


def test_a_table_of_mostly_codes_is_labelled_enum_dominant():
    """Label 6: strictly more than half the columns are enumerations."""

    columns = [
        _column("country", "text", "country", unique_count=4, unique_ratio=0.04),
        _column("status", "text", "status", unique_count=3, unique_ratio=0.03),
        _column("note", "text", "note", unique_count=100, unique_ratio=1.0),
    ]
    profile = profile_table("t", pd.DataFrame({c["name"]: range(100) for c in columns}), columns)

    assert "is_enum_containing_most" in _labels(profile)


def test_a_flattened_multi_row_header_counts_as_hierarchy():
    """Label 7 — and the one source of it no other phase can see again.

    Phase 0 flattens "Order Info / Order ID" into one name.  That the author
    drew a group band across three columns is a statement about structure, and
    the header record is the only place it survives.
    """

    df = pd.DataFrame({"order_info_order_id": [1, 2], "amounts_total": [10.0, 20.0]})
    columns = [
        _column("order_info_order_id", "identifier", "identifier"),
        _column("amounts_total", "currency", "revenue"),
    ]

    with_header = profile_table(
        "t", df, columns, header_info={"rows": [2, 3], "is_multi_row": True}
    )
    without = profile_table("t", df, columns, header_info={"rows": [1], "is_multi_row": False})

    assert "is_label_having_hierarchy_column" in _labels(with_header)
    assert "is_label_having_hierarchy_column" not in _labels(without)


def test_two_rungs_of_one_ladder_are_hierarchy_and_two_unrelated_labels_are_not():
    """Label 7 again: city+country is a tree, email+phone is two attributes."""

    df = pd.DataFrame({"a": range(20), "b": range(20)})
    ladder = profile_table(
        "t",
        df,
        [
            _column("a", "text", "city", unique_count=6, unique_ratio=0.3),
            _column("b", "text", "country", unique_count=2, unique_ratio=0.1),
        ],
    )
    flat = profile_table(
        "t",
        df,
        [
            _column("a", "text", "email", unique_count=20, unique_ratio=1.0),
            _column("b", "text", "phone", unique_count=20, unique_ratio=1.0),
        ],
    )

    assert "is_label_having_hierarchy_column" in _labels(ladder)
    assert "is_label_having_hierarchy_column" not in _labels(flat)


def test_a_table_two_others_point_at_is_labelled_mostly_referenced():
    """Label 8, counting distinct referencing tables rather than edges."""

    df = pd.DataFrame({"customer_id": range(30), "name": [f"n{i}" for i in range(30)]})
    columns = [
        _column("customer_id", "identifier", "identifier", unique_count=30, unique_ratio=1.0),
        _column("name", "text", "person_name", unique_count=30, unique_ratio=1.0),
    ]
    relationships = [
        _fk("orders", "customer_id", "customers", "customer_id"),
        _fk("invoices", "customer_id", "customers", "customer_id"),
    ]

    profile = profile_table(
        "customers", df, columns, relationships=relationships, peer_table_count=3
    )

    assert "is_mostly_referenced" in _labels(profile)
    assert profile.features["referenced_by_tables"] == ["invoices", "orders"]


def test_a_column_pointing_at_its_own_table_is_labelled_self_reference():
    """Label 9 — and the branch of the tree that makes it a hierarchy."""

    df = pd.DataFrame({"employee_id": range(20), "manager_id": [0] + list(range(19))})
    profile = profile_table(
        "employees",
        df,
        [
            _column("employee_id", "identifier", "identifier", unique_count=20, unique_ratio=1.0),
            _column("manager_id", "identifier", "foreign_identifier", unique_count=19),
        ],
        key_analysis={"primary_key": ["employee_id"], "source": "detected"},
        relationships=[_fk("employees", "manager_id", "employees", "employee_id")],
    )

    assert "is_self_reference" in _labels(profile)
    assert profile.table_type is TableType.HIERARCHY


def test_a_table_with_no_single_column_key_says_so():
    """Label 10, in both directions."""

    df = pd.DataFrame({"order_id": [1, 1, 2], "sku": ["a", "b", "a"]})
    columns = [_column("order_id", "identifier"), _column("sku", "text", "sku")]

    keyless = profile_table("t", df, columns, key_analysis={"primary_key": []})
    composite = profile_table("t", df, columns, key_analysis={"primary_key": ["order_id", "sku"]})
    single = profile_table("t", df, columns, key_analysis={"primary_key": ["order_id"]})

    assert "is_no_single_value_as_primary_key" in _labels(keyless)
    assert "is_no_single_value_as_primary_key" in _labels(composite)
    assert "is_no_single_value_as_primary_key" not in _labels(single)


def test_exactly_two_foreign_keys_is_labelled_and_three_is_not():
    """Label 11: the junction signature, which is a count and not "several"."""

    df = pd.DataFrame({"a_id": range(20), "b_id": range(20), "c_id": range(20)})
    columns = [_column(name, "identifier", "foreign_identifier") for name in df.columns]

    two = profile_table(
        "j",
        df[["a_id", "b_id"]],
        columns[:2],
        relationships=[_fk("j", "a_id", "a", "id"), _fk("j", "b_id", "b", "id")],
    )
    three = profile_table(
        "j",
        df,
        columns,
        relationships=[
            _fk("j", "a_id", "a", "id"),
            _fk("j", "b_id", "b", "id"),
            _fk("j", "c_id", "c", "id"),
        ],
    )

    assert "is_exactly_two_foreign_keys_existing" in _labels(two)
    assert "is_exactly_two_foreign_keys_existing" not in _labels(three)


def test_a_chain_of_three_dependencies_is_labelled_and_a_pair_is_not():
    """Label 12: the paper's threshold is a chain of three or more."""

    df = pd.DataFrame({"a": range(9), "b": range(9), "c": range(9)})
    columns = [_column(name, "text") for name in "abc"]

    chain = profile_table(
        "t", df, columns, relationships=[_fd("t", "a", "b"), _fd("t", "b", "c")]
    )
    pair = profile_table("t", df, columns, relationships=[_fd("t", "a", "b")])

    assert "is_having_dependency_chain" in _labels(chain)
    assert chain.features["dependency_chain"] == ["a", "b", "c"]
    assert "is_having_dependency_chain" not in _labels(pair)


def test_a_dependency_through_a_mostly_empty_column_is_not_stable():
    """Label 13: stability means dropping the sparse columns changes nothing."""

    df = pd.DataFrame({"city": ["dhaka"] * 10, "region": ["central"] * 10})
    stable = profile_table(
        "t",
        df,
        [_column("city", "text", "city"), _column("region", "text", "region")],
        relationships=[_fd("t", "city", "region")],
    )
    fragile = profile_table(
        "t",
        df,
        [
            _column("city", "text", "city"),
            _column("region", "text", "region", null_ratio=0.8),
        ],
        relationships=[_fd("t", "city", "region")],
    )

    assert "is_fd_stable_after_null_drop" in _labels(stable)
    assert "is_fd_stable_after_null_drop" not in _labels(fragile)
    assert fragile.features["fragile_dependencies"] == ["city → region"]


# ---------------------------------------------------------------------------
# the decision tree
# ---------------------------------------------------------------------------


def test_two_foreign_keys_and_nothing_else_is_a_bridge():
    df = pd.DataFrame({"student_id": range(50), "course_id": range(50)})
    profile = profile_table(
        "enrolments",
        df,
        [
            _column("student_id", "identifier", "foreign_identifier"),
            _column("course_id", "identifier", "foreign_identifier"),
        ],
        relationships=[
            _fk("enrolments", "student_id", "students", "id"),
            _fk("enrolments", "course_id", "courses", "id"),
        ],
    )

    assert profile.table_type is TableType.BRIDGE
    assert profile.type_confidence >= 0.9


def test_measures_hanging_off_a_reference_are_a_fact_table():
    df = pd.DataFrame({"order_id": range(45), "customer_id": range(45), "total": [1.0] * 45})
    profile = profile_table(
        "orders",
        df,
        [
            _column("order_id", "identifier", "order_number"),
            _column("customer_id", "identifier", "foreign_identifier"),
            _column("total", "currency", "revenue"),
        ],
        key_analysis={"primary_key": ["order_id"], "source": "detected"},
        relationships=[_fk("orders", "customer_id", "customers", "customer_id")],
    )

    assert profile.table_type is TableType.FACT
    assert "customers" in profile.rationale


def test_a_keyed_descriptive_table_others_point_at_is_a_dimension():
    df = pd.DataFrame({"customer_id": range(30), "name": [f"n{i}" for i in range(30)]})
    profile = profile_table(
        "customers",
        df,
        [
            _column("customer_id", "identifier", "identifier"),
            _column("name", "text", "person_name"),
        ],
        key_analysis={"primary_key": ["customer_id"], "source": "detected"},
        relationships=[_fk("orders", "customer_id", "customers", "customer_id")],
        peer_table_count=2,
    )

    assert profile.table_type is TableType.DIMENSION


def test_a_measured_series_on_a_periodic_key_is_a_time_series():
    df = pd.DataFrame(
        {
            "day": pd.date_range("2024-01-01", periods=60, freq="D"),
            "revenue": [100.0 + i for i in range(60)],
        }
    )
    profile = profile_table(
        "daily_revenue",
        df,
        [
            _column("day", "date", "date", unique_count=60, unique_ratio=1.0),
            _column("revenue", "currency", "revenue", unique_count=60, unique_ratio=1.0),
        ],
        key_analysis={"primary_key": ["day"], "source": "detected"},
    )

    assert profile.table_type is TableType.TIME_SERIES


def test_a_short_list_of_codes_is_a_lookup():
    df = pd.DataFrame({"code": list("abcdefgh"), "name": [f"n{i}" for i in range(8)]})
    profile = profile_table(
        "status_codes",
        df,
        [
            _column("code", "text", "status", unique_count=8, unique_ratio=1.0),
            _column("name", "text", "description", unique_count=8, unique_ratio=1.0),
        ],
    )

    assert profile.table_type is TableType.LOOKUP


def test_a_thirty_column_sheet_is_wide():
    columns = [_column(f"c{i}", "text", "description", unique_count=50) for i in range(30)]
    df = pd.DataFrame({c["name"]: ["x"] * 200 for c in columns})

    profile = profile_table("everything", df, columns)

    assert profile.table_type is TableType.WIDE


def test_a_shape_the_tree_does_not_recognise_is_unknown_not_a_guess():
    """The paper assigns ``unknown`` below the confidence floor.  So does this.

    Four free-text columns, no key, no measure, nothing pointing anywhere:
    there is no honest answer, and inventing one would be a label the user has
    to notice is wrong before it reaches the semantic layer.
    """

    columns = [_column(f"note{i}", "text", "note", unique_count=500) for i in range(4)]
    df = pd.DataFrame({c["name"]: [f"v{i}" for i in range(500)] for c in columns})

    profile = profile_table("notes", df, columns)

    assert profile.table_type is TableType.UNKNOWN
    assert profile.type_confidence < MIN_TYPE_CONFIDENCE


def test_an_empty_table_profiles_without_claiming_anything():
    profile = profile_table("empty", pd.DataFrame({"a": [], "b": []}), [_column("a", "text")])

    assert profile.table_type is TableType.UNKNOWN
    assert _labels(profile) <= {"is_no_single_value_as_primary_key"}


# ---------------------------------------------------------------------------
# what the user's decisions do to it
# ---------------------------------------------------------------------------


def test_a_rejected_relationship_does_not_hold_up_a_label():
    """The verdict the user gave in Phase 3 has to reach Phase 4 intact.

    ``is_self_reference`` rests entirely on one edge.  Rejecting that edge has
    to take the label — and the table type it produced — with it.
    """

    df = pd.DataFrame({"employee_id": range(20), "manager_id": [0] + list(range(19))})
    columns = [
        _column("employee_id", "identifier", "identifier"),
        _column("manager_id", "identifier", "foreign_identifier"),
    ]
    edge = _fk("employees", "manager_id", "employees", "employee_id")

    confirmed = profile_table("employees", df, columns, relationships=[edge])
    rejected = profile_table(
        "employees", df, columns, relationships=[{**edge, "status": "rejected"}]
    )

    assert confirmed.table_type is TableType.HIERARCHY
    assert "is_self_reference" not in _labels(rejected)
    assert rejected.table_type is not TableType.HIERARCHY


def test_a_label_resting_on_an_unconfirmed_proposal_is_counted_as_such():
    """Profiling reads proposals, so it has to say how many are still proposals."""

    df = pd.DataFrame({"a_id": range(20), "b_id": range(20)})
    columns = [_column(name, "identifier", "foreign_identifier") for name in ("a_id", "b_id")]
    profile = profile_table(
        "j",
        df,
        columns,
        relationships=[
            _fk("j", "a_id", "a", "id", status="proposed"),
            _fk("j", "b_id", "b", "id", status="confirmed"),
        ],
    )

    assert profile.features["unconfirmed_edges"] == 1


def test_user_edits_survive_a_recomputation():
    """Re-running detection must not undo a correction the user made.

    This is the same contract ``ColumnSemantics`` keeps for types and taxonomy
    labels: the detected value and the human's value are stored side by side,
    and re-detection may overwrite only the first.
    """

    df = pd.DataFrame({"code": list("abcdefgh"), "name": list("abcdefgh")})
    columns = [
        _column("code", "text", "status", unique_count=8),
        _column("name", "text", "description", unique_count=8),
    ]
    first = profile_table("t", df, columns)
    first.user_table_type = TableType.DIMENSION.value
    first.removed_labels = ["is_enum_containing_most"]
    first.added_labels = ["is_mostly_referenced"]

    second = profile_table("t", df, columns, previous=first)

    assert second.table_type is TableType.LOOKUP  # the detector still says what it says
    assert second.effective_type == TableType.DIMENSION.value  # the user still wins
    assert "is_enum_containing_most" not in second.effective_labels
    assert "is_mostly_referenced" in second.effective_labels


def test_a_profile_survives_a_round_trip_through_json():
    """It is stored as JSON on the sheet record and read back by Phase 4."""

    df = pd.DataFrame({"day": pd.date_range("2024-01-01", periods=10), "v": range(10)})
    profile = profile_table(
        "t",
        df,
        [_column("day", "date", "date"), _column("v", "float", "revenue")],
        key_analysis={"primary_key": ["day"]},
    )
    profile.user_table_type = TableType.FACT.value

    restored = TableProfile.from_dict(profile.to_dict())

    assert restored.to_dict() == profile.to_dict()
    assert restored.effective_type == TableType.FACT.value


def test_an_unrecognised_stored_type_reads_back_as_unknown():
    """A profile written by a later build must not crash an earlier one."""

    restored = TableProfile.from_dict({"table": "t", "table_type": "galaxy"})

    assert restored is not None and restored.table_type is TableType.UNKNOWN


# ---------------------------------------------------------------------------
# internals worth pinning
# ---------------------------------------------------------------------------


def test_the_chain_walk_terminates_on_a_cycle():
    """Mutually dependent columns are ordinary — a → b and b → a both hold when
    two columns are one-to-one — and must not be walked forever."""

    assert _longest_chain([("a", "b"), ("b", "a")]) == ["a", "b"]
    assert _longest_chain([("a", "b"), ("b", "c"), ("c", "a")]) == ["a", "b", "c"]


def test_the_whole_relational_fixture_profiles_as_a_person_would_read_it():
    """End to end on the Phase 3 fixture, with the detectors' real output.

    This is the test that would catch a threshold change quietly reclassifying
    a customer list as a fact table.
    """

    from app.ingestion.keys import discover_keys
    from app.relationships.dependencies import detect_all_dependencies
    from app.relationships.foreign_keys import detect_foreign_keys
    from app.semantics.pipeline import analyze_tables
    from tests.fixtures.make_fixtures import build_relational_workbook
    from app.ingestion.loader import load_workbook_sheets

    results = {r.name: r for r in load_workbook_sheets(build_relational_workbook())}
    tables = {name: result.dataframe for name, result in results.items()}
    keys = {name: discover_keys(df) for name, df in tables.items()}
    semantics = analyze_tables(tables)

    relationships = [
        {**candidate.to_dict(), "status": "confirmed"}
        for candidate in detect_foreign_keys(tables, keys)
    ] + [
        {**dependency.to_dict(), "status": "confirmed"}
        for dependency in detect_all_dependencies(
            tables, key_columns={n: k.primary_key for n, k in keys.items()}
        )
    ]

    profiles = profile_tables(
        tables,
        {name: [c.to_dict() for c in sem.columns] for name, sem in semantics.items()},
        key_analyses={name: analysis.to_dict() for name, analysis in keys.items()},
        relationships=relationships,
        header_info={name: result.summary()["header"] for name, result in results.items()},
    )

    assert profiles["customers"].table_type is TableType.DIMENSION
    assert profiles["orders"].table_type is TableType.FACT
    assert profiles["order_lines"].table_type is TableType.FACT
    # customers is what orders points at, and it describes rather than measures
    assert profiles["customers"].features["referenced_by_tables"] == ["orders"]
    assert profiles["customers"].features["measure_columns"] == []
    assert profiles["orders"].features["measure_columns"] == ["total_amount"]


# ---------------------------------------------------------------------------
# the API: profiles are computed, corrected and kept
# ---------------------------------------------------------------------------


def _profiled(client, workbook) -> tuple[str, dict]:
    session_id = client.post("/api/sessions", json={"name": "phase 4"}).json()["id"]
    with workbook.open("rb") as handle:
        upload = client.post(
            f"/api/sessions/{session_id}/upload", files={"file": (workbook.name, handle)}
        )
    assert upload.status_code == 200, upload.text
    detection = client.post(f"/api/sessions/{session_id}/relationships")
    assert detection.status_code == 200, detection.text
    return session_id, detection.json()


def test_relationship_detection_ends_with_step_four(client, relational_workbook):
    """SemTabla's fourth step runs off the back of the first three."""

    session_id, payload = _profiled(client, relational_workbook)

    assert {p["table"] for p in payload["profiles"]} >= {"customers", "orders", "order_lines"}
    listed = client.get(f"/api/sessions/{session_id}/profiles").json()
    assert listed["counts"]["total"] == len(payload["profiles"])
    assert all(profile["rationale"] for profile in listed["profiles"])


def test_the_vocabulary_carries_the_labels_the_ui_has_to_offer(client):
    vocabulary = client.get("/api/vocabulary").json()

    assert len(vocabulary["table_labels"]) == 13
    assert "fact" in vocabulary["table_types"] and "unknown" in vocabulary["table_types"]


def test_a_corrected_table_type_outlives_a_recomputation(client, relational_workbook):
    """The paper's Table Profiling Validation, end to end."""

    session_id, _ = _profiled(client, relational_workbook)

    corrected = client.patch(
        f"/api/sessions/{session_id}/tables/orders/profile",
        json={"table_type": "wide", "remove_label": "is_single_enum_column"},
    )
    assert corrected.status_code == 200, corrected.text
    assert corrected.json()["effective_type"] == "wide"

    again = client.post(f"/api/sessions/{session_id}/profiles").json()
    orders = next(p for p in again["profiles"] if p["table"] == "orders")

    assert orders["table_type"] == "fact", "the detector's answer is still recorded"
    assert orders["effective_type"] == "wide", "and the user's still wins"
    assert "is_single_enum_column" not in orders["effective_labels"]
    assert again["counts"]["corrected"] == 1


def test_an_override_can_be_withdrawn(client, relational_workbook):
    session_id, _ = _profiled(client, relational_workbook)

    client.patch(f"/api/sessions/{session_id}/tables/orders/profile", json={"table_type": "wide"})
    restored = client.patch(
        f"/api/sessions/{session_id}/tables/orders/profile", json={"table_type": ""}
    )

    assert restored.json()["effective_type"] == "fact"


def test_a_label_outside_the_thirteen_is_refused(client, relational_workbook):
    """The label set is closed, so the UI can always explain what it shows."""

    session_id, _ = _profiled(client, relational_workbook)

    response = client.patch(
        f"/api/sessions/{session_id}/tables/orders/profile",
        json={"add_label": "is_definitely_important"},
    )

    assert response.status_code == 422
    assert "thirteen" in response.json()["detail"]


def test_an_invented_table_type_is_refused(client, relational_workbook):
    session_id, _ = _profiled(client, relational_workbook)

    response = client.patch(
        f"/api/sessions/{session_id}/tables/orders/profile", json={"table_type": "galaxy"}
    )

    assert response.status_code == 422


def test_rejecting_a_relationship_reprofiles_the_tables_it_touched(client, relational_workbook):
    """A verdict in Phase 3 changes what a table is, in the same request."""

    session_id, payload = _profiled(client, relational_workbook)
    edge = next(
        r
        for r in payload["relationships"]
        if r["rel_type"] == "foreign_key" and r["to_table"] == "customers"
    )

    response = client.post(
        f"/api/sessions/{session_id}/relationships/{edge['id']}", json={"confirmed": False}
    )

    assert response.status_code == 200, response.text
    customers = next(p for p in response.json()["profiles"] if p["table"] == "customers")
    assert "orders" not in customers["features"]["referenced_by_tables"]


def test_profiles_belong_to_their_owner(client, other_client, relational_workbook):
    session_id, _ = _profiled(client, relational_workbook)

    assert other_client.get(f"/api/sessions/{session_id}/profiles").status_code == 404
    assert other_client.post(f"/api/sessions/{session_id}/profiles").status_code == 404
    assert (
        other_client.patch(
            f"/api/sessions/{session_id}/tables/orders/profile", json={"table_type": "wide"}
        ).status_code
        == 404
    )


def test_profiling_needs_data_first(client):
    session_id = client.post("/api/sessions", json={"name": "empty"}).json()["id"]

    assert client.post(f"/api/sessions/{session_id}/profiles").status_code == 409
