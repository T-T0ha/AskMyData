"""Phase 2 — cleaning executors, planner and the co-planning graph."""

from __future__ import annotations

import pandas as pd
import pytest
from langgraph.checkpoint.memory import InMemorySaver

from app.cleaning.graph import build_graph, interrupt_payload, resume, thread_config
from app.cleaning.planner import heuristic_plan, normalize_claude_plan, sort_plan
from app.cleaning.steps import StepError, execute_step
from app.cleaning.store import TableStore, drop_store, get_store
from app.core.schemas import CleaningStep, StepType


@pytest.fixture
def store(tmp_path) -> TableStore:
    store = TableStore("unit-test", root=tmp_path / "uploads")
    store.put(
        "orders",
        pd.DataFrame(
            {
                "order_id": ["A", "B", "C", "C"],
                "status": ["Shipped", "shipped", "PENDING", "PENDING"],
                "amount": [10.0, None, 30.0, 30.0],
                "customer": ["c1", "c2", "c3", "c3"],
            }
        ),
    )
    store.put(
        "customers",
        pd.DataFrame({"customer_id": ["c1", "c2", "c3"], "city": ["Dhaka", "Sylhet", "Khulna"]}),
    )
    return store


def _step(step_type: StepType, table: str, **params) -> CleaningStep:
    return CleaningStep(
        id="s1", type=step_type, table=table, description="test", params=params
    )


# ---------------------------------------------------------------------------
# executors
# ---------------------------------------------------------------------------


def test_all_ten_step_types_have_an_executor():
    from app.cleaning.steps import EXECUTORS

    assert set(EXECUTORS) == set(StepType)
    assert len(EXECUTORS) == 10


def test_no_step_can_fill_a_missing_value():
    """The platform has no imputation capability, by construction.

    A blank cell means "not known" and has to still mean that in the exported
    database: PostgreSQL excludes NULL from AVG and SUM, so a filled-in median
    would silently change every total the Phase 5 query interface computes, and
    nothing downstream could separate the invented value from a measured one.
    This is not a default the user can flip — the step type does not exist.
    """

    import ast
    import inspect

    from app.cleaning import steps as steps_module

    assert not [s for s in StepType if "fill" in s.value or "impute" in s.value]

    # Parsed, not grepped: prose about not filling nulls must not fail the test
    # that checks nothing fills nulls.
    tree = ast.parse(inspect.getsource(steps_module))
    called = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert not called & {"fillna", "ffill", "bfill", "pad", "backfill", "interpolate"}


def test_rename_column(store):
    outcome = execute_step(store, _step(StepType.RENAME_COLUMN, "orders", column="amount", new_name="total"))
    assert "total" in store.get("orders").columns
    assert outcome.after_preview


def test_rename_rejects_existing_name(store):
    with pytest.raises(StepError, match="already has a column"):
        execute_step(store, _step(StepType.RENAME_COLUMN, "orders", column="amount", new_name="status"))


def test_drop_column(store):
    execute_step(store, _step(StepType.DROP_COLUMN, "orders", column="customer"))
    assert "customer" not in store.get("orders").columns


def test_standardize_format_collapses_case_variants(store):
    outcome = execute_step(store, _step(StepType.STANDARDIZE_FORMAT, "orders", column="status", format="title"))
    assert set(store.get("orders")["status"]) == {"Shipped", "Pending"}
    assert outcome.cells_changed == 3


def test_deduplicate(store):
    outcome = execute_step(store, _step(StepType.DEDUPLICATE, "orders", subset=None))
    assert len(store.get("orders")) == 3
    assert outcome.rows_before - outcome.rows_after == 1


def test_deduplicate_rejects_table_without_duplicates(store):
    with pytest.raises(StepError, match="no duplicate rows"):
        execute_step(store, _step(StepType.DEDUPLICATE, "customers", subset=None))


def test_type_cast_reports_values_lost(store):
    store.put("t", pd.DataFrame({"v": ["1", "2", "not a number"]}))
    outcome = execute_step(store, _step(StepType.TYPE_CAST, "t", column="v", to="numeric"))
    assert pd.api.types.is_numeric_dtype(store.get("t")["v"])
    assert any("could not be converted" in n for n in outcome.notes)


def test_split_column(store):
    store.put("t", pd.DataFrame({"name": ["Rahim Uddin", "Karim Ali"]}))
    execute_step(store, _step(StepType.SPLIT_COLUMN, "t", column="name", delimiter=" ", into=["first", "last"]))
    assert list(store.get("t")["first"]) == ["Rahim", "Karim"]


def test_add_synthetic_key(store):
    execute_step(store, _step(StepType.ADD_SYNTHETIC_KEY, "orders", column="row_id"))
    assert list(store.get("orders")["row_id"]) == [1, 2, 3, 4]
    assert list(store.get("orders").columns)[0] == "row_id"


def test_merge_sheets(store):
    execute_step(
        store,
        _step(
            StepType.MERGE_SHEETS,
            "orders",
            left_table="orders",
            right_table="customers",
            left_key="customer",
            right_key="customer_id",
            how="left",
            result_table="joined",
        ),
    )
    joined = store.get("joined")
    assert "city" in joined.columns
    assert len(joined) == 4


def test_unknown_column_is_rejected_with_a_helpful_message(store):
    with pytest.raises(StepError, match="available:"):
        execute_step(store, _step(StepType.DROP_COLUMN, "orders", column="nope"))


# ---------------------------------------------------------------------------
# planner
# ---------------------------------------------------------------------------


def test_heuristic_plan_covers_the_obvious_problems(store):
    steps = heuristic_plan(store.tables())
    kinds = {(s.type, s.table) for s in steps}

    assert (StepType.DEDUPLICATE, "orders") in kinds
    # Capitalisation mismatches are now proposed as standardize_casing, which
    # names the fix instead of hiding it behind a generic "format" parameter.
    assert (StepType.STANDARDIZE_CASING, "orders") in kinds


def test_planner_leaves_gaps_alone_instead_of_inventing_or_deleting(store):
    """A partially-empty column gets no step: not a fill, and not a drop.

    Filling a missing SKU manufactures a duplicate key, filling a missing date
    manufactures an event that never happened, and filling a missing quantity
    manufactures a sale — the last one is the tempting case, and it is wrong for
    the same reason as the other two. Deleting the column instead is no better:
    a 25%-empty order_id still identifies three quarters of the rows.
    """

    store.put(
        "sales",
        pd.DataFrame(
            {
                "sku": ["MEN-1001", "MEN-1002", None, "MEN-1004"],
                "sold_on": ["2024-01-02", None, "2024-01-04", "2024-01-05"],
                "qty": [1, 2, None, 4],
            }
        ),
    )
    semantics = {
        "sales": {
            "sku": {"column_type": "identifier", "taxonomy_label": "sku"},
            "sold_on": {"column_type": "date", "taxonomy_label": "date"},
            "qty": {"column_type": "integer_continuous", "taxonomy_label": "quantity"},
        }
    }
    steps = heuristic_plan({"sales": store.get("sales")}, semantics=semantics)

    assert not [s for s in steps if s.type is StepType.DROP_COLUMN]
    assert not [s for s in steps if "fill" in s.type.value]


def test_planner_still_offers_to_drop_an_almost_empty_column():
    """Emptiness is only actionable when there is nothing left to lose."""

    df = pd.DataFrame({"legacy_note": [None] * 9 + ["x"], "id": list(range(10))})
    steps = heuristic_plan({"t": df})
    dropped = {s.params.get("column") for s in steps if s.type is StepType.DROP_COLUMN}

    assert dropped == {"legacy_note"}


def test_casing_step_quotes_real_values_and_keeps_the_dominant_spelling():
    """The description is evidence the user can check against their own data."""

    df = pd.DataFrame({"grade": ["XL"] * 8 + ["xl", "M"]})
    step = next(
        s for s in heuristic_plan({"t": df}) if s.type is StepType.STANDARDIZE_CASING
    )

    assert "'XL'" in step.description and "'xl'" in step.description
    assert "Shipped" not in step.description
    # 'XL' dominates, so upper-casing preserves it instead of writing 'Xl'.
    assert step.params["casing"] == "upper"


def test_standardize_casing_merges_duplicate_categories(store):
    outcome = execute_step(
        store, _step(StepType.STANDARDIZE_CASING, "orders", column="status", casing="lower")
    )
    assert set(store.get("orders")["status"]) == {"shipped", "pending"}
    # 'Shipped' plus both 'PENDING' rows.
    assert outcome.cells_changed == 3
    assert any("merged" in note for note in outcome.notes)


def test_strip_whitespace(store):
    store.put("orders", store.get("orders").assign(status=[" Shipped", "shipped ", "PENDING", "PENDING"]))
    outcome = execute_step(store, _step(StepType.STRIP_WHITESPACE, "orders", column="status"))

    assert list(store.get("orders")["status"])[:2] == ["Shipped", "shipped"]
    assert outcome.cells_changed == 2


def test_an_imputation_step_from_the_model_is_refused_with_its_reason(store):
    """Claude asking to fill nulls must not read as a typo to the user.

    ``unknown step type 'fill_nulls'`` looks like a bug in the platform. The
    rejection has to say that this is a decision, and why, because that message
    is what the user reads on the plan screen.
    """

    accepted, rejected = normalize_claude_plan(
        [
            {
                "type": "fill_nulls",
                "table": "orders",
                "params": {"column": "amount", "strategy": "median"},
            },
            {"type": "interpolate", "table": "orders", "params": {"column": "amount"}},
        ],
        store.tables(),
    )

    assert accepted == []
    assert len(rejected) == 2
    assert all("refused" in reason for reason in rejected)
    assert any("NULL" in reason for reason in rejected)


def test_synthetic_key_is_the_last_resort_not_the_first(store):
    """A row_id is added only when nothing in the table identifies a row.

    Three cases, in the order the proposal's no-key handling lists them:
    a single-column key (orders.order_id, once its duplicate row is gone), a
    composite key — which is the *user's* call and so produces no step — and a
    table where no combination works at all.
    """

    steps = heuristic_plan(store.tables())
    assert not [s for s in steps if s.type is StepType.ADD_SYNTHETIC_KEY]

    # order_id + line_no identify a line item; bolting a row_id onto it would
    # bury the real business key under a meaningless one.
    composite = pd.DataFrame(
        {"order_id": ["A1", "A1", "A2", "A2"], "line_no": [1, 2, 1, 2], "sku": ["x", "y", "x", "z"]}
    )
    steps = heuristic_plan({"line_items": composite})
    assert not [s for s in steps if s.type is StepType.ADD_SYNTHETIC_KEY]

    # Four two-valued columns: every combination of three repeats, so nothing
    # up to the search depth identifies a row.
    keyless = pd.DataFrame(
        {
            "a": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            "b": [0, 0, 0, 0, 1, 1, 1, 1] * 2,
            "c": [0, 0, 1, 1] * 4,
            "d": [0, 1] * 8,
        }
    )
    steps = heuristic_plan({"keyless": keyless})
    assert [s for s in steps if s.type is StepType.ADD_SYNTHETIC_KEY]


def test_a_confirmed_key_overrules_a_fresh_guess(store):
    """The user's answer on the triage screen wins over re-derivation."""

    keyless = pd.DataFrame({"a": [1, 1, 2, 2], "b": ["x", "y", "x", "y"]})
    stored = {
        "keyless": {
            "primary_key": ["a", "b"],
            "source": "confirmed",
            "confirmed": True,
            "needs_synthetic_key": False,
        }
    }
    steps = heuristic_plan({"keyless": keyless}, keys=stored)
    assert not [s for s in steps if s.type is StepType.ADD_SYNTHETIC_KEY]


def test_plan_is_ordered_structure_then_rows_then_joins():
    steps = sort_plan(
        [
            _step(StepType.MERGE_SHEETS, "a"),
            _step(StepType.DEDUPLICATE, "a"),
            _step(StepType.RENAME_COLUMN, "a"),
        ]
    )
    assert [s.type for s in steps] == [
        StepType.RENAME_COLUMN,
        StepType.DEDUPLICATE,
        StepType.MERGE_SHEETS,
    ]


def test_claude_plan_validation_rejects_hallucinated_references(store):
    accepted, rejected = normalize_claude_plan(
        [
            {"type": "drop_column", "table": "orders", "params": {"column": "customer"}},
            {"type": "drop_column", "table": "ghost_table", "params": {"column": "x"}},
            {"type": "drop_column", "table": "orders", "params": {"column": "ghost_column"}},
            {"type": "teleport_column", "table": "orders", "params": {}},
        ],
        store.tables(),
    )

    assert len(accepted) == 1
    assert accepted[0].params["column"] == "customer"
    assert len(rejected) == 3
    assert any("ghost_table" in r for r in rejected)
    assert any("ghost_column" in r for r in rejected)
    assert any("teleport_column" in r for r in rejected)


def test_merge_step_requires_both_sides_to_exist(store):
    accepted, rejected = normalize_claude_plan(
        [
            {
                "type": "merge_sheets",
                "table": "orders",
                "params": {"right_table": "customers", "left_key": "nope", "right_key": "customer_id"},
            }
        ],
        store.tables(),
    )
    assert accepted == []
    assert "nope" in rejected[0]


# ---------------------------------------------------------------------------
# graph
# ---------------------------------------------------------------------------


@pytest.fixture
def graph_session(messy_sheets):
    session_id = "graph-test"
    drop_store(session_id)
    store = get_store(session_id)
    store.put_many(
        {
            name: sheet.dataframe
            for name, sheet in messy_sheets.items()
            if not sheet.skipped and sheet.row_count > 2
        }
    )
    store.persist()
    yield session_id, store, build_graph(InMemorySaver())
    drop_store(session_id)


def _start(graph, session_id):
    return graph.invoke(
        {"session_id": session_id, "summary": {}, "equivalences": [], "excluded_tables": []},
        thread_config(session_id),
    )


def test_graph_pauses_for_plan_review(graph_session):
    session_id, _, graph = graph_session
    payload = interrupt_payload(_start(graph, session_id))

    assert payload is not None
    assert payload["kind"] == "plan_review"
    assert payload["plan"]
    assert all("description" in s for s in payload["plan"])


def test_graph_pauses_after_every_step(graph_session):
    session_id, _, graph = graph_session
    payload = interrupt_payload(_start(graph, session_id))
    plan = payload["plan"]

    result = resume(graph, session_id, {"action": "confirm", "plan": plan})
    pauses = 0
    while (pending := interrupt_payload(result)) is not None:
        pauses += 1
        assert pending["kind"] in {"step_validation", "step_failed"}
        if pending["kind"] == "step_validation":
            assert "outcome" in pending
            assert pending["outcome"]["summary"]
        result = resume(graph, session_id, {"action": "approve"})

    assert pauses == len(plan)
    assert result["status"] == "completed"


def test_user_edited_plan_replaces_the_proposal(graph_session):
    session_id, store, graph = graph_session
    payload = interrupt_payload(_start(graph, session_id))
    edited = [s for s in payload["plan"] if s["type"] == "deduplicate"][:1]

    result = resume(graph, session_id, {"action": "confirm", "plan": edited})
    pending = interrupt_payload(result)
    assert pending["cursor"] == 0
    assert pending["step"]["type"] == "deduplicate"

    result = resume(graph, session_id, {"action": "approve"})
    assert result["status"] == "completed"
    assert len(result["plan"]) == 1


def test_revert_restores_the_snapshot(graph_session):
    session_id, store, graph = graph_session
    payload = interrupt_payload(_start(graph, session_id))
    step = [s for s in payload["plan"] if s["type"] == "deduplicate"][:1]
    before = len(store.get("sales_data"))

    result = resume(graph, session_id, {"action": "confirm", "plan": step})
    assert len(get_store(session_id).get("sales_data")) == before - 1

    result = resume(graph, session_id, {"action": "revert"})
    assert len(get_store(session_id).get("sales_data")) == before
    assert result["plan"][0]["status"] == "reverted"


def test_retry_reruns_the_step_with_new_parameters(graph_session):
    session_id, store, graph = graph_session
    payload = interrupt_payload(_start(graph, session_id))
    step = [
        s
        for s in payload["plan"]
        if s["type"] == "standardize_casing" and s["table"] == "sales_data"
    ][:1]
    assert step, "the fixture sheet has capitalisation variants to fix"

    resume(graph, session_id, {"action": "confirm", "plan": step})
    result = resume(
        graph,
        session_id,
        {
            "action": "retry",
            "params": {"column": step[0]["params"]["column"], "casing": "upper"},
        },
    )
    pending = interrupt_payload(result)
    assert "uppercase" in pending["outcome"]["summary"]


def test_a_whole_cleaning_run_never_fills_a_blank_cell(graph_session):
    """The end-to-end guarantee, measured rather than argued.

    Approving every step of a real plan over the messy fixture must not turn a
    single blank into a value. Steps may remove rows, drop columns, or *create*
    nulls (a cast that cannot parse a value), so the invariant is stated per
    surviving column: its count of filled cells can only go down.
    """

    session_id, store, graph = graph_session
    before = {
        (table, str(column)): int(df[column].notna().sum())
        for table, df in store.tables().items()
        for column in df.columns
    }

    payload = interrupt_payload(_start(graph, session_id))
    assert payload["plan"], "the fixture must produce a non-empty plan"
    result = resume(graph, session_id, {"action": "confirm", "plan": payload["plan"]})
    while interrupt_payload(result) is not None:
        result = resume(graph, session_id, {"action": "approve"})
    assert result["status"] == "completed"

    for table, df in get_store(session_id).tables().items():
        for column in df.columns:
            filled_before = before.get((table, str(column)))
            if filled_before is None:  # a synthetic key or a merged-in column
                continue
            assert int(df[column].notna().sum()) <= filled_before, (
                f"{table}.{column} holds more values after cleaning than the source did"
            )


def test_cancelling_the_plan_changes_nothing(graph_session):
    session_id, store, graph = graph_session
    before = {name: len(df) for name, df in store.tables().items()}
    _start(graph, session_id)

    result = resume(graph, session_id, {"action": "cancel"})
    assert result["status"] == "cancelled"
    assert {name: len(df) for name, df in get_store(session_id).tables().items()} == before


def test_failed_step_is_reported_not_raised(graph_session):
    session_id, _, graph = graph_session
    _start(graph, session_id)
    broken = [
        {
            "id": "x1",
            "type": "drop_column",
            "table": "sales_data",
            "description": "drop a column that does not exist",
            "params": {"column": "does_not_exist"},
            "origin": "user",
            "status": "pending",
        }
    ]

    result = resume(graph, session_id, {"action": "confirm", "plan": broken})
    pending = interrupt_payload(result)
    assert pending["kind"] == "step_failed"
    assert "does_not_exist" in pending["error"]

    result = resume(graph, session_id, {"action": "skip"})
    assert result["status"] == "completed"


def test_state_survives_a_new_graph_instance(graph_session, tmp_path):
    """Checkpointing is what lets a session outlive the browser tab."""

    session_id, _, graph = graph_session
    checkpointer = InMemorySaver()
    first = build_graph(checkpointer)
    first.invoke(
        {"session_id": session_id, "summary": {}, "equivalences": [], "excluded_tables": []},
        thread_config(session_id),
    )

    second = build_graph(checkpointer)  # a "restarted server"
    state = second.get_state(thread_config(session_id))
    assert state.values["status"] == "awaiting_plan_review"
    assert state.interrupts
