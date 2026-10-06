"""Doing the obvious work without asking.

Duplicate tables are set aside, the cross-table column matches and the clear
relationships are decided from their evidence, cleaning can run unattended, and
the query path finds a column by the words a question uses.  Every automatic
decision is a default the user can reverse, and one the user made is never
overridden — the tests hold both ends of that.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy import select

from app.cleaning.graph import (
    build_graph,
    interrupt_payload,
    resume,
    thread_config,
)
from app.cleaning.store import drop_store, get_store
from app.db.base import get_session_factory
from app.db.models import ColumnSemantics, EquivalenceCandidateRecord, SheetRecord
from app.ingestion.duplicates import find_duplicate_tables, table_fingerprint
from app.ingestion.equivalence import ColumnRef, EquivalenceCandidate, auto_verdict
from app.query.retrieval import question_terms, select_columns
from app.relationships.auto import decide_dependency, decide_foreign_key, settle_foreign_keys
from app.semantics.descriptions import is_template_description, template_description


def _upload(client, session_id, workbook):
    with workbook.open("rb") as handle:
        return client.post(
            f"/api/sessions/{session_id}/upload", files={"file": (workbook.name, handle)}
        )


def _session(client, name="automation") -> str:
    return client.post("/api/sessions", json={"name": name}).json()["id"]


# ---------------------------------------------------------------------------
# identical tables
# ---------------------------------------------------------------------------


def test_identical_tables_are_found_and_the_original_is_the_first_one():
    frame = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", None]})
    tables = {
        "frpm_3": frame.copy(),
        "frpm": frame.copy(),
        "frpm_2": frame.copy(),
        "other": pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}),
    }

    assert find_duplicate_tables(tables) == {"frpm_2": "frpm", "frpm_3": "frpm"}


def test_tables_that_differ_in_one_cell_are_not_duplicates():
    left = pd.DataFrame({"q": [1, 2, 3]})
    right = pd.DataFrame({"q": [1, 2, 4]})

    assert table_fingerprint(left) != table_fingerprint(right)
    assert find_duplicate_tables({"sales_q1": left, "sales_q2": right}) == {}


def test_a_table_holding_lists_can_still_be_fingerprinted():
    frame = pd.DataFrame({"tags": [["a"], ["b"]]})

    assert table_fingerprint(frame) == table_fingerprint(frame.copy())


def test_uploading_the_same_file_twice_into_one_session_keeps_one_table(client, clean_workbook):
    session_id = _session(client)
    assert _upload(client, session_id, clean_workbook).status_code == 200
    second = _upload(client, session_id, clean_workbook)
    assert second.status_code == 200

    triage = client.get(f"/api/sessions/{session_id}/triage").json()
    by_name = {sheet["table_name"]: sheet for sheet in triage["sheets"]}

    assert set(by_name) == {"products", "products_2"}
    assert not by_name["products"]["skipped"]
    assert by_name["products_2"]["skipped"]
    assert by_name["products_2"]["skip_reason"].startswith("Identical copy of 'products'")
    # The copy is not a table of the dataset, so it is not counted as one.
    assert triage["summary"]["sheet_count"] == 1
    assert triage["summary"]["total_rows"] == by_name["products"]["row_count"]
    assert get_store(session_id).names() == ["products"]

    client.post(f"/api/sessions/{session_id}/semantics")
    semantics = client.get(f"/api/sessions/{session_id}/semantics").json()
    assert [t["table"] for t in semantics["tables"]] == ["products"]

    equivalences = client.get(f"/api/sessions/{session_id}/equivalences").json()["equivalences"]
    assert equivalences == []


def test_an_older_session_holding_a_copy_is_cleaned_up_when_opened(client, clean_workbook):
    """Sessions made before copies were set aside still have their ``_2`` tables."""

    session_id = _session(client)
    assert _upload(client, session_id, clean_workbook).status_code == 200
    store = get_store(session_id)
    store.put("products_2", store.get("products").copy())
    store.persist()
    with get_session_factory()() as db:
        original = db.execute(
            select(SheetRecord).where(SheetRecord.session_id == session_id)
        ).scalar_one()
        db.add(
            SheetRecord(
                session_id=session_id,
                table_name="products_2",
                source_name=original.source_name,
                source_file=original.source_file,
                source_kind=original.source_kind,
                triage=original.triage,
                row_count=original.row_count,
                column_count=original.column_count,
                key_analysis=original.key_analysis,
                issues=[],
                clean_reports=[],
                header_info={},
            )
        )
        db.add(
            ColumnSemantics(
                session_id=session_id,
                table_name="products_2",
                column_name="product_id",
                column_type="identifier",
                taxonomy_label="identifier",
            )
        )
        db.commit()

    triage = client.get(f"/api/sessions/{session_id}/triage").json()
    copy = next(s for s in triage["sheets"] if s["table_name"] == "products_2")

    assert copy["skipped"] and "Identical copy of 'products'" in copy["skip_reason"]
    assert triage["summary"]["sheet_count"] == 1
    assert get_store(session_id).names() == ["products"]
    with get_session_factory()() as db:
        left = db.execute(
            select(ColumnSemantics).where(
                ColumnSemantics.session_id == session_id,
                ColumnSemantics.table_name == "products_2",
            )
        ).scalars().all()
        assert left == []


# ---------------------------------------------------------------------------
# column matches across tables
# ---------------------------------------------------------------------------


def _candidate(score, *, overlap=0.8, compatible=True):
    return EquivalenceCandidate(
        left=ColumnRef("a", "x"),
        right=ColumnRef("b", "y"),
        score=score,
        embedding_similarity=score,
        lexical_similarity=score,
        type_compatible=compatible,
        left_dtype="object",
        right_dtype="object",
        value_overlap=overlap,
    )


@pytest.mark.parametrize(
    ("score", "overlap", "compatible", "expected"),
    [
        (0.97, 1.0, True, True),  # the same column in two tables
        (0.97, 0.0, True, False),  # same name, nothing in common: orders.name / products.name
        (0.97, 1.0, False, False),  # storage types differ
        (0.85, 0.9, True, True),  # middling name, but the values are the same values
        (0.85, 0.1, True, None),  # middling name, little overlap: a person should look
        (0.78, 0.9, True, False),  # too different in name to bother anyone with
        (0.95, None, True, True),  # no values to compare: the name decides
    ],
)
def test_a_column_match_is_decided_from_its_evidence(score, overlap, compatible, expected):
    assert auto_verdict(_candidate(score, overlap=overlap, compatible=compatible)) is expected


def test_matches_are_decided_on_upload_and_the_users_answer_survives_a_re_run(
    client, messy_workbook
):
    session_id = _session(client)
    payload = _upload(client, session_id, messy_workbook).json()
    decided = [e for e in payload["equivalences"] if e["decided_by"] == "auto"]
    assert decided, "the messy workbook has an obvious cust_id / customer_id match"

    target = decided[0]
    flipped = not target["confirmed"]
    client.post(
        f"/api/sessions/{session_id}/equivalences/{target['id']}", json={"confirmed": flipped}
    )

    # A second upload re-runs detection over the whole session.
    other = pd.DataFrame({"unrelated_thing": range(5)})
    path = messy_workbook.parent / "other.csv"
    other.to_csv(path, index=False)
    assert _upload(client, session_id, path).status_code == 200

    after = {
        (e["left_table"], e["left_column"], e["right_table"], e["right_column"]): e
        for e in client.get(f"/api/sessions/{session_id}/equivalences").json()["equivalences"]
    }
    kept = after[
        (target["left_table"], target["left_column"], target["right_table"], target["right_column"])
    ]
    assert kept["confirmed"] is flipped
    assert kept["decided_by"] == "user"


# ---------------------------------------------------------------------------
# relationships
# ---------------------------------------------------------------------------


def _evidence(overlap=1.0, name=0.9, targets=100):
    return {"ratio_overlap": overlap, "embedding_similarity": name, "target_distinct": targets}


def test_a_link_every_value_of_which_is_real_and_is_named_like_one_is_confirmed():
    verdict = decide_foreign_key("detected", _evidence(), to_table="customers")

    assert verdict.status == "confirmed"
    assert "customers" in verdict.reason


def test_a_declared_link_is_confirmed_whatever_its_numbers_say():
    verdict = decide_foreign_key("declared", {}, to_table="customers")

    assert verdict.status == "confirmed"


def test_an_overlap_by_coincidence_of_range_is_turned_down():
    """Quantities 1..20 sit inside the ids 1..50; nothing says they refer to them."""

    verdict = decide_foreign_key(
        "detected", _evidence(overlap=1.0, name=0.1, targets=50), to_table="products"
    )

    assert verdict.status == "rejected"


def test_an_identifier_may_point_at_a_key_with_a_different_name():
    verdict = decide_foreign_key(
        "detected",
        _evidence(overlap=1.0, name=0.2, targets=500),
        to_table="customers",
        child_is_identifier=True,
    )

    assert verdict.status == "confirmed"


def test_a_link_that_nearly_holds_is_the_only_thing_asked_about():
    asked = decide_foreign_key("detected", _evidence(overlap=0.93, name=0.6), to_table="t")
    same_name = decide_foreign_key("detected", _evidence(overlap=0.93, name=0.95), to_table="t")
    partial = decide_foreign_key("fuzzy", _evidence(overlap=0.7, name=0.9), to_table="t")

    assert asked.status == "proposed"
    assert same_name.status == "confirmed"
    assert partial.status == "rejected"


def test_patterns_inside_a_table_never_ask_anything():
    assert decide_dependency("functional_dependency").status == "confirmed"
    assert decide_dependency("approximate_dependency").status == "proposed"


def _fk(id_, frm, fcol, to, tcol, score, **evidence):
    return {
        "id": id_,
        "from_table": frm,
        "from_column": fcol,
        "to_table": to,
        "to_column": tcol,
        "score": score,
        "origin": "detected",
        "evidence": _evidence(**evidence),
    }


def test_two_tables_get_one_link_and_a_link_is_not_counted_from_both_ends():
    verdicts = settle_foreign_keys(
        [
            _fk("a", "frpm", "cdscode", "schools", "cdscode", 0.99),
            # A second, weaker route between the same two tables.
            _fk("b", "frpm", "district_code", "schools", "district", 0.80, name=0.5),
            # The same link seen from the other side.
            _fk("c", "schools", "cdscode", "frpm", "cdscode", 0.90),
        ]
    )

    assert verdicts["a"].status == "confirmed"
    assert verdicts["b"].status == "rejected"
    assert verdicts["c"].status == "rejected"
    assert "other table" in verdicts["c"].reason


def test_two_clearly_named_links_between_the_same_tables_are_both_kept():
    verdicts = settle_foreign_keys(
        [
            _fk("a", "orders", "billing_address_id", "addresses", "address_id", 0.98, name=0.9),
            _fk("b", "orders", "shipping_address_id", "addresses", "address_id", 0.97, name=0.9),
        ]
    )

    assert {v.status for v in verdicts.values()} == {"confirmed"}


def test_a_link_a_person_already_confirmed_holds_its_place():
    verdicts = settle_foreign_keys(
        [_fk("a", "frpm", "district_code", "schools", "district", 0.95, name=0.5)],
        taken=[("frpm", "cdscode", "schools", "cdscode")],
    )

    assert verdicts["a"].status == "rejected"


def test_what_a_person_decided_is_not_revisited_by_a_re_run(client, relational_workbook):
    session_id = _session(client)
    assert _upload(client, session_id, relational_workbook).status_code == 200
    first = client.post(f"/api/sessions/{session_id}/relationships").json()
    edge = next(
        r for r in first["relationships"] if r["rel_type"] == "foreign_key" and r["status"] == "confirmed"
    )
    assert edge["decided_by"] == "auto"

    client.post(f"/api/sessions/{session_id}/relationships/{edge['id']}", json={"confirmed": False})
    second = client.post(f"/api/sessions/{session_id}/relationships").json()

    same = next(r for r in second["relationships"] if r["id"] == edge["id"])
    assert same["status"] == "rejected"
    assert same["decided_by"] == "user"


# ---------------------------------------------------------------------------
# descriptions and retrieval
# ---------------------------------------------------------------------------


def test_the_template_is_recognised_and_a_written_sentence_is_not():
    template = template_description("avgscrwrite", "satscores", "unknown", "text", "fact table")

    assert is_template_description(template)
    assert is_template_description("")
    assert is_template_description(None)
    assert not is_template_description(
        "The average SAT writing score for students who took the test."
    )


def test_the_exported_description_carries_the_sentence_written_for_the_column(shop_plan):
    from app.export.metadata import describe

    table = shop_plan.table("orders")
    column = table.column("customer_id")
    column.description = "Which customer placed the order, used to look up their details."

    assert "Which customer placed the order" in describe(column, table)

    column.description = template_description(
        "customer_id", "orders", "foreign identifier", "text", "fact table"
    )
    assert "in table" not in describe(column, table)


@pytest.fixture
def shop_plan(relational_tables):
    from app.export.schema import build_plan
    from app.ingestion.keys import discover_keys
    from app.semantics.pipeline import analyze_tables

    semantics = analyze_tables(relational_tables, {})
    columns = {
        name: [
            {
                "column_name": c.name,
                "column_type": c.column_type.value,
                "taxonomy_label": c.taxonomy_label,
                "null_ratio": c.null_ratio,
                "sample_values": c.sample_values,
                "semantic_description": "",
            }
            for c in table.columns
        ]
        for name, table in semantics.items()
    }
    keys = {name: discover_keys(df).to_dict() for name, df in relational_tables.items()}
    return build_plan(relational_tables, columns, keys)


def _row(table, column, text, vector):
    return {
        "table_name": table,
        "column_name": column,
        "semantic_description": text,
        "embedding": vector,
        "is_primary_key": False,
        "is_foreign_key": False,
    }


def test_a_column_is_found_by_the_words_of_the_question_even_when_its_vector_is_far():
    """``avgscrwrite`` ranks below noise on meaning alone; the word 'writing' finds it."""

    vector = [1.0, 0.0]
    rows = [_row("s", f"filler_{i}", "unrelated filler column", [0.8, 0.6]) for i in range(10)]
    rows.append(
        _row("s", "avgscrwrite", "The average SAT writing score for the school", [0.6, 0.8])
    )

    by_meaning = select_columns(vector, rows, ["s"], max_ranked=3)
    by_words = select_columns(
        vector, rows, ["s"], max_ranked=3, question="What is the average writing score?"
    )

    assert "avgscrwrite" not in {r["column_name"] for r in by_meaning}
    assert "avgscrwrite" in {r["column_name"] for r in by_words}


def test_question_terms_drop_filler_and_stem_lightly():
    assert question_terms("What are the writing scores of the schools?") == [
        "writ",
        "scor",
        "school",
    ]


def test_every_column_fits_when_the_table_is_small():
    rows = [_row("t", f"c{i}", "x", [1.0, 0.0]) for i in range(30)]

    assert len(select_columns([1.0, 0.0], rows, ["t"])) == 30


# ---------------------------------------------------------------------------
# cleaning, unattended
# ---------------------------------------------------------------------------


@pytest.fixture
def many_steps():
    session_id = "auto-approve-test"
    drop_store(session_id)
    store = get_store(session_id)
    frame = pd.DataFrame(
        {f"c{i}": [" a ", "b ", " c"] * 4 for i in range(15)}
    ).assign(id=range(12))
    store.put("wide", frame)
    store.persist()
    yield session_id, store, build_graph(InMemorySaver())
    drop_store(session_id)


def _plan_for(graph, session_id):
    result = graph.invoke(
        {"session_id": session_id, "summary": {}, "equivalences": [], "excluded_tables": []},
        thread_config(session_id),
    )
    return interrupt_payload(result)["plan"]


def test_a_long_plan_runs_to_the_end_without_stopping_when_approved_up_front(many_steps):
    session_id, store, graph = many_steps
    plan = _plan_for(graph, session_id)
    assert len(plan) >= 15  # more steps than LangGraph's default 25-node limit allows at 2 per step

    result = resume(graph, session_id, {"action": "confirm", "plan": plan, "auto_approve": True})

    assert interrupt_payload(result) is None
    state = graph.get_state(thread_config(session_id)).values
    assert state["status"] == "completed"
    assert all(step["status"] == "approved" for step in state["plan"])
    assert all(value == value.strip() for value in store.get("wide")["c0"])
    # Nothing is left lying around to revert to.
    assert not list(store.root.glob("snapshot-*.pkl"))


def test_approving_all_from_the_middle_finishes_the_rest(many_steps):
    session_id, _, graph = many_steps
    plan = _plan_for(graph, session_id)
    paused = resume(graph, session_id, {"action": "confirm", "plan": plan})
    assert interrupt_payload(paused)["kind"] == "step_validation"

    result = resume(graph, session_id, {"action": "approve_all"})

    assert interrupt_payload(result) is None
    state = graph.get_state(thread_config(session_id)).values
    assert state["status"] == "completed"
    assert all(step["status"] == "approved" for step in state["plan"])


def test_an_unattended_run_still_stops_for_a_step_that_fails(many_steps):
    session_id, _, graph = many_steps
    plan = _plan_for(graph, session_id)
    bad = {
        "id": "bad",
        "type": "type_cast",
        "table": "wide",
        "description": "cast text to numbers",
        "params": {"column": "c0", "to": "numeric", "strict": True},
        "origin": "user",
        "status": "pending",
    }
    # An unknown column fails validation in review, so break it at execution time.
    from app.cleaning import steps as steps_module

    original = steps_module.execute_step

    def explode(store, step):
        if step.id == "bad":
            raise steps_module.StepError("cannot cast this")
        return original(store, step)

    import app.cleaning.graph as graph_module

    graph_module.execute_step = explode
    try:
        result = resume(
            graph, session_id, {"action": "confirm", "plan": [*plan[:2], bad, *plan[2:4]], "auto_approve": True}
        )
    finally:
        graph_module.execute_step = original

    pending = interrupt_payload(result)
    assert pending is not None and pending["kind"] == "step_failed"
    assert pending["step"]["id"] == "bad"
