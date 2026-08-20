"""Phase 0 — what identifies a row.

A table with no key cannot be referenced, joined to, or exported with a
``PRIMARY KEY`` clause, so these tests are about the one Phase 0 answer the
relational database depends on.  They check three things the naive version of
this feature gets wrong: that a key is never made out of nulls or measurements,
that a composite key is *proposed* rather than adopted, and that the user's
answer survives everything downstream.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.ingestion.keys import (
    MAX_KEY_DEPTH,
    KeyAnalysis,
    confirm_key,
    discover_keys,
    is_unique_key,
)
from app.ingestion.triage import triage_sheet
from app.ingestion.headers import HeaderAnalysis


def _header(columns: list[str]) -> HeaderAnalysis:
    return HeaderAnalysis(
        header_rows=[1],
        data_start_row=2,
        columns=columns,
        original_labels=[[c] for c in columns],
        confidence=1.0,
    )


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------


def test_a_unique_column_is_adopted_as_the_primary_key():
    frame = pd.DataFrame({"order_id": ["A1", "A2", "A3"], "city": ["dhaka", "ctg", "dhaka"]})
    analysis = discover_keys(frame)

    assert analysis.primary_key == ["order_id"]
    assert analysis.source == "detected"
    assert analysis.needs_synthetic_key is False


def test_a_column_containing_a_blank_is_never_a_key():
    """``NULL`` is not equal to itself, so it cannot identify anything.

    pandas is happy to report three distinct values here; SQL would not let the
    column be a primary key, and neither does this.
    """

    frame = pd.DataFrame({"code": ["A1", None, "A3"], "seq": [1, 2, 3]})

    assert is_unique_key(frame, ["code"]) is False
    assert discover_keys(frame).primary_key == ["seq"]


def test_a_measurement_that_happens_to_be_unique_is_not_a_key():
    """Every amount being distinct is an accident of the data, not a design.

    Accepting it would put ``PRIMARY KEY (amount)`` on the exported table,
    which breaks the first time two orders are for the same money.
    """

    frame = pd.DataFrame(
        {"amount": [10.5, 20.25, 30.75, 40.5], "city": ["a", "a", "b", "b"]}
    )
    analysis = discover_keys(frame)

    assert analysis.primary_key == []
    assert analysis.needs_synthetic_key is True
    assert any("amount" in note for note in analysis.notes)


def test_a_composite_key_is_proposed_and_not_adopted():
    """order_id + line_no is a claim about the business, so the user confirms it."""

    frame = pd.DataFrame(
        {
            "order_id": ["A1", "A1", "A2", "A2"],
            "line_no": [1, 2, 1, 2],
            "sku": ["x", "y", "x", "z"],
        }
    )
    analysis = discover_keys(frame)

    assert analysis.needs_confirmation is True
    assert analysis.primary_key == []
    assert analysis.needs_synthetic_key is False
    assert analysis.candidates[0].columns == ["order_id", "line_no"]
    assert analysis.candidates[0].kind == "composite"


def test_a_proposed_composite_key_contains_no_smaller_key():
    """Minimality: a key with a redundant column is not a key worth having."""

    frame = pd.DataFrame(
        {
            "order_id": ["A1", "A2", "A3", "A4"],  # already unique on its own
            "line_no": [1, 1, 2, 2],
            "sku": ["x", "y", "x", "y"],
        }
    )
    analysis = discover_keys(frame)

    assert analysis.primary_key == ["order_id"]
    assert all(len(candidate.columns) == 1 for candidate in analysis.candidates)


def test_a_table_where_nothing_identifies_a_row_asks_for_a_synthetic_key():
    """Four two-valued columns: no three of them can separate sixteen rows."""

    frame = pd.DataFrame(
        {
            "a": [0] * 8 + [1] * 8,
            "b": ([0] * 4 + [1] * 4) * 2,
            "c": ([0] * 2 + [1] * 2) * 4,
            "d": [0, 1] * 8,
        }
    )
    analysis = discover_keys(frame)

    assert analysis.needs_synthetic_key is True
    assert analysis.candidates == []
    assert any(str(MAX_KEY_DEPTH) in note for note in analysis.notes)


def test_duplicate_rows_do_not_hide_a_key():
    """``deduplicate`` runs before ``add_synthetic_key``, so the key survives it."""

    frame = pd.DataFrame({"order_id": ["A1", "A2", "A2"], "qty": [1, 2, 2]})
    analysis = discover_keys(frame)

    assert analysis.duplicate_rows == 1
    assert analysis.primary_key == ["order_id"]
    assert any("duplicate" in note for note in analysis.notes)


def test_sampling_a_large_table_still_finds_and_verifies_the_key():
    """The sample can only add candidates to check, never hide one.

    Uniqueness over a table implies uniqueness over any subset of its rows, so
    a combination that fails in the sample cannot succeed overall — and the
    ones that pass are re-checked against every row before being offered.
    """

    rows = 40_000
    frame = pd.DataFrame(
        {
            "order_id": np.repeat(np.arange(rows // 2), 2),
            "line_no": np.tile([1, 2], rows // 2),
            "amount": np.random.default_rng(0).random(rows),
        }
    )
    analysis = discover_keys(frame, sample_rows=5_000)

    assert analysis.sampled is True
    assert analysis.candidates[0].columns == ["order_id", "line_no"]
    assert is_unique_key(frame, ["order_id", "line_no"])


# ---------------------------------------------------------------------------
# declared keys
# ---------------------------------------------------------------------------


def test_a_declared_key_is_trusted_rather_than_re_derived():
    frame = pd.DataFrame({"customer_id": ["C1", "C2"], "email": ["a@x.com", "b@x.com"]})
    analysis = discover_keys(frame, declared_primary_key=["customer_id"])

    assert analysis.primary_key == ["customer_id"]
    assert analysis.source == "declared"
    assert analysis.declared_key_holds is True


def test_a_declared_key_that_does_not_hold_is_reported_not_believed():
    """A dump can lose its constraints; a key that fails is news, not noise."""

    frame = pd.DataFrame({"customer_id": ["C1", "C1"], "email": ["a@x.com", "b@x.com"]})
    analysis = discover_keys(frame, declared_primary_key=["customer_id"])

    assert analysis.declared_key_holds is False
    assert analysis.primary_key != ["customer_id"]
    assert any("not unique" in note for note in analysis.notes)


# ---------------------------------------------------------------------------
# triage
# ---------------------------------------------------------------------------


def test_a_table_with_no_key_is_orange_not_green():
    """The proposal's own example of a sheet needing a human decision."""

    frame = pd.DataFrame(
        {
            "a": [0] * 8 + [1] * 8,
            "b": ([0] * 4 + [1] * 4) * 2,
            "c": ([0] * 2 + [1] * 2) * 4,
            "d": [0, 1] * 8,
        }
    )
    keys = discover_keys(frame)
    status, issues = triage_sheet(frame, _header(list(frame.columns)), [], keys)

    assert status.value == "needs_attention"
    assert any(issue.code == "no_primary_key" for issue in issues)


def test_a_composite_candidate_asks_the_user_on_the_triage_screen():
    frame = pd.DataFrame(
        {"order_id": ["A1", "A1", "A2"], "line_no": [1, 2, 1], "qty": [5, 6, 5]}
    )
    keys = discover_keys(frame)
    status, issues = triage_sheet(frame, _header(list(frame.columns)), [], keys)

    issue = next(i for i in issues if i.code == "composite_key_candidate")
    assert status.value == "needs_attention"
    assert "order_id + line_no" in issue.message
    assert issue.columns == ["order_id", "line_no"]


def test_a_table_with_an_obvious_key_stays_green():
    """Having a key is not an issue, and must not colour a clean sheet."""

    frame = pd.DataFrame({"product_id": ["P1", "P2", "P3"], "name": ["a", "b", "c"]})
    keys = discover_keys(frame)
    status, issues = triage_sheet(frame, _header(list(frame.columns)), [], keys)

    assert status.value == "clean"
    assert issues == []


# ---------------------------------------------------------------------------
# the user's answer
# ---------------------------------------------------------------------------


def test_confirming_a_key_survives_a_round_trip_through_json():
    """The analysis is persisted as JSON and read back by the planner."""

    frame = pd.DataFrame({"a": ["x", "x", "y"], "b": [1, 2, 1]})
    analysis = confirm_key(discover_keys(frame), ["a", "b"])
    restored = KeyAnalysis.from_dict(analysis.to_dict())

    assert restored.primary_key == ["a", "b"]
    assert restored.source == "confirmed"
    assert restored.confirmed is True
    assert restored.needs_synthetic_key is False


@pytest.mark.parametrize(
    "columns",
    [["order_id"], ["order_id", "line_no"], []],
)
def test_is_unique_key_is_the_single_source_of_truth(columns):
    """Every acceptance path — detection, confirmation, the API — uses this."""

    frame = pd.DataFrame({"order_id": ["A1", "A1"], "line_no": [1, 2]})
    expected = columns == ["order_id", "line_no"]

    assert is_unique_key(frame, columns) is expected
