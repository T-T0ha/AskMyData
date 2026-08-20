"""Phase 0 — cross-sheet column name equivalence."""

from __future__ import annotations

import pandas as pd
import pytest

from app.ingestion.equivalence import (
    detect_equivalences,
    lexical_similarity,
    value_overlap,
)
from app.semantics.embeddings import (
    HashingEmbedder,
    expand_abbreviations,
    humanize,
    normalize_column_name,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Cust_ID", "cust id"),
        ("customerID", "customer id"),
        ("order-date", "order date"),
        ("Total  Amount", "total amount"),
    ],
)
def test_humanize(raw, expected):
    assert humanize(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("cust id", "customer identifier"),
        ("qty", "quantity"),
        ("amt", "amount"),
        ("total amount", "total amount"),
    ],
)
def test_expand_abbreviations(raw, expected):
    assert expand_abbreviations(raw) == expected


def test_abbreviations_make_equivalent_names_identical():
    """The whole point: MiniLM alone scores cust_id ~ customer_id at only 0.57."""

    assert normalize_column_name("Cust_ID") == normalize_column_name("customer_id")
    assert normalize_column_name("qty") == normalize_column_name("quantity")


def test_lexical_similarity_separates_shared_suffixes():
    same = lexical_similarity("customer identifier", "customer identifier")
    different = lexical_similarity("customer identifier", "order identifier")
    assert same == 1.0
    assert different < 0.8


def test_detects_true_equivalences(messy_sheets):
    tables = {
        name: sheet.dataframe
        for name, sheet in messy_sheets.items()
        if not sheet.skipped and sheet.row_count > 2
    }
    pairs = {
        (c.left.qualified, c.right.qualified) for c in detect_equivalences(tables)
    }

    assert ("sales_data.order_info_cust_id", "customers.customer_id") in pairs
    assert ("sales_data.order_info_order_id", "line_items.order_id") in pairs


def test_rejects_unrelated_columns_that_share_a_token(messy_sheets):
    """cust_id and order_id both expand to '... identifier' but are not the same."""

    tables = {
        name: sheet.dataframe
        for name, sheet in messy_sheets.items()
        if not sheet.skipped and sheet.row_count > 2
    }
    pairs = {
        (c.left.qualified, c.right.qualified) for c in detect_equivalences(tables)
    }
    assert ("sales_data.order_info_cust_id", "line_items.order_id") not in pairs


def test_same_sheet_pairs_are_never_suggested():
    df = pd.DataFrame({"customer_id": ["C-1", "C-2"], "cust_id": ["C-1", "C-2"]})
    assert detect_equivalences({"only": df}) == []


def test_type_mismatch_is_penalised():
    text_table = pd.DataFrame({"customer_id": ["C-1", "C-2", "C-3"]})
    numeric_table = pd.DataFrame({"customer_id": [1, 2, 3]})
    candidates = detect_equivalences(
        {"a": text_table, "b": numeric_table}, threshold=0.0
    )
    assert candidates[0].type_compatible is False
    assert candidates[0].score < 0.9


def test_value_overlap():
    left = pd.Series(["A", "B", "C"])
    right = pd.Series(["A", "B", "C", "D"])
    assert value_overlap(left, right) == pytest.approx(1.0)
    assert value_overlap(left, pd.Series(["X"])) == pytest.approx(0.0)


def test_hashing_fallback_is_deterministic_and_normalised():
    embedder = HashingEmbedder(384)
    first = embedder.encode(["customer id"])
    second = embedder.encode(["customer id"])

    assert first.shape == (1, 384)
    assert (first == second).all()
    assert abs(float((first[0] ** 2).sum()) - 1.0) < 1e-5
