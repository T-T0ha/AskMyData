"""Phase 1 — column types, taxonomy and distributions."""

from __future__ import annotations

import pandas as pd
import pytest

from app.core.schemas import ColumnType
from app.semantics.column_types import detect_column_type
from app.semantics.pipeline import analyze_tables
from app.semantics.profiling import profile_series
from app.semantics.taxonomy import RULES, classify_by_rules, classify_column


# ---------------------------------------------------------------------------
# nine types
# ---------------------------------------------------------------------------


def _detect(values, name="col", **kwargs) -> ColumnType:
    return detect_column_type(pd.Series(values), name, **kwargs).column_type


def test_six_paper_types_and_three_extensions_exist():
    assert sum(1 for t in ColumnType if t.is_paper_type) == 6
    assert {t for t in ColumnType if not t.is_paper_type} == {
        ColumnType.CURRENCY,
        ColumnType.BOOLEAN,
        ColumnType.IDENTIFIER,
    }


@pytest.mark.parametrize(
    ("values", "name", "expected"),
    [
        (["Yes", "No", "Yes", "No"], "active", ColumnType.BOOLEAN),
        ([True, False, True], "flag", ColumnType.BOOLEAN),
        (["1", "0", "1", "0"], "is_paid", ColumnType.BOOLEAN),
        (["2024-01-15", "2024-02-16", "2024-03-17"], "order_date", ColumnType.DATE),
        (["ORD-1001", "ORD-1002", "ORD-1003"], "order_id", ColumnType.IDENTIFIER),
        (
            ["3f2504e0-4f89-11d3-9a0c-0305e82c3301", "3f2504e0-4f89-11d3-9a0c-0305e82c3302"],
            "uuid",
            ColumnType.IDENTIFIER,
        ),
        (["01711223344", "01822334455", "01933445566"], "phone", ColumnType.IDENTIFIER),
        ([10.5, 20.25, 30.75], "measurement", ColumnType.FLOAT),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "sequence", ColumnType.INTEGER_ORDINAL),
        ([1, 1, 2, 2, 1, 2, 1, 2, 1, 2], "region_code", ColumnType.INTEGER_NOMINAL),
        ([17, 43, 91, 128, 245, 12, 88, 301], "views", ColumnType.INTEGER_CONTINUOUS),
        (["Dhaka", "Chittagong", "Sylhet"], "city", ColumnType.TEXT),
    ],
)
def test_type_cascade(values, name, expected):
    assert _detect(values, name) is expected


def test_currency_detected_from_stripped_symbol():
    decision = detect_column_type(
        pd.Series([1200.0, 45.5, 2300.0]), "unit_price", currency_symbol="৳"
    )
    assert decision.column_type is ColumnType.CURRENCY
    assert decision.currency_symbol == "৳"


def test_currency_detected_from_financial_name():
    assert _detect([1200.0, 4500.0, 780.0], "total_amount") is ColumnType.CURRENCY
    assert _detect([1200.0, 4500.0, 780.0], "views") is not ColumnType.CURRENCY


def test_fraction_named_pct_is_not_currency():
    """'discount_pct' holds a ratio; calling it money would corrupt Phase 4 types."""

    assert _detect([0.05, 0.0, 0.1, 0.025], "discount_pct") is ColumnType.FLOAT


def test_empty_column_is_text_with_zero_confidence():
    decision = detect_column_type(pd.Series([None, None], dtype=object), "blank")
    assert decision.column_type is ColumnType.TEXT
    assert decision.confidence == 0.0


def test_every_decision_carries_evidence():
    for values, name in (
        (["Yes", "No"], "flag"),
        (["ORD-1", "ORD-2"], "order_id"),
        ([1.5, 2.5], "amount_paid"),
    ):
        assert detect_column_type(pd.Series(values), name).evidence


# ---------------------------------------------------------------------------
# taxonomy
# ---------------------------------------------------------------------------


def test_rule_engine_has_at_least_forty_rules():
    assert len(RULES) >= 40


@pytest.mark.parametrize(
    ("name", "values", "column_type", "expected"),
    [
        ("email_address", ["a@b.com", "c@d.org"], ColumnType.TEXT, "email"),
        ("phone", ["01711223344"], ColumnType.IDENTIFIER, "phone"),
        ("website", ["https://x.com"], ColumnType.TEXT, "url"),
        ("customer_name", ["Rahim"], ColumnType.TEXT, "person_name"),
        ("city", ["Dhaka"], ColumnType.TEXT, "city"),
        ("unit_price", [10.0], ColumnType.CURRENCY, "price"),
        ("total_amount", [10.0], ColumnType.CURRENCY, "revenue"),
        ("qty", [3], ColumnType.INTEGER_CONTINUOUS, "quantity"),
        ("discount_pct", [0.05], ColumnType.FLOAT, "percentage"),
        ("order_date", ["2024-01-01"], ColumnType.DATE, "date"),
        ("status", ["Shipped"], ColumnType.TEXT, "status"),
        ("line_note", ["bulk"], ColumnType.TEXT, "note"),
        ("product_sku", ["SKU-A1"], ColumnType.TEXT, "sku"),
        ("order_id", ["ORD-1"], ColumnType.IDENTIFIER, "order_number"),
        ("active", ["Yes"], ColumnType.BOOLEAN, "boolean_flag"),
    ],
)
def test_rule_engine_labels(name, values, column_type, expected):
    decision = classify_by_rules(name, column_type, pd.Series(values))
    assert decision is not None, f"no rule matched {name}"
    assert decision.label == expected
    assert decision.source == "rule"


def test_unmatched_column_without_claude_becomes_unknown():
    decision = classify_column("zxqv", "t", ColumnType.TEXT, pd.Series(["a", "b"]), claude=None)
    assert decision.label == "unknown"
    assert decision.source == "fallback"


def test_additive_labels_are_flagged():
    assert classify_by_rules("unit_price", ColumnType.CURRENCY, pd.Series([1.0])).is_additive
    assert not classify_by_rules("city", ColumnType.TEXT, pd.Series(["Dhaka"])).is_additive


class _StubClaude:
    """Stands in for the API so the fallback path is testable offline."""

    available = True

    def __init__(self, decision) -> None:
        self.decision = decision
        self.calls: list[dict] = []

    def classify_taxonomy(self, **kwargs):
        self.calls.append(kwargs)
        return self.decision


def test_claude_is_only_consulted_when_no_rule_matches():
    from app.semantics.taxonomy import TaxonomyDecision

    claude = _StubClaude(TaxonomyDecision("category", 0.8, "claude"))

    classify_column("city", "t", ColumnType.TEXT, pd.Series(["Dhaka"]), claude=claude)
    assert claude.calls == []

    decision = classify_column("zxqv", "t", ColumnType.TEXT, pd.Series(["a"]), claude=claude)
    assert len(claude.calls) == 1
    assert decision.label == "category"
    assert decision.source == "claude"


def test_claude_receives_only_ten_sample_values_and_no_rows():
    from app.semantics.taxonomy import TaxonomyDecision

    claude = _StubClaude(TaxonomyDecision("unknown", 0.0, "claude"))
    classify_column(
        "zxqv", "t", ColumnType.TEXT, pd.Series([f"v{i}" for i in range(500)]), claude=claude
    )
    call = claude.calls[0]
    assert len(call["samples"]) <= 10
    assert set(call) == {"column_name", "table_name", "column_type", "samples"}


# ---------------------------------------------------------------------------
# distributions
# ---------------------------------------------------------------------------


def test_numeric_distribution_has_stats_and_histogram():
    profile = profile_series(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]), ColumnType.FLOAT)
    assert profile["kind"] == "numeric"
    assert profile["min"] == 1 and profile["max"] == 5
    assert profile["median"] == 3
    assert sum(profile["histogram"]["counts"]) == 5


def test_categorical_distribution_returns_top_values():
    series = pd.Series(["a"] * 5 + ["b"] * 3 + ["c"])
    profile = profile_series(series, ColumnType.TEXT)
    assert profile["kind"] == "categorical"
    assert profile["top_values"][0] == {"value": "a", "count": 5, "share": pytest.approx(0.5556, abs=1e-3)}
    assert profile["distinct"] == 3


def test_temporal_distribution_spans_dates():
    series = pd.to_datetime(pd.Series(["2024-01-01", "2024-03-01"]))
    profile = profile_series(series, ColumnType.DATE)
    assert profile["kind"] == "temporal"
    assert profile["span_days"] == 60


def test_distribution_is_json_safe():
    import json

    profile = profile_series(pd.Series([1.0, float("nan"), 3.0]), ColumnType.FLOAT)
    json.dumps(profile)  # must not raise


# ---------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------


def test_pipeline_labels_every_column_of_the_messy_workbook(messy_sheets):
    sheets = {
        name: sheet
        for name, sheet in messy_sheets.items()
        if not sheet.skipped and sheet.row_count > 2
    }
    semantics = analyze_tables(
        {n: s.dataframe for n, s in sheets.items()},
        {n: s.clean_reports for n, s in sheets.items()},
    )

    sales = semantics["sales_data"]
    assert sales.column("amounts_unit_price").column_type is ColumnType.CURRENCY
    assert sales.column("amounts_unit_price").is_additive
    assert sales.column("order_info_order_date").column_type is ColumnType.DATE

    customers = semantics["customers"]
    assert customers.column("email_address").taxonomy_label == "email"
    assert customers.column("phone").column_type is ColumnType.IDENTIFIER
    assert customers.column("active").column_type is ColumnType.BOOLEAN

    unknown = [
        c.name for table in semantics.values() for c in table.columns if c.taxonomy_label == "unknown"
    ]
    assert unknown == [], f"rule engine left {unknown} unlabelled"
