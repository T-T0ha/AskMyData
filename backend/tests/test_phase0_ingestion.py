"""Phase 0 — structural repair, cleaning and triage."""

from __future__ import annotations

import pandas as pd
import pytest

from app.ingestion.cleaning import (
    NULL_TOKENS,
    clean_series,
    is_null_token,
    parse_percent,
    strip_currency,
    try_parse_date,
)
from app.ingestion.headers import sanitize_name, value_shape
from app.core.schemas import TriageStatus


# ---------------------------------------------------------------------------
# header flattening
# ---------------------------------------------------------------------------


def test_banner_row_is_skipped_not_treated_as_header(messy_sheets):
    sales = messy_sheets["sales_data"]
    assert sales.header.banner_rows == [1]
    assert 1 not in sales.header.header_rows


def test_multi_row_merged_header_is_flattened(messy_sheets):
    sales = messy_sheets["sales_data"]
    assert sales.header.is_multi_row
    assert sales.header.header_rows == [2, 3]
    # Merged group label propagates across every column it spans.
    assert "order_info_order_id" in sales.dataframe.columns
    assert "order_info_cust_id" in sales.dataframe.columns
    assert "amounts_unit_price" in sales.dataframe.columns


def test_single_row_header_is_not_over_absorbed(messy_sheets):
    """The regression that matters most: data rows must not become header rows."""

    for name in ("customers", "line_items"):
        assert messy_sheets[name].header.header_rows == [1]
    assert list(messy_sheets["line_items"].dataframe.columns) == [
        "order_id",
        "product_sku",
        "qty",
        "discount_pct",
        "line_note",
    ]


def test_all_data_rows_are_kept(messy_sheets):
    assert messy_sheets["sales_data"].row_count == 12  # 11 unique + 1 duplicate
    assert messy_sheets["customers"].row_count == 8
    assert messy_sheets["line_items"].row_count == 12


def test_empty_trailing_column_is_dropped(messy_sheets):
    assert "col_8" not in messy_sheets["customers"].dataframe.columns


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Order ID", "order_id"),
        ("Q1 2024 Revenue", "q1_2024_revenue"),
        ("% Margin", "pct_margin"),
        ("Price/Unit", "price_unit"),
        ("  spaced  out  ", "spaced_out"),
        ("2024", "c_2024"),
        ("Order #", "order_no"),
        ("---", ""),
    ],
)
def test_sanitize_name(raw, expected):
    assert sanitize_name(raw) == expected


@pytest.mark.parametrize(
    ("value", "shape"),
    [
        ("ORD-1001", "a-9"),
        ("Order ID", "a a"),
        ("৳1,200.00", "৳9"),
        ("2024-01-15", "9-9-9"),
        ("15/01/2024", "9/9/9"),
        (1200, "9"),
    ],
)
def test_value_shape(value, shape):
    assert value_shape(value) == shape


# ---------------------------------------------------------------------------
# mixed-type cleaning
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("token", ["N/A", "n/a", "—", "-", "NULL", "None", "#REF!", "#N/A", "?"])
def test_documented_null_tokens_are_recognised(token):
    assert token.lower() in NULL_TOKENS
    assert is_null_token(token)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("৳1,200.00", "1200.00"),
        ("$45.50", "45.50"),
        ("£1,000", "1000"),
        ("€2,500.75", "2500.75"),
        ("(1,200)", "-1200"),
        ("1 200", "1200"),
    ],
)
def test_strip_currency(raw, expected):
    assert strip_currency(raw)[0] == expected


def test_currency_column_becomes_numeric_and_reports_symbol():
    series = pd.Series(["৳1,200.00", "৳2,300", "৳780.25", "N/A", "৳950"])
    cleaned, report = clean_series(series, "unit_price")

    assert pd.api.types.is_numeric_dtype(cleaned)
    assert report.detected_currency_symbol == "৳"
    assert report.retyped_to == "numeric"
    assert cleaned.iloc[0] == 1200.0
    assert pd.isna(cleaned.iloc[3])


def test_mixed_date_formats_parse_to_datetime():
    series = pd.Series(["2024-01-15", "16/01/2024", "2024-01-17", "18/01/2024", "2024-01-19"])
    cleaned, report = clean_series(series, "order_date")

    assert pd.api.types.is_datetime64_any_dtype(cleaned)
    assert report.retyped_to == "datetime"
    assert cleaned.iloc[1].day == 16


def test_percent_strings_become_fractions():
    series = pd.Series(["5%", "0%", "10%", "2.5%", "15%"])
    cleaned, report = clean_series(series, "discount_pct")

    assert report.retyped_to == "float"
    assert cleaned.iloc[0] == pytest.approx(0.05)
    assert cleaned.iloc[3] == pytest.approx(0.025)


def test_leading_zero_codes_are_not_cast_to_numbers():
    """A phone number is a code; casting it would silently delete the zero."""

    series = pd.Series(["01711223344", "01822334455", "01933445566", "01644556677"])
    cleaned, report = clean_series(series, "phone")

    assert report.retyped_to is None
    assert cleaned.iloc[0] == "01711223344"
    assert any("leading zero" in note for note in report.notes)


def test_column_below_threshold_stays_text():
    series = pd.Series(["12", "13", "not a number", "also text", "15"])
    cleaned, report = clean_series(series, "mixed")

    assert report.retyped_to is None
    assert report.mixed_type_ratio == 0.0  # all values are strings on the way in
    assert any("mixed content" in note for note in report.notes)


def test_whitespace_is_trimmed_and_counted():
    series = pd.Series(["  Dhaka", "Dhaka  ", "Dhaka", " Sylhet "])
    cleaned, report = clean_series(series, "city")

    assert report.whitespace_trimmed == 3
    assert set(cleaned.dropna()) == {"Dhaka", "Sylhet"}


@pytest.mark.parametrize(
    ("raw", "day", "month", "year"),
    [
        ("2024-01-15", 15, 1, 2024),
        ("15/01/2024", 15, 1, 2024),
        ("15-01-2024", 15, 1, 2024),
        ("15.01.2024", 15, 1, 2024),
        ("15 Jan 2024", 15, 1, 2024),
        ("Jan 15, 2024", 15, 1, 2024),
    ],
)
def test_date_formats(raw, day, month, year):
    parsed, _ = try_parse_date(raw)
    assert parsed is not None
    assert (parsed.day, parsed.month, parsed.year) == (day, month, year)


def test_parse_percent_rejects_plain_numbers():
    assert parse_percent("45%") == pytest.approx(0.45)
    assert parse_percent("45") is None


# ---------------------------------------------------------------------------
# triage
# ---------------------------------------------------------------------------


def test_triage_buckets(messy_sheets, clean_workbook):
    from app.ingestion.loader import load_workbook_sheets

    assert messy_sheets["notes"].triage is TriageStatus.STRUCTURAL_ISSUES
    assert messy_sheets["line_items"].triage is TriageStatus.NEEDS_ATTENTION
    assert messy_sheets["sales_data"].triage is TriageStatus.FIXABLE

    products = load_workbook_sheets(clean_workbook)[0]
    assert products.triage is TriageStatus.CLEAN
    assert products.issues == []


def test_triage_reports_actionable_issues(messy_sheets):
    codes = {i.code for i in messy_sheets["sales_data"].issues}
    assert "duplicate_rows" in codes
    assert "multi_row_header" in codes
    assert "banner_rows_skipped" in codes


# ---------------------------------------------------------------------------
# which date, not just whether a date
# ---------------------------------------------------------------------------


def test_one_column_is_read_under_one_date_convention():
    """The bug this guards: 06/05 and 06/15 read by two different rules.

    Both entries below are American. Taking the first format that happens to
    match, value by value, makes ``06/05/2021`` day-first (6 May) because
    ``%d/%m/%Y`` is tried first, while ``06/15/2021`` can only be month-first
    (15 June) — one column, two conventions, and the row that moved by a month
    looks exactly like the rows that did not.
    """

    from app.ingestion.cleaning import clean_series

    series = pd.Series(["06/05/2021", "06/15/2021", "07/04/2021", "12/25/2021"])
    cleaned, report = clean_series(series, "sold_on")

    assert report.date_format == "%m/%d/%Y"
    assert cleaned.iloc[0].month == 6 and cleaned.iloc[0].day == 5
    assert cleaned.iloc[1].month == 6 and cleaned.iloc[1].day == 15


def test_a_day_first_column_is_still_read_day_first():
    """The same evidence pointing the other way must reach the other answer."""

    from app.ingestion.cleaning import clean_series

    series = pd.Series(["06/05/2021", "15/06/2021", "04/07/2021", "25/12/2021"])
    cleaned, report = clean_series(series, "sold_on")

    assert report.date_format == "%d/%m/%Y"
    assert cleaned.iloc[0].month == 5 and cleaned.iloc[0].day == 6


def test_two_digit_years_are_understood():
    """``06-05-21`` is what a real export writes; text here means VARCHAR later."""

    from app.ingestion.cleaning import clean_series

    series = pd.Series(["06-05-21", "06-15-21", "07-04-21", "12-25-21"])
    cleaned, report = clean_series(series, "date")

    assert report.retyped_to == "datetime"
    assert report.date_format == "%m-%d-%y"
    assert cleaned.iloc[1].year == 2021 and cleaned.iloc[1].day == 15


def test_a_column_that_is_half_dates_is_reported_not_silently_left_as_text():
    """Half dates and half product codes is a second table stacked in the file.

    Every value is a string, so the python-type check sees one consistent type
    and says nothing; without this the column exports as VARCHAR and no screen
    ever explains why.
    """

    from app.ingestion.cleaning import clean_series
    from app.ingestion.triage import triage_sheet
    from app.ingestion.headers import HeaderAnalysis

    series = pd.Series(["06-05-21", "06-08-21", "JNE3826", "JNE3827"])
    cleaned, report = clean_series(series, "date")

    assert report.retyped_to is None
    assert report.unconverted_type == "date"
    assert report.unconverted_ratio == pytest.approx(0.5)

    frame = pd.DataFrame({"date": cleaned})
    header = HeaderAnalysis(
        header_rows=[1], data_start_row=2, columns=["date"],
        original_labels=[["date"]], confidence=1.0,
    )
    status, issues = triage_sheet(frame, header, [report])

    issue = next(i for i in issues if i.code == "partly_typed")
    assert status.value == "needs_attention"
    assert issue.columns == ["date"]
    assert "50% date" in issue.message
