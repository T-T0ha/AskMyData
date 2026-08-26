"""Phase 4 — identifiers and column types for the export.

Both modules exist to stop the same class of failure: something the user typed
into a spreadsheet changing meaning on its way into the database.  A header
becomes an identifier without becoming SQL, and a value is never rounded,
truncated or overflowed by the type it is declared with.
"""

from __future__ import annotations

import pandas as pd
import pytest
from sqlalchemy import Integer, Numeric, String, Text

from app.core.schemas import ColumnType
from app.export.naming import (
    MAX_IDENTIFIER_LENGTH,
    is_managed_schema,
    sanitize_identifier,
    schema_name,
    unique_identifiers,
)
from app.export.types import (
    CURRENCY_PRECISION,
    CURRENCY_SCALE,
    INT32_MAX,
    coerce_series,
    default_sql_type,
    plan_type,
    render_type,
    to_records,
)

# ---------------------------------------------------------------------------
# identifiers
# ---------------------------------------------------------------------------


def test_a_spreadsheet_header_becomes_something_a_person_would_type():
    assert sanitize_identifier("Total Amount (৳)") == "total_amount"
    assert sanitize_identifier("  Order Date  ") == "order_date"
    assert sanitize_identifier("Q1_2024_Revenue") == "q1_2024_revenue"
    assert sanitize_identifier("Prénom") == "prenom", "accents fold, the letter survives"


def test_nothing_that_could_end_a_statement_survives_sanitisation():
    """Identifiers are quoted by SQLAlchemy; this is the second line of defence.

    Whatever the header said, what comes out has no quote, semicolon,
    backslash, whitespace or comment marker in it to break out of a quoted
    identifier with.
    """

    hostile = [
        '"; DROP TABLE users; --',
        "'); DELETE FROM ingestion_sessions; --",
        "a\\b\"c'd`e",
        "col--comment",
        "/* block */ name",
        "tab\tand\nnewline",
        "\x00null byte",
    ]

    for raw in hostile:
        identifier = sanitize_identifier(raw)
        assert identifier
        assert set(identifier) <= set("abcdefghijklmnopqrstuvwxyz0123456789_")
        assert not identifier[0].isdigit()


def test_a_name_that_says_nothing_in_ascii_still_yields_a_usable_column():
    """Bengali and CJK headers have no ASCII form.  The export needs a name
    anyway; the original is kept in the metadata beside it."""

    assert sanitize_identifier("গ্রাহক", fallback="column") == "column"
    assert sanitize_identifier("", fallback="table") == "table"
    assert sanitize_identifier(None) == "column"


def test_reserved_words_and_leading_digits_are_made_legal():
    assert sanitize_identifier("Order") == "order_col"
    assert sanitize_identifier("group") == "group_col"
    assert sanitize_identifier("2024 Total") == "c_2024_total"


def test_a_long_header_is_cut_to_what_postgresql_will_hold():
    identifier = sanitize_identifier("Extremely " * 20 + "Long Header")

    assert len(identifier) <= MAX_IDENTIFIER_LENGTH
    assert not identifier.endswith("_")


def test_sanitising_twice_changes_nothing():
    """Re-exporting a session must not rename its columns a second time."""

    for raw in ("Total Amount (৳)", "Order", "2024 Total", "già_così"):
        once = sanitize_identifier(raw)
        assert sanitize_identifier(once) == once


def test_two_headers_that_reduce_to_one_name_still_get_two_columns():
    mapping = unique_identifiers(["Total (BDT)", "Total $", "total"])

    assert len(set(mapping.values())) == 3
    assert mapping["Total $"] == "total"
    assert mapping["total"] == "total_2"


def test_disambiguation_still_fits_the_length_limit():
    long_name = "Quarterly Revenue Recognised Against Deferred Subscription Contracts"
    mapping = unique_identifiers([long_name, long_name + " (restated)"])

    assert len(set(mapping.values())) == 2
    assert all(len(name) <= MAX_IDENTIFIER_LENGTH for name in mapping.values())


def test_a_name_the_exporter_reserves_for_itself_is_not_handed_out():
    mapping = unique_identifiers(["row_id", "Row ID"], reserved={"row_id"})

    assert "row_id" not in mapping.values()


def test_the_same_header_asked_about_twice_is_one_column():
    assert unique_identifiers(["city", "city"]) == {"city": "city"}


# ---------------------------------------------------------------------------
# the schema the exporter owns
# ---------------------------------------------------------------------------


def test_a_session_gets_a_schema_named_after_it():
    name = schema_name("0af8ff24f4e308722f6c0b0e75ab7c11")

    assert name == "ds_0af8ff24f4e308722f6c0b0e75ab7c11"
    assert is_managed_schema(name)
    assert len(name) <= MAX_IDENTIFIER_LENGTH


def test_only_a_schema_this_package_made_is_recognised_as_its_own():
    """``DROP SCHEMA`` is checked against this, so the answer matters."""

    assert not is_managed_schema("public")
    assert not is_managed_schema("pg_catalog")
    assert not is_managed_schema("ds_public; DROP SCHEMA public CASCADE")
    assert not is_managed_schema("ds_")
    assert not is_managed_schema("")
    assert not is_managed_schema("information_schema")


def test_a_session_id_that_is_not_one_gets_no_schema_at_all():
    for bad in ("", "public", "../../etc", "x"):
        with pytest.raises(ValueError):
            schema_name(bad)


# ---------------------------------------------------------------------------
# types
# ---------------------------------------------------------------------------


def test_the_nine_types_map_as_the_specification_says():
    rendered = {
        column_type: render_type(default_sql_type(column_type)) for column_type in ColumnType
    }

    assert rendered[ColumnType.INTEGER_CONTINUOUS] == "NUMERIC"
    assert rendered[ColumnType.INTEGER_ORDINAL] == "INTEGER"
    assert rendered[ColumnType.INTEGER_NOMINAL] == "VARCHAR(50)"
    assert rendered[ColumnType.CURRENCY] == f"NUMERIC({CURRENCY_PRECISION}, {CURRENCY_SCALE})"
    assert rendered[ColumnType.DATE] == "TIMESTAMP WITHOUT TIME ZONE"
    assert rendered[ColumnType.TEXT] == "TEXT"
    assert rendered[ColumnType.BOOLEAN] == "BOOLEAN"
    assert rendered[ColumnType.IDENTIFIER] == "VARCHAR(100)"
    assert rendered[ColumnType.FLOAT] == "DOUBLE PRECISION"


def test_a_value_too_long_for_the_declared_width_widens_the_column():
    """VARCHAR(100) against a 140-character value truncates or refuses.  Both
    lose the value, so the column becomes TEXT and the report says why."""

    plan = plan_type(ColumnType.IDENTIFIER, pd.Series(["x" * 140, "short"]))

    assert isinstance(plan.sql_type, Text)
    assert plan.widened and "140 characters" in plan.notes[0]


def test_a_value_that_fits_leaves_the_declared_width_alone():
    plan = plan_type(ColumnType.IDENTIFIER, pd.Series(["INV-2024-0001"]))

    assert isinstance(plan.sql_type, String) and plan.sql_type.length == 100
    assert not plan.widened


def test_an_ordinal_past_the_32_bit_limit_becomes_a_bigint():
    plan = plan_type(ColumnType.INTEGER_ORDINAL, pd.Series([1, INT32_MAX + 1]))

    assert plan.render() == "BIGINT"
    assert plan.widened


def test_money_with_four_decimal_places_is_not_rounded_to_two():
    """The whole point.  NUMERIC(15,2) would store 12.3456 as 12.35, and
    nothing downstream could tell that from a measured 12.35."""

    plan = plan_type(ColumnType.CURRENCY, pd.Series([12.3456, 1.0]))

    assert plan.render() == "NUMERIC(15, 4)"
    assert "rounding to 2 would change them" in plan.notes[0]


def test_ordinary_money_keeps_the_declared_currency_type():
    plan = plan_type(ColumnType.CURRENCY, pd.Series([1250.00, 37.50, 0.99]))

    assert plan.render() == "NUMERIC(15, 2)"
    assert not plan.widened


def test_money_larger_than_the_declared_precision_gets_more_digits():
    plan = plan_type(ColumnType.CURRENCY, pd.Series([9.9e14]))

    assert isinstance(plan.sql_type, Numeric)
    assert plan.sql_type.precision > CURRENCY_PRECISION
    assert plan.widened


def test_a_column_of_only_blanks_keeps_its_declared_type():
    plan = plan_type(ColumnType.CURRENCY, pd.Series([None, None], dtype="object"))

    assert plan.render() == "NUMERIC(15, 2)"
    assert not plan.widened


def test_planning_a_type_without_any_data_is_the_declared_mapping():
    assert isinstance(plan_type(ColumnType.INTEGER_ORDINAL).sql_type, Integer)


# ---------------------------------------------------------------------------
# values
# ---------------------------------------------------------------------------


def test_a_blank_stays_blank_through_every_coercion():
    """No branch of the coercion may substitute a zero, an empty string or an
    epoch date for a value that was not there."""

    for column_type in ColumnType:
        series = pd.Series([None, float("nan")], dtype="object")
        coerced = coerce_series(series, column_type)
        assert coerced.isna().all(), column_type


def test_a_value_that_will_not_convert_becomes_null_rather_than_a_guess():
    coerced = coerce_series(pd.Series(["12.5", "not a number"]), ColumnType.CURRENCY)

    assert coerced.tolist()[0] == 12.5
    assert pd.isna(coerced.tolist()[1])


def test_a_nominal_code_does_not_arrive_with_a_decimal_point():
    """pandas types an integer column that ever held a blank as float, and
    ``1001.0`` is not the code anybody typed."""

    coerced = coerce_series(pd.Series([1001.0, None, 1002.0]), ColumnType.INTEGER_NOMINAL)

    assert coerced.tolist() == ["1001", None, "1002"]


def test_yes_and_no_become_real_booleans():
    coerced = coerce_series(pd.Series(["Yes", "no", "TRUE", "maybe"]), ColumnType.BOOLEAN)

    assert coerced.tolist() == [True, False, True, None]


def test_records_carry_none_and_never_a_floating_point_nan():
    """A bound ``nan`` reaches PostgreSQL as a value, in a column meant to be
    empty — the one way a missing measurement can become a present one."""

    frame = pd.DataFrame(
        {
            "a": [1.0, float("nan")],
            "b": ["x", None],
            "d": pd.to_datetime(["2024-01-01", None]),
        }
    )

    records = to_records(frame)

    assert records[1]["a"] is None
    assert records[1]["b"] is None
    assert records[1]["d"] is None
    assert records[0]["a"] == 1.0
