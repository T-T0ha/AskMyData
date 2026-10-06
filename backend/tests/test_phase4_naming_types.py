"""Phase 4 — identifiers and column types for the export.

Both modules exist to stop the same class of failure: something the user typed
into a spreadsheet changing meaning on its way into the database.  A header
becomes an identifier without becoming SQL, and a value is never rounded,
truncated or overflowed by the type it is declared with.
"""

from __future__ import annotations

import pandas as pd
import pytest
from sqlalchemy import BigInteger, Boolean, Date, DateTime, Double, Integer, Numeric, String, Text

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
    declared_family,
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


# ---------------------------------------------------------------------------
# a type the source declared
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "declared, family",
    [
        ("INTEGER", "integer"),
        ("BIGINT", "integer"),
        ("UNSIGNED BIG INT", "integer"),
        ("TINYINT(1)", "integer"),
        ("REAL", "float"),
        ("DOUBLE PRECISION", "float"),
        ("NUMERIC(10, 2)", "numeric"),
        ("DECIMAL(5,0)", "numeric"),
        ("VARCHAR(20)", "text"),
        ("TEXT", "text"),
        ("DATE", "date"),
        ("DATETIME", "datetime"),
        ("TIMESTAMP WITHOUT TIME ZONE", "datetime"),
        ("INTERVAL", None),
        ("NULL", None),
        ("", None),
        (None, None),
    ],
)
def test_a_declared_type_is_read_the_way_the_database_reads_it(declared, family):
    assert declared_family(declared) == family


def test_a_code_the_source_declared_as_a_number_stays_a_number():
    """``height INTEGER`` has 3 distinct values here, so Phase 1 reads it as a
    category code.  Storing it as VARCHAR makes ``height > 200`` compare text."""

    heights = pd.Series([180, 200, 180, 200, 175, 180])

    kept = plan_type(ColumnType.INTEGER_NOMINAL, heights, declared="INTEGER")

    assert isinstance(kept.sql_type, Integer) and not isinstance(kept.sql_type, BigInteger)
    assert kept.column_type is ColumnType.INTEGER_NOMINAL, "it is still *read* as a code"
    assert kept.storage is ColumnType.INTEGER_ORDINAL, "but stored as the number it is"
    assert kept.kept_source_type == "INTEGER"
    assert not kept.widened, "keeping the source's type is not a widening"
    assert coerce_series(heights, kept.storage).tolist() == [180, 200, 180, 200, 175, 180]


def test_a_spreadsheet_column_has_no_declaration_and_is_typed_by_the_mapping():
    heights = pd.Series([180, 200, 180, 200])

    planned = plan_type(ColumnType.INTEGER_NOMINAL, heights)

    assert isinstance(planned.sql_type, String)
    assert planned.kept_source_type is None
    assert planned.storage is ColumnType.INTEGER_NOMINAL


def test_a_column_the_source_declared_as_text_stays_text():
    zips = pd.Series(["02134", "02139", "02134"])

    planned = plan_type(ColumnType.INTEGER_NOMINAL, zips, declared="TEXT")

    assert isinstance(planned.sql_type, String)
    assert planned.kept_source_type is None


def test_a_numeric_declaration_does_not_survive_a_value_that_is_not_a_number():
    """SQLite lets a stray ``'n/a'`` into an INTEGER column.  The declaration
    cannot hold the column then, so the ordinary mapping applies."""

    messy = pd.Series([1, 2, "n/a", 2, 1])

    planned = plan_type(ColumnType.INTEGER_NOMINAL, messy, declared="INTEGER")

    assert isinstance(planned.sql_type, String)
    assert planned.kept_source_type is None


def test_an_identifier_the_source_declared_as_an_integer_stays_an_integer():
    ids = pd.Series([1, 2, 3, 4])

    planned = plan_type(ColumnType.IDENTIFIER, ids, declared="INTEGER")

    assert isinstance(planned.sql_type, Integer)


def test_a_declared_integer_past_the_32_bit_limit_is_a_bigint():
    planned = plan_type(
        ColumnType.INTEGER_NOMINAL, pd.Series([INT32_MAX + 5, 7, 7]), declared="BIGINT"
    )

    assert isinstance(planned.sql_type, BigInteger)


def test_a_declared_real_and_a_declared_numeric_keep_their_own_kind():
    values = pd.Series([2.0, 3.0, 2.0])

    assert isinstance(plan_type(ColumnType.INTEGER_NOMINAL, values, declared="REAL").sql_type, Double)
    assert isinstance(
        plan_type(ColumnType.INTEGER_NOMINAL, values, declared="NUMERIC(6,1)").sql_type, Numeric
    )


def test_a_declared_integer_holding_fractions_is_not_forced_to_an_integer():
    planned = plan_type(ColumnType.INTEGER_NOMINAL, pd.Series([1.5, 1.5, 2.5]), declared="INTEGER")

    assert isinstance(planned.sql_type, String)


def test_a_declared_date_of_whole_days_stays_a_date():
    days = pd.Series(["2015-11-29", "2016-03-20"])

    planned = plan_type(ColumnType.DATE, days, declared="DATE")

    assert isinstance(planned.sql_type, Date)
    assert render_type(planned.sql_type) == "DATE"


def test_a_declared_date_that_carries_a_time_keeps_it():
    stamped = pd.Series(["2015-11-29 08:30:00", "2016-03-20 00:00:00"])

    assert isinstance(plan_type(ColumnType.DATE, stamped, declared="DATE").sql_type, DateTime)


def test_a_date_with_no_declaration_is_still_a_timestamp():
    days = pd.Series(["2015-11-29", "2016-03-20"])

    assert isinstance(plan_type(ColumnType.DATE, days).sql_type, DateTime)


@pytest.mark.parametrize(
    "column_type, values",
    [
        (ColumnType.DATE, ["2019-09-14", "2019-10-17"]),
        (ColumnType.FLOAT, ["218.300", "91.610", "257.320"]),
        (ColumnType.INTEGER_CONTINUOUS, ["1200", "40", "7"]),
        (ColumnType.INTEGER_ORDINAL, ["1", "2", "3"]),
        (ColumnType.CURRENCY, ["10.25", "3.5"]),
        (ColumnType.BOOLEAN, ["true", "false", "true"]),
    ],
)
def test_a_column_the_source_declared_as_text_is_not_stored_as_what_it_looks_like(
    column_type, values
):
    """``'218.300'`` in a TEXT column sorts as text (``'91.610'`` is the largest),
    and the question the source answers that way must be answered the same way
    from the export.  Phase 1's reading of the contents is a description of the
    column, not an instruction to rewrite it."""

    series = pd.Series(values)

    planned = plan_type(column_type, series, declared="TEXT")

    assert isinstance(planned.sql_type, Text)
    assert planned.column_type is column_type, "it is still *read* as a number, day or flag"
    assert planned.storage is ColumnType.TEXT
    assert planned.kept_source_type == "TEXT"
    assert not planned.widened
    assert coerce_series(series, planned.storage).tolist() == values, "byte for byte"


@pytest.mark.parametrize("declared", ["VARCHAR(20)", "CHAR(3)", "TEXT", "character varying"])
def test_every_spelling_of_a_text_declaration_keeps_the_column_text(declared):
    planned = plan_type(ColumnType.DATE, pd.Series(["2019-09-14"]), declared=declared)

    assert isinstance(planned.sql_type, Text)


def test_a_text_column_with_nothing_in_it_is_still_a_text_column():
    assert isinstance(plan_type(ColumnType.DATE, pd.Series([None, None]), declared="TEXT").sql_type, Text)
    assert isinstance(plan_type(ColumnType.FLOAT, None, declared="TEXT").sql_type, Text)


def test_a_spreadsheet_column_read_as_a_number_or_a_day_is_still_typed_by_the_mapping():
    """No declaration, no override: a spreadsheet's only voice is the mapping."""

    assert isinstance(plan_type(ColumnType.FLOAT, pd.Series(["1.5", "2.5"])).sql_type, Double)
    assert isinstance(plan_type(ColumnType.DATE, pd.Series(["2019-09-14"])).sql_type, DateTime)
    assert isinstance(plan_type(ColumnType.BOOLEAN, pd.Series(["true", "false"])).sql_type, Boolean)


def test_the_declaration_only_replaces_the_reading_that_changes_what_is_stored():
    """A currency or a float keeps the mapping — only code/identifier and date do
    anything with a declaration."""

    amounts = pd.Series([10.25, 3.5])

    planned = plan_type(ColumnType.CURRENCY, amounts, declared="REAL")

    assert render_type(planned.sql_type) == "NUMERIC(15, 2)"
    assert planned.kept_source_type is None
