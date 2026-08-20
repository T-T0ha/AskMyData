"""Phase 0 — ingestion from databases and SQL dumps.

The proposal accepts five kinds of source. Excel and CSV are covered by
``test_phase0_ingestion``; this module covers the other three — SQLite files,
live database connections and SQL dump files — plus the rules that only apply
to sources that arrive with a schema of their own.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from app.ingestion.loader import load_file_source
from app.ingestion.relational import (
    SourceError,
    inspect_source,
    load_relational_tables,
    sqlite_url,
)
from app.ingestion.sqldump import DumpError, extract_copy_blocks, guess_dialect, load_sql_dump


@pytest.fixture
def shop_db(tmp_path: Path) -> Path:
    """A small SQLite database with a declared key, a foreign key and gaps."""

    path = tmp_path / "shop.sqlite"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE customers (
            customer_id TEXT PRIMARY KEY,
            "Full Name" TEXT NOT NULL,
            city TEXT
        );
        CREATE TABLE orders (
            order_id INTEGER PRIMARY KEY,
            customer_id TEXT REFERENCES customers(customer_id),
            amount REAL,
            placed_on TEXT
        );
        INSERT INTO customers VALUES
            ('C-1', 'Rahim Uddin', 'Dhaka'),
            ('C-2', 'Karim Ali', NULL),
            ('C-3', 'Nusrat J', 'Sylhet');
        INSERT INTO orders VALUES
            (1, 'C-1', 1200.5, '2024-01-02'),
            (2, 'C-2', NULL, '2024-01-03'),
            (3, 'C-1', 300.0, '2024-02-11');
        """
    )
    connection.commit()
    connection.close()
    return path


# ---------------------------------------------------------------------------
# SQLite files and live connections
# ---------------------------------------------------------------------------


def test_a_sqlite_file_loads_as_tables(shop_db: Path):
    results = {r.name: r for r in load_file_source(shop_db)}

    assert set(results) == {"customers", "orders"}
    assert results["orders"].row_count == 3
    assert results["customers"].source_kind == "sqlite"
    # A quoted, spaced column name is sanitised exactly like a spreadsheet header.
    assert "full_name" in results["customers"].dataframe.columns


def test_declared_keys_are_captured_as_evidence(shop_db: Path):
    """Phase 3 should not have to rediscover a key the database declares."""

    results = {r.name: r for r in load_file_source(shop_db)}

    assert results["customers"].native_schema["primary_key"] == ["customer_id"]
    foreign_keys = results["orders"].native_schema["foreign_keys"]
    assert foreign_keys == [
        {
            "columns": ["customer_id"],
            "references_table": "customers",
            "references_columns": ["customer_id"],
            "source": "declared",
        }
    ]
    assert any("declared foreign key" in note for note in results["orders"].notes)


def test_a_declared_key_becomes_the_tables_primary_key(shop_db: Path):
    """The declared key is adopted directly — nothing is asked or re-derived."""

    results = {r.name: r for r in load_file_source(shop_db)}
    keys = results["customers"].key_analysis

    assert keys.primary_key == ["customer_id"]
    assert keys.source == "declared"
    assert keys.needs_confirmation is False
    assert keys.needs_synthetic_key is False


def test_missing_values_from_a_database_stay_missing(shop_db: Path):
    """NULL in, NULL out — the same guarantee the cleaning phase makes."""

    results = {r.name: r for r in load_file_source(shop_db)}

    assert results["customers"].dataframe["city"].isna().sum() == 1
    assert results["orders"].dataframe["amount"].isna().sum() == 1


def test_declared_types_are_not_second_guessed(shop_db: Path):
    """A text column full of dates stays text when the source declared it text.

    Re-typing here is not a harmless improvement: ``orders.customer_id`` and
    ``customers.customer_id`` are joined by a declared foreign key, and
    converting one side to a number would break it. Phase 1 still detects the
    semantic type from the values, so nothing is lost by leaving the dtype be.
    """

    results = {r.name: r for r in load_file_source(shop_db)}
    orders = results["orders"].dataframe

    assert orders["placed_on"].dtype == object
    assert str(orders["amount"].dtype).startswith("float")
    assert any(
        "declared by the source schema" in note
        for report in results["orders"].clean_reports
        for note in report.notes
    )


def test_inspecting_a_source_reports_tables_without_loading_them(shop_db: Path):
    inspection = inspect_source(sqlite_url(shop_db))

    assert inspection.dialect == "sqlite"
    assert {t.name for t in inspection.tables} == {"customers", "orders"}
    assert {t.row_count for t in inspection.tables} == {3}
    assert inspection.to_dict()["table_count"] == 2


def test_only_the_requested_tables_are_ingested(shop_db: Path):
    results = load_relational_tables(sqlite_url(shop_db), tables=["orders"])

    assert [r.name for r in results] == ["orders"]


def test_an_unknown_table_is_named_in_the_error(shop_db: Path):
    with pytest.raises(SourceError, match="ghost_table"):
        load_relational_tables(sqlite_url(shop_db), tables=["ghost_table"])


def test_a_large_table_is_truncated_and_says_so(tmp_path: Path):
    path = tmp_path / "big.sqlite"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE readings (id INTEGER, value REAL)")
    connection.executemany(
        "INSERT INTO readings VALUES (?, ?)", [(i, float(i)) for i in range(50)]
    )
    connection.commit()
    connection.close()

    result = load_relational_tables(sqlite_url(path), row_limit=10)[0]

    assert result.row_count == 10
    codes = {issue.code for issue in result.issues}
    assert "truncated_read" in codes
    assert any("first 10 rows" in issue.message for issue in result.issues)


def test_unsupported_backends_are_refused():
    """The URL is user input; SQLAlchemy would load any dialect plugin named in it."""

    with pytest.raises(SourceError, match="unsupported database type"):
        inspect_source("mssql+pyodbc://user:pass@host/db")
    with pytest.raises(SourceError, match="not a valid database URL"):
        inspect_source("this is not a url")
    with pytest.raises(SourceError, match="no SQLite database at"):
        inspect_source("sqlite:////nowhere/missing.sqlite")


def test_the_display_url_never_carries_the_password():
    from app.ingestion.relational import display_url

    shown = display_url("postgresql://analyst:s3cret@db.example.com:5432/shop")

    assert "s3cret" not in shown
    assert "analyst" in shown and "shop" in shown


# ---------------------------------------------------------------------------
# SQL dumps
# ---------------------------------------------------------------------------


MYSQL_DUMP = """
-- MySQL dump 10.13  Distrib 8.0.36
SET NAMES utf8mb4;
DROP TABLE IF EXISTS `products`;
CREATE TABLE `products` (
  `sku` varchar(20) NOT NULL,
  `title` varchar(100) DEFAULT NULL,
  `price` decimal(10,2) DEFAULT NULL,
  PRIMARY KEY (`sku`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
INSERT INTO `products` VALUES ('MEN-1','Shirt',1200.00),('MEN-2','Trouser',NULL),('MEN-3','Cap',350.00);
CREATE INDEX idx_title ON products (title);
"""

# pg_dump's plain-text format: schema-qualified names, constraints declared
# afterwards with ALTER TABLE, and rows in a COPY block rather than INSERTs.
PG_DUMP = """
SET statement_timeout = 0;
CREATE TABLE public.staff (
    staff_id integer NOT NULL,
    name text,
    email text,
    note text
);
ALTER TABLE ONLY public.staff ADD CONSTRAINT staff_pkey PRIMARY KEY (staff_id);
COPY public.staff (staff_id, name, email, note) FROM stdin;
1\tRahim\trahim@example.com\ttop\\tseller
2\tKarim\t\\N\tone line\\nsecond line
3\tNusrat\tnusrat@example.com\t\\N
\\.
"""
# Real tabs separate the fields above; ``\\t`` and ``\\n`` inside a value are
# the two-character escapes COPY uses for a tab and a newline in the data, and
# ``\\N`` is COPY's spelling of NULL.


def test_dialect_is_guessed_from_syntax_only_that_dialect_produces():
    assert guess_dialect(MYSQL_DUMP) == "mysql"
    assert guess_dialect(PG_DUMP) == "postgres"


def test_a_mysql_dump_loads_with_its_rows_and_key(tmp_path: Path):
    path = tmp_path / "dump.sql"
    path.write_text(MYSQL_DUMP)

    result = load_sql_dump(path)[0]

    assert result.name == "products"
    assert result.source_kind == "sql_dump"
    assert result.row_count == 3
    assert result.native_schema["primary_key"] == ["sku"]
    # NULL in the dump is a missing value, not the string "NULL".
    assert result.dataframe["price"].isna().sum() == 1


def test_a_pg_dump_copy_block_becomes_rows(tmp_path: Path):
    """The format ``pg_dump`` actually writes, not the one that is easy to parse.

    A reader that only understands INSERT loads this file's table definition
    and none of its data — a silent empty table that looks like a broken input
    file to the user.
    """

    path = tmp_path / "pg.sql"
    path.write_text(PG_DUMP)

    result = load_sql_dump(path)[0]

    assert result.name == "staff"
    assert result.row_count == 3
    assert list(result.dataframe["name"]) == ["Rahim", "Karim", "Nusrat"]
    # \N is NULL; \t and \n inside a value are escaped, not field separators.
    assert result.dataframe["email"].isna().sum() == 1
    assert result.dataframe["note"].iloc[0] == "top\tseller"
    assert result.dataframe["note"].iloc[1] == "one line\nsecond line"
    assert result.dataframe["note"].isna().sum() == 1


def test_keys_declared_after_the_table_are_still_captured(tmp_path: Path):
    """SQLite cannot ALTER a key onto a table, but Phase 3 still needs to know."""

    path = tmp_path / "pg.sql"
    path.write_text(PG_DUMP)

    result = load_sql_dump(path)[0]

    assert result.native_schema["primary_key"] == ["staff_id"]


def test_foreign_keys_declared_after_the_table_are_captured(tmp_path: Path):
    path = tmp_path / "pg_fk.sql"
    path.write_text(
        """
CREATE TABLE public.customers (customer_id integer NOT NULL, name text);
CREATE TABLE public.orders (order_id integer NOT NULL, customer_id integer, total numeric);
ALTER TABLE ONLY public.customers ADD CONSTRAINT customers_pkey PRIMARY KEY (customer_id);
ALTER TABLE ONLY public.orders ADD CONSTRAINT orders_customer_fkey
    FOREIGN KEY (customer_id) REFERENCES public.customers(customer_id);
INSERT INTO public.orders VALUES (1, 10, 500), (2, 11, 250);
INSERT INTO public.customers VALUES (10, 'Rahim'), (11, 'Karim');
"""
    )

    results = {r.name: r for r in load_sql_dump(path)}

    assert results["orders"].native_schema["foreign_keys"] == [
        {
            "columns": ["customer_id"],
            "references_table": "customers",
            "references_columns": ["customer_id"],
            "source": "declared",
        }
    ]
    assert results["customers"].native_schema["primary_key"] == ["customer_id"]


def test_copy_extraction_leaves_the_rest_of_the_dump_parseable():
    body, blocks = extract_copy_blocks(PG_DUMP)

    assert "COPY" not in body
    assert "Rahim" not in body, "data lines must not reach the SQL parser"
    assert len(blocks) == 1
    table, columns, rows = blocks[0]
    assert table == "staff"
    assert columns == ["staff_id", "name", "email", "note"]
    assert len(rows) == 3


def test_a_dump_with_nothing_replayable_is_reported(tmp_path: Path):
    path = tmp_path / "roles.sql"
    path.write_text("CREATE ROLE analyst;\nGRANT SELECT ON ALL TABLES TO analyst;\n")

    with pytest.raises(DumpError, match="no CREATE TABLE"):
        load_sql_dump(path)


def test_an_empty_dump_is_reported(tmp_path: Path):
    path = tmp_path / "empty.sql"
    path.write_text("   \n")

    with pytest.raises(DumpError, match="empty"):
        load_sql_dump(path)
