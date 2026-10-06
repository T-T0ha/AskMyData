"""The nine column types, as PostgreSQL columns.

The mapping itself is the project's, and is fixed:

============================  ==========================
``integer_continuous``        ``NUMERIC``
``integer_ordinal``           ``INTEGER``
``integer_nominal``           ``VARCHAR(50)``
``currency``                  ``NUMERIC(15, 2)``
``date``                      ``TIMESTAMP``
``text``                      ``TEXT``
``boolean``                   ``BOOLEAN``
``identifier``                ``VARCHAR(100)``
``float``                     ``DOUBLE PRECISION``
============================  ==========================

What this module adds is the part a table of types cannot express: **the
declared width has to fit the data that is going into it.**  ``VARCHAR(100)``
against a 140-character order reference either truncates the value or refuses
the insert; ``NUMERIC(15, 2)`` against a unit price of 12.3456 rounds it,
quietly, and the rounded number is then indistinguishable from a measured one.
That is the same failure the no-imputation rule exists to prevent, arriving
through the type system instead of through a cleaning step — so every type here
is *checked against the column* and widened if it does not fit, with a note
saying so.

**A type the source declared is kept.**  A spreadsheet has no types, so the
mapping above is the only voice the column has.  A database or SQL dump has
one: ``height INTEGER`` is a number that gets compared, sorted and summed, and
exporting it as ``VARCHAR(50)`` because few of its values are distinct makes
``WHERE height > 200`` compare text and ``ORDER BY height`` put 99 after 100.
So when the source declared a numeric type and every value fits it, a column
Phase 1 read as a nominal code or an identifier keeps that type; a column the
source declared ``DATE``, and whose values are whole days, stays a ``DATE``.
The same goes for the other direction: a column the source declared ``TEXT``
stays text however Phase 1 reads what is in it.  ``'218.300'`` in a ``TEXT``
column sorts as text, ``'2019-09-14'`` is compared with ``'2019-09-14'``, and
``'true'`` is what a ``WHERE approved = 'true'`` is written against — a
database that stored them that way is queried that way, and rewriting them as
a float, a timestamp (``'2019-09-14 00:00:00.000000'`` on SQLite) or a ``1``
silently changes the answer to questions the source answered correctly.
Turning such a column into a number or a date is a cleaning decision, and
Phase 2's reviewed ``type_cast`` is where it is made.
The *semantic* type is untouched — ``sem_metadata`` still says
``integer_nominal`` or ``date`` — only the storage follows the declaration,
and a type the user corrected by hand is never overridden.

The types are SQLAlchemy's rather than literal SQL, so the same plan renders as
PostgreSQL DDL and executes against the SQLite the test suite runs on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from sqlalchemy import (
    BigInteger,
    Boolean,
    Date,
    DateTime,
    Double,
    Integer,
    Numeric,
    String,
    Text,
)
from sqlalchemy.dialects import postgresql
from sqlalchemy.types import TypeEngine

from app.core.schemas import ColumnType
from app.semantics.column_types import BOOLEAN_FALSE, BOOLEAN_TRUE

#: ``INTEGER`` is 32-bit.  A count of milliseconds, a national ID number or a
#: phone number typed as a figure goes past it, and PostgreSQL raises rather
#: than wrapping — but the export has already failed by then.
INT32_MAX = 2_147_483_647

#: The declared scale of a currency column.  Widened, never narrowed, and only
#: when the data actually carries more decimal places than this.
CURRENCY_SCALE = 2
CURRENCY_PRECISION = 15

#: Scales tried, in order, when a currency column does not fit ``NUMERIC(15,2)``.
_SCALE_LADDER = (2, 4, 6, 8)

#: Longest value a ``VARCHAR(n)`` column type is declared with by default.
DEFAULT_WIDTHS: dict[ColumnType, int] = {
    ColumnType.INTEGER_NOMINAL: 50,
    ColumnType.IDENTIFIER: 100,
}


def default_sql_type(column_type: ColumnType) -> TypeEngine:
    """The mapping above, with no reference to any data."""

    if column_type is ColumnType.INTEGER_CONTINUOUS:
        return Numeric()
    if column_type is ColumnType.INTEGER_ORDINAL:
        return Integer()
    if column_type is ColumnType.INTEGER_NOMINAL:
        return String(DEFAULT_WIDTHS[ColumnType.INTEGER_NOMINAL])
    if column_type is ColumnType.CURRENCY:
        return Numeric(CURRENCY_PRECISION, CURRENCY_SCALE)
    if column_type is ColumnType.DATE:
        return DateTime()
    if column_type is ColumnType.BOOLEAN:
        return Boolean()
    if column_type is ColumnType.IDENTIFIER:
        return String(DEFAULT_WIDTHS[ColumnType.IDENTIFIER])
    if column_type is ColumnType.FLOAT:
        return Double()
    return Text()


@dataclass(slots=True)
class TypePlan:
    """The type a column will be created with, and why it is not the default."""

    column_type: ColumnType
    sql_type: TypeEngine
    #: Empty when the declared mapping was used unchanged.  Every entry is
    #: something the user should see in the export report.
    notes: list[str] = field(default_factory=list)
    #: The type whose coercion fills the column, when it is not ``column_type``:
    #: a nominal code the source declared ``INTEGER`` is *read* as a code but
    #: *stored* as the integer it was.
    stored_as: ColumnType | None = None
    #: The source's own declaration, when it was kept over the default mapping.
    kept_source_type: str | None = None

    @property
    def widened(self) -> bool:
        return bool(self.notes)

    @property
    def storage(self) -> ColumnType:
        """The column type whose :func:`coerce_series` rules produce the stored values."""

        return self.stored_as or self.column_type

    def render(self) -> str:
        """The type as PostgreSQL would spell it, for the DDL preview."""

        return render_type(self.sql_type)

    def to_dict(self) -> dict[str, Any]:
        return {
            "column_type": self.column_type.value,
            "sql_type": self.render(),
            "widened": self.widened,
            "notes": self.notes,
            "kept_source_type": self.kept_source_type,
        }


def render_type(sql_type: TypeEngine) -> str:
    return str(sql_type.compile(dialect=postgresql.dialect()))


def declared_family(data_type: str | None) -> str | None:
    """What kind of column a source's declaration describes.

    Reads the type the way SQLite's affinity rules do, which is also how
    ``INT``, ``BIGINT`` and ``UNSIGNED BIG INT`` are told from ``VARCHAR(20)``
    on PostgreSQL and MySQL.  ``None`` means the source said nothing usable —
    a spreadsheet, or a SQLite column declared with no type — and the mapping
    by semantic type applies unchanged.
    """

    text = str(data_type or "").upper()
    if not text or text == "NULL":
        return None
    if any(word in text for word in ("CHAR", "CLOB", "TEXT", "STRING", "UUID", "JSON")):
        return "text"
    if "DATETIME" in text or "TIMESTAMP" in text:
        return "datetime"
    if "DATE" in text:
        return "date"
    if "INT" in text and not any(word in text for word in ("INTERVAL", "POINT")):
        return "integer"
    if any(word in text for word in ("REAL", "FLOA", "DOUB")):
        return "float"
    if any(word in text for word in ("NUMERIC", "DECIMAL", "NUMBER", "MONEY")):
        return "numeric"
    return None


#: Readings Phase 1 gives a column whose *contents* look like numbers, days or
#: truth values.  Over a ``TEXT`` declaration the source has already said what
#: the column is, so none of them changes how it is stored.
_READ_BEYOND_TEXT = frozenset(
    {
        ColumnType.INTEGER_ORDINAL,
        ColumnType.INTEGER_CONTINUOUS,
        ColumnType.CURRENCY,
        ColumnType.FLOAT,
        ColumnType.DATE,
        ColumnType.BOOLEAN,
    }
)


def _keep_declared(
    column_type: ColumnType, values: pd.Series, family: str
) -> tuple[TypeEngine, ColumnType] | None:
    """The source's own numeric or date type, if it applies and every value fits it.

    Only the readings that *change what kind of thing is stored* are
    overridden: a code or identifier over a numeric declaration, and a
    timestamp over a ``DATE`` declaration.  Anything else is the ordinary
    mapping, and a value that does not fit the declaration (a stray ``'n/a'``
    in an ``INTEGER`` column, which SQLite allows) sends the column back to it.
    (A ``TEXT`` declaration needs no check of the values and is handled in
    :func:`plan_type`.)
    """


    if column_type in (ColumnType.INTEGER_NOMINAL, ColumnType.IDENTIFIER) and family in {
        "integer",
        "float",
        "numeric",
    }:
        numbers = pd.to_numeric(values, errors="coerce")
        if bool(numbers.isna().any()):
            return None
        if family == "integer":
            if not bool((numbers % 1 == 0).all()):
                return None
            largest = _max_absolute(numbers)
            return (
                BigInteger() if largest is not None and largest > INT32_MAX else Integer(),
                ColumnType.INTEGER_ORDINAL,
            )
        if family == "float":
            return Double(), ColumnType.FLOAT
        return Numeric(), ColumnType.INTEGER_CONTINUOUS

    if column_type is ColumnType.DATE and family == "date":
        parsed = pd.to_datetime(values, errors="coerce")
        if bool(parsed.notna().all()) and bool((parsed == parsed.dt.normalize()).all()):
            return Date(), ColumnType.DATE
    return None


def plan_type(
    column_type: ColumnType,
    series: pd.Series | None = None,
    declared: str | None = None,
) -> TypePlan:
    """Pick the column's type, widening the default if the values need it.

    ``declared`` is the type the *source* gave the column, when it had one.  Pass
    ``None`` for a spreadsheet, and for any column whose type the user chose.
    """

    sql_type = default_sql_type(column_type)
    notes: list[str] = []
    family = declared_family(declared)

    if family == "text" and column_type in _READ_BEYOND_TEXT:
        # The source said text.  Nothing in the values can contradict that, so
        # this is decided before they are looked at — a column of only blanks
        # is as much a text column as any other.
        return TypePlan(
            column_type,
            Text(),
            notes,
            stored_as=ColumnType.TEXT,
            kept_source_type=str(declared),
        )

    if series is None or series.empty:
        return TypePlan(column_type, sql_type, notes)

    values = series.dropna()
    if values.empty:
        return TypePlan(column_type, sql_type, notes)

    kept = _keep_declared(column_type, values, family) if family else None
    if kept is not None:
        return TypePlan(
            column_type, kept[0], notes, stored_as=kept[1], kept_source_type=str(declared)
        )

    if column_type in DEFAULT_WIDTHS:
        width = DEFAULT_WIDTHS[column_type]
        longest = int(values.map(lambda value: len(str(value))).max())
        if longest > width:
            sql_type = Text()
            notes.append(
                f"declared VARCHAR({width}) by type, widened to TEXT: the longest value is "
                f"{longest} characters and would not fit"
            )

    elif column_type is ColumnType.INTEGER_ORDINAL:
        largest = _max_absolute(values)
        if largest is not None and largest > INT32_MAX:
            sql_type = BigInteger()
            notes.append(
                f"declared INTEGER by type, widened to BIGINT: {largest:,.0f} is past the "
                "32-bit limit"
            )

    elif column_type is ColumnType.CURRENCY:
        sql_type, note = _fit_currency(values)
        if note:
            notes.append(note)

    return TypePlan(column_type, sql_type, notes)


def _max_absolute(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return None
    largest = float(numeric.abs().max())
    return None if math.isnan(largest) or math.isinf(largest) else largest


def _fit_currency(values: pd.Series) -> tuple[TypeEngine, str]:
    """``NUMERIC(15,2)`` unless the money has more digits than that.

    Scale is decided by asking the only question that matters — does rounding
    to this many places change any value in the column? — rather than by
    parsing decimal strings, which is both slower and wrong for values that
    arrived as floats.
    """

    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return Numeric(CURRENCY_PRECISION, CURRENCY_SCALE), ""

    scale = CURRENCY_SCALE
    for candidate in _SCALE_LADDER:
        if bool(((numeric - numeric.round(candidate)).abs() < 1e-9).all()):
            scale = candidate
            break
    else:
        # More than eight decimal places is no longer money; store it exactly
        # and let the value speak for itself.
        return (
            Numeric(),
            "declared NUMERIC(15,2) by type, widened to unconstrained NUMERIC: the values "
            "carry more than eight decimal places and rounding would change them",
        )

    largest = float(numeric.abs().max())
    integer_digits = 1 if largest < 1 else int(math.floor(math.log10(largest))) + 1
    precision = max(CURRENCY_PRECISION, integer_digits + scale)

    if (precision, scale) == (CURRENCY_PRECISION, CURRENCY_SCALE):
        return Numeric(CURRENCY_PRECISION, CURRENCY_SCALE), ""
    return (
        Numeric(precision, scale),
        f"declared NUMERIC(15,2) by type, widened to NUMERIC({precision},{scale}): "
        + (
            f"values carry {scale} decimal places and rounding to 2 would change them"
            if scale > CURRENCY_SCALE
            else f"the largest value needs {integer_digits} digits before the point"
        ),
    )


# ---------------------------------------------------------------------------
# values
# ---------------------------------------------------------------------------


def coerce_series(series: pd.Series, column_type: ColumnType) -> pd.Series:
    """Values as the target column wants them, with blanks left blank.

    Every branch keeps missing values missing.  ``errors="coerce"`` turns a
    value that will not convert into ``NaT``/``NaN``, which becomes ``NULL`` —
    the export never substitutes a zero, an empty string or an epoch date for
    something it could not read, because a reader could not then tell that
    value from a real one.
    """

    if column_type is ColumnType.DATE:
        return pd.to_datetime(series, errors="coerce")
    if column_type is ColumnType.BOOLEAN:
        return _mapped(series, _to_bool)
    if column_type is ColumnType.INTEGER_NOMINAL:
        return _mapped(series, _to_code)
    if column_type in {ColumnType.IDENTIFIER, ColumnType.TEXT}:
        return _mapped(series, lambda value: None if _is_blank(value) else str(value))
    if column_type.is_numeric:
        return pd.to_numeric(series, errors="coerce")
    return series


def _mapped(series: pd.Series, convert) -> pd.Series:
    """``Series.map`` with the result dtype stated rather than inferred.

    Left to itself, pandas re-reads a list of strings and ``None`` and may hand
    back ``NaN`` where the ``None`` was.  Both mean "missing" downstream, but a
    coercion whose output depends on what the input happened to contain is not
    something the rest of the export can reason about.
    """

    return pd.Series(
        [convert(value) for value in series], index=series.index, dtype="object"
    )


def _is_blank(value: Any) -> bool:
    if value is None or value is pd.NaT:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return value is pd.NA


def _to_bool(value: Any) -> bool | None:
    if _is_blank(value):
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in BOOLEAN_TRUE:
        return True
    if text in BOOLEAN_FALSE:
        return False
    return None


def _to_code(value: Any) -> str | None:
    """A nominal integer is a code, and codes are stored as text.

    ``1001.0`` is what pandas hands back for an integer column that ever held a
    blank, and ``"1001.0"`` is not the code anybody typed.
    """

    if _is_blank(value):
        return None
    if isinstance(value, float) and float(value).is_integer():
        return str(int(value))
    return str(value)


def to_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Rows as parameter dictionaries, with every missing value as ``None``.

    ``DataFrame.to_dict("records")`` leaves ``NaN`` and ``NaT`` in place, and a
    float ``nan`` bound as a parameter reaches PostgreSQL as the floating-point
    NaN — a *value*, in a column that is supposed to be empty.
    """

    columns = list(frame.columns)
    records: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False, name=None):
        record: dict[str, Any] = {}
        for column, value in zip(columns, row):
            record[str(column)] = None if _is_missing(value) else value
        records.append(record)
    return records


def _is_missing(value: Any) -> bool:
    if _is_blank(value):
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):  # arrays and other unhashables are not NA
        return False
