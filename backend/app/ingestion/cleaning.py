"""Phase 0 — mixed-type column cleaning.

Excel columns arrive as ``object`` dtype holding a soup of ``"৳1,200"``,
``"N/A"``, ``"1200"`` and real numbers.  This module normalises null markers,
strips currency/thousands decoration and re-types a column to numeric or
datetime when the overwhelming majority of its values agree.

Every decision is reported back as a :class:`ColumnCleanReport` so the UI can
explain *why* a column changed type — the transparency the human-in-the-loop
design depends on.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

from app.core.config import get_settings

#: Textual stand-ins for "no value" seen across real business spreadsheets,
#: including Excel's own error literals.
NULL_TOKENS: frozenset[str] = frozenset(
    {
        "",
        "-",
        "--",
        "---",
        "—",
        "–",
        "n/a",
        "n.a.",
        "na",
        "null",
        "none",
        "nil",
        "nan",
        "?",
        "??",
        "unknown",
        "not available",
        "not applicable",
        "tbd",
        "#ref!",
        "#n/a",
        "#value!",
        "#div/0!",
        "#name?",
        "#null!",
        "#num!",
        "\\n",
    }
)

#: Symbols stripped before attempting a numeric parse.  ৳ (BDT) is first
#: because the target users are Bangladeshi SMEs.
CURRENCY_SYMBOLS: tuple[str, ...] = ("৳", "$", "£", "€", "₹", "¥", "₦", "R$", "Tk", "TK", "tk")

#: Date patterns covering the formats Excel and CSV exports produce in
#: practice.  The two-digit-year variants come last so that a four-digit
#: format wins any tie, and because they are the most likely to claim a value
#: that is not a date at all.
DATE_FORMATS: tuple[str, ...] = (
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%d-%m-%Y",
    "%m-%d-%Y",
    "%Y/%m/%d",
    "%d.%m.%Y",
    "%d %b %Y",
    "%d %B %Y",
    "%b %d, %Y",
    "%Y-%m-%d %H:%M:%S",
    "%d/%m/%Y %H:%M",
    "%d/%m/%y",
    "%m/%d/%y",
    "%d-%m-%y",
    "%m-%d-%y",
)

#: Distinct values scored when deciding which format a column is written in.
#: Ranking is about which convention the column follows, and a couple of
#: thousand distinct values settle that as well as a million would.
DATE_FORMAT_SAMPLE = 2_000

_TRAILING_MINUS_RE = re.compile(r"^\((.*)\)$")
_PERCENT_RE = re.compile(r"^\s*[-+]?[\d,]*\.?\d+\s*%\s*$")
_THOUSANDS_RE = re.compile(r"(?<=\d),(?=\d{3}\b)")
_SPACES_RE = re.compile(r"[\s ]+")


@dataclass(slots=True)
class ColumnCleanReport:
    """What the cleaner did to one column, and what it noticed."""

    column: str
    original_dtype: str
    final_dtype: str
    nulls_normalized: int = 0
    currency_stripped: int = 0
    percent_converted: int = 0
    whitespace_trimmed: int = 0
    retyped_to: str | None = None
    parse_success_ratio: float | None = None
    coerced_to_null: int = 0
    detected_currency_symbol: str | None = None
    date_format: str | None = None
    mixed_type_ratio: float = 0.0
    #: Share of a *text* column that parses as some other type, when that share
    #: was too small to convert on.  ``mixed_type_ratio`` cannot see this: every
    #: value in a CSV arrives as a string, so a column holding half dates and
    #: half product codes has one python type and looks perfectly consistent.
    unconverted_ratio: float = 0.0
    unconverted_type: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(
            self.nulls_normalized
            or self.currency_stripped
            or self.percent_converted
            or self.whitespace_trimmed
            or self.retyped_to
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "original_dtype": self.original_dtype,
            "final_dtype": self.final_dtype,
            "nulls_normalized": self.nulls_normalized,
            "currency_stripped": self.currency_stripped,
            "percent_converted": self.percent_converted,
            "whitespace_trimmed": self.whitespace_trimmed,
            "retyped_to": self.retyped_to,
            "parse_success_ratio": (
                round(self.parse_success_ratio, 4) if self.parse_success_ratio is not None else None
            ),
            "coerced_to_null": self.coerced_to_null,
            "detected_currency_symbol": self.detected_currency_symbol,
            "date_format": self.date_format,
            "mixed_type_ratio": round(self.mixed_type_ratio, 4),
            "unconverted_ratio": round(self.unconverted_ratio, 4),
            "unconverted_type": self.unconverted_type,
            "changed": self.changed,
            "notes": self.notes,
        }


def is_null_token(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    if isinstance(value, str):
        return value.strip().lower() in NULL_TOKENS
    return False


def strip_currency(text: str) -> tuple[str, str | None]:
    """Remove one currency symbol and thousands separators.

    Returns the cleaned text plus the symbol that was found, if any.  Accounting
    negatives — ``(1,200)`` — become ``-1200``.
    """

    cleaned = _SPACES_RE.sub("", text)
    symbol: str | None = None
    for candidate in CURRENCY_SYMBOLS:
        if candidate in cleaned:
            symbol = candidate
            cleaned = cleaned.replace(candidate, "")
            break
    negated = _TRAILING_MINUS_RE.match(cleaned)
    if negated:
        cleaned = "-" + negated.group(1)
    cleaned = _THOUSANDS_RE.sub("", cleaned)
    return cleaned, symbol


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    cleaned, _ = strip_currency(text)
    if not cleaned or cleaned in {"-", "+", "."}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_percent(value: Any) -> float | None:
    """``"45%"`` -> ``0.45``.  Returns ``None`` when the value is not a percent."""

    if not isinstance(value, str) or not _PERCENT_RE.match(value):
        return None
    number = _to_float(value.strip().rstrip("%"))
    return None if number is None else number / 100.0


def try_parse_date(
    value: Any, formats: Sequence[str] = DATE_FORMATS
) -> tuple[datetime | None, str | None]:
    """Parse one value against the known formats.  Returns (value, format used).

    ``formats`` is normally the column's ranking from :func:`rank_date_formats`
    rather than the declaration order — see there for why the order decides
    whether a date is right or wrong.
    """

    if isinstance(value, datetime):
        return value, None
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day), None
    if not isinstance(value, str):
        return None, None
    text = value.strip()
    if not text or text.lower() in NULL_TOKENS:
        return None, None
    for fmt in formats:
        try:
            return datetime.strptime(text, fmt), fmt
        except ValueError:
            continue
    return None, None


def rank_date_formats(values: Iterable[Any]) -> list[str]:
    """Order the known formats by how much of *this column* each one explains.

    Two formats can both parse the same string and disagree about what it
    means: ``06/05/2021`` is 6 May under ``%d/%m/%Y`` and 5 June under
    ``%m/%d/%Y``.  Taking the first format that happens to match, value by
    value, therefore reads a single American column two different ways — the
    ambiguous ``06/05/2021`` becomes 6 May while the unambiguous ``06/15/2021``
    becomes 15 June, and the row that moved by a month looks exactly like the
    rows that did not.

    Counting first and parsing second removes the guess: whichever format
    explains the most of the column is the convention the column was written
    in.  Formats that are not ambiguous with each other (``2024-01-15`` and
    ``16/01/2024`` cannot both describe one string) all keep non-zero scores,
    so a column that genuinely mixes them still parses in full.
    """

    seen: dict[str, None] = {}
    for value in values:
        if not isinstance(value, str):
            continue
        text = value.strip()
        if text and text.lower() not in NULL_TOKENS:
            seen.setdefault(text, None)
        if len(seen) >= DATE_FORMAT_SAMPLE:
            break

    scored: list[tuple[int, int, str]] = []
    for position, fmt in enumerate(DATE_FORMATS):
        hits = 0
        for text in seen:
            try:
                datetime.strptime(text, fmt)
            except ValueError:
                continue
            hits += 1
        if hits:
            scored.append((-hits, position, fmt))  # ties keep declaration order
    return [fmt for _, _, fmt in sorted(scored)]


def _dominant_python_type(series: pd.Series) -> tuple[str, float]:
    """Share of values that do *not* belong to the most common python type."""

    kinds: dict[str, int] = {}
    for value in series:
        if isinstance(value, bool):
            kind = "bool"
        elif isinstance(value, (int, np.integer)):
            kind = "number"
        elif isinstance(value, (float, np.floating)):
            kind = "number"
        elif isinstance(value, (datetime, date)):
            kind = "date"
        else:
            kind = "text"
        kinds[kind] = kinds.get(kind, 0) + 1
    if not kinds:
        return "empty", 0.0
    total = sum(kinds.values())
    dominant, count = max(kinds.items(), key=lambda item: item[1])
    return dominant, 1.0 - (count / total)


def clean_series(
    series: pd.Series, name: str, retype: bool = True
) -> tuple[pd.Series, ColumnCleanReport]:
    """Normalise one column and report every transformation applied.

    ``retype=False`` keeps the null-marker and whitespace normalisation but
    leaves the column's type alone.  That is what relational sources want: a
    database has already committed to a type, and converting one side of a
    declared foreign key from text to numeric would break the join the source
    itself documented.  Phase 1 reads semantic types from values, not dtypes,
    so nothing downstream is weakened by it.
    """

    settings = get_settings()
    report = ColumnCleanReport(
        column=name,
        original_dtype=str(series.dtype),
        final_dtype=str(series.dtype),
    )

    working = series.copy()

    if working.dtype == object or pd.api.types.is_string_dtype(working):
        trimmed = 0
        nulls = 0
        values: list[Any] = []
        for value in working:
            if isinstance(value, str):
                stripped = value.strip()
                if stripped != value:
                    trimmed += 1
                if stripped.lower() in NULL_TOKENS:
                    nulls += 1
                    values.append(np.nan)
                    continue
                values.append(stripped)
                continue
            if is_null_token(value):
                if value is not None and not (
                    isinstance(value, float) and np.isnan(value)
                ):
                    nulls += 1
                values.append(np.nan)
                continue
            values.append(value)
        working = pd.Series(values, index=working.index, name=working.name, dtype=object)
        report.whitespace_trimmed = trimmed
        report.nulls_normalized = nulls

    non_null = working.dropna()
    if non_null.empty:
        report.final_dtype = str(working.dtype)
        report.notes.append("column is entirely empty")
        return working, report

    _, report.mixed_type_ratio = _dominant_python_type(non_null)

    if working.dtype != object and not pd.api.types.is_string_dtype(working):
        report.final_dtype = str(working.dtype)
        return working, report

    if not retype:
        report.final_dtype = str(working.dtype)
        report.notes.append("type kept as declared by the source schema")
        return working, report

    total = len(non_null)

    # --- percentages -----------------------------------------------------
    percents = non_null.map(parse_percent)
    percent_hits = int(percents.notna().sum())
    if percent_hits / total >= settings.numeric_convert_threshold:
        converted = working.map(lambda v: parse_percent(v) if pd.notna(v) else np.nan)
        result = pd.to_numeric(converted, errors="coerce")
        report.percent_converted = percent_hits
        report.retyped_to = "float"
        report.parse_success_ratio = percent_hits / total
        report.final_dtype = str(result.dtype)
        report.notes.append("percent strings converted to fractions")
        return result, report

    # --- numeric (incl. currency) ---------------------------------------
    symbols: dict[str, int] = {}
    numeric_hits = 0
    leading_zero_hits = 0
    for value in non_null:
        if isinstance(value, str):
            _, symbol = strip_currency(value)
            if symbol:
                symbols[symbol] = symbols.get(symbol, 0) + 1
            text = value.strip()
            # "01711223344" is a phone number, not the integer 1711223344 —
            # a leading zero is meaningful and casting would destroy it.  Only
            # pure digit strings count, so "01/03/2023" still reaches the date
            # branch below.
            if len(text) > 1 and text[0] == "0" and text.isdigit():
                leading_zero_hits += 1
        if _to_float(value) is not None:
            numeric_hits += 1

    numeric_ratio = numeric_hits / total
    zero_padded = leading_zero_hits / total > 0.10
    if zero_padded:
        report.notes.append(
            f"{leading_zero_hits / total:.0%} of values are digit strings with a leading zero — "
            "kept as text so codes are not corrupted"
        )

    if numeric_ratio >= settings.numeric_convert_threshold and not zero_padded:
        converted = working.map(lambda v: _to_float(v) if pd.notna(v) else np.nan)
        result = pd.to_numeric(converted, errors="coerce")
        report.currency_stripped = sum(symbols.values())
        report.detected_currency_symbol = (
            max(symbols.items(), key=lambda item: item[1])[0] if symbols else None
        )
        report.parse_success_ratio = numeric_ratio
        report.coerced_to_null = int(result.isna().sum() - working.isna().sum())
        if pd.notna(result).any() and float(result.dropna().mod(1).abs().max()) == 0.0:
            result = result.astype("Int64")
        report.retyped_to = "numeric"
        report.final_dtype = str(result.dtype)
        if report.coerced_to_null:
            report.notes.append(
                f"{report.coerced_to_null} value(s) could not be parsed as numbers and became null"
            )
        return result, report

    # --- dates -----------------------------------------------------------
    # Rank the formats over the whole column before parsing anything: which
    # format wins decides what ``06/05/2021`` means.
    ranked = rank_date_formats(non_null)
    parsed: list[datetime | None] = []
    formats: dict[str, int] = {}
    date_hits = 0
    for value in non_null:
        value_dt, fmt = try_parse_date(value, ranked)
        if value_dt is not None:
            date_hits += 1
            if fmt:
                formats[fmt] = formats.get(fmt, 0) + 1
    date_ratio = date_hits / total
    if date_ratio >= settings.date_parse_threshold:
        parsed = [
            try_parse_date(v, ranked)[0] if pd.notna(v) else None for v in working
        ]
        result = pd.Series(parsed, index=working.index, name=working.name, dtype="datetime64[ns]")
        report.retyped_to = "datetime"
        report.parse_success_ratio = date_ratio
        report.date_format = (
            max(formats.items(), key=lambda item: item[1])[0] if formats else None
        )
        report.coerced_to_null = int(result.isna().sum() - working.isna().sum())
        report.final_dtype = str(result.dtype)
        if len(formats) > 1:
            report.notes.append(
                "column mixes " + ", ".join(sorted(formats)) + " date formats"
            )
        return result, report

    report.final_dtype = str(working.dtype)
    # Neither conversion reached its threshold.  Say which one came closest and
    # by how much: a column that is 50% dates is not a text column, it is the
    # loudest evidence there is that the file holds two tables stacked on each
    # other, and leaving it as text with no comment exports it as VARCHAR and
    # tells nobody why.
    if date_ratio >= numeric_ratio and date_ratio > 0.0:
        report.unconverted_ratio, report.unconverted_type = date_ratio, "date"
    elif numeric_ratio > 0.0:
        report.unconverted_ratio, report.unconverted_type = numeric_ratio, "numeric"
    if report.unconverted_type:
        report.notes.append(
            f"{report.unconverted_ratio:.0%} of values parse as "
            f"{report.unconverted_type} — column left as text (mixed content)"
        )
    return working, report


def clean_dataframe(
    df: pd.DataFrame, retype: bool = True
) -> tuple[pd.DataFrame, list[ColumnCleanReport]]:
    """Run :func:`clean_series` across every column of a sheet."""

    cleaned: dict[str, pd.Series] = {}
    reports: list[ColumnCleanReport] = []
    for column in df.columns:
        series, report = clean_series(df[column], str(column), retype=retype)
        cleaned[str(column)] = series
        reports.append(report)
    out = pd.DataFrame(cleaned, index=df.index)
    return out, reports
