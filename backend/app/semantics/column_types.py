"""Phase 1 — column type detection (nine types).

SemTabla (Jin et al., CHI '26) defines six types: ``text``,
``integer_ordinal``, ``integer_nominal``, ``integer_continuous``, ``float`` and
``date``.  This project adds ``currency``, ``boolean`` and ``identifier``,
because business spreadsheets are full of money columns, yes/no flags and
order codes that all collapse to "text" or "float" under the paper's scheme
and lose their meaning in the process.

Detection is a cascade run over the **full** column (not a sample), most
specific test first.  Each test returns a confidence and a human-readable
piece of evidence, so the UI can always answer "why did you call this a
date?".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from app.core.config import get_settings
from app.core.schemas import ColumnType
from app.ingestion.cleaning import DATE_FORMATS, _to_float, try_parse_date

BOOLEAN_TRUE = frozenset({"true", "t", "yes", "y", "1", "1.0", "on", "active", "enabled"})
BOOLEAN_FALSE = frozenset({"false", "f", "no", "n", "0", "0.0", "off", "inactive", "disabled"})

#: Column-name keywords that make a numeric column monetary.
FINANCIAL_KEYWORDS: tuple[str, ...] = (
    "price",
    "cost",
    "revenue",
    "amount",
    "total",
    "fee",
    "salary",
    "wage",
    "budget",
    "payment",
    "balance",
    "charge",
    "income",
    "expense",
    "profit",
    "discount",
    "tax",
    "subtotal",
    "paid",
    "due",
    "invoice_value",
    "turnover",
)

UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE
)
#: ``ORD-1001``, ``SKU_A1``, ``INV/2024/003`` — a prefix plus a code.
CODE_RE = re.compile(r"^[A-Za-z]{1,6}[-_/#]?\d{2,}[-_/]?[A-Za-z0-9]*$")
HEX_RE = re.compile(r"^[0-9a-fA-F]{12,}$")

IDENTIFIER_NAME_HINTS = ("_id", "id_", "code", "sku", "uuid", "guid", "ref", "number", "no")

#: Names that mark a numeric column as a ratio rather than money, even when a
#: financial keyword is also present ("discount_pct").
RATIO_NAME_HINTS = ("pct", "percent", "percentage", "ratio", "rate", "margin", "share")


@dataclass(slots=True)
class TypeDecision:
    column_type: ColumnType
    confidence: float
    evidence: list[str] = field(default_factory=list)
    currency_symbol: str | None = None


def _non_null(series: pd.Series) -> pd.Series:
    return series.dropna()


def _as_text(values: pd.Series) -> list[str]:
    return [str(v).strip() for v in values]


def _name_has(name: str, keywords: tuple[str, ...]) -> str | None:
    lowered = re.sub(r"[^a-z0-9]+", "_", str(name).lower())
    for keyword in keywords:
        if keyword in lowered:
            return keyword
    return None


def _is_boolean(series: pd.Series, values: pd.Series) -> TypeDecision | None:
    if pd.api.types.is_bool_dtype(series):
        return TypeDecision(ColumnType.BOOLEAN, 1.0, ["column is stored as a boolean dtype"])
    distinct = {str(v).strip().lower() for v in values}
    if not distinct or len(distinct) > 2:
        return None
    if distinct <= BOOLEAN_TRUE | BOOLEAN_FALSE:
        # A single distinct value of "0"/"1" is more likely a flag than a
        # measurement, but two complementary values are conclusive.
        confidence = 0.98 if len(distinct) == 2 else 0.75
        return TypeDecision(
            ColumnType.BOOLEAN,
            confidence,
            [f"only {len(distinct)} distinct value(s): {sorted(distinct)}"],
        )
    return None


def _is_date(series: pd.Series, values: pd.Series, threshold: float) -> TypeDecision | None:
    if pd.api.types.is_datetime64_any_dtype(series):
        return TypeDecision(ColumnType.DATE, 1.0, ["column is stored as datetime64"])
    if pd.api.types.is_numeric_dtype(series):
        return None
    text_values = _as_text(values)
    if not text_values:
        return None
    hits = sum(1 for v in text_values if try_parse_date(v)[0] is not None)
    ratio = hits / len(text_values)
    if ratio >= threshold:
        return TypeDecision(
            ColumnType.DATE,
            round(ratio, 3),
            [f"{ratio:.0%} of values parse against {len(DATE_FORMATS)} known date formats"],
        )
    return None


def _is_identifier(
    name: str, series: pd.Series, values: pd.Series, unique_ratio: float
) -> TypeDecision | None:
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series):
        return None
    text_values = _as_text(values)
    if not text_values:
        return None
    total = len(text_values)

    # Digit strings that keep a leading zero are codes, not quantities: no one
    # writes the number 1711223344 as "01711223344".
    zero_padded = [v for v in text_values if len(v) > 1 and v[0] == "0" and v.isdigit()]
    if len(zero_padded) / total > 0.5:
        return TypeDecision(
            ColumnType.IDENTIFIER,
            0.9,
            [
                f"{len(zero_padded) / total:.0%} of values are digit strings with a "
                "significant leading zero"
            ],
        )

    uuid_hits = sum(1 for v in text_values if UUID_RE.match(v))
    if uuid_hits / total >= 0.9:
        return TypeDecision(ColumnType.IDENTIFIER, 0.99, ["values match the UUID pattern"])

    hex_hits = sum(1 for v in text_values if HEX_RE.match(v))
    if hex_hits / total >= 0.9 and unique_ratio > 0.9:
        return TypeDecision(
            ColumnType.IDENTIFIER, 0.9, ["values are long unique hexadecimal codes"]
        )

    code_hits = sum(1 for v in text_values if CODE_RE.match(v))
    code_ratio = code_hits / total
    lengths = {len(v) for v in text_values}
    consistent_length = len(lengths) <= 2

    if code_ratio >= 0.9 and unique_ratio >= 0.5:
        evidence = [f"{code_ratio:.0%} of values are prefix+digits codes (e.g. {text_values[0]!r})"]
        confidence = 0.9 if consistent_length else 0.82
        if _name_has(name, IDENTIFIER_NAME_HINTS):
            confidence = min(0.97, confidence + 0.06)
            evidence.append("column name suggests an identifier")
        return TypeDecision(ColumnType.IDENTIFIER, round(confidence, 3), evidence)

    if (
        consistent_length
        and unique_ratio > 0.95
        and _name_has(name, IDENTIFIER_NAME_HINTS)
        and all(v.isalnum() or "-" in v or "_" in v for v in text_values)
    ):
        return TypeDecision(
            ColumnType.IDENTIFIER,
            0.8,
            ["values are unique fixed-length alphanumeric codes and the name suggests an identifier"],
        )
    return None


def _numeric_values(series: pd.Series, values: pd.Series) -> np.ndarray | None:
    if pd.api.types.is_numeric_dtype(series):
        return values.astype("float64").to_numpy()
    converted = [_to_float(v) for v in values]
    if not converted or any(c is None for c in converted):
        return None
    return np.asarray(converted, dtype="float64")


def _is_currency(name: str, numbers: np.ndarray, currency_symbol: str | None) -> TypeDecision | None:
    if currency_symbol:
        return TypeDecision(
            ColumnType.CURRENCY,
            0.97,
            [f"values carried the currency symbol {currency_symbol!r}"],
            currency_symbol,
        )
    if _name_has(name, RATIO_NAME_HINTS):
        return None
    keyword = _name_has(name, FINANCIAL_KEYWORDS)
    if keyword is None:
        return None
    # Money is not confined to [0, 1]; a column that is means a rate/fraction.
    if numbers.size and float(np.nanmax(numbers)) <= 1.0 and float(np.nanmin(numbers)) >= 0.0:
        return None
    return TypeDecision(
        ColumnType.CURRENCY,
        0.85,
        [f"numeric column whose name contains the financial keyword {keyword!r}"],
    )


def _integer_subtype(
    numbers: np.ndarray, unique_ratio: float, nominal_threshold: float
) -> TypeDecision:
    """Distinguish the paper's three integer flavours."""

    unique_values = np.unique(numbers)
    if unique_values.size >= 3:
        diffs = np.diff(unique_values)
        if np.all(diffs == diffs[0]) and diffs[0] > 0 and unique_values.size >= 5:
            return TypeDecision(
                ColumnType.INTEGER_ORDINAL,
                0.9,
                [f"distinct values form a constant step of {diffs[0]:g}"],
            )
    if unique_ratio <= nominal_threshold:
        return TypeDecision(
            ColumnType.INTEGER_NOMINAL,
            round(0.7 + (nominal_threshold - unique_ratio), 3),
            [
                f"only {unique_values.size} distinct integer value(s) over "
                f"{numbers.size} rows — reads as a category code"
            ],
        )
    return TypeDecision(
        ColumnType.INTEGER_CONTINUOUS,
        0.85,
        [f"{unique_ratio:.0%} of integer values are distinct — reads as a measurement"],
    )


def detect_column_type(
    series: pd.Series,
    name: str,
    currency_symbol: str | None = None,
) -> TypeDecision:
    """Run the full cascade for one column.

    Order matters: boolean before integer (0/1 flags), date before text,
    identifier before text, currency before the numeric subtypes.
    """

    settings = get_settings()
    values = _non_null(series)
    row_count = int(len(series))
    if values.empty:
        return TypeDecision(ColumnType.TEXT, 0.0, ["column is entirely null"])

    unique_count = int(values.nunique())
    unique_ratio = unique_count / len(values) if len(values) else 0.0

    decision = _is_boolean(series, values)
    if decision:
        return decision

    decision = _is_date(series, values, settings.date_type_threshold)
    if decision:
        return decision

    decision = _is_identifier(name, series, values, unique_ratio)
    if decision:
        return decision

    numbers = _numeric_values(series, values)
    if numbers is not None and numbers.size:
        decision = _is_currency(name, numbers, currency_symbol)
        if decision:
            return decision

        is_integral = bool(np.all(np.mod(numbers, 1) == 0))
        if not is_integral:
            return TypeDecision(
                ColumnType.FLOAT,
                0.95,
                ["values include a fractional part"],
            )

        # A high-cardinality integer column named like a key is an identifier,
        # not a measurement (e.g. an 11-digit phone number or account code).
        if unique_ratio > 0.98 and _name_has(name, IDENTIFIER_NAME_HINTS):
            return TypeDecision(
                ColumnType.IDENTIFIER,
                0.85,
                ["integer values are unique and the column name suggests an identifier"],
            )
        return _integer_subtype(numbers, unique_ratio, settings.nominal_unique_ratio)

    lengths = [len(t) for t in _as_text(values)]
    average_length = sum(lengths) / len(lengths)
    return TypeDecision(
        ColumnType.TEXT,
        0.8 if average_length > 3 else 0.65,
        [f"free text, average length {average_length:.0f} characters"],
    )


def type_summary(decision: TypeDecision) -> dict[str, Any]:
    return {
        "column_type": decision.column_type.value,
        "confidence": round(decision.confidence, 4),
        "evidence": decision.evidence,
        "currency_symbol": decision.currency_symbol,
        "from_paper": decision.column_type.is_paper_type,
    }
