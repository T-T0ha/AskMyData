"""Phase 1 — value distribution profiling.

Numeric columns get min/max/mean/median/std plus a histogram; categorical
columns get their top-k values by frequency; every column gets a null ratio.
These feed the editable grid in the UI and, crucially, the compact statistical
summary that Phase 2 sends to Claude *instead of* the data itself.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from app.core.config import get_settings
from app.core.schemas import ColumnType

HISTOGRAM_BINS = 10


def _clean_number(value: Any) -> float | int | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    if number.is_integer() and abs(number) < 1e15:
        return int(number)
    return round(number, 6)


def _json_safe(value: Any) -> Any:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return _clean_number(float(value))
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float):
        return _clean_number(value)
    return value


def numeric_distribution(series: pd.Series) -> dict[str, Any]:
    values = pd.to_numeric(series.dropna(), errors="coerce").dropna()
    if values.empty:
        return {"kind": "numeric", "empty": True}
    array = values.to_numpy(dtype="float64")
    counts, edges = np.histogram(array, bins=min(HISTOGRAM_BINS, max(1, len(np.unique(array)))))
    return {
        "kind": "numeric",
        "min": _clean_number(array.min()),
        "max": _clean_number(array.max()),
        "mean": _clean_number(array.mean()),
        "median": _clean_number(np.median(array)),
        "std": _clean_number(array.std(ddof=1)) if array.size > 1 else 0,
        "sum": _clean_number(array.sum()),
        "zero_count": int((array == 0).sum()),
        "negative_count": int((array < 0).sum()),
        "histogram": {
            "counts": [int(c) for c in counts],
            "edges": [_clean_number(e) for e in edges],
        },
    }


def categorical_distribution(series: pd.Series, top_k: int) -> dict[str, Any]:
    values = series.dropna()
    if values.empty:
        return {"kind": "categorical", "empty": True}
    counts = values.astype(str).value_counts()
    total = int(counts.sum())
    top = counts.head(top_k)
    return {
        "kind": "categorical",
        "distinct": int(counts.size),
        "top_values": [
            {"value": str(index), "count": int(count), "share": round(count / total, 4)}
            for index, count in top.items()
        ],
        "covered_share": round(float(top.sum()) / total, 4),
        "longest_value": int(values.astype(str).str.len().max()),
    }


def temporal_distribution(series: pd.Series) -> dict[str, Any]:
    values = pd.to_datetime(series.dropna(), errors="coerce").dropna()
    if values.empty:
        return {"kind": "temporal", "empty": True}
    by_month = values.dt.to_period("M").value_counts().sort_index()
    return {
        "kind": "temporal",
        "min": values.min().isoformat(),
        "max": values.max().isoformat(),
        "span_days": int((values.max() - values.min()).days),
        "by_month": [
            {"period": str(period), "count": int(count)} for period, count in by_month.items()
        ][:24],
    }


def boolean_distribution(series: pd.Series) -> dict[str, Any]:
    values = series.dropna().astype(str).str.strip().str.lower()
    if values.empty:
        return {"kind": "boolean", "empty": True}
    counts = values.value_counts()
    total = int(counts.sum())
    return {
        "kind": "boolean",
        "values": [
            {"value": str(index), "count": int(count), "share": round(count / total, 4)}
            for index, count in counts.items()
        ],
    }


def profile_series(series: pd.Series, column_type: ColumnType) -> dict[str, Any]:
    """Distribution appropriate to the detected type."""

    settings = get_settings()
    if column_type == ColumnType.BOOLEAN:
        return boolean_distribution(series)
    if column_type == ColumnType.DATE:
        return temporal_distribution(series)
    if column_type.is_numeric:
        return numeric_distribution(series)
    return categorical_distribution(series, settings.profile_top_k)


def sample_values(series: pd.Series, limit: int) -> list[Any]:
    """Distinct-first sampling, so the samples are actually informative."""

    values = series.dropna()
    if values.empty:
        return []
    distinct = values.drop_duplicates().head(limit)
    return [_json_safe(v) for v in distinct]


def basic_stats(series: pd.Series) -> dict[str, Any]:
    total = int(len(series))
    non_null = int(series.notna().sum())
    unique = int(series.nunique(dropna=True))
    return {
        "row_count": total,
        "non_null_count": non_null,
        "null_count": total - non_null,
        "null_ratio": round(1 - (non_null / total), 4) if total else 0.0,
        "unique_count": unique,
        "unique_ratio": round(unique / non_null, 4) if non_null else 0.0,
    }
