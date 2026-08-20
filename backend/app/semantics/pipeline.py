"""Phase 1 orchestration — the Field Semantic View's data source.

Runs type detection, taxonomy classification and distribution profiling over
every column of every loaded sheet, and produces the compact statistical
summary that Phase 2 hands to Claude.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import pandas as pd

from app.core.config import get_settings
from app.core.schemas import ColumnProfile, ColumnType
from app.ingestion.cleaning import ColumnCleanReport
from app.semantics.column_types import detect_column_type
from app.semantics.profiling import basic_stats, profile_series, sample_values
from app.semantics.taxonomy import classify_column


@dataclass(slots=True)
class TableSemantics:
    table: str
    columns: list[ColumnProfile] = field(default_factory=list)
    row_count: int = 0

    def column(self, name: str) -> ColumnProfile | None:
        return next((c for c in self.columns if c.name == name), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "table": self.table,
            "row_count": self.row_count,
            "columns": [c.to_dict() for c in self.columns],
        }


def _currency_symbols(reports: Iterable[ColumnCleanReport] | None) -> dict[str, str]:
    if not reports:
        return {}
    return {
        r.column: r.detected_currency_symbol
        for r in reports
        if r.detected_currency_symbol
    }


def analyze_table(
    table: str,
    df: pd.DataFrame,
    clean_reports: Iterable[ColumnCleanReport] | None = None,
    claude=None,
    original_names: Mapping[str, str] | None = None,
) -> TableSemantics:
    """Full Phase 1 pass over one sheet."""

    settings = get_settings()
    symbols = _currency_symbols(clean_reports)
    originals = dict(original_names or {})
    semantics = TableSemantics(table=table, row_count=int(len(df)))

    for raw_column in df.columns:
        column = str(raw_column)
        series = df[raw_column]
        stats = basic_stats(series)

        type_decision = detect_column_type(series, column, symbols.get(column))
        taxonomy = classify_column(
            name=column,
            table=table,
            column_type=type_decision.column_type,
            series=series,
            claude=claude,
            sample_limit=settings.taxonomy_sample_values,
        )

        semantics.columns.append(
            ColumnProfile(
                table=table,
                name=column,
                original_name=originals.get(column, column),
                column_type=type_decision.column_type,
                type_confidence=type_decision.confidence,
                taxonomy_label=taxonomy.label,
                taxonomy_confidence=taxonomy.confidence,
                taxonomy_source=taxonomy.source,
                taxonomy_rule=taxonomy.rule,
                nullable=stats["null_count"] > 0,
                null_ratio=stats["null_ratio"],
                unique_count=stats["unique_count"],
                unique_ratio=stats["unique_ratio"],
                row_count=stats["row_count"],
                sample_values=sample_values(series, settings.plan_sample_values),
                distribution=profile_series(series, type_decision.column_type),
                type_evidence=type_decision.evidence,
                is_additive=taxonomy.is_additive,
            )
        )
    return semantics


def analyze_tables(
    tables: Mapping[str, pd.DataFrame],
    clean_reports: Mapping[str, Iterable[ColumnCleanReport]] | None = None,
    claude=None,
) -> dict[str, TableSemantics]:
    reports = clean_reports or {}
    return {
        name: analyze_table(name, df, reports.get(name), claude=claude)
        for name, df in tables.items()
    }


def statistical_summary(
    semantics: Mapping[str, TableSemantics],
    equivalences: list[dict[str, Any]] | None = None,
    duplicates: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """The compact, row-free payload sent to Claude in Phase 2.

    Contains column names, detected types, taxonomy labels, null ratios, five
    sample values per column and the confirmed cross-sheet equivalences —
    never a data row.
    """

    settings = get_settings()
    duplicate_counts = duplicates or {}
    tables = []
    for name, table in semantics.items():
        columns = []
        for column in table.columns:
            entry: dict[str, Any] = {
                "name": column.name,
                "type": column.column_type.value,
                "taxonomy": column.taxonomy_label,
                "null_ratio": column.null_ratio,
                "unique_ratio": column.unique_ratio,
                "samples": column.sample_values[: settings.plan_sample_values],
            }
            distribution = column.distribution or {}
            if distribution.get("kind") == "numeric" and not distribution.get("empty"):
                entry["range"] = [distribution.get("min"), distribution.get("max")]
            if distribution.get("kind") == "categorical" and not distribution.get("empty"):
                entry["distinct"] = distribution.get("distinct")
            columns.append(entry)
        tables.append(
            {
                "name": name,
                "row_count": table.row_count,
                "duplicate_row_count": duplicate_counts.get(name, 0),
                "columns": columns,
            }
        )

    return {
        "tables": tables,
        "confirmed_equivalences": [
            {
                "left": f"{e['left_table']}.{e['left_column']}",
                "right": f"{e['right_table']}.{e['right_column']}",
                "score": e["score"],
            }
            for e in (equivalences or [])
            if e.get("confirmed")
        ],
        "candidate_equivalences": [
            {
                "left": f"{e['left_table']}.{e['left_column']}",
                "right": f"{e['right_table']}.{e['right_column']}",
                "score": e["score"],
            }
            for e in (equivalences or [])
            if not e.get("confirmed")
        ],
    }


def unknown_columns(semantics: Mapping[str, TableSemantics]) -> list[dict[str, str]]:
    """Columns the UI must highlight for manual labelling."""

    return [
        {"table": table.table, "column": column.name}
        for table in semantics.values()
        for column in table.columns
        if column.taxonomy_label == "unknown"
    ]


def type_counts(semantics: Mapping[str, TableSemantics]) -> dict[str, int]:
    counts: dict[str, int] = {t.value: 0 for t in ColumnType}
    for table in semantics.values():
        for column in table.columns:
            counts[column.column_type.value] += 1
    return counts
