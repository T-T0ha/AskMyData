"""Phase 0 — file loading.

``load_file_source`` is the single entry point for turning a file on disk into
clean ``DataFrame``s plus a full account of what had to be repaired.  Four
shapes of file arrive here and each takes the shortest correct path:

===================  =================================================
``.xlsx`` ``.xlsm``  openpyxl, so merged and multi-row headers survive
``.csv`` ``.tsv``    pandas, then the same cleaning pass
``.db`` ``.sqlite``  reflected as a database (:mod:`app.ingestion.relational`)
``.sql``             replayed into SQLite first (:mod:`app.ingestion.sqldump`)
===================  =================================================

A live PostgreSQL/MySQL connection is the fifth source and has no file, so it
enters through :func:`app.ingestion.relational.load_relational_tables` directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from openpyxl import load_workbook

from app.core.schemas import SheetIssue, TriageStatus
from app.ingestion.cleaning import ColumnCleanReport, clean_dataframe
from app.ingestion.headers import (
    HeaderAnalysis,
    build_merged_value_map,
    describe_span,
    detect_header_block,
    sanitize_name,
)
from app.ingestion.keys import KeyAnalysis, discover_keys
from app.ingestion.relational import SQLITE_SUFFIXES, load_relational_tables, sqlite_url
from app.ingestion.sqldump import load_sql_dump
from app.ingestion.triage import triage_sheet

EXCEL_SUFFIXES = {".xlsx", ".xlsm", ".xltx", ".xltm"}
CSV_SUFFIXES = {".csv", ".tsv", ".txt"}
SQL_DUMP_SUFFIXES = {".sql"}

#: Everything the upload endpoint accepts.
FILE_SUFFIXES = EXCEL_SUFFIXES | CSV_SUFFIXES | SQLITE_SUFFIXES | SQL_DUMP_SUFFIXES


@dataclass(slots=True)
class SheetLoadResult:
    """One table, repaired, with the provenance of every repair."""

    name: str
    source_name: str
    dataframe: pd.DataFrame
    header: HeaderAnalysis
    clean_reports: list[ColumnCleanReport]
    triage: TriageStatus
    issues: list[SheetIssue]
    row_count: int
    column_count: int
    span: str
    skipped: bool = False
    skip_reason: str | None = None
    #: Which kind of source this came from: excel, csv, sqlite, postgresql,
    #: mysql or sql_dump.  Downstream phases treat a declared schema as
    #: stronger evidence than a detected one, so they have to know.
    source_kind: str = "excel"
    #: The source's own schema (declared types, primary and foreign keys) when
    #: it had one.  Empty for spreadsheets, where there is nothing to declare.
    native_schema: dict[str, Any] = field(default_factory=dict)
    #: What identifies a row here — declared, detected, or nothing yet.
    key_analysis: KeyAnalysis = field(default_factory=KeyAnalysis)
    notes: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source_name": self.source_name,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "span": self.span,
            "triage": self.triage.value,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
            "source_kind": self.source_kind,
            "native_schema": self.native_schema,
            "key_analysis": self.key_analysis.to_dict(),
            "header": {
                "rows": self.header.header_rows,
                "data_start_row": self.header.data_start_row,
                "is_multi_row": self.header.is_multi_row,
                "merged_ranges": self.header.merged_ranges,
                "banner_rows": self.header.banner_rows,
                "confidence": self.header.confidence,
                "columns": self.header.columns,
                "original_labels": self.header.original_labels,
                "notes": self.header.notes,
            },
            "clean_reports": [r.to_dict() for r in self.clean_reports],
            "issues": [i.to_dict() for i in self.issues],
            "notes": self.notes,
        }


def _drop_empty_edges(df: pd.DataFrame) -> pd.DataFrame:
    """Remove all-empty trailing rows and all-empty columns."""

    if df.empty:
        return df
    df = df.dropna(axis=0, how="all")
    keep = [c for c in df.columns if not df[c].isna().all()]
    if keep:
        df = df[keep]
    return df.reset_index(drop=True)


def _load_excel_sheet(path: Path, sheet_name: str) -> SheetLoadResult:
    workbook = load_workbook(path, data_only=True, read_only=False)
    sheet = workbook[sheet_name]
    merged = build_merged_value_map(sheet)
    header = detect_header_block(sheet, merged)
    span = describe_span(sheet)

    if not header.columns:
        workbook.close()
        empty = pd.DataFrame()
        return SheetLoadResult(
            name=sanitize_name(sheet_name) or "sheet",
            source_name=sheet_name,
            dataframe=empty,
            header=header,
            clean_reports=[],
            triage=TriageStatus.STRUCTURAL_ISSUES,
            issues=[
                SheetIssue(
                    code="empty_sheet",
                    severity="error",
                    message="Sheet has no readable header or data.",
                )
            ],
            row_count=0,
            column_count=0,
            span=span,
            skipped=True,
            skip_reason="empty sheet",
        )

    width = len(header.columns)
    records: list[list[Any]] = []
    for row in sheet.iter_rows(
        min_row=header.data_start_row, max_col=width, values_only=True
    ):
        if row is None:
            continue
        values = list(row) + [None] * (width - len(row))
        if all(v is None or (isinstance(v, str) and not v.strip()) for v in values):
            continue
        records.append(values[:width])
    workbook.close()

    raw = pd.DataFrame(records, columns=header.columns)
    raw = _drop_empty_edges(raw)
    cleaned, reports = clean_dataframe(raw)
    keys = discover_keys(cleaned)
    triage, issues = triage_sheet(cleaned, header, reports, keys)

    return SheetLoadResult(
        name=sanitize_name(sheet_name) or "sheet",
        source_name=sheet_name,
        dataframe=cleaned,
        header=header,
        clean_reports=reports,
        triage=triage,
        issues=issues,
        row_count=int(len(cleaned)),
        column_count=int(len(cleaned.columns)),
        span=span,
        key_analysis=keys,
    )


def _load_csv(path: Path) -> SheetLoadResult:
    separator = "\t" if path.suffix.lower() == ".tsv" else ","
    raw = pd.read_csv(path, sep=separator, dtype=object, keep_default_na=False)
    original = [str(c) for c in raw.columns]
    raw.columns = [
        sanitize_name(c) or f"col_{i + 1}" for i, c in enumerate(original)
    ]
    seen: dict[str, int] = {}
    unique: list[str] = []
    for column in raw.columns:
        if column in seen:
            seen[column] += 1
            unique.append(f"{column}_{seen[column]}")
        else:
            seen[column] = 1
            unique.append(column)
    raw.columns = unique
    raw = _drop_empty_edges(raw)
    cleaned, reports = clean_dataframe(raw)
    header = HeaderAnalysis(
        header_rows=[1],
        data_start_row=2,
        columns=list(cleaned.columns),
        original_labels=[[o] for o in original],
        confidence=1.0,
    )
    keys = discover_keys(cleaned)
    triage, issues = triage_sheet(cleaned, header, reports, keys)
    return SheetLoadResult(
        name=sanitize_name(path.stem) or "table",
        source_name=path.name,
        dataframe=cleaned,
        header=header,
        clean_reports=reports,
        triage=triage,
        issues=issues,
        row_count=int(len(cleaned)),
        column_count=int(len(cleaned.columns)),
        span=f"A1:{len(cleaned.columns)}x{len(cleaned)}",
        source_kind="csv",
        key_analysis=keys,
    )


def load_file_source(path: str | Path) -> list[SheetLoadResult]:
    """Load any supported file — workbook, CSV, SQLite database or SQL dump."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    if suffix in CSV_SUFFIXES:
        return [_load_csv(path)]
    if suffix in SQLITE_SUFFIXES:
        return load_relational_tables(sqlite_url(path), kind="sqlite")
    if suffix in SQL_DUMP_SUFFIXES:
        return load_sql_dump(path)
    if suffix not in EXCEL_SUFFIXES:
        raise ValueError(
            f"Unsupported file type '{suffix}'. Supported: " + ", ".join(sorted(FILE_SUFFIXES))
        )
    return _load_excel_workbook(path)


def load_workbook_sheets(path: str | Path) -> list[SheetLoadResult]:
    """Backwards-compatible alias for :func:`load_file_source`."""

    return load_file_source(path)


def _load_excel_workbook(path: Path) -> list[SheetLoadResult]:
    """Load every sheet of one workbook through Phase 0."""

    probe = load_workbook(path, data_only=True, read_only=True)
    sheet_names = list(probe.sheetnames)
    probe.close()

    results: list[SheetLoadResult] = []
    used: set[str] = set()
    for sheet_name in sheet_names:
        result = _load_excel_sheet(path, sheet_name)
        # Sheet names are unique inside a workbook but sanitising can collide.
        base = result.name
        counter = 2
        while result.name in used:
            result.name = f"{base}_{counter}"
            counter += 1
        used.add(result.name)
        results.append(result)
    return results
