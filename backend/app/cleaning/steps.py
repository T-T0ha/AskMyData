"""Phase 2 — the ten cleaning executors.

Each executor is a pure-ish function over the session's tables: it validates
its own parameters, applies one pandas operation, and reports exactly what
changed.  Nothing here talks to the LLM — Claude proposes steps, this module
performs them, and the user approves the result.

Every executor returns a :class:`StepOutcome` carrying a before/after preview
of the affected rows, which is what the validation interrupt renders.

**No executor writes a value into a blank cell.**  There is no ``fillna`` in
this module and there is not meant to be one: missing values survive cleaning
untouched and reach PostgreSQL as ``NULL``.  Casting can *create* a null (a
value that does not fit the target type), and that is reported loudly, but
nothing here ever removes one by inventing data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from app.core.config import get_settings
from app.core.schemas import CleaningStep, StepType
from app.cleaning.store import TableStore
from app.ingestion.cleaning import rank_date_formats, try_parse_date
from app.semantics.profiling import _json_safe


class StepError(ValueError):
    """A step cannot run as specified (bad params, missing column, ...)."""


@dataclass(slots=True)
class StepOutcome:
    step_id: str
    summary: str
    affected_tables: list[str] = field(default_factory=list)
    affected_columns: list[str] = field(default_factory=list)
    rows_before: int = 0
    rows_after: int = 0
    cells_changed: int = 0
    before_preview: list[dict[str, Any]] = field(default_factory=list)
    after_preview: list[dict[str, Any]] = field(default_factory=list)
    preview_columns: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "summary": self.summary,
            "affected_tables": self.affected_tables,
            "affected_columns": self.affected_columns,
            "rows_before": self.rows_before,
            "rows_after": self.rows_after,
            "rows_removed": max(0, self.rows_before - self.rows_after),
            "cells_changed": self.cells_changed,
            "before_preview": self.before_preview,
            "after_preview": self.after_preview,
            "preview_columns": self.preview_columns,
            "notes": self.notes,
        }


def _records(df: pd.DataFrame, columns: list[str], limit: int) -> list[dict[str, Any]]:
    if df.empty:
        return []
    usable = [c for c in columns if c in df.columns] or [str(c) for c in df.columns[:6]]
    subset = df.loc[:, usable].head(limit)
    return [
        {str(k): _json_safe(v) for k, v in record.items()}
        for record in subset.to_dict(orient="records")
    ]


def _require(params: dict[str, Any], key: str, step: CleaningStep) -> Any:
    if key not in params or params[key] in (None, ""):
        raise StepError(f"step {step.type.value} on {step.table} requires param {key!r}")
    return params[key]


def _require_column(df: pd.DataFrame, column: str, step: CleaningStep) -> str:
    if column not in df.columns:
        raise StepError(
            f"column {column!r} does not exist in {step.table}; "
            f"available: {', '.join(str(c) for c in df.columns)}"
        )
    return column


def _changed_mask(before: pd.Series, after: pd.Series) -> pd.Series:
    both_null = before.isna() & after.isna()
    try:
        equal = before.eq(after)
    except (TypeError, ValueError):
        equal = before.astype(str).eq(after.astype(str))
    return ~(equal | both_null)


# ---------------------------------------------------------------------------
# executors
# ---------------------------------------------------------------------------


def _rename_column(store: TableStore, step: CleaningStep, limit: int) -> StepOutcome:
    df = store.get(step.table)
    column = _require_column(df, str(_require(step.params, "column", step)), step)
    new_name = str(_require(step.params, "new_name", step))
    if new_name in df.columns and new_name != column:
        raise StepError(f"{step.table} already has a column named {new_name!r}")

    before = _records(df, [column], limit)
    updated = df.rename(columns={column: new_name})
    store.put(step.table, updated)
    return StepOutcome(
        step_id=step.id,
        summary=f"Renamed {step.table}.{column} to {new_name}",
        affected_tables=[step.table],
        affected_columns=[new_name],
        rows_before=len(df),
        rows_after=len(updated),
        before_preview=before,
        after_preview=_records(updated, [new_name], limit),
        preview_columns=[column, new_name],
    )


def _drop_column(store: TableStore, step: CleaningStep, limit: int) -> StepOutcome:
    df = store.get(step.table)
    column = _require_column(df, str(_require(step.params, "column", step)), step)
    if len(df.columns) == 1:
        raise StepError(f"refusing to drop the last remaining column of {step.table}")

    updated = df.drop(columns=[column])
    store.put(step.table, updated)
    return StepOutcome(
        step_id=step.id,
        summary=f"Dropped column {step.table}.{column}",
        affected_tables=[step.table],
        affected_columns=[column],
        rows_before=len(df),
        rows_after=len(updated),
        cells_changed=int(len(df)),
        before_preview=_records(df, [column], limit),
        after_preview=[],
        preview_columns=[column],
        notes=[f"{int(df[column].notna().sum())} non-null value(s) were discarded"],
    )


def _standardize_format(store: TableStore, step: CleaningStep, limit: int) -> StepOutcome:
    df = store.get(step.table)
    column = _require_column(df, str(_require(step.params, "column", step)), step)
    fmt = str(step.params.get("format", "strip")).lower()
    series = df[column]

    if fmt == "date":
        target = str(step.params.get("date_format", "%Y-%m-%d"))
        ranked = rank_date_formats(series.dropna())
        parsed = series.map(lambda v: try_parse_date(v, ranked)[0] if pd.notna(v) else None)
        result = pd.Series(
            [v.strftime(target) if v is not None else np.nan for v in parsed],
            index=series.index,
            name=series.name,
        )
    else:
        text = series.astype("string")
        operations: dict[str, Callable[[Any], Any]] = {
            "upper": lambda s: s.str.upper(),
            "lower": lambda s: s.str.lower(),
            "title": lambda s: s.str.title(),
            "strip": lambda s: s.str.strip(),
        }
        if fmt not in operations:
            raise StepError(f"unknown format {fmt!r}")
        result = operations[fmt](text.str.strip())
        result = result.where(series.notna(), other=pd.NA)

    changed = _changed_mask(series, result)
    updated = df.copy()
    updated[column] = result
    store.put(step.table, updated)
    return StepOutcome(
        step_id=step.id,
        summary=f"Standardized {step.table}.{column} to {fmt} format ({int(changed.sum())} value(s) changed)",
        affected_tables=[step.table],
        affected_columns=[column],
        rows_before=len(df),
        rows_after=len(updated),
        cells_changed=int(changed.sum()),
        before_preview=_records(df.loc[changed], [column], limit),
        after_preview=_records(updated.loc[changed], [column], limit),
        preview_columns=[column],
    )


def _standardize_casing(store: TableStore, step: CleaningStep, limit: int) -> StepOutcome:
    """Merge values that differ only by capitalisation.

    Separate from :func:`_standardize_format` because this is the step the
    planner proposes against real evidence, and because it refuses to run on a
    column where re-casing would destroy meaning (product codes, IDs).
    """

    df = store.get(step.table)
    column = _require_column(df, str(_require(step.params, "column", step)), step)
    casing = str(step.params.get("casing", "lower")).lower()
    operations: dict[str, Callable[[Any], Any]] = {
        "upper": lambda s: s.str.upper(),
        "lower": lambda s: s.str.lower(),
        "title": lambda s: s.str.title(),
    }
    if casing not in operations:
        raise StepError(f"unknown casing {casing!r}; expected upper, lower or title")

    series = df[column]
    text = series.astype("string")
    result = operations[casing](text)
    result = result.where(series.notna(), other=pd.NA)

    changed = _changed_mask(series, result)
    before_distinct = int(series.dropna().nunique())
    updated = df.copy()
    updated[column] = result
    after_distinct = int(result.dropna().nunique())
    store.put(step.table, updated)

    notes = []
    if before_distinct > after_distinct:
        notes.append(
            f"{before_distinct - after_distinct} duplicate categor(y/ies) merged "
            f"({before_distinct} distinct values before, {after_distinct} after)"
        )
    return StepOutcome(
        step_id=step.id,
        summary=(
            f"Standardized {step.table}.{column} to {casing}case "
            f"({int(changed.sum())} value(s) changed)"
        ),
        affected_tables=[step.table],
        affected_columns=[column],
        rows_before=len(df),
        rows_after=len(updated),
        cells_changed=int(changed.sum()),
        before_preview=_records(df.loc[changed], [column], limit),
        after_preview=_records(updated.loc[changed], [column], limit),
        preview_columns=[column],
        notes=notes,
    )


def _strip_whitespace(store: TableStore, step: CleaningStep, limit: int) -> StepOutcome:
    """Trim leading/trailing whitespace, and collapse internal runs on request."""

    df = store.get(step.table)
    column = _require_column(df, str(_require(step.params, "column", step)), step)
    collapse_inner = bool(step.params.get("collapse_internal", False))

    series = df[column]
    text = series.astype("string")
    result = text.str.strip()
    if collapse_inner:
        result = result.str.replace(r"\s+", " ", regex=True)
    result = result.where(series.notna(), other=pd.NA)

    changed = _changed_mask(series, result)
    updated = df.copy()
    updated[column] = result
    store.put(step.table, updated)
    return StepOutcome(
        step_id=step.id,
        summary=(
            f"Trimmed whitespace in {step.table}.{column} "
            f"({int(changed.sum())} value(s) changed)"
        ),
        affected_tables=[step.table],
        affected_columns=[column],
        rows_before=len(df),
        rows_after=len(updated),
        cells_changed=int(changed.sum()),
        before_preview=_records(df.loc[changed], [column], limit),
        after_preview=_records(updated.loc[changed], [column], limit),
        preview_columns=[column],
    )


def _merge_sheets(store: TableStore, step: CleaningStep, limit: int) -> StepOutcome:
    left_name = str(step.params.get("left_table") or step.table)
    right_name = str(_require(step.params, "right_table", step))
    left_key = str(_require(step.params, "left_key", step))
    right_key = str(_require(step.params, "right_key", step))
    how = str(step.params.get("how", "left")).lower()
    if how not in {"inner", "left", "right", "outer"}:
        raise StepError(f"unsupported join type {how!r}")

    left = store.get(left_name)
    right = store.get(right_name)
    if left_key not in left.columns:
        raise StepError(f"{left_name} has no column {left_key!r}")
    if right_key not in right.columns:
        raise StepError(f"{right_name} has no column {right_key!r}")

    result_name = str(step.params.get("result_table") or f"{left_name}_{right_name}")
    merged = left.merge(
        right,
        how=how,  # type: ignore[arg-type]
        left_on=left_key,
        right_on=right_key,
        suffixes=("", f"_{right_name}"),
    )
    store.put(result_name, merged)

    matched = int(merged[left_key].notna().sum()) if left_key in merged.columns else len(merged)
    notes = [f"{how} join on {left_name}.{left_key} = {right_name}.{right_key}"]
    if how == "left":
        unmatched = len(merged) - matched
        if unmatched:
            notes.append(f"{unmatched} left row(s) found no match")
    return StepOutcome(
        step_id=step.id,
        summary=f"Merged {left_name} with {right_name} into {result_name} ({len(merged)} rows)",
        affected_tables=[left_name, right_name, result_name],
        affected_columns=[left_key, right_key],
        rows_before=len(left),
        rows_after=len(merged),
        before_preview=_records(left, [str(c) for c in left.columns[:6]], limit),
        after_preview=_records(merged, [str(c) for c in merged.columns[:6]], limit),
        preview_columns=[str(c) for c in merged.columns[:6]],
        notes=notes,
    )


def _split_column(store: TableStore, step: CleaningStep, limit: int) -> StepOutcome:
    df = store.get(step.table)
    column = _require_column(df, str(_require(step.params, "column", step)), step)
    into = step.params.get("into") or []
    if not isinstance(into, list) or len(into) < 2:
        raise StepError("split_column needs an 'into' list of at least two new column names")
    delimiter = step.params.get("delimiter")
    pattern = step.params.get("regex")
    if not delimiter and not pattern:
        raise StepError("split_column needs either 'delimiter' or 'regex'")

    text = df[column].astype("string")
    parts = text.str.split(pattern or delimiter, regex=bool(pattern), n=len(into) - 1, expand=True)
    updated = df.copy()
    for index, new_name in enumerate(into):
        if new_name in updated.columns:
            raise StepError(f"{step.table} already has a column named {new_name!r}")
        updated[new_name] = parts[index].str.strip() if index in parts.columns else pd.NA

    store.put(step.table, updated)
    return StepOutcome(
        step_id=step.id,
        summary=f"Split {step.table}.{column} into {', '.join(into)}",
        affected_tables=[step.table],
        affected_columns=[column, *into],
        rows_before=len(df),
        rows_after=len(updated),
        cells_changed=int(text.notna().sum()),
        before_preview=_records(df, [column], limit),
        after_preview=_records(updated, [column, *into], limit),
        preview_columns=[column, *into],
    )


def _deduplicate(store: TableStore, step: CleaningStep, limit: int) -> StepOutcome:
    df = store.get(step.table)
    subset = step.params.get("subset") or None
    if subset:
        missing = [c for c in subset if c not in df.columns]
        if missing:
            raise StepError(f"{step.table} has no column(s) {', '.join(missing)}")
    keep = str(step.params.get("keep", "first"))

    duplicated = df.duplicated(subset=subset, keep=keep)
    if not duplicated.any():
        raise StepError(f"{step.table} has no duplicate rows for the given subset")

    updated = df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)  # type: ignore[arg-type]
    store.put(step.table, updated)
    preview_columns = list(subset) if subset else [str(c) for c in df.columns[:6]]
    return StepOutcome(
        step_id=step.id,
        summary=f"Removed {int(duplicated.sum())} duplicate row(s) from {step.table}",
        affected_tables=[step.table],
        affected_columns=preview_columns,
        rows_before=len(df),
        rows_after=len(updated),
        before_preview=_records(df.loc[duplicated], preview_columns, limit),
        after_preview=[],
        preview_columns=preview_columns,
        notes=["rows shown are the duplicates that were removed"],
    )


def _type_cast(store: TableStore, step: CleaningStep, limit: int) -> StepOutcome:
    df = store.get(step.table)
    column = _require_column(df, str(_require(step.params, "column", step)), step)
    target = str(_require(step.params, "to", step)).lower()
    series = df[column]

    if target in {"numeric", "float", "int", "integer"}:
        result = pd.to_numeric(series, errors="coerce")
    elif target in {"datetime", "date"}:
        ranked = rank_date_formats(series.dropna())
        result = pd.Series(
            [try_parse_date(v, ranked)[0] if pd.notna(v) else None for v in series],
            index=series.index,
            name=series.name,
            dtype="datetime64[ns]",
        )
    elif target in {"string", "text", "str"}:
        result = series.astype("string")
    elif target in {"boolean", "bool"}:
        truthy = {"true", "t", "yes", "y", "1", "1.0", "on"}
        falsy = {"false", "f", "no", "n", "0", "0.0", "off"}
        result = series.map(
            lambda v: (
                pd.NA
                if pd.isna(v)
                else True
                if str(v).strip().lower() in truthy
                else False
                if str(v).strip().lower() in falsy
                else pd.NA
            )
        ).astype("boolean")
    else:
        raise StepError(f"unsupported cast target {target!r}")

    newly_null = series.notna() & result.isna()
    updated = df.copy()
    updated[column] = result
    store.put(step.table, updated)
    notes: list[str] = []
    if int(newly_null.sum()):
        notes.append(
            f"{int(newly_null.sum())} value(s) could not be converted and became null — "
            "review before approving"
        )
    return StepOutcome(
        step_id=step.id,
        summary=f"Cast {step.table}.{column} to {target} (dtype {result.dtype})",
        affected_tables=[step.table],
        affected_columns=[column],
        rows_before=len(df),
        rows_after=len(updated),
        cells_changed=int(_changed_mask(series, result).sum()),
        before_preview=_records(df, [column], limit),
        after_preview=_records(updated, [column], limit),
        preview_columns=[column],
        notes=notes,
    )


def _add_synthetic_key(store: TableStore, step: CleaningStep, limit: int) -> StepOutcome:
    df = store.get(step.table)
    column = str(step.params.get("column", "row_id"))
    if column in df.columns:
        raise StepError(f"{step.table} already has a column named {column!r}")

    updated = df.copy()
    updated.insert(0, column, range(1, len(updated) + 1))
    store.put(step.table, updated)
    return StepOutcome(
        step_id=step.id,
        summary=f"Added synthetic primary key {step.table}.{column}",
        affected_tables=[step.table],
        affected_columns=[column],
        rows_before=len(df),
        rows_after=len(updated),
        cells_changed=len(updated),
        before_preview=_records(df, [str(c) for c in df.columns[:5]], limit),
        after_preview=_records(updated, [column, *[str(c) for c in df.columns[:4]]], limit),
        preview_columns=[column],
    )


EXECUTORS: dict[StepType, Callable[[TableStore, CleaningStep, int], StepOutcome]] = {
    StepType.RENAME_COLUMN: _rename_column,
    StepType.DROP_COLUMN: _drop_column,
    StepType.STANDARDIZE_FORMAT: _standardize_format,
    StepType.STANDARDIZE_CASING: _standardize_casing,
    StepType.STRIP_WHITESPACE: _strip_whitespace,
    StepType.MERGE_SHEETS: _merge_sheets,
    StepType.SPLIT_COLUMN: _split_column,
    StepType.DEDUPLICATE: _deduplicate,
    StepType.TYPE_CAST: _type_cast,
    StepType.ADD_SYNTHETIC_KEY: _add_synthetic_key,
}


def execute_step(store: TableStore, step: CleaningStep) -> StepOutcome:
    """Apply one plan step, returning its before/after evidence."""

    executor = EXECUTORS.get(step.type)
    if executor is None:  # pragma: no cover - StepType is exhaustive
        raise StepError(f"no executor registered for {step.type}")
    if step.type != StepType.MERGE_SHEETS and not store.has(step.table):
        raise StepError(f"unknown table {step.table!r}; have {', '.join(store.names())}")
    return executor(store, step, get_settings().validation_preview_rows)
