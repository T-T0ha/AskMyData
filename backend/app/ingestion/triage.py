"""Phase 0 — data quality triage.

Every sheet lands in one of four buckets before any semantic work begins, so
the user knows up front which files will just work and which need attention:

``clean``              nothing worth mentioning
``fixable``            issues the cleaning plan can resolve automatically
``needs_attention``    issues that require a human decision — including the
                       proposal's own example, a table with no primary key
``structural_issues``  the sheet itself is broken (no header, no data)
"""

from __future__ import annotations

import pandas as pd

from app.core.schemas import SheetIssue, TriageStatus
from app.ingestion.cleaning import ColumnCleanReport
from app.ingestion.headers import HeaderAnalysis
from app.ingestion.keys import KeyAnalysis

HIGH_NULL_RATIO = 0.50
MODERATE_NULL_RATIO = 0.20
MIXED_TYPE_RATIO = 0.05
#: A text column has to be substantially some *other* type before it is worth
#: reporting.  A handful of stray numbers in a notes column is normal; a fifth
#: of the column parsing as dates is not.
PARTLY_TYPED_RATIO = 0.20
MIN_ROWS_FOR_ANALYSIS = 3

_SEVERITY_RANK = {"info": 0, "warning": 1, "error": 2}


def triage_sheet(
    df: pd.DataFrame,
    header: HeaderAnalysis,
    reports: list[ColumnCleanReport],
    keys: KeyAnalysis | None = None,
) -> tuple[TriageStatus, list[SheetIssue]]:
    """Classify a sheet and list the concrete reasons for the classification."""

    issues: list[SheetIssue] = []

    # ---- structural -----------------------------------------------------
    if df.empty or not len(df.columns):
        issues.append(
            SheetIssue("no_data", "error", "Sheet contains no data rows after parsing.")
        )
        return TriageStatus.STRUCTURAL_ISSUES, issues

    if len(df) < MIN_ROWS_FOR_ANALYSIS:
        issues.append(
            SheetIssue(
                "too_few_rows",
                "error",
                f"Only {len(df)} data row(s) — too small for reliable semantic analysis.",
            )
        )

    if header.confidence < 0.6:
        issues.append(
            SheetIssue(
                "uncertain_header",
                "error",
                "Header row could not be identified confidently — check the column names.",
            )
        )

    auto_named = [c for c in df.columns if str(c).startswith("col_")]
    if auto_named:
        issues.append(
            SheetIssue(
                "unnamed_columns",
                "warning",
                f"{len(auto_named)} column(s) had no header text and were auto-named.",
                [str(c) for c in auto_named],
            )
        )

    if header.is_multi_row:
        issues.append(
            SheetIssue(
                "multi_row_header",
                "info",
                f"Header spanned rows {header.header_rows} and was flattened into single names.",
            )
        )

    if header.banner_rows:
        issues.append(
            SheetIssue(
                "banner_rows_skipped",
                "info",
                f"Skipped {len(header.banner_rows)} title/banner row(s) above the header.",
            )
        )

    # ---- content --------------------------------------------------------
    empty_columns = [str(c) for c in df.columns if df[c].isna().all()]
    if empty_columns:
        issues.append(
            SheetIssue(
                "empty_columns",
                "warning",
                f"{len(empty_columns)} column(s) are completely empty.",
                empty_columns,
            )
        )

    high_null = [
        str(c)
        for c in df.columns
        if str(c) not in empty_columns and df[c].isna().mean() > HIGH_NULL_RATIO
    ]
    if high_null:
        issues.append(
            SheetIssue(
                "high_null_ratio",
                "warning",
                f"{len(high_null)} column(s) are more than {HIGH_NULL_RATIO:.0%} empty — "
                "the cleaning plan will offer to drop them, or you can keep them as they are.",
                high_null,
            )
        )

    moderate_null = [
        str(c)
        for c in df.columns
        if str(c) not in empty_columns
        and str(c) not in high_null
        and df[c].isna().mean() > MODERATE_NULL_RATIO
    ]
    if moderate_null:
        issues.append(
            SheetIssue(
                "missing_values",
                "info",
                f"{len(moderate_null)} column(s) have missing values. They are kept as "
                "blanks (NULL in the database) — nothing is filled in on your behalf.",
                moderate_null,
            )
        )

    duplicate_count = int(df.duplicated().sum())
    if duplicate_count:
        issues.append(
            SheetIssue(
                "duplicate_rows",
                "info",
                f"{duplicate_count} duplicate row(s) can be removed automatically.",
            )
        )

    # ---- keys -----------------------------------------------------------
    # A table nothing can identify a row in cannot be referenced by another
    # table, so this is the proposal's own example of an orange sheet: not
    # broken, but not usable as a relational table until a human decides.
    # Nothing is raised for a table whose key is obvious — a green sheet stays
    # green, and the key it does have is shown on the triage card regardless.
    issues.extend(_key_issues(keys))

    mixed = [r.column for r in reports if r.mixed_type_ratio > MIXED_TYPE_RATIO and not r.retyped_to]
    if mixed:
        issues.append(
            SheetIssue(
                "mixed_types",
                "warning",
                f"{len(mixed)} column(s) mix text and numbers and could not be retyped automatically.",
                mixed,
            )
        )

    # A column that is half dates and half product codes is one python type all
    # the way down, so ``mixed_types`` above never sees it.  It is worth its own
    # warning: the usual cause is a second header — a second table — partway
    # down the file, and the column exports as text until that is dealt with.
    partly = [
        r.column
        for r in reports
        if r.unconverted_type and r.unconverted_ratio > PARTLY_TYPED_RATIO
    ]
    if partly:
        worst = max(
            (r for r in reports if r.column in partly), key=lambda r: r.unconverted_ratio
        )
        issues.append(
            SheetIssue(
                "partly_typed",
                "warning",
                f"{len(partly)} column(s) hold a mix of types — {worst.column} is "
                f"{worst.unconverted_ratio:.0%} {worst.unconverted_type} and the rest is "
                "something else. Often a second table stacked in the same file.",
                partly,
            )
        )

    retyped = [r.column for r in reports if r.retyped_to]
    if retyped:
        issues.append(
            SheetIssue(
                "retyped_columns",
                "info",
                f"{len(retyped)} column(s) were converted to a proper numeric/date type.",
                retyped,
            )
        )

    coerced = [r.column for r in reports if r.coerced_to_null > 0]
    if coerced:
        issues.append(
            SheetIssue(
                "values_lost_in_cast",
                "warning",
                "Some values did not fit the detected type and became null — review before export.",
                coerced,
            )
        )

    duplicate_names = _duplicate_base_names(df)
    if duplicate_names:
        issues.append(
            SheetIssue(
                "duplicate_column_names",
                "warning",
                "Duplicate column names were made unique with a numeric suffix.",
                duplicate_names,
            )
        )

    return classify_issues(issues), issues


def _key_issues(keys: KeyAnalysis | None) -> list[SheetIssue]:
    """Everything the user has to decide about this table's identity."""

    if keys is None:
        return []

    issues: list[SheetIssue] = []

    if keys.declared_key_holds is False:
        issues.append(
            SheetIssue(
                "declared_key_broken",
                "warning",
                "The primary key this database declares does not hold over the rows that "
                "were read — check the note below before relying on it.",
            )
        )

    if keys.needs_confirmation and keys.candidates:
        best = keys.candidates[0]
        alternatives = len(keys.candidates) - 1
        message = (
            f"No single column identifies a row, but {best.label} together do. "
            "Confirm that this is the table's key, or a numbered row_id will be added instead."
        )
        if alternatives > 0:
            message += f" {alternatives} other combination(s) would also work."
        issues.append(
            SheetIssue("composite_key_candidate", "warning", message, list(best.columns))
        )
    elif keys.needs_synthetic_key:
        issues.append(
            SheetIssue(
                "no_primary_key",
                "warning",
                "Nothing in this table identifies a row uniquely, so no other table can "
                "reference it. The cleaning plan will offer to add a numbered row_id.",
            )
        )

    return issues


def _duplicate_base_names(df: pd.DataFrame) -> list[str]:
    suffixed = [str(c) for c in df.columns if str(c).rsplit("_", 1)[-1].isdigit()]
    bases = {str(c).rsplit("_", 1)[0] for c in suffixed}
    return sorted(c for c in suffixed if c.rsplit("_", 1)[0] in bases)


def classify_issues(issues: list[SheetIssue] | list[dict[str, str]]) -> TriageStatus:
    """Bucket a sheet from its issues.

    Public because a sheet can be re-classified after ingestion: confirming a
    composite key resolves the issue that made the sheet orange, and the card
    has to stop being orange.
    """

    if not issues:
        return TriageStatus.CLEAN
    severities = [i.severity if isinstance(i, SheetIssue) else str(i.get("severity", "info")) for i in issues]
    worst = max(_SEVERITY_RANK.get(s, 0) for s in severities)
    if worst == 2:
        return TriageStatus.STRUCTURAL_ISSUES
    if worst == 1:
        return TriageStatus.NEEDS_ATTENTION
    return TriageStatus.FIXABLE
