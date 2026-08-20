"""Phase 2 — cleaning plan generation.

Two producers, one output format:

* :func:`heuristic_plan` derives steps directly from the Phase 0/1 statistics.
  It is deterministic, needs no API key, and is what the platform falls back
  to when Claude is unavailable.
* :func:`normalize_claude_plan` takes Claude's JSON and rejects anything that
  references a table or column that does not exist, or that names an unknown
  step type.  An LLM plan is a *proposal*: it is validated before the user ever
  sees it, and the user validates it again before it runs.

Neither producer may propose filling in a missing value, and the vocabulary
they draw from has no step that could (see :class:`~app.core.schemas.StepType`).
A gap is evidence about the data; closing it with a mean, a median or a mode
replaces that evidence with a number nobody measured, and the exported database
keeps no record of which is which.
"""

from __future__ import annotations

import uuid
from typing import Any, Iterable, Mapping

import pandas as pd

from app.core.schemas import FORBIDDEN_STEP_TYPES, CleaningStep, StepType
from app.ingestion.keys import KeyAnalysis, discover_keys

#: Steps are executed in this order regardless of how they were proposed:
#: structure first, then values, then rows, then joins.
STEP_ORDER: dict[StepType, int] = {
    StepType.DROP_COLUMN: 0,
    StepType.RENAME_COLUMN: 1,
    StepType.SPLIT_COLUMN: 2,
    StepType.TYPE_CAST: 3,
    StepType.STRIP_WHITESPACE: 4,
    StepType.STANDARDIZE_CASING: 5,
    StepType.STANDARDIZE_FORMAT: 6,
    StepType.DEDUPLICATE: 7,
    StepType.ADD_SYNTHETIC_KEY: 8,
    StepType.MERGE_SHEETS: 9,
}

#: A column this empty is a leftover, and the plan offers to delete it.  Note
#: what the planner does *not* offer for anything below this line: nothing.  A
#: gap is reported (triage screen, null-ratio bar in the field semantic view)
#: and then left alone, because the only way to close it is to invent the value
#: that is missing.
HIGH_NULL_RATIO = 0.60


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def sort_plan(steps: list[CleaningStep]) -> list[CleaningStep]:
    return sorted(steps, key=lambda s: (STEP_ORDER.get(s.type, 99), s.table))


# ---------------------------------------------------------------------------
# heuristic planner
# ---------------------------------------------------------------------------


def _case_collisions(series: pd.Series) -> tuple[int, list[str], str]:
    """Find values that differ only by capitalisation.

    Returns the number of collapsing groups, up to two real example spellings
    taken *from this column*, and the casing style that already dominates it.
    The examples matter: a plan step that says "for example 'Shipped' and
    'shipped'" when the column contains neither is not evidence, it is noise,
    and the user cannot validate a claim they cannot see.
    """

    values = series.dropna().astype(str).str.strip()
    if values.empty:
        return 0, [], "lower"

    groups: dict[str, list[str]] = {}
    for value in values.unique():
        groups.setdefault(value.lower(), []).append(value)
    colliding = [variants for variants in groups.values() if len(variants) > 1]
    if not colliding:
        return 0, [], "lower"

    # The two spellings of the single worst group are the clearest evidence.
    example = max(colliding, key=len)
    return len(colliding), sorted(example)[:2], _dominant_casing(values)


def _dominant_casing(values: pd.Series) -> str:
    """Which casing the column already mostly uses.

    Forcing title case on everything is what mangles ``MEN-1234`` into
    ``Men-1234``; the safe target is whatever the column is already mostly
    written in.
    """

    counts = {
        "upper": int((values == values.str.upper()).sum()),
        "lower": int((values == values.str.lower()).sum()),
        "title": int((values == values.str.title()).sum()),
    }
    return max(counts, key=lambda key: counts[key])


def _whitespace_offenders(series: pd.Series) -> int:
    """Values carrying leading or trailing whitespace."""

    values = series.dropna().astype(str)
    if values.empty:
        return 0
    return int((values != values.str.strip()).sum())


def _column_facts(
    semantics: Mapping[str, Any] | None, table: str, column: str
) -> dict[str, Any]:
    """Phase 1's verdict for one column, or an empty verdict when unavailable.

    The planner degrades to type-agnostic behaviour when Phase 1 has not run,
    but it never *ignores* Phase 1 when it has.
    """

    if not semantics:
        return {}
    table_entry = semantics.get(table) or {}
    return dict(table_entry.get(column) or {})


def _rationale(facts: dict[str, Any]) -> str | None:
    """The Phase 1 verdict behind a step, shown on the plan card."""

    column_type = facts.get("column_type")
    label = facts.get("taxonomy_label")
    if not column_type and not label:
        return None
    parts = [p for p in (label, column_type) if p and p != "unknown"]
    return "detected as " + " / ".join(str(p).replace("_", " ") for p in parts) if parts else None


def heuristic_plan(
    tables: Mapping[str, pd.DataFrame],
    semantics: Mapping[str, Any] | None = None,
    equivalences: Iterable[Mapping[str, Any]] = (),
    keys: Mapping[str, Any] | None = None,
) -> list[CleaningStep]:
    """Derive a cleaning plan from statistics alone.

    ``keys`` carries Phase 0's key analysis per table, including any key the
    user confirmed on the triage screen.  Passing it is what stops the plan
    proposing a synthetic ``row_id`` for a table that already has a perfectly
    good composite key.
    """

    steps: list[CleaningStep] = []
    key_analyses = keys or {}

    for name, df in tables.items():
        if df.empty:
            continue

        for raw_column in df.columns:
            column = str(raw_column)
            series = df[raw_column]
            facts = _column_facts(semantics, name, column)
            null_ratio = float(series.isna().mean())

            if null_ratio == 1.0:
                steps.append(
                    CleaningStep(
                        id=_new_id(),
                        type=StepType.DROP_COLUMN,
                        table=name,
                        description=(
                            f"'{column}' is completely empty, so it carries no information "
                            "and would only add noise to the database."
                        ),
                        params={"column": column},
                        origin="heuristic",
                    )
                )
                continue

            if null_ratio > HIGH_NULL_RATIO:
                steps.append(
                    CleaningStep(
                        id=_new_id(),
                        type=StepType.DROP_COLUMN,
                        table=name,
                        description=(
                            f"'{column}' is {null_ratio:.0%} empty — usually a leftover column. "
                            "Delete this step if the few values it does hold matter to you."
                        ),
                        params={"column": column},
                        origin="heuristic",
                    )
                )
                continue

            # A partially-empty column gets no step at all.  Its gaps cannot be
            # closed without inventing the values that are missing, and it must
            # not be dropped either — deleting a 5%-null order_id to "fix" it
            # destroys the key.  The gap is already reported on the triage
            # screen and as a null-ratio bar in the field semantic view; the
            # only remedy the plan can offer is ``add_synthetic_key`` below,
            # which fires precisely because the gappy column no longer
            # identifies a row.

            if series.dtype == object or pd.api.types.is_string_dtype(series):
                stray = _whitespace_offenders(series)
                if stray:
                    steps.append(
                        CleaningStep(
                            id=_new_id(),
                            type=StepType.STRIP_WHITESPACE,
                            table=name,
                            description=(
                                f"{stray:,} value(s) in '{column}' start or end with a space, so "
                                "they look identical on screen but count as different values."
                            ),
                            params={"column": column},
                            origin="heuristic",
                        )
                    )

                collisions, examples, casing = _case_collisions(series)
                if collisions:
                    shown = " and ".join(repr(v) for v in examples)
                    steps.append(
                        CleaningStep(
                            id=_new_id(),
                            type=StepType.STANDARDIZE_CASING,
                            table=name,
                            description=(
                                f"'{column}' writes the same value with different capitalisation "
                                f"(for example {shown}), which splits one category into "
                                f"{collisions + 1 if collisions == 1 else 'several'}. Matching the "
                                f"{casing}case spelling the column already mostly uses merges them."
                            ),
                            params={"column": column, "casing": casing},
                            origin="heuristic",
                            # Phase 1's verdict rides along because case is not
                            # always noise: on a product code or an identifier
                            # 'MEN-1001' and 'men-1001' may be two real values,
                            # and the person approving the step is the only one
                            # who can tell.  Say what the column was detected as
                            # rather than decide for them.
                            rationale=", ".join(
                                part
                                for part in (
                                    f"{collisions} value group(s) differ only by case",
                                    _rationale(facts),
                                )
                                if part
                            ),
                        )
                    )

        duplicate_count = int(df.duplicated().sum())
        if duplicate_count:
            steps.append(
                CleaningStep(
                    id=_new_id(),
                    type=StepType.DEDUPLICATE,
                    table=name,
                    description=(
                        f"{duplicate_count} row(s) in '{name}' are exact copies of another row, "
                        "which would double-count them in every total."
                    ),
                    params={"subset": None},
                    origin="heuristic",
                )
            )

        key_analysis = _key_analysis(key_analyses, name, df)
        if key_analysis.needs_synthetic_key:
            steps.append(
                CleaningStep(
                    id=_new_id(),
                    type=StepType.ADD_SYNTHETIC_KEY,
                    table=name,
                    description=(
                        f"No column or combination of columns in '{name}' uniquely identifies a "
                        "row, so the database has no way to reference one. A simple numbered id "
                        "column fixes that."
                    ),
                    params={"column": "row_id"},
                    origin="heuristic",
                    rationale=(
                        "; ".join(key_analysis.notes) if key_analysis.notes else None
                    ),
                )
            )

    # A confirmed equivalence is a *relationship*, and the proposal says so:
    # "confirmed equivalences passed forward as signals to FK detection".
    # Joining orders into customers because they share a customer id would
    # denormalise exactly the structure this platform exists to recover — the
    # repeated customer columns are what a foreign key is for.  The one case
    # where a join really is the answer is a single table split across two
    # sheets (Q1 and Q2), which is recognisable: the columns are the same.
    for equivalence in equivalences:
        if not equivalence.get("confirmed"):
            continue
        left_table = equivalence["left_table"]
        right_table = equivalence["right_table"]
        if left_table not in tables or right_table not in tables:
            continue
        if not _same_shape(tables[left_table], tables[right_table]):
            continue
        steps.append(
            CleaningStep(
                id=_new_id(),
                type=StepType.MERGE_SHEETS,
                table=left_table,
                description=(
                    f"'{left_table}' and '{right_table}' hold the same columns and you confirmed "
                    f"that {equivalence['left_column']} and {equivalence['right_column']} mean the "
                    "same thing, so they look like one table split in two."
                ),
                params={
                    "left_table": equivalence["left_table"],
                    "right_table": equivalence["right_table"],
                    "left_key": equivalence["left_column"],
                    "right_key": equivalence["right_column"],
                    "how": "left",
                    "result_table": f"{equivalence['left_table']}_with_{equivalence['right_table']}",
                },
                origin="heuristic",
            )
        )

    return sort_plan(steps)


def _key_analysis(
    stored: Mapping[str, Any], table: str, df: pd.DataFrame
) -> KeyAnalysis:
    """Phase 0's key verdict for one table, recomputed only if we have none.

    The stored analysis is preferred because it is the one the user saw and
    answered on the triage screen.  Recomputing here would silently overrule a
    confirmed composite key with a fresh guess.
    """

    payload = stored.get(table)
    if payload:
        return KeyAnalysis.from_dict(dict(payload))
    return discover_keys(df)


def _same_shape(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    """True when two tables hold the same columns — one table split in two."""

    return {str(c) for c in left.columns} == {str(c) for c in right.columns}


# ---------------------------------------------------------------------------
# LLM plan validation
# ---------------------------------------------------------------------------


def normalize_claude_plan(
    raw_steps: Iterable[Mapping[str, Any]],
    tables: Mapping[str, pd.DataFrame],
) -> tuple[list[CleaningStep], list[str]]:
    """Convert Claude's JSON into validated steps.

    Returns the accepted steps plus a list of human-readable rejection reasons,
    which the UI shows so the user knows what the agent tried to do.
    """

    accepted: list[CleaningStep] = []
    rejected: list[str] = []

    for index, raw in enumerate(raw_steps):
        label = f"step {index + 1}"
        step_type = str(raw.get("type", "")).strip()
        table = str(raw.get("table", "")).strip()
        params = raw.get("params") or {}
        if not isinstance(params, dict):
            rejected.append(f"{label}: params must be an object")
            continue

        try:
            parsed_type = StepType(step_type)
        except ValueError:
            refusal = FORBIDDEN_STEP_TYPES.get(step_type.lower())
            if refusal:
                # Not "unknown step type": the model asked for something the
                # platform refuses on principle, and the user is entitled to
                # read the principle on the plan screen.
                rejected.append(f"{label}: {step_type} was refused — {refusal}")
            else:
                rejected.append(f"{label}: unknown step type {step_type!r}")
            continue

        if table not in tables:
            rejected.append(f"{label} ({step_type}): unknown table {table!r}")
            continue

        columns = {str(c) for c in tables[table].columns}
        column = params.get("column")
        if column is not None and str(column) not in columns:
            rejected.append(
                f"{label} ({step_type}): {table} has no column {str(column)!r}"
            )
            continue

        if parsed_type is StepType.MERGE_SHEETS:
            right_table = str(params.get("right_table", ""))
            if right_table not in tables:
                rejected.append(f"{label} (merge_sheets): unknown table {right_table!r}")
                continue
            left_key = str(params.get("left_key", ""))
            right_key = str(params.get("right_key", ""))
            if left_key not in columns:
                rejected.append(f"{label} (merge_sheets): {table} has no column {left_key!r}")
                continue
            if right_key not in {str(c) for c in tables[right_table].columns}:
                rejected.append(
                    f"{label} (merge_sheets): {right_table} has no column {right_key!r}"
                )
                continue

        if parsed_type is StepType.SPLIT_COLUMN:
            into = params.get("into")
            if not isinstance(into, list) or len(into) < 2:
                rejected.append(f"{label} (split_column): 'into' must list two or more names")
                continue

        accepted.append(
            CleaningStep(
                id=_new_id(),
                type=parsed_type,
                table=table,
                description=str(raw.get("description", "")).strip()
                or f"{step_type.replace('_', ' ')} on {table}",
                params=params,
                origin="agent",
            )
        )

    return sort_plan(accepted), rejected


def merge_plans(
    primary: list[CleaningStep], secondary: list[CleaningStep]
) -> list[CleaningStep]:
    """Add heuristic steps the agent missed, without duplicating its work."""

    seen = {(s.type, s.table, str(s.params.get("column", ""))) for s in primary}
    extra = [
        step
        for step in secondary
        if (step.type, step.table, str(step.params.get("column", ""))) not in seen
    ]
    return sort_plan([*primary, *extra])
