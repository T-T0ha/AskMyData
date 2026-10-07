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

#: The plan only ever offers to delete a column with **nothing** in it.  A column
#: that is 88 % empty still holds values somebody typed, and deleting it is a
#: loss the user did not ask for.  Any gap short of that is reported (triage
#: screen, null-ratio bar in the field semantic view) and then left alone, because
#: the only way to close it is to invent the value that is missing.


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
                            f"The column '{column}' in '{name}' is completely empty — every row "
                            "is blank. It holds no information, so removing it makes the table "
                            "tidier and loses nothing."
                        ),
                        params={"column": column},
                        origin="heuristic",
                        rationale="no value in any row, so nothing is lost by removing it",
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
                                f"{stray:,} value(s) in '{column}' ({name}) have extra spaces at "
                                "the start or end. They look the same on screen, but a database "
                                "counts 'Oak ' and 'Oak' as different, so totals and filters would "
                                "miss some. Trimming the spaces fixes that."
                            ),
                            params={"column": column},
                            origin="heuristic",
                            rationale="only spaces at the edges are removed; the words are untouched",
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
                                f"'{column}' ({name}) spells the same value with different "
                                f"capitalisation (for example {shown}), so one category is counted "
                                f"as {'two' if collisions == 1 else 'several'}. Using the "
                                f"{casing}case spelling the column already mostly uses brings them "
                                "together."
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
                        f"{duplicate_count:,} row(s) in '{name}' are exact copies of another "
                        "row. Keeping them would count the same thing twice in every total. "
                        "One copy of each is kept."
                    ),
                    rationale="only rows identical in every column are removed",
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
                        f"Nothing in '{name}' uniquely identifies a row, so no other table can "
                        "point at one. Adding a simple counter column (1, 2, 3, …) fixes that "
                        "and changes none of your existing data."
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
    merged_pairs: set[frozenset[str]] = set()
    for equivalence in sorted(
        (e for e in equivalences if e.get("confirmed")),
        key=lambda e: float(e.get("score") or 0.0),
        reverse=True,
    ):
        left_table = equivalence["left_table"]
        right_table = equivalence["right_table"]
        if left_table not in tables or right_table not in tables:
            continue
        if not _same_shape(tables[left_table], tables[right_table]):
            continue
        # One step per pair of tables, however many of their columns match —
        # every column of a split table matches its twin.  The strongest pair
        # is the join key.
        pair = frozenset((left_table, right_table))
        if pair in merged_pairs:
            continue
        merged_pairs.add(pair)
        steps.append(
            CleaningStep(
                id=_new_id(),
                type=StepType.MERGE_SHEETS,
                table=left_table,
                description=(
                    f"'{left_table}' and '{right_table}' have exactly the same columns, and "
                    f"{equivalence['left_column']} / {equivalence['right_column']} match, so "
                    "they look like one table that was split in two. This combines them into one."
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
                rationale="check this one — it is the only step that joins two tables",
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
# reference validation — shared by an LLM's proposal and the user's edits
# ---------------------------------------------------------------------------


def _validate_references(
    step_type: StepType, table: str, params: Mapping[str, Any], tables: Mapping[str, pd.DataFrame]
) -> str | None:
    """``None`` if every table/column ``step_type``'s ``params`` names on
    ``table`` actually exists and the step is otherwise structurally sound;
    the human-readable reason it is not, otherwise.

    Shared between validating an LLM's own proposal
    (:func:`normalize_claude_plan`) and re-validating a plan the user edited or
    added to by hand (:func:`validate_plan_references`) — a step is a step
    regardless of who wrote it, and neither producer gets to skip the check.
    """

    if table not in tables:
        return f"unknown table {table!r}"

    columns = {str(c) for c in tables[table].columns}
    column = params.get("column")
    if step_type is StepType.ADD_SYNTHETIC_KEY:
        # The one step whose column is the one it creates, not one it finds.
        if column is not None and str(column) in columns:
            return f"{table} already has a column named {str(column)!r}"
    elif column is not None and str(column) not in columns:
        return f"{table} has no column {str(column)!r}"

    if step_type is StepType.MERGE_SHEETS:
        right_table = str(params.get("right_table", ""))
        if right_table not in tables:
            return f"unknown table {right_table!r}"
        left_key = str(params.get("left_key", ""))
        right_key = str(params.get("right_key", ""))
        if left_key not in columns:
            return f"{table} has no column {left_key!r}"
        right_columns = {str(c) for c in tables[right_table].columns}
        if right_key not in right_columns:
            return f"{right_table} has no column {right_key!r}"
        if not _same_shape(tables[table], tables[right_table]):
            return (
                f"{table} and {right_table} do not share the same columns — "
                "merge_sheets only joins a table split across sheets, not two "
                "different entities that merely share a key"
            )

    if step_type is StepType.SPLIT_COLUMN:
        into = params.get("into")
        if not isinstance(into, list) or len(into) < 2:
            return "'into' must list two or more names"

    return None


def validate_plan_references(
    steps: Iterable[CleaningStep], tables: Mapping[str, pd.DataFrame]
) -> tuple[list[CleaningStep], list[str]]:
    """Re-check a plan the user edited, reordered or added to by hand.

    ``normalize_claude_plan`` is the only gate an LLM's own proposal passes
    through before the user ever sees it; nothing then re-checked what the
    user did to it afterward.  A step the user adds or edits draws from the
    same vocabulary and names the same kind of references, so it is held to
    the same standard here — dropped with a reason before it runs, rather than
    surfacing only as a soft "step failed" interrupt once execution reaches it.
    """

    accepted: list[CleaningStep] = []
    rejected: list[str] = []
    for step in steps:
        reason = _validate_references(step.type, step.table, step.params, tables)
        if reason:
            rejected.append(f"step {step.id} ({step.type.value}) on {step.table}: {reason}")
            continue
        accepted.append(step)
    return accepted, rejected


# ---------------------------------------------------------------------------
# LLM plan validation
# ---------------------------------------------------------------------------


#: What each executor in ``steps.py`` refuses to run without.  A model's step
#: that lacks one of these would parse and validate, then fail on the step
#: screen — for a plan of dozens of steps, on every one of them.
REQUIRED_PARAMS: dict[StepType, tuple[str, ...]] = {
    StepType.RENAME_COLUMN: ("column", "new_name"),
    StepType.DROP_COLUMN: ("column",),
    StepType.STANDARDIZE_FORMAT: ("column",),
    StepType.STANDARDIZE_CASING: ("column",),
    StepType.STRIP_WHITESPACE: ("column",),
    StepType.MERGE_SHEETS: ("right_table", "left_key", "right_key"),
    StepType.SPLIT_COLUMN: ("column",),
    StepType.TYPE_CAST: ("column", "to"),
}

#: Steps that act on one column and so can be split from a "columns": [...] list.
_PER_COLUMN_STEPS = frozenset(
    {
        StepType.DROP_COLUMN,
        StepType.STANDARDIZE_FORMAT,
        StepType.STANDARDIZE_CASING,
        StepType.STRIP_WHITESPACE,
    }
)

#: Every key a step's ``params`` may hold; a model that puts one at the top
#: level of the step instead of inside ``params`` is still understood.
_PARAM_KEYS = (
    "column", "new_name", "format", "date_format", "casing", "collapse_internal",
    "left_table", "right_table", "left_key", "right_key", "how", "result_table",
    "delimiter", "regex", "into", "subset", "keep", "to",
)  # fmt: skip

_COLUMN_ALIASES = ("column_name", "col", "field", "column_names", "columns")


def _params_variants(
    raw: Mapping[str, Any], step_type: StepType
) -> tuple[list[dict[str, Any]], str | None]:
    """Read a model's step into one ``params`` dict per column it acts on.

    Models are loose about where a step's arguments go: inside ``params`` as
    asked, at the top level of the step, under another name for the column, or
    as a ``columns`` list for several at once.  Reading all of them is cheaper
    than rejecting a step whose meaning is plain.  The second value is the
    reason the step cannot be read at all.
    """

    holder = next(
        (raw[key] for key in ("params", "parameters", "arguments", "args") if raw.get(key)),
        {},
    )
    if not isinstance(holder, Mapping):
        return [], "params must be an object"
    params = dict(holder)

    for key in _PARAM_KEYS:
        if params.get(key) in (None, "") and raw.get(key) not in (None, ""):
            params[key] = raw[key]

    if "column" not in REQUIRED_PARAMS.get(step_type, ()):
        # deduplicate names its columns as ``subset``; merge_sheets and
        # add_synthetic_key have no column to find.
        if step_type is StepType.DEDUPLICATE and not params.get("subset"):
            listed = next(
                (v for k in ("columns", "column_names") if (v := params.get(k, raw.get(k)))),
                None,
            )
            params["subset"] = listed if isinstance(listed, list) else None
        for key in ("columns", "column_names"):
            params.pop(key, None)
        return [params], None

    columns: list[Any] = []
    if params.get("column") not in (None, ""):
        columns = params["column"] if isinstance(params["column"], list) else [params["column"]]
    else:
        for key in _COLUMN_ALIASES:
            value = params.get(key, raw.get(key))
            if value not in (None, "", []):
                columns = value if isinstance(value, list) else [value]
                break
    columns = [str(c) for c in columns if c not in (None, "")]
    for key in _COLUMN_ALIASES:
        params.pop(key, None)

    if not columns:
        params.pop("column", None)
        return [params], None
    if len(columns) > 1 and step_type in _PER_COLUMN_STEPS:
        return [{**params, "column": column} for column in columns], None
    return [{**params, "column": columns[0]}], None


def _missing_params(step_type: StepType, params: Mapping[str, Any]) -> list[str]:
    return [
        key for key in REQUIRED_PARAMS.get(step_type, ()) if params.get(key) in (None, "")
    ]



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

        variants, unreadable = _params_variants(raw, parsed_type)
        if unreadable:
            rejected.append(f"{label}: {unreadable}")
            continue

        for params in variants:
            missing = _missing_params(parsed_type, params)
            if missing:
                rejected.append(
                    f"{label} ({step_type}): no {' or '.join(repr(m) for m in missing)} "
                    f"was given for {table}, so it could not be run"
                )
                continue

            reason = _validate_references(parsed_type, table, params, tables)
            if reason:
                rejected.append(f"{label} ({step_type}): {reason}")
                continue

            if (
                parsed_type is StepType.DROP_COLUMN
                and not tables[table][str(params.get("column"))].isna().all()
            ):
                # The planner's own rule, applied to the model too: a column that
                # holds any value at all is the user's data, however sparse.
                rejected.append(
                    f"{label}: '{params.get('column')}' in {table} still holds values, so it was "
                    "kept — only completely empty columns are ever dropped"
                )
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
