"""The automated data-quality report (§5.4) — a cheap read, not a fresh pass.

Every number here is already sitting in :class:`~app.db.models.ColumnSemantics`
from Phase 1/3 enrichment: ``null_ratio``, ``unique_ratio``, ``type_confidence``
and ``taxonomy_confidence``. This module only aggregates what enrichment
already computed into the four dimensions the Report names — completeness,
uniqueness, consistency, validity — so producing the report costs a
``SELECT`` and some arithmetic, never a second analysis of the data itself.

A before/after comparison (also promised by §5.4) can only be as rich as what
is actually persisted across a re-enrichment: only the *exported* semantic
layer is archived (:class:`~app.db.models.SemanticLayerVersion`), and the
exported ``sem_metadata`` rows do not carry ``unique_ratio``,
``type_confidence`` or ``taxonomy_confidence`` — those are control-plane-only
fields that never leave :class:`~app.db.models.ColumnSemantics`. So an
archived version's score is honestly reported on the two dimensions the
export *does* carry (completeness, validity via the unknown-taxonomy rate)
rather than inventing numbers for the other two.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

#: Weakest-dimension labels, in the order a summary sentence checks them.
_DIMENSION_LABELS = {
    "completeness": "how many cells are filled in",
    "uniqueness": "how distinct the data in each column is",
    "consistency": "how confidently each column's type was detected",
    "validity": "how confidently each column's meaning was classified",
}


def _avg(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _round(value: float) -> float:
    return round(max(0.0, min(value, 100.0)), 1)


def _summary(scores: Mapping[str, float]) -> str:
    weakest = min(scores, key=lambda key: scores[key])
    return (
        f"Strongest on {max(scores, key=lambda key: scores[key])}; "
        f"weakest on {weakest} ({_DIMENSION_LABELS[weakest]})."
    )


def score_table(columns: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The four-dimension score for one table's columns, from live enrichment state.

    ``columns`` are :meth:`~app.db.models.ColumnSemantics.to_dict` (or
    equivalent) mappings — the full control-plane fields, not the reduced
    exported ``sem_metadata`` shape.
    """

    if not columns:
        return {
            "completeness": 0.0,
            "uniqueness": 0.0,
            "consistency": 0.0,
            "validity": 0.0,
            "overall": 0.0,
            "summary": "No columns to score yet.",
        }

    scores = {
        "completeness": _round(100 * (1 - _avg([float(c.get("null_ratio") or 0.0) for c in columns]))),
        "uniqueness": _round(100 * _avg([float(c.get("unique_ratio") or 0.0) for c in columns])),
        "consistency": _round(100 * _avg([float(c.get("type_confidence") or 0.0) for c in columns])),
        "validity": _round(100 * _avg([float(c.get("taxonomy_confidence") or 0.0) for c in columns])),
    }
    overall = _round(_avg(list(scores.values())))
    return {**scores, "overall": overall, "summary": _summary(scores)}


def score_table_basic(columns: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The two dimensions an *archived export* can still support.

    ``columns`` are exported ``sem_metadata`` rows (see
    :func:`app.export.metadata.build_rows`): they carry ``null_ratio`` and
    ``taxonomy_label`` but not the confidence fields ``score_table`` also
    uses, so ``consistency``/``uniqueness`` are reported as ``None`` — an
    honest "not available for this comparison" rather than a fabricated
    number.
    """

    if not columns:
        return {"completeness": None, "validity": None}
    completeness = _round(100 * (1 - _avg([float(c.get("null_ratio") or 0.0) for c in columns])))
    known = sum(1 for c in columns if str(c.get("taxonomy_label") or "unknown") != "unknown")
    validity = _round(100 * known / len(columns))
    return {"completeness": completeness, "validity": validity}


def build_quality_report(
    current_tables: Sequence[Mapping[str, Any]],
    previous_tables: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Per-table scores for the live layer, plus a before/after when a prior build exists.

    ``current_tables``/``previous_tables`` are each ``[{"name": str, "columns":
    [...]}, ...]`` — the current side scored with :func:`score_table` (full
    control-plane fields), the previous side with :func:`score_table_basic`
    (archived export fields only).
    """

    previous_by_name = {str(t.get("name")): t.get("columns") or [] for t in (previous_tables or [])}
    tables: list[dict[str, Any]] = []
    for table in current_tables:
        name = str(table.get("name"))
        entry: dict[str, Any] = {"name": name, **score_table(table.get("columns") or [])}
        if name in previous_by_name:
            entry["previous"] = score_table_basic(previous_by_name[name])
        tables.append(entry)

    overall = _round(_avg([t["overall"] for t in tables])) if tables else 0.0
    return {"tables": tables, "overall": overall}
