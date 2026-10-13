"""Phase 6 — dashboard cards: pinning, re-running and editing saved queries."""

from __future__ import annotations

import re
from datetime import date, datetime, timezone
from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.api.services.export import export_record
from app.core.config import get_settings
from app.db.models import DashboardCard, IngestionSession
from app.export.runner import read_semantic_layer, run_readonly
from app.query.guard import QueryRejected, validate_select_only


_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def pin_dashboard_card(
    db: Session, record: IngestionSession, payload: dict[str, Any]
) -> DashboardCard:
    """Save one answered question as a dashboard card.

    The SQL is re-validated here, independently of whatever validated it when
    the question was first answered: a pin request is client-supplied input
    like any other, and this is the write path that turns a stored statement
    into something that runs, unattended, on every future dashboard load.
    Re-checked against every exported table rather than only the ones that
    question's retrieval step picked — a card outlives the request that made
    it, so the wider, still-honest bound is the one worth keeping.
    """

    question = str(payload.get("question") or "").strip()
    sql = str(payload.get("sql") or "").strip()
    if not question or not sql:
        raise ValueError("a card needs both a question and its SQL")

    export = export_record(db, record.id)
    if export is None:
        raise ValueError("export the semantic layer before pinning a card")

    rows = read_semantic_layer(record.id)
    exported_tables = {str(row.get("table_name")).lower() for row in rows}
    try:
        validated_sql = validate_select_only(sql, exported_tables, export.dialect)
    except QueryRejected as exc:
        raise ValueError(f"this query can no longer be pinned: {exc}") from exc

    position = db.execute(
        select(func.count())
        .select_from(DashboardCard)
        .where(DashboardCard.session_id == record.id)
    ).scalar_one()

    card = DashboardCard(
        session_id=record.id,
        title=str(payload.get("title") or "").strip() or question,
        question=question,
        sql=validated_sql,
        dialect=export.dialect,
        explanation=str(payload.get("explanation") or ""),
        tables_used=list(payload.get("tables_used") or []),
        visualization=dict(payload.get("visualization") or {}),
        position=position,
    )
    db.add(card)
    db.flush()
    return card


def _quote_identifier(name: str) -> str | None:
    if not _IDENTIFIER_RE.match(name or ""):
        return None
    return f'"{name}"'


def _date_filtered_sql(card: DashboardCard, start: date | None, end: date | None) -> tuple[str, bool]:
    """Wrap a card's SQL in a date-range ``WHERE``, when it unambiguously can be.

    Only applies when the card's own shape names exactly one date column —
    for anything else there is no column a *global* filter could mean, so the
    card runs unfiltered rather than guessing one.  ``start``/``end`` arrive as
    parsed :class:`datetime.date` objects (the route's own parameter type), so
    interpolating their ``isoformat()`` never carries anything but digits and
    hyphens into the statement.
    """

    if start is None and end is None:
        return card.sql, False
    date_columns = (card.visualization or {}).get("date_columns") or []
    if len(date_columns) != 1:
        return card.sql, False
    column = _quote_identifier(date_columns[0])
    if column is None:
        return card.sql, False

    clauses = []
    if start is not None:
        clauses.append(f"{column} >= '{start.isoformat()}'")
    if end is not None:
        clauses.append(f"{column} < '{end.isoformat()}'")
    where = " AND ".join(clauses)
    return f"SELECT * FROM ({card.sql}) AS _dashboard_filtered WHERE {where}", True


def list_dashboard_cards(
    db: Session, record: IngestionSession, start: date | None = None, end: date | None = None
) -> list[dict[str, Any]]:
    """Every pinned card, re-executed against the live export.

    "Results always current, not snapshots" (§Phase 6) means the stored row is
    never trusted for data — even a card whose SQL now fails (a column renamed
    by a later re-export) is reported per-card as ``ok: False``, rather than
    one bad card taking the whole dashboard down.
    """

    cards = (
        db.execute(
            select(DashboardCard)
            .where(DashboardCard.session_id == record.id)
            .order_by(DashboardCard.position)
        )
        .scalars()
        .all()
    )
    settings = get_settings()
    refreshed_at = datetime.now(timezone.utc).isoformat()
    payload: list[dict[str, Any]] = []
    for card in cards:
        item = card.to_dict()
        statement, filtered = _date_filtered_sql(card, start, end)
        try:
            columns, rows, truncated = run_readonly(record.id, statement, settings.query_row_cap)
            item.update(
                ok=True,
                columns=columns,
                rows=rows,
                row_count=len(rows),
                truncated=truncated,
                error=None,
            )
        except Exception as exc:  # the driver's own exception type, dialect-dependent
            item.update(ok=False, columns=[], rows=[], row_count=0, truncated=False, error=str(exc))
        item["date_filtered"] = filtered
        item["refreshed_at"] = refreshed_at
        payload.append(item)
    return payload


def update_dashboard_card(
    db: Session, record: IngestionSession, card_id: str, patch: dict[str, Any]
) -> DashboardCard:
    """Rename a card or move/resize it.  Never touches its question or SQL —
    re-pin to change what a card answers."""

    card = db.execute(
        select(DashboardCard).where(
            DashboardCard.id == card_id, DashboardCard.session_id == record.id
        )
    ).scalar_one_or_none()
    if card is None:
        raise KeyError(f"dashboard card {card_id} not found")
    if patch.get("title") is not None:
        card.title = str(patch["title"]).strip() or card.question
    if patch.get("layout") is not None:
        card.layout = dict(patch["layout"])
    db.flush()
    return card


def delete_dashboard_card(db: Session, record: IngestionSession, card_id: str) -> None:
    db.execute(
        delete(DashboardCard).where(
            DashboardCard.id == card_id, DashboardCard.session_id == record.id
        )
    )
