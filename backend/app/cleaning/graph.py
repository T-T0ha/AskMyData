"""Phase 2 — the co-planning / co-execution graph (LangGraph).

This is the Cocoa (Feng et al., CHI '26) interaction pattern applied to data
cleaning: the agent proposes a plan, the human edits and approves it, and then
each step executes one at a time with the human approving the result before
the next step starts.  Planning and execution interleave — a rejected step can
be re-parameterised and retried without discarding the rest of the plan.

    analyze ──▶ review ⟨interrupt⟩ ──▶ execute ──▶ validate ⟨interrupt⟩ ──┐
                                          ▲                              │
                                          └──────── more steps ──────────┘
                                                                         ▼
                                                                     finalize

The two ``interrupt()`` calls are what make this co-planning rather than
automation: LangGraph persists the whole graph state to PostgreSQL at each
pause, so a session survives a browser close, a server restart, or a week-long
gap.  DataFrames stay out of the state (see :mod:`app.cleaning.store`) — the
graph carries only names, step definitions and previews.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any, Literal, Mapping, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from app.cleaning.planner import (
    heuristic_plan,
    merge_plans,
    normalize_claude_plan,
)
from app.cleaning.steps import StepError, execute_step
from app.cleaning.store import TableStore, get_store
from app.core.schemas import CleaningStep

logger = logging.getLogger(__name__)


def semantics_from_summary(summary: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Index the Phase 1 statistical summary by table and column.

    ``summary`` is already in the graph state (it is what Claude is sent), so
    the planner gets Phase 1's types and taxonomy labels without a database
    round-trip and without ever touching a data row.
    """

    indexed: dict[str, dict[str, Any]] = {}
    for table in summary.get("tables") or []:
        table_name = str(table.get("name", ""))
        columns: dict[str, Any] = {}
        for column in table.get("columns") or []:
            columns[str(column.get("name", ""))] = {
                "column_type": column.get("type"),
                "taxonomy_label": column.get("taxonomy"),
                "null_ratio": column.get("null_ratio"),
                "unique_ratio": column.get("unique_ratio"),
            }
        indexed[table_name] = columns
    return indexed


def _last(_current: Any, incoming: Any) -> Any:
    """Reducer: latest write wins (the default, made explicit)."""

    return incoming


def _append(current: list[Any] | None, incoming: list[Any] | None) -> list[Any]:
    return [*(current or []), *(incoming or [])]


class CleaningState(TypedDict, total=False):
    """Everything the graph persists.  JSON-serialisable by construction."""

    session_id: str
    summary: dict[str, Any]
    equivalences: list[dict[str, Any]]
    #: Phase 0's key verdict per table, including anything the user confirmed
    #: on the triage screen.  Carried in the state so the planner never has to
    #: re-derive — or accidentally overrule — a decision the user already made.
    keys: dict[str, Any]
    #: Sheets ingested but excluded from cleaning (too small / structurally broken).
    excluded_tables: list[str]
    plan: Annotated[list[dict[str, Any]], _last]
    plan_source: str
    plan_rejections: list[str]
    cursor: int
    status: str
    history: Annotated[list[dict[str, Any]], _append]
    pending: dict[str, Any] | None
    errors: Annotated[list[str], _append]


# ---------------------------------------------------------------------------
# nodes
# ---------------------------------------------------------------------------


def analyze_node(state: CleaningState) -> dict[str, Any]:
    """Node 1 — propose a cleaning plan from statistical summaries only.

    Claude receives ``state["summary"]``: column names, detected types,
    taxonomy labels, null ratios, five sample values per column and the
    confirmed cross-sheet equivalences.  Never a data row.
    """

    from app.semantics.claude_client import get_claude_client  # local: keeps import cheap

    store = get_store(state["session_id"])
    excluded = set(state.get("excluded_tables") or [])
    tables = {name: df for name, df in store.tables().items() if name not in excluded}
    equivalences = state.get("equivalences") or []
    summary = state.get("summary") or {}

    # Phase 1's verdict is what makes the plan sane: without it the planner
    # cannot tell an order code from a quantity, and proposes filling missing
    # SKUs with the most common SKU.
    semantics = semantics_from_summary(summary)
    fallback = heuristic_plan(
        tables,
        semantics=semantics,
        equivalences=equivalences,
        keys=state.get("keys") or {},
    )

    claude = get_claude_client()
    if not claude.available:
        return {
            "plan": [s.to_dict() for s in fallback],
            "plan_source": "heuristic",
            "plan_rejections": [],
            "status": "awaiting_plan_review",
            "cursor": 0,
        }

    raw = claude.propose_cleaning_plan(summary)
    if raw is None:
        return {
            "plan": [s.to_dict() for s in fallback],
            "plan_source": "heuristic_after_llm_error",
            "plan_rejections": [claude.last_error or "the model did not return a plan"],
            "status": "awaiting_plan_review",
            "cursor": 0,
        }

    accepted, rejections = normalize_claude_plan(raw, tables)
    combined = merge_plans(accepted, fallback)
    return {
        "plan": [s.to_dict() for s in combined],
        "plan_source": "claude" if accepted else "heuristic_after_llm_rejected",
        "plan_rejections": rejections,
        "status": "awaiting_plan_review",
        "cursor": 0,
    }


def review_node(state: CleaningState) -> Command[Literal["execute", "finalize"]]:
    """Node 2 — human review interrupt.

    The graph stops here and persists.  The frontend renders the plan as
    editable, reorderable cards; the resume payload is the plan the user
    actually approved, which may bear little resemblance to what was proposed.
    """

    decision = interrupt(
        {
            "kind": "plan_review",
            "plan": state.get("plan", []),
            "plan_source": state.get("plan_source"),
            "plan_rejections": state.get("plan_rejections", []),
            "tables": get_store(state["session_id"]).names(),
        }
    )
    decision = decision or {}
    action = str(decision.get("action", "confirm"))

    if action == "cancel":
        return Command(
            goto="finalize",
            update={"status": "cancelled", "plan": state.get("plan", [])},
        )

    plan = decision.get("plan")
    if plan is None:
        plan = state.get("plan", [])
    # The user's ordering is authoritative — it is not re-sorted.
    plan = [CleaningStep.from_dict(step).to_dict() for step in plan]

    if not plan:
        return Command(goto="finalize", update={"status": "completed", "plan": []})

    return Command(
        goto="execute",
        update={"plan": plan, "cursor": 0, "status": "executing"},
    )


def execute_node(state: CleaningState) -> dict[str, Any]:
    """Node 3 — run the step at the cursor and snapshot for revert."""

    session_id = state["session_id"]
    store: TableStore = get_store(session_id)
    plan = state.get("plan", [])
    cursor = int(state.get("cursor", 0))

    if cursor >= len(plan):
        return {"status": "completed", "pending": None}

    step = CleaningStep.from_dict(plan[cursor])
    snapshot_tag = f"{cursor}-{step.id}"
    store.snapshot(snapshot_tag)

    try:
        outcome = execute_step(store, step)
    except StepError as exc:
        store.persist()
        failed = [dict(s) for s in plan]
        failed[cursor] = {**failed[cursor], "status": "failed"}
        return {
            "plan": failed,
            "pending": {
                "kind": "step_failed",
                "step": failed[cursor],
                "cursor": cursor,
                "error": str(exc),
            },
            "status": "awaiting_step_validation",
            "errors": [f"{step.type.value} on {step.table}: {exc}"],
        }

    store.persist()
    updated = [dict(s) for s in plan]
    updated[cursor] = {**updated[cursor], "status": "executed"}
    return {
        "plan": updated,
        "pending": {
            "kind": "step_validation",
            "step": updated[cursor],
            "cursor": cursor,
            "snapshot": snapshot_tag,
            "outcome": outcome.to_dict(),
            "tables": [t.to_dict() for t in store.meta()],
        },
        "status": "awaiting_step_validation",
    }


def validate_node(state: CleaningState) -> Command[Literal["execute", "finalize"]]:
    """Node 4 — validation interrupt after every executed step.

    Resume payloads:

    ``{"action": "approve"}``                  keep the change, move on
    ``{"action": "revert"}``                   restore the snapshot, skip the step
    ``{"action": "retry", "params": {...}}``   restore, re-parameterise, run again
    ``{"action": "skip"}``                     leave the step unexecuted
    """

    session_id = state["session_id"]
    store = get_store(session_id)
    plan = [dict(s) for s in state.get("plan", [])]
    cursor = int(state.get("cursor", 0))
    pending = state.get("pending") or {}

    decision = interrupt({**pending, "kind": pending.get("kind", "step_validation")}) or {}
    action = str(decision.get("action", "approve"))
    snapshot = str(pending.get("snapshot") or f"{cursor}-{plan[cursor]['id']}")

    if action == "abort":
        store.restore(snapshot)
        return Command(goto="finalize", update={"status": "aborted", "pending": None})

    if action in {"revert", "retry"}:
        restored = store.restore(snapshot)
        note = "reverted" if restored else "revert failed: snapshot missing"
        if action == "retry":
            params = decision.get("params")
            if isinstance(params, dict):
                plan[cursor] = {**plan[cursor], "params": params}
            plan[cursor] = {**plan[cursor], "status": "pending"}
            return Command(
                goto="execute",
                update={
                    "plan": plan,
                    "cursor": cursor,  # same step, new parameters
                    "pending": None,
                    "status": "executing",
                    "history": [
                        {"cursor": cursor, "step_id": plan[cursor]["id"], "action": "retry"}
                    ],
                },
            )
        plan[cursor] = {**plan[cursor], "status": "reverted"}
        return Command(
            goto="execute",
            update={
                "plan": plan,
                "cursor": cursor + 1,
                "pending": None,
                "status": "executing",
                "history": [
                    {"cursor": cursor, "step_id": plan[cursor]["id"], "action": note}
                ],
            },
        )

    if action == "skip":
        store.restore(snapshot)
        plan[cursor] = {**plan[cursor], "status": "skipped"}
    else:
        plan[cursor] = {**plan[cursor], "status": "approved"}

    store.persist()
    return Command(
        goto="execute",
        update={
            "plan": plan,
            "cursor": cursor + 1,
            "pending": None,
            "status": "executing",
            "history": [
                {
                    "cursor": cursor,
                    "step_id": plan[cursor]["id"],
                    "action": plan[cursor]["status"],
                    "summary": (pending.get("outcome") or {}).get("summary"),
                }
            ],
        },
    )


def finalize_node(state: CleaningState) -> dict[str, Any]:
    """Terminal node — persist the cleaned tables and report."""

    store = get_store(state["session_id"])
    store.persist()
    status = state.get("status")
    if status not in {"cancelled", "aborted"}:
        status = "completed"
    return {
        "status": status,
        "pending": None,
        "history": [{"action": "finalized", "tables": [t.to_dict() for t in store.meta()]}],
    }


def _after_execute(state: CleaningState) -> Literal["validate", "finalize"]:
    if state.get("pending"):
        return "validate"
    return "finalize"


# ---------------------------------------------------------------------------
# graph construction
# ---------------------------------------------------------------------------


def build_graph(checkpointer=None):
    """Compile the cleaning graph.

    ``checkpointer`` must be supplied for the interrupts to work; the API layer
    passes a PostgreSQL saver (or a SQLite saver in local development).
    """

    graph = StateGraph(CleaningState)
    graph.add_node("analyze", analyze_node)
    graph.add_node("review", review_node)
    graph.add_node("execute", execute_node)
    graph.add_node("validate", validate_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "analyze")
    graph.add_edge("analyze", "review")
    graph.add_conditional_edges(
        "execute", _after_execute, {"validate": "validate", "finalize": "finalize"}
    )
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=checkpointer)


def thread_config(session_id: str) -> dict[str, Any]:
    """LangGraph thread id == cleaning session id."""

    return {"configurable": {"thread_id": f"cleaning-{session_id}"}}


def interrupt_payload(result: Any) -> dict[str, Any] | None:
    """Pull the pending interrupt out of an invoke/resume result."""

    interrupts = result.get("__interrupt__") if isinstance(result, dict) else None
    if not interrupts:
        return None
    first = interrupts[0]
    value = getattr(first, "value", first)
    return value if isinstance(value, dict) else {"value": value}


def resume(graph, session_id: str, payload: dict[str, Any]) -> Any:
    """Answer the pending interrupt and run until the next one."""

    return graph.invoke(Command(resume=payload), thread_config(session_id))
