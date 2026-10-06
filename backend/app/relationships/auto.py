"""Phase 3 — settling the obvious relationships without asking.

Detection proposes; it does not decide, and for a schema of a dozen tables that
meant a dozen-odd cards for someone who may never have heard of a foreign key.
Most of them are not close calls.  A column whose every value is a real key in
another table, and which is *named* like it, is a reference; a column that only
partly overlaps, or overlaps by accident of range, is not.  Those are settled
here, with the reason recorded so the screen can say why.

What stays a question is deliberately narrow: a link that nearly holds (a few
orphaned rows) and is named like a reference.  Everything else is decided, and a
decision made here is only ever a default — the user can reverse any of them,
and a decision the user made is never revisited.

Pure functions over plain values; the service layer feeds them rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Collection, Mapping, Sequence

from app.core.schemas import RelationshipOrigin, RelationshipStatus, RelationshipType

#: Every value of the referencing column is a real key.  A single stray value in
#: ten thousand still counts: messy data, not a different relationship.
CONFIRM_OVERLAP = 0.98
#: Mostly a reference, with a visible fraction of orphans.  Confirmed outright
#: when the two columns carry the same name, a question otherwise.  (A link with
#: orphans is still exported — as a join hint rather than an enforced constraint,
#: and the export says why — so confirming it costs the data nothing.)
ASK_OVERLAP = 0.90
#: Names this alike are, for practical purposes, the same name
#: (``customer_id`` / ``customer_id``, ``cds`` / ``cds_code``).
SAME_NAME_SIMILARITY = 0.85
#: Column names this alike read as "the same kind of thing".
MIN_NAME_SIMILARITY = 0.45
#: An identifier column pointing at a key with at least this many values does
#: not need its *name* to agree (``orders.cust`` → ``customers.id``): ids are
#: rarely coincidences once there are this many to coincide with.
MIN_TARGET_FOR_ID_MATCH = 10
#: A second link between the same two tables needs to be named very clearly
#: (``billing_address_id`` and ``shipping_address_id`` → ``addresses``).
SECOND_LINK_NAME_SIMILARITY = 0.70

IDENTIFIER_LABELS = frozenset(
    {"identifier", "foreign_identifier", "order_number", "invoice_number", "sku"}
)

CONFIRMED = RelationshipStatus.CONFIRMED.value
REJECTED = RelationshipStatus.REJECTED.value
PROPOSED = RelationshipStatus.PROPOSED.value


@dataclass(frozen=True, slots=True)
class Verdict:
    status: str
    reason: str

    @property
    def decided(self) -> bool:
        return self.status != PROPOSED


def _percent(value: float) -> str:
    return f"{value:.0%}"


def decide_foreign_key(
    origin: str,
    evidence: Mapping[str, Any],
    *,
    to_table: str,
    child_is_identifier: bool = False,
) -> Verdict:
    """The verdict for one foreign-key candidate considered on its own."""

    if origin == RelationshipOrigin.DECLARED.value:
        return Verdict(CONFIRMED, "the source database itself declares this link")
    if origin == RelationshipOrigin.MANUAL.value:
        return Verdict(CONFIRMED, "you drew this link")

    overlap = float(evidence.get("ratio_overlap") or 0.0)
    name = float(evidence.get("embedding_similarity") or 0.0)
    target_values = int(evidence.get("target_distinct") or 0)

    name_agrees = name >= MIN_NAME_SIMILARITY
    id_agrees = child_is_identifier and target_values >= MIN_TARGET_FOR_ID_MATCH
    if not (name_agrees or id_agrees):
        return Verdict(
            REJECTED,
            "the values overlap, but nothing about the column names or kinds suggests "
            "one refers to the other",
        )
    if overlap >= CONFIRM_OVERLAP:
        return Verdict(
            CONFIRMED,
            f"{_percent(overlap)} of the values are real {to_table} entries and the "
            "column names agree",
        )
    if overlap >= ASK_OVERLAP and name >= SAME_NAME_SIMILARITY:
        return Verdict(
            CONFIRMED,
            f"{_percent(overlap)} of the values are real {to_table} entries and the "
            "column names match; the rest are listed as orphans",
        )
    if overlap >= ASK_OVERLAP:
        return Verdict(
            PROPOSED,
            f"{_percent(overlap)} of the values are real {to_table} entries — close, "
            "but the rest match nothing",
        )
    return Verdict(
        REJECTED,
        f"only {_percent(overlap)} of the values are real {to_table} entries",
    )


def decide_dependency(rel_type: str) -> Verdict:
    """Patterns inside one table are facts about the rows, not decisions.

    An exact one holds for every row there is, so it is simply recorded.  An
    approximate one is left as it is: it describes the data and needs no verdict
    from anyone.
    """

    if rel_type == RelationshipType.FUNCTIONAL_DEPENDENCY.value:
        return Verdict(CONFIRMED, "true of every row in the table")
    return Verdict(PROPOSED, "true of most rows, not all")


def settle_foreign_keys(
    candidates: Sequence[Mapping[str, Any]],
    taken: Collection[tuple[str, str, str, str]] = (),
) -> dict[str, Verdict]:
    """Verdicts for a whole schema's candidates, best score first.

    Each candidate carries ``id``, ``from_table``, ``from_column``, ``to_table``,
    ``to_column``, ``score``, ``origin``, ``evidence`` and ``child_is_identifier``.
    ``taken`` is the links a person already confirmed (as
    ``(from_table, from_column, to_table, to_column)``): they occupy their slot,
    so an automatic confirmation never duplicates one.

    Two tables normally have *one* link between them, and a link seen from both
    ends is one link.  Going strongest-first means the best-supported candidate
    wins the slot and the echoes are turned down with a reason.
    """

    confirmed_pairs: dict[tuple[str, str], int] = {}
    confirmed_edges: set[tuple[str, str, str, str]] = set()
    for from_table, from_column, to_table, to_column in taken:
        confirmed_pairs[(from_table, to_table)] = confirmed_pairs.get((from_table, to_table), 0) + 1
        confirmed_edges.add((from_table, from_column, to_table, to_column))

    verdicts: dict[str, Verdict] = {}
    for candidate in sorted(candidates, key=lambda c: float(c.get("score") or 0.0), reverse=True):
        evidence = candidate.get("evidence") or {}
        verdict = decide_foreign_key(
            str(candidate.get("origin") or ""),
            evidence,
            to_table=str(candidate["to_table"]),
            child_is_identifier=bool(candidate.get("child_is_identifier")),
        )
        edge = (
            str(candidate["from_table"]),
            str(candidate["from_column"]),
            str(candidate["to_table"]),
            str(candidate["to_column"]),
        )
        pair = (edge[0], edge[2])
        reverse = (edge[2], edge[3], edge[0], edge[1])

        if verdict.status == CONFIRMED and candidate.get("origin") not in (
            RelationshipOrigin.DECLARED.value,
            RelationshipOrigin.MANUAL.value,
        ):
            if reverse in confirmed_edges:
                verdict = Verdict(
                    REJECTED,
                    "the same link, seen from the other table, is already in place",
                )
            elif confirmed_pairs.get(pair) and (
                float(evidence.get("embedding_similarity") or 0.0) < SECOND_LINK_NAME_SIMILARITY
            ):
                verdict = Verdict(
                    REJECTED,
                    f"{edge[0]} is already linked to {edge[2]} by a stronger match",
                )

        if verdict.status == CONFIRMED:
            confirmed_pairs[pair] = confirmed_pairs.get(pair, 0) + 1
            confirmed_edges.add(edge)
        verdicts[str(candidate["id"])] = verdict
    return verdicts
