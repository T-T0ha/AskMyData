"""Phase 3 — relationship detection and interactive validation.

Three modules, matching the three things the platform can say about how data
relates:

``foreign_keys``  which column in one table points at another table's key
``dependencies``  which column inside a table determines another
``evidence``      the rows that support or contradict either claim

Detection is algorithmic, not model-based.  Claude's only role in this phase is
explaining a borderline candidate in plain English (see
``ClaudeClient.explain_relationship``) so that a non-technical user can decide;
it never produces or scores a relationship.  Every relationship arrives as a
proposal and stays one until a human confirms it.
"""
