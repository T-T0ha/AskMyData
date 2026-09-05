"""Phase 5 — natural language questions answered against the semantic layer.

Nothing in this package touches a data row until :mod:`app.export.runner`
executes the one validated SELECT at the very end.  Everything before that —
retrieval, the SQL guard, result shaping — works on schema, metadata and
embeddings, the same privacy boundary Phases 1-4 hold.
"""
