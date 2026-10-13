"""Service layer — the phase orchestration the API routes call into.

Keeps FastAPI handlers thin: a route parses the request, calls one function
here, and serialises the result.

Split into one module per phase.  Every public name the single
``services.py`` defined is re-exported here, along with the two private
helpers the routes call (``_semantic_description``, ``_embed_columns``), so
``services.answer_question`` and the rest are unchanged for callers.

Patch a function where it is *looked up*.  The routes call through this
package (``services.ingest_file``), so patching an attribute here reaches them.
The functions inside the package call each other, and ``explain`` /
``run_readonly``, through their own module — to change what
:func:`answer_question` runs, patch ``services.query``, not this package.
"""

from __future__ import annotations

from app.api.services.ingestion import (
    MIN_ANALYZABLE_ROWS,
    create_session,
    list_sessions,
    DuplicateSourceError,
    find_duplicate_session,
    ingest_file,
    ingest_database,
    skip_duplicate_tables,
    triage_stats,
    analyzable_tables,
    table_keys,
    set_table_key,
    sync_synthetic_keys,
    detect_and_store_equivalences,
)
from app.api.services.profiling import (
    profile_tables,
    profiles_payload,
    override_profile,
    table_type_of,
    sync_key_roles,
    describe_tables,
)
from app.api.services.field_semantics import (
    _semantic_description,
    DESCRIBE_CHUNK_COLUMNS,
    DESCRIBE_MAX_WORKERS,
    run_field_semantics,
    _embed_columns,
    semantics_payload,
    build_summary,
    table_preview,
)
from app.api.services.relationships import (
    LLM_EXPLAIN_BAND,
    detect_relationships,
    apply_auto_relationship_decisions,
    relationships_payload,
    relationship_graph,
    get_relationship,
    decide_relationship,
    add_manual_relationship,
    relationship_evidence,
)
from app.api.services.export import (
    MAX_LAYER_VERSIONS,
    export_semantic_layer,
    layer_versions,
    layer_bundle,
    get_quality_report,
    get_suggested_questions,
    layer_documentation,
    export_record,
    export_payload,
    forget_export,
)
from app.api.services.query import (
    recent_conversation,
    question_history,
    list_business_rules,
    add_business_rule,
    delete_business_rule,
    list_query_examples,
    save_query_example,
    delete_query_example,
    answer_question,
)
from app.api.services.dashboard import (
    pin_dashboard_card,
    list_dashboard_cards,
    update_dashboard_card,
    delete_dashboard_card,
)
from app.api.services.status import (
    backend_status,
)

__all__ = [
    "MIN_ANALYZABLE_ROWS",
    "create_session",
    "list_sessions",
    "DuplicateSourceError",
    "find_duplicate_session",
    "ingest_file",
    "ingest_database",
    "skip_duplicate_tables",
    "triage_stats",
    "analyzable_tables",
    "table_keys",
    "set_table_key",
    "sync_synthetic_keys",
    "detect_and_store_equivalences",
    "profile_tables",
    "profiles_payload",
    "override_profile",
    "table_type_of",
    "sync_key_roles",
    "describe_tables",
    "_semantic_description",
    "DESCRIBE_CHUNK_COLUMNS",
    "DESCRIBE_MAX_WORKERS",
    "run_field_semantics",
    "_embed_columns",
    "semantics_payload",
    "build_summary",
    "table_preview",
    "LLM_EXPLAIN_BAND",
    "detect_relationships",
    "apply_auto_relationship_decisions",
    "relationships_payload",
    "relationship_graph",
    "get_relationship",
    "decide_relationship",
    "add_manual_relationship",
    "relationship_evidence",
    "MAX_LAYER_VERSIONS",
    "export_semantic_layer",
    "layer_versions",
    "layer_bundle",
    "get_quality_report",
    "get_suggested_questions",
    "layer_documentation",
    "export_record",
    "export_payload",
    "forget_export",
    "recent_conversation",
    "question_history",
    "list_business_rules",
    "add_business_rule",
    "delete_business_rule",
    "list_query_examples",
    "save_query_example",
    "delete_query_example",
    "answer_question",
    "pin_dashboard_card",
    "list_dashboard_cards",
    "update_dashboard_card",
    "delete_dashboard_card",
    "backend_status",
]
