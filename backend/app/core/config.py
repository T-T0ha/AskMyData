"""Application configuration.

Everything the platform needs to run is expressed here so that the rest of the
code never reads ``os.environ`` directly.  Defaults target the docker-compose
PostgreSQL + pgvector service described in ``docker-compose.yml``.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_ROOT.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(PROJECT_ROOT / ".env", BACKEND_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- storage -------------------------------------------------------
    database_url: str = "postgresql+psycopg://semantic:semantic@localhost:5433/semanticlayer"
    upload_dir: Path = BACKEND_ROOT / "var" / "uploads"
    #: The credential Phase 5's query execution connects with — a role that
    #: physically cannot write or drop anything, so a generated SELECT that
    #: somehow got past the application-level guard (app/query/guard.py)
    #: still fails at the database, not merely at the application. ``None``
    #: (the default) falls back to ``database_url``, matching every backend
    #: that has no such role — SQLite, and a PostgreSQL instance nobody has
    #: provisioned one on yet.
    readonly_database_url: str | None = None
    #: The role name ``app/export/runner.py`` grants per-dataset SELECT to
    #: whenever a session's schema is (re)built. Provisioned once by
    #: ``ops/init-readonly-role.sql``, not by the application at request time.
    readonly_role_name: str = "semantic_readonly"

    # ---- Phase 0 : ingestion -------------------------------------------
    header_scan_rows: int = 5
    header_text_ratio: float = 0.60
    numeric_convert_threshold: float = 0.80
    date_parse_threshold: float = 0.80
    equivalence_threshold: float = 0.75

    # ---- Phase 0 : database sources ------------------------------------
    #: Rows read per table from a database or dump.  A source table larger than
    #: this is loaded up to the limit and flagged as truncated rather than
    #: silently sampled — semantic detection stays valid on a prefix, but the
    #: user has to know the profile is not the whole table.
    max_rows_per_table: int = 200_000
    #: Tables accepted from one connection in a single ingest.  A production
    #: schema with hundreds of tables should be narrowed by the user first.
    max_tables_per_source: int = 50
    #: Seconds to wait for a database connection before giving up.
    source_connect_timeout: int = 10
    #: Seconds a single read may run on the source server (PostgreSQL/MySQL).
    source_statement_timeout: int = 60

    # ---- Phase 1 : field semantics -------------------------------------
    date_type_threshold: float = 0.85
    nominal_unique_ratio: float = 0.20
    taxonomy_sample_values: int = 10
    profile_top_k: int = 10

    # ---- Phase 2 : co-planned cleaning ---------------------------------
    plan_sample_values: int = 5
    validation_preview_rows: int = 10

    # ---- Phase 5 : natural language query ------------------------------
    #: Tables sent to the SQL prompt, at most.  A written-down bound rather than
    #: "all of them" so a wide database cannot blow the prompt budget; a
    #: business dataset's table count rarely reaches it.
    query_max_tables: int = 8
    #: Non-key columns kept by embedding similarity, at most — SemTabla's "top
    #: 15 most relevant column metadata records".  Key columns are never
    #: subject to this cap; see app.query.retrieval.
    query_max_ranked_columns: int = 15
    #: generate → validate → EXPLAIN attempts before giving up and reporting
    #: the last error to the user instead of a fifth guess.
    query_max_attempts: int = 3
    #: Rows returned to the browser, at most, regardless of what the query
    #: itself asked for — a forgotten LIMIT must not pull a 200k-row table into
    #: one HTTP response.
    query_row_cap: int = 1000
    #: PostgreSQL only: aborts a runaway query (a bad join, an expensive scan)
    #: instead of holding the connection for the request's full timeout.
    query_statement_timeout_seconds: int = 20
    #: Business rules retrieved into the SQL prompt, at most, ranked by
    #: similarity to the question — the third retrieval index alongside tables
    #: and columns.
    query_max_business_rules: int = 5
    #: Verified question/SQL examples retrieved into the SQL prompt, at most.
    query_max_examples: int = 3
    #: Below this, a generated query is withheld rather than shown — an
    #: abstention, not an answer.  Same shape as the taxonomy classifier's own
    #: confidence floor (0.5): a written-down number a caller can reason about,
    #: not a magic threshold buried in an if-statement.
    query_confidence_threshold: float = 0.45

    # ---- Phase 6 : dashboard --------------------------------------------
    #: Questions kept per session for the history sidebar, newest first —
    #: claude.md's "last 20 questions, clickable to re-run".
    query_history_limit: int = 20

    # ---- OpenRouter --------------------------------------------------------
    #: The module is still called ``claude_client.py`` / ``ClaudeClient`` —
    #: renaming it would ripple into routes, services, the frontend's status
    #: labels and the ``taxonomy_source == "claude"`` values already stored in
    #: the database — but the API calls it actually makes go through
    #: OpenRouter's OpenAI-compatible endpoint.
    openrouter_api_key: str | None = None
    openrouter_model: str = "nex-agi/nex-n2.5-pro:free"
    #: Current models think before answering, and max_tokens caps thinking
    #: plus the answer together — a budget sized only for the JSON reply
    #: truncates mid-object and the response fails to parse.
    openrouter_max_tokens: int = 16000
    openrouter_timeout_seconds: float = 120.0

    # ---- authentication -------------------------------------------------
    #: scrypt work factors.  ``n`` dominates both time and memory: the hash
    #: costs roughly ``128 * n * r`` bytes, so 2**14 with r=8 is ~16 MB and
    #: ~60 ms per verification on a laptop.  Raising ``n`` invalidates nothing —
    #: the parameters are stored in every hash and old ones keep verifying.
    auth_scrypt_n: int = 1 << 14
    auth_scrypt_r: int = 8
    auth_scrypt_p: int = 1
    #: Long rather than complex: length is the only password rule that reliably
    #: buys entropy, and composition rules mostly buy "Password1!".
    auth_min_password_length: int = 10
    #: How long a login lasts before the user has to sign in again.
    auth_token_ttl_hours: int = 24 * 14
    #: Failed logins tolerated per email within the window before the account
    #: stops answering.  Per-process and in-memory — see app/auth/service.py.
    auth_max_failed_logins: int = 8
    auth_failed_login_window_seconds: int = 900
    #: Set false to close registration once the intended users have accounts.
    auth_registration_open: bool = True

    # ---- embeddings ----------------------------------------------------
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    allow_embedding_fallback: bool = True

    @property
    def is_postgres(self) -> bool:
        return self.database_url.startswith("postgres")

    def resolved_api_key(self) -> str | None:
        return (
            self.openrouter_api_key
            or os.environ.get("OPENROUTER_API_KEY")
            or os.environ.get("openrouter_api_key")
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    return settings
