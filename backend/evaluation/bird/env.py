"""Pointing the product at an evaluation-only workspace, *before* it is imported.

``app.db.models`` reads the settings while its classes are being defined, so
these variables must be in the environment before the first ``import app.db``.
:func:`configure` is therefore the first thing the CLI does, and everything that
imports the product does so lazily afterwards.

The evaluation runs on the product's **SQLite** backend (the same code path the
test-suite uses), in its own database file under ``work/``:

* the exported database is a plain file the harness can open read-only to score
  a prediction, with no server to provision;
* BIRD's gold SQL is SQLite SQL, so the prediction and the gold run on the same
  engine and compare like with like;
* the evaluation never touches the application's own sessions, users or
  PostgreSQL schemas.
"""

from __future__ import annotations

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_WORK_DIR = HERE / "work"


def configure(work_dir: Path | None = None, *, product_gates: bool = False) -> Path:
    """Set the environment for one evaluation process.  Returns the work directory."""

    work = Path(work_dir or DEFAULT_WORK_DIR).resolve()
    work.mkdir(parents=True, exist_ok=True)

    # Isolated stores: an eval database, uploaded copies, per-session table
    # pickles (``<upload_dir>/../sessions``) and exported databases
    # (``<upload_dir>/../exports``) all land inside ``work``.
    os.environ["DATABASE_URL"] = f"sqlite:///{work / 'eval.db'}"
    os.environ["UPLOAD_DIR"] = str(work / "uploads")
    os.environ.pop("READONLY_DATABASE_URL", None)

    # A truncated table would make the gold and the export disagree for reasons
    # that have nothing to do with the SQL.  formula_1's largest table has 420k
    # rows, over the product's 200k default.
    os.environ["MAX_ROWS_PER_TABLE"] = "2000000"
    os.environ["MAX_TABLES_PER_SOURCE"] = "50"

    # BIRD questions are independent: a "conversation" would only leak one
    # question into the next and add a rewrite call per question.
    os.environ["QUERY_CONVERSATION_TURNS"] = "0"

    # The product withholds a low-confidence answer; a benchmark that scores
    # "whatever SQL comes out" (as BIRD's own baselines are) must not, or the
    # arm with the gate is compared with an arm without one.  The model's
    # confidence is still recorded for every answer, so a selective-accuracy
    # curve can be drawn afterwards.  ``--product-gates`` restores the default.
    if not product_gates:
        os.environ["QUERY_CONFIDENCE_THRESHOLD"] = "0.0"

    # A hashing fallback would silently replace the embedder that retrieval
    # quality depends on.  Fail loudly instead.
    os.environ.setdefault("ALLOW_EMBEDDING_FALLBACK", "false")

    from app.core.config import get_settings  # noqa: PLC0415

    get_settings.cache_clear()
    return work
