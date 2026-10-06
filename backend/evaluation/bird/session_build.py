"""One product session per BIRD database — built the way a user builds one.

A BIRD ``.sqlite`` file goes through the same stages the UI walks a user through,
by calling the same service functions the API routes call:

    upload (Phase 0)  →  field semantics (1)  →  relationships + table profiles (3)
                      →  semantic-layer export (4)

after which :func:`app.api.services.answer_question` (Phase 5) can be asked
questions of the session exactly as the browser does.  Phase 2 (cleaning) is not
run: it rewrites the data, and BIRD's gold SQL is defined on the data as shipped.

Every stage commits and is recorded in ``work/sessions.json``, so a run that dies
(credit exhausted, Ctrl-C) resumes at the stage it stopped in instead of paying
for the LLM calls of the earlier ones again.  A stage whose LLM calls *failed* is
never accepted: the product degrades silently to template descriptions when a
call fails, and a benchmark must not quietly measure that degraded layer.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from sqlalchemy import event, select

from app.api import services
from app.auth import service as auth
from app.cleaning import store as table_store
from app.db.base import get_engine, get_session_factory, init_db
from app.db.models import ColumnSemantics, IngestionSession, SemanticLayerExport, SheetRecord
from app.export.runner import read_semantic_layer, resolve_target
from app.semantics.descriptions import is_template_description

from . import dataset
from .llm import UsageMeter
from .metric import normalized_value, strict_value

EVAL_EMAIL = "bird-eval@eval.local"
STAGES = ("ingested", "semantics", "relationships", "exported")

_manifest_lock = threading.Lock()


#: Bump when a change to ingestion, typing or the export changes what a built
#: session's database *holds*.  A session built under an older revision stored
#: its tables then (ingestion writes them, and re-exporting cannot undo it), so
#: reusing it would measure the old pipeline under the new one's name.
#:
#: 1  a source's missing-data markers were turned into NULL and its numeric
#:    columns exported as VARCHAR when few of their values were distinct.
#: 2  a database source keeps its stored values, and the export keeps the numeric
#:    and DATE types the source declared.
#: 3  the export also keeps a declared TEXT column as text, however Phase 1 reads what
#:    is in it — ``'218.300'``, ``'2019-09-14'`` and ``'true'`` were being stored as a
#:    float, a timestamp and a ``1``.
DATA_REVISION = 3


class BuildDegraded(RuntimeError):
    """A build stage's LLM calls failed, so the layer it produced is not the product's."""


class StaleSession(BuildDegraded):
    """A built session predates a change to what the pipeline does to the data."""


def check_revision(db_id: str, entry: dict[str, Any]) -> None:
    """Refuse to reuse a session built under another data revision."""

    if not entry.get("session_id"):
        return
    built = entry.get("data_revision", 1)
    if built != DATA_REVISION:
        raise StaleSession(
            f"{db_id}: this session was built under data revision {built} and the pipeline is now "
            f"at {DATA_REVISION} (a database source's stored values and declared types, text "
            "included, are kept now), so its tables differ from the BIRD file in ways the "
            "current pipeline would not. Re-run with --rebuild-sessions; the semantic build is "
            "repeated, which uses credit."
        )


# ---------------------------------------------------------------------------
# workspace
# ---------------------------------------------------------------------------


#: How long a lane waits for the one write lock SQLite has before giving up.
#:
#: Each build stage is a single transaction, so the lane running it holds the
#: write lock from its first write to its commit — about a minute for formula_1's
#: ingestion on an idle machine, and longer while three lanes compete for the same
#: CPU.  The first real run waited 60 s for it, gave up ("database is locked") in
#: toxicology's export, and lost that database's semantic arm.  Waiting is always
#: the right answer to a lock the other lanes will release, so wait for minutes.
BUSY_TIMEOUT_MS = 10 * 60 * 1000


def tune_sqlite(engine: Any, busy_timeout_ms: int = BUSY_TIMEOUT_MS) -> None:
    """Make every connection of ``engine`` wait for a held lock instead of failing."""

    @event.listens_for(engine, "connect")
    def _pragmas(connection: Any, _record: Any) -> None:  # pragma: no cover - driver hook
        cursor = connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute(f"PRAGMA busy_timeout={int(busy_timeout_ms)}")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.close()


def prepare_database() -> None:
    """Tune SQLite for several threads writing one file, then create the tables."""

    tune_sqlite(get_engine())
    init_db()


def _manifest_path(work: Path) -> Path:
    return work / "sessions.json"


def read_manifest(work: Path) -> dict[str, Any]:
    try:
        return json.loads(_manifest_path(work).read_text())
    except (OSError, ValueError):
        return {}


def _update_manifest(work: Path, db_id: str, **fields: Any) -> dict[str, Any]:
    with _manifest_lock:
        manifest = read_manifest(work)
        entry = manifest.setdefault(db_id, {})
        entry.update(fields)
        tmp = _manifest_path(work).with_suffix(".tmp")
        tmp.write_text(json.dumps(manifest, indent=2, default=str))
        os.replace(tmp, _manifest_path(work))
        return entry


_user_lock = threading.Lock()


def _ensure_user(db: Any) -> str:
    """The evaluation's own account.  Lanes start together, so creation is serialised
    *and committed inside the lock* — a second lane must see the row, not race to insert it."""

    with _user_lock:
        user = auth.get_user_by_email(db, EVAL_EMAIL)
        if user is None:
            user = auth.register(db, EVAL_EMAIL, secrets.token_urlsafe(24), "BIRD evaluation")
            db.commit()
        return user.id


def _discard(db: Any, record: IngestionSession) -> None:
    session_id = record.id
    db.delete(record)
    table_store.drop_store(session_id)
    services.forget_export(session_id)


def _release_memory(session_id: str) -> None:
    """Forget the in-memory frames of a session (they stay on disk).

    ``get_store`` keeps every session's DataFrames for the life of the process;
    after the build only the exported database is read, and formula_1 alone is
    several hundred thousand rows.
    """

    with table_store._registry_lock:
        table_store._stores.pop(session_id, None)


# ---------------------------------------------------------------------------
# names: BIRD's original identifiers -> the exported ones
# ---------------------------------------------------------------------------


@dataclass
class NameMap:
    """How BIRD's table/column names read in the product's exported database.

    Phase 0 sanitises names (``Free Meal Count (K-12)`` → ``free_meal_count_k_12``)
    and the export may do so again.  BIRD's evidence text, and its gold SQL, use
    the original names; the semantic arm's schema uses the exported ones.
    """

    #: exported table -> BIRD table name
    table_origin: dict[str, str] = field(default_factory=dict)
    #: (BIRD table lower, BIRD column lower) -> (exported table, exported column)
    columns: dict[tuple[str, str], tuple[str, str]] = field(default_factory=dict)
    #: BIRD column label (lower) -> every exported name it became
    labels: dict[str, set[str]] = field(default_factory=dict)
    _pattern: re.Pattern[str] | None = None
    _replace: dict[str, str] = field(default_factory=dict)

    def _build_rewrite(self) -> None:
        """Names that changed by more than case — the only ones evidence needs fixed.

        A label that became several different exported names (the same header in
        two tables, disambiguated differently) is left alone rather than guessed.
        """

        replace: dict[str, str] = {}
        for label, exported in self.labels.items():
            if len(exported) != 1:
                continue
            (new,) = exported
            if new.lower() != label:
                replace[label] = new
        for exported_table, original in self.table_origin.items():
            if exported_table.lower() != original.lower():
                replace.setdefault(original.lower(), exported_table)
        self._replace = replace
        if not replace:
            self._pattern = None
            return
        ordered = sorted(replace, key=len, reverse=True)
        self._pattern = re.compile(
            r"(?<![A-Za-z0-9_])(" + "|".join(re.escape(label) for label in ordered) + r")(?![A-Za-z0-9_])",
            re.IGNORECASE,
        )

    def translate(self, text: str) -> str:
        """``text`` with BIRD's original identifiers replaced by the exported ones."""

        if not text:
            return text
        if self._pattern is None and not self._replace:
            self._build_rewrite()
        if self._pattern is None:
            return text
        return self._pattern.sub(lambda m: self._replace.get(m.group(1).lower(), m.group(1)), text)

    def original_tables(self, exported: list[str]) -> set[str]:
        return {self.table_origin.get(name, name).lower() for name in exported}

    def exported_column_names(self, original_column: str) -> set[str]:
        """What a BIRD column name is called in the export (itself, if never renamed)."""

        return self.labels.get(original_column.lower()) or {original_column.lower()}


def build_name_map(session_id: str) -> NameMap:
    names = NameMap()
    layer = read_semantic_layer(session_id)
    exported = {
        (str(row["source_table"]), str(row["source_column"])): (
            str(row["table_name"]),
            str(row["column_name"]),
        )
        for row in layer
    }
    Session = get_session_factory()
    with Session() as db:
        sheets = (
            db.execute(select(SheetRecord).where(SheetRecord.session_id == session_id)).scalars().all()
        )
        for sheet in sheets:
            native = sheet.native_schema or {}
            original_table = str(native.get("source_name") or sheet.table_name)
            ingested_table = sheet.table_name
            for column in native.get("columns", []):
                key = (ingested_table, str(column.get("name")))
                if key not in exported:
                    continue
                exported_table, exported_column = exported[key]
                original_column = str(column.get("source_name"))
                names.table_origin[exported_table] = original_table
                names.columns[(original_table.lower(), original_column.lower())] = (
                    exported_table,
                    exported_column,
                )
                names.labels.setdefault(original_column.lower(), set()).add(exported_column)
    names._build_rewrite()
    return names


# ---------------------------------------------------------------------------
# what the pipeline did to the data
# ---------------------------------------------------------------------------


def _storage_class(connection: sqlite3.Connection, table: str, column: str) -> str | None:
    """``numeric`` or ``text`` — whichever SQLite storage class most of the column's values have."""

    counts = {"numeric": 0, "text": 0}
    for kind, n in connection.execute(
        f'SELECT typeof("{column}"), count(*) FROM "{table}" WHERE "{column}" IS NOT NULL GROUP BY 1'
    ):
        if kind in ("integer", "real"):
            counts["numeric"] += n
        elif kind == "text":
            counts["text"] += n
    if not any(counts.values()):
        return None
    return max(counts, key=counts.get)


def column_drift(original: Path, exported: Path, names: NameMap, limit: int = 60) -> dict[str, Any]:
    """Columns the pipeline changed between BIRD's file and the export, and how.

    The semantic arm queries the export, so anything the pipeline rewrote can make
    a correct query return something other than the gold.  Three kinds, in order of
    how much they matter:

    ``values``
        the set of distinct values changed after the scorer's own normalisation —
        a ``'na'`` or ``'-'`` read as missing, a trimmed string.  No metric can
        forgive this.
    ``retyped``
        the storage class flipped between number and text.  Values still *read* the
        same, but ``ORDER BY``, ``MAX``/``MIN`` and ``> 500`` now compare as text, and
        ``normalized`` / ``relaxed`` cannot see it.
    ``reformatted``
        values differ only in how they are spelled (a date gaining ``00:00:00``, ``1.0``
        becoming ``1``).  Forgiven in the result, but not in a ``WHERE ... = '2002-09-08'``.

    Measured once per database, so a failure can be put down to the SQL or the data.
    """

    drifted: list[dict[str, Any]] = []
    by_kind = {"values": 0, "retyped": 0, "reformatted": 0}
    compared = 0
    with sqlite3.connect(f"file:{original}?mode=ro", uri=True) as src, sqlite3.connect(
        f"file:{exported}?mode=ro", uri=True
    ) as dst:
        for (table, column), (exported_table, exported_column) in sorted(names.columns.items()):
            # The BIRD file keeps the declared case of its identifiers; SQLite
            # resolves them case-insensitively, so the lower-cased key works.
            try:
                raw_before = [r[0] for r in src.execute(f'SELECT DISTINCT "{column}" FROM "{table}"')]
                raw_after = [
                    r[0] for r in dst.execute(f'SELECT DISTINCT "{exported_column}" FROM "{exported_table}"')
                ]
                class_before = _storage_class(src, table, column)
                class_after = _storage_class(dst, exported_table, exported_column)
            except sqlite3.Error:
                continue
            compared += 1
            before = {normalized_value(v) for v in raw_before}
            after = {normalized_value(v) for v in raw_after}
            kinds: list[str] = []
            if before != after:
                kinds.append("values")
            if class_before and class_after and class_before != class_after:
                kinds.append("retyped")
            if not kinds and {strict_value(v) for v in raw_before} != {strict_value(v) for v in raw_after}:
                kinds.append("reformatted")
            if not kinds:
                continue
            for kind in kinds:
                by_kind[kind] += 1
            lost, gained = before - after, after - before
            drifted.append(
                {
                    "table": table,
                    "column": column,
                    "exported_as": f"{exported_table}.{exported_column}",
                    "kinds": kinds,
                    "stored_as": f"{class_before} -> {class_after}",
                    "values_lost": len(lost),
                    "values_gained": len(gained),
                    "lost_sample": [str(v) for v in list(lost)[:3]],
                    "gained_sample": [str(v) for v in list(gained)[:3]],
                }
            )
    return {
        "columns_compared": compared,
        "columns_drifted": len(drifted),
        "by_kind": by_kind,
        "drifted": drifted[:limit],
        # Every pair with its kinds, uncapped: the runner flags the questions whose gold SQL names one.
        "drifted_columns": [[d["table"], d["column"], d["kinds"]] for d in drifted],
    }


# ---------------------------------------------------------------------------
# the build
# ---------------------------------------------------------------------------


@dataclass
class EvalSession:
    db_id: str
    session_id: str
    export_path: Path
    names: NameMap
    build: dict[str, Any]


def _stage_ingest(db: Any, record: IngestionSession, source: Path) -> None:
    sheets = services.ingest_file(db, record, source)
    truncated = [s.table_name for s in sheets if "truncated_read" in json.dumps(s.issues or [])]
    if truncated:
        raise RuntimeError(
            f"{source.stem}: {truncated} were read only in part (raise MAX_ROWS_PER_TABLE in "
            "evaluation/bird/env.py) — the gold SQL and the export would disagree"
        )
    services.detect_and_store_equivalences(db, record)


def _stage_semantics(db: Any, record: IngestionSession, claude: Any) -> None:
    services.run_field_semantics(db, record, claude=claude)


def _stage_relationships(db: Any, record: IngestionSession, claude: Any) -> None:
    services.detect_relationships(db, record, claude=claude)


def _stage_export(db: Any, record: IngestionSession, claude: Any) -> None:
    services.export_semantic_layer(db, record, claude=claude)


def _exported_ok(session_id: str) -> bool:
    Session = get_session_factory()
    with Session() as db:
        export = db.execute(
            select(SemanticLayerExport).where(SemanticLayerExport.session_id == session_id)
        ).scalar_one_or_none()
    if export is None:
        return False
    target = resolve_target(session_id)
    try:
        return target.path is not None and target.path.exists()
    finally:
        target.dispose()


def _open(db_id: str, session_id: str, build: dict[str, Any]) -> EvalSession:
    target = resolve_target(session_id)
    path = target.path
    target.dispose()
    if target.dialect != "sqlite" or path is None:
        raise RuntimeError("the BIRD evaluation needs the SQLite backend (evaluation/bird/env.py)")
    return EvalSession(
        db_id=db_id,
        session_id=session_id,
        export_path=path,
        names=build_name_map(session_id),
        build=build,
    )


def build_session(
    db_id: str,
    *,
    claude: Any,
    meter: UsageMeter,
    work: Path,
    log: Callable[[str], None],
    rebuild: bool = False,
) -> EvalSession:
    """Return the product session for ``db_id``, building or finishing it as needed."""

    source = dataset.DATABASES_DIR / db_id / f"{db_id}.sqlite"
    if not source.exists():
        raise FileNotFoundError(f"{source} — run `python -m evaluation.bird.download` first")

    Session = get_session_factory()
    entry = {} if rebuild else read_manifest(work).get(db_id, {})
    check_revision(db_id, entry)
    session_id = entry.get("session_id")

    with Session() as db:
        user_id = _ensure_user(db)
        db.commit()
        existing = db.get(IngestionSession, session_id) if session_id else None
        if existing is not None and entry.get("stage") == "exported" and _exported_ok(session_id):
            log(f"[{db_id}] reusing the built session ({session_id[:8]})")
            return _open(db_id, session_id, entry)
        if existing is not None and entry.get("stage") == "exported":
            # The record says "done" but the export itself is gone: redo that stage.
            entry = _update_manifest(work, db_id, stage="relationships")
        if existing is None:
            # Anything an earlier, abandoned build left under this name goes first:
            # ``ingest_file`` refuses a source another session of the account holds.
            for stale in services.list_sessions(db, user_id):
                if stale.name == f"bird:{db_id}":
                    _discard(db, stale)
            record = services.create_session(db, f"bird:{db_id}", user_id)
            session_id = record.id
            entry = _update_manifest(
                work,
                db_id,
                session_id=session_id,
                stage="created",
                stages={},
                started_at=time.time(),
                data_revision=DATA_REVISION,
            )
        db.commit()

    log(f"[{db_id}] building the semantic layer ({session_id[:8]}) from {entry.get('stage')}")
    steps: list[tuple[str, Callable[[Any, IngestionSession], None]]] = [
        ("ingested", lambda db, rec: _stage_ingest(db, rec, source)),
        ("semantics", lambda db, rec: _stage_semantics(db, rec, claude)),
        ("relationships", lambda db, rec: _stage_relationships(db, rec, claude)),
        ("exported", lambda db, rec: _stage_export(db, rec, claude)),
    ]
    done = STAGES.index(entry["stage"]) if entry.get("stage") in STAGES else -1
    stage_info: dict[str, Any] = dict(entry.get("stages") or {})
    for index, (name, run) in enumerate(steps):
        if index <= done:
            continue
        started = time.monotonic()
        with meter.scope("build") as usage, Session() as db:
            record = db.get(IngestionSession, session_id)
            run(db, record)
            if usage.api_errors:
                raise BuildDegraded(
                    f"{db_id}: {usage.api_errors} model call(s) failed during '{name}' "
                    f"({usage.last_api_error}); the layer would silently fall back to templates. "
                    "Nothing was saved for this stage — re-run to retry it."
                )
            db.commit()
        stage_info[name] = {
            "seconds": round(time.monotonic() - started, 1),
            "usage": usage.to_dict(),
        }
        _update_manifest(work, db_id, stage=name, stages=stage_info)
        log(f"[{db_id}] {name} done in {stage_info[name]['seconds']}s ({usage.calls} model calls)")
        _release_memory(session_id)

    session = _open(db_id, session_id, {})
    drift = column_drift(source, session.export_path, session.names)
    exported_tables = {r["table_name"] for r in read_semantic_layer(session_id)}
    bird_tables = {t.lower() for t in _bird_tables(source)}
    missing = sorted(bird_tables - {session.names.table_origin.get(t, t).lower() for t in exported_tables})
    totals = {key: sum(s["usage"][key] for s in stage_info.values()) for key in
              ("calls", "prompt_tokens", "completion_tokens", "reasoning_tokens", "total_tokens")}
    totals["cost_usd"] = round(sum(s["usage"]["cost_usd"] for s in stage_info.values()), 6)
    quality = model_written_share(session_id)
    entry = _update_manifest(
        work,
        db_id,
        stage="exported",
        built_at=time.time(),
        model_written=quality,
        drift=drift,
        tables_missing_from_export=missing,
        build_usage=totals,
    )
    if missing:
        log(f"[{db_id}] WARNING: tables absent from the export: {missing}")
    if quality["columns"] and not quality["described_by_model"]:
        log(
            f"[{db_id}] WARNING: none of the {quality['columns']} column descriptions was written by "
            "the model — this is the template-only layer. Check the model and its replies "
            "(--rebuild-sessions after fixing) before trusting a semantic-arm result."
        )
    log(
        f"[{db_id}] ready — model wrote {quality['described_by_model']}/{quality['columns']} descriptions and "
        f"{quality['labelled_by_model']} labels; {drift['columns_drifted']}/{drift['columns_compared']} columns "
        f"altered by the export ({drift['by_kind']['values']} values, {drift['by_kind']['retyped']} retyped, "
        f"{drift['by_kind']['reformatted']} reformatted); build used {totals['total_tokens']:,} tokens"
    )
    session.build = entry
    return session


def model_written_share(session_id: str) -> dict[str, int]:
    """How much of the layer the *model* wrote, as opposed to the rule-based fallback.

    The product falls back to rules and templates when a model reply cannot be
    used, without any HTTP error for the harness to see.  A layer with no
    model-written descriptions or labels is the template-only layer, and a
    benchmark of "the semantic layer" must not be quietly measuring that.
    """

    Session = get_session_factory()
    with Session() as db:
        rows = db.execute(
            select(ColumnSemantics).where(ColumnSemantics.session_id == session_id)
        ).scalars().all()
    return {
        "columns": len(rows),
        "described_by_model": sum(1 for r in rows if not is_template_description(r.semantic_description)),
        "labelled_by_model": sum(1 for r in rows if r.taxonomy_source == "claude"),
    }


def _bird_tables(source: Path) -> list[str]:
    with sqlite3.connect(f"file:{source}?mode=ro", uri=True) as conn:
        return [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        ]
