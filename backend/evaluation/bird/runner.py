"""One evaluation run: sampling, the two arms, checkpointing.

Layout of a run, ``results/<run-name>/``::

    config.json     every setting that defines the run (the resume contract)
    questions.json  the exact sampled questions, in the order they are asked
    records.jsonl   one line per (question, arm), appended and fsynced as each
                    finishes — this file *is* the checkpoint
    build.json      per-database semantic-layer build cost and data-drift report
    summary.json / records.csv / paired.csv    written by :mod:`evaluation.bird.report`

Re-running the same command resumes: finished (question, arm) pairs are skipped,
and each database's saved-example store is rebuilt from ``records.jsonl`` so the
semantic arm continues in exactly the state it stopped in.

Concurrency.  The raw arm's questions are independent and run across
``raw_workers`` threads.  The semantic arm is **stateful** — a question answered
correctly becomes an example for the questions after it — so each database is
one sequential lane in a fixed order (reproducible), and ``db_workers`` lanes run
at once.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import os
import platform
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from sqlalchemy import delete

from app.api import services
from app.core.config import get_settings
from app.db.base import get_session_factory
from app.db.models import IngestionSession, QueryExample, QueryHistoryRecord
from app.export.runner import resolve_target
from app.query.guard import referenced_names
from app.semantics.claude_client import SQL_SYSTEM_PROMPT
from app.semantics.embeddings import get_embedder

from . import dataset, metric, predict, report, schema_extract, session_build, sqlite_exec
from .llm import CreditsExhausted, EvalModel, UsageMeter, install_meter
from .session_build import BuildDegraded, EvalSession

RESULTS_DIR = Path(__file__).resolve().parent / "results"
SCHEMA_VERSION = 1

#: Settings that change what a *result means*.  A resumed run must agree on all
#: of them, or its records would mix two different experiments.
CONDITION_KEYS = (
    "seed",
    "per_difficulty",
    "databases",
    "arms",
    "model",
    "max_attempts",
    "max_tokens",
    "reasoning_effort",
    "compact",
    "product_gates",
    "save_examples",
    "save_on",
    "primary_metric",
    "max_extra_columns",
    "sql_timeout",
    "limit",
    "data_revision",
    "use_evidence",
    "retrieval",
)


def retrieval_settings() -> dict[str, Any]:
    """The product settings that decide what the semantic arm's model is shown.

    Part of the experiment: a run resumed after the cap on columns, the similarity an
    example needs or the embedding model changed would have half its semantic answers
    from one pipeline and half from another.
    """

    settings = get_settings()
    return {
        "embedding_model": settings.embedding_model,
        "max_tables": settings.query_max_tables,
        "max_ranked_columns": settings.query_max_ranked_columns,
        "max_business_rules": settings.query_max_business_rules,
        "max_examples": settings.query_max_examples,
        "example_min_similarity": settings.query_example_min_similarity,
        "example_match_threshold": settings.query_example_match_threshold,
    }


@dataclass
class RunConfig:
    run_name: str = "run"
    seed: int = 0
    per_difficulty: dict[str, int] = field(
        default_factory=lambda: {"simple": 100, "moderate": 100, "challenging": 100}
    )
    databases: list[str] = field(default_factory=lambda: list(dataset.DEFAULT_DATABASES))
    arms: list[str] = field(default_factory=lambda: ["raw", "semantic"])
    model: str = ""
    max_attempts: int = 3
    max_tokens: int = 8192
    reasoning_effort: str | None = None
    compact: bool = True
    product_gates: bool = False
    save_examples: bool = True
    #: Which verdict counts as "passes" for saving an example (strict|normalized|relaxed).
    save_on: str = "relaxed"
    primary_metric: str = "relaxed"
    max_extra_columns: int = metric.DEFAULT_MAX_EXTRA_COLUMNS
    sql_timeout: float = 30.0
    limit: int | None = None
    #: What the pipeline does to the data (``session_build.DATA_REVISION``).  Part of
    #: the experiment: answers measured on one export are not comparable with answers
    #: measured on another, so a run is never resumed across a change to it.
    data_revision: int = field(default_factory=lambda: session_build.DATA_REVISION)
    #: Whether BIRD's ``evidence`` is sent to the model.  On by default (BIRD's own
    #: protocol); off is the product's real condition, where nobody writes a hint
    #: beside each question, and what shows what the semantic layer adds by itself.
    use_evidence: bool = True
    retrieval: dict[str, Any] = field(default_factory=retrieval_settings)
    # -- not part of the experiment's definition --
    raw_workers: int = 4
    db_workers: int = 3
    max_infra_failures: int = 6


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parent,
            timeout=5,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _atomic_write(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    os.replace(tmp, path)


def _install_query_timeouts(paths: dict[str, Path], timeout: float) -> None:
    """Give the product's SQLite query execution the deadline it does not have.

    ``app.export.runner`` bounds a runaway query only on PostgreSQL
    (``statement_timeout``).  On SQLite one bad cross join would hold a lane for
    as long as it likes.  ``answer_question`` calls ``services.explain`` and
    ``services.run_readonly``; these stand in for them with the same contract
    (same exceptions to the retry loop, same ``(columns, row dicts, truncated)``
    result) plus a deadline, whose message goes back to the model like any other
    error.
    """

    def explain(session_id: str, statement: str) -> None:
        sqlite_exec.explain(paths[session_id], statement, timeout)

    def run_readonly(session_id: str, statement: str, row_cap: int):
        columns, rows, truncated = sqlite_exec.run_readonly(
            paths[session_id], statement, row_cap, timeout
        )
        return columns, [dict(zip(columns, row)) for row in rows], truncated

    services.explain = explain  # type: ignore[assignment]
    services.run_readonly = run_readonly  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------


class Run:
    def __init__(self, cfg: RunConfig, claude: Any, work: Path, rebuild: bool = False) -> None:
        self.cfg = cfg
        self.claude = claude
        self.work = work
        self.rebuild = rebuild
        self.dir = RESULTS_DIR / cfg.run_name
        self.meter = UsageMeter()
        install_meter(claude, self.meter, cfg.reasoning_effort)

        self.questions: list[dataset.BirdExample] = []
        self.done: dict[tuple[str, str, int], dict[str, Any]] = {}
        self._io_lock = threading.Lock()
        self._log_lock = threading.Lock()
        self.stop = threading.Event()
        self.fatal: str | None = None
        self._infra_streak = 0
        self.problems: list[str] = []
        self._bare: dict[str, list[dict[str, Any]]] = {}
        self._allowed: dict[str, set[str]] = {}
        self._export_paths: dict[str, Path] = {}
        self._started = time.monotonic()
        self._tally = {arm: [0, 0] for arm in ("raw", "semantic")}  # [done this session, correct]

    # -- output ------------------------------------------------------------
    def log(self, message: str) -> None:
        stamp = dt.datetime.now().strftime("%H:%M:%S")
        with self._log_lock:
            print(f"{stamp} {message}", file=sys.stderr, flush=True)

    @property
    def records_path(self) -> Path:
        return self.dir / "records.jsonl"

    def _commit(self, record: dict[str, Any]) -> None:
        key = (record["arm"], record["db_id"], record["question_id"])
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._io_lock:
            with self.records_path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            self.done[key] = record
            tally = self._tally[record["arm"]]
            tally[0] += 1
            tally[1] += int(bool(record["correct"]))
            self._infra_streak = 0

    def _infra_failure(self, arm: str, q: dataset.BirdExample, detail: str | None) -> None:
        """A question that got no answer because the *API* failed — never recorded.

        Scoring it would turn a rate limit or an empty wallet into a wrong
        answer; leaving it out means the next run simply asks it again.
        """

        with self._io_lock:
            self._infra_streak += 1
            streak = self._infra_streak
        self.log(f"[{arm}] {q.db_id} #{q.question_id}: API failure, not recorded — {detail}")
        if streak >= self.cfg.max_infra_failures:
            self.fatal = (
                f"{streak} questions in a row failed at the API ({detail}); stopping so no more "
                "time is spent. Progress is saved — re-run the same command to continue."
            )
            self.stop.set()

    # -- setup ---------------------------------------------------------------
    def prepare(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        config_path = self.dir / "config.json"
        questions_path = self.dir / "questions.json"
        stored = json.loads(config_path.read_text()) if config_path.exists() else None

        if stored is not None:
            old = stored["config"]
            new = asdict(self.cfg)
            changed = {k: (old.get(k), new.get(k)) for k in CONDITION_KEYS if old.get(k) != new.get(k)}
            if changed:
                lines = "\n".join(f"  {k}: stored {a!r}, now {b!r}" for k, (a, b) in changed.items())
                hint = ""
                if "data_revision" in changed:
                    hint = (
                        "\nThe pipeline has changed what the export holds since this run began (data "
                        "revision), so its earlier semantic answers were measured on different data: "
                        "start a new run (a new --run-name) instead of resuming it."
                    )
                elif "retrieval" in changed:
                    hint = (
                        "\nThe settings that decide what the semantic arm is shown (column cap, example "
                        "similarity, embedding model) differ from when this run began, so its earlier "
                        "semantic answers came from a different pipeline: start a new run (a new "
                        "--run-name) instead of resuming it."
                    )
                raise SystemExit(
                    f"run '{self.cfg.run_name}' already exists with different settings:\n{lines}{hint}\n"
                    "Use the same settings to resume it, or pick a new --run-name."
                )
            sample = json.loads(questions_path.read_text())
            by_id = {(e.db_id, e.question_id): e for e in dataset.load_all()}
            self.questions = [by_id[(row["db_id"], row["question_id"])] for row in sample]
            self.log(f"resuming run '{self.cfg.run_name}' ({len(self.questions)} questions)")
        else:
            examples = dataset.load_all()
            sample = dataset.sample_by_difficulty(
                examples, self.cfg.per_difficulty, self.cfg.databases, self.cfg.seed
            )
            if self.cfg.limit:
                sample = sample[: self.cfg.limit]
            self.questions = sample
            _atomic_write(
                questions_path,
                [
                    {
                        "question_id": e.question_id,
                        "db_id": e.db_id,
                        "difficulty": e.difficulty,
                        "question": e.question,
                        "evidence": e.evidence,
                        "gold_sql": e.gold_sql,
                    }
                    for e in sample
                ],
            )
            _atomic_write(
                config_path,
                {
                    "config": asdict(self.cfg),
                    "created_at": _now(),
                    "git_commit": _git_commit(),
                    "python": platform.python_version(),
                    "embedder": get_embedder().name,
                    "system_prompt_sha": hashlib.sha256(SQL_SYSTEM_PROMPT.encode()).hexdigest()[:12],
                    "question_count": len(sample),
                    "by_difficulty": {
                        d: sum(1 for e in sample if e.difficulty == d) for d in dataset.DIFFICULTIES
                    },
                    "by_database": {
                        db: sum(1 for e in sample if e.db_id == db)
                        for db in sorted({e.db_id for e in sample})
                    },
                },
            )
            self.log(f"new run '{self.cfg.run_name}': {len(sample)} questions")

        self.done = {
            (r["arm"], r["db_id"], r["question_id"]): r for r in report.load_records(self.records_path)
        }
        if self.done:
            self.log(f"{len(self.done)} (question, arm) results already on disk")
        for db_id in sorted({e.db_id for e in self.questions}):
            path = dataset.DATABASES_DIR / db_id / f"{db_id}.sqlite"
            self._bare[db_id] = schema_extract.bare_schema(path)
            self._allowed[db_id] = schema_extract.allowed_tables(path)

    def pending(self, arm: str) -> list[dataset.BirdExample]:
        return [q for q in self.questions if (arm, q.db_id, q.question_id) not in self.done]

    # -- record assembly -------------------------------------------------
    def _base_record(
        self,
        arm: str,
        q: dataset.BirdExample,
        *,
        outcome: str,
        sql: str | None,
        attempts: list[dict[str, Any]],
        error: str | None,
        scored: metric.ScoredQuery,
        usage: dict[str, Any],
        elapsed: float,
        explanation: str = "",
        confidence: float | None = None,
        source: str = "generated",
    ) -> dict[str, Any]:
        correct = bool(getattr(scored, self.cfg.primary_metric))
        return {
            "schema_version": SCHEMA_VERSION,
            "run_name": self.cfg.run_name,
            "recorded_at": _now(),
            "arm": arm,
            "db_id": q.db_id,
            "question_id": q.question_id,
            "difficulty": q.difficulty,
            "question": q.question,
            "evidence": q.evidence,
            "gold_sql": q.gold_sql,
            "model": self.cfg.model,
            "outcome": outcome,
            "source": source,
            "predicted_sql": sql,
            "explanation": explanation,
            "confidence": confidence,
            "attempts": attempts,
            "n_attempts": len(attempts),
            "first_attempt_ok": bool(attempts and attempts[0]["error"] is None),
            "generation_error": error,
            "correct": correct,
            **scored.to_dict(),
            "usage": usage,
            "elapsed_seconds": round(elapsed, 2),
        }

    def _log_result(self, record: dict[str, Any]) -> None:
        tally = self._tally[record["arm"]]
        mark = "PASS" if record["correct"] else "fail"
        detail = ""
        if not record["correct"]:
            detail = f" — {(record['predicted_error'] or record['generation_error'] or 'wrong result')[:70]}"
        elif record.get("extra_columns"):
            detail = f" (+{record['extra_columns']} extra col)"
        self.log(
            f"[{record['arm']:<8}] {record['db_id']:<21} #{record['question_id']:<5} "
            f"{record['difficulty']:<11} {mark}{detail}  "
            f"{record['usage']['total_tokens']:>6} tok {record['elapsed_seconds']:>5.0f}s  "
            f"(session {tally[1]}/{tally[0]})"
        )

    # -- raw arm ------------------------------------------------------------
    def raw_question(self, q: dataset.BirdExample) -> None:
        if self.stop.is_set():
            return
        model = EvalModel(
            self.claude,
            evidence=q.evidence if self.cfg.use_evidence else "",
            max_tokens=self.cfg.max_tokens,
            compact=self.cfg.compact,
        )
        started = time.monotonic()
        try:
            with self.meter.scope("raw") as usage:
                prediction = predict.generate_with_retry(
                    model,
                    q.question,
                    self._bare[q.db_id],
                    q.db_path,
                    self._allowed[q.db_id],
                    self.cfg.max_attempts,
                    self.cfg.sql_timeout,
                )
        except CreditsExhausted as exc:
            self.fatal = str(exc)
            self.stop.set()
            return
        if prediction["sql"] is None and usage.api_errors:
            self._infra_failure("raw", q, usage.last_api_error)
            return
        scored = metric.score(
            prediction["sql"], q.gold_sql, q.db_path, None, self.cfg.max_extra_columns, self.cfg.sql_timeout
        )
        record = self._base_record(
            "raw",
            q,
            outcome="answered" if prediction["sql"] else "no_sql",
            sql=prediction["sql"],
            attempts=prediction["attempts"],
            error=prediction["error"],
            scored=scored,
            usage=usage.to_dict(),
            elapsed=time.monotonic() - started,
            explanation=prediction["explanation"],
            confidence=prediction["confidence"],
        )
        self._commit(record)
        self._log_result(record)

    # -- semantic arm ----------------------------------------------------
    def _restore_examples(self, session: EvalSession, lane: list[dataset.BirdExample]) -> int:
        """Empty the session's example store, then replay what this run saved in it.

        Wiping first is what keeps one run's examples from leaking into the next;
        replaying from ``records.jsonl`` is what makes a resume continue in the
        same state — not whatever the eval database happens to hold.
        """

        Session = get_session_factory()
        with Session() as db:
            db.execute(delete(QueryExample).where(QueryExample.session_id == session.session_id))
            db.execute(delete(QueryHistoryRecord).where(QueryHistoryRecord.session_id == session.session_id))
            db.commit()
        replayed = 0
        if not self.cfg.save_examples:
            return 0
        for q in lane:
            record = self.done.get(("semantic", q.db_id, q.question_id))
            if record and record.get("example_saved"):
                if self._save_example(session, q.question, record["predicted_sql"], record["explanation"]) is None:
                    replayed += 1
        return replayed

    def _save_example(self, session: EvalSession, question: str, sql: str, explanation: str) -> str | None:
        """Save a verified pair the way the product's "save as example" does.  Returns an error or ``None``."""

        Session = get_session_factory()
        try:
            with Session() as db:
                record = db.get(IngestionSession, session.session_id)
                services.save_query_example(db, record, question, sql, explanation=explanation)
                db.commit()
        except ValueError as exc:
            return str(exc)
        return None

    def semantic_lane(self, db_id: str) -> None:
        lane = [q for q in self.questions if q.db_id == db_id]
        if not [q for q in lane if ("semantic", db_id, q.question_id) not in self.done]:
            return
        if self.stop.is_set():
            return
        try:
            session = session_build.build_session(
                db_id,
                claude=self.claude,
                meter=self.meter,
                work=self.work,
                log=self.log,
                rebuild=self.rebuild,
            )
        except CreditsExhausted as exc:
            self.fatal = str(exc)
            self.stop.set()
            return
        except BuildDegraded as exc:
            self.problems.append(str(exc))
            self.log(f"[semantic] {exc}")
            return
        self._export_paths[session.session_id] = session.export_path

        in_store = self._restore_examples(session, lane)
        if in_store:
            self.log(f"[{db_id}] restored {in_store} saved example(s) from earlier in this run")

        for q in lane:
            if self.stop.is_set():
                return
            if ("semantic", db_id, q.question_id) in self.done:
                continue
            try:
                record = self._semantic_question(session, q, in_store)
            except CreditsExhausted as exc:
                self.fatal = str(exc)
                self.stop.set()
                return
            if record is None:
                continue
            if record.get("example_saved"):
                in_store += 1
            self._commit(record)
            self._log_result(record)

    def _semantic_question(
        self, session: EvalSession, q: dataset.BirdExample, examples_in_store: int
    ) -> dict[str, Any] | None:
        # The evidence is BIRD's domain rule for this question, so the product is
        # given it as one (``extra_rules``).  It reads such a rule the way it reads a
        # recorded one: sent to the model ahead of the rest *and* searched for the
        # columns it names, which are then kept in the schema the model sees however
        # the retrieval scored them — "followed up at the outpatient clinic refers to
        # Admission = '-'" is no use if ``admission`` was cut from the context.  (The
        # raw arm has no retrieval to feed: its model wrapper sends the evidence itself.)
        evidence = session.names.translate(q.evidence) if self.cfg.use_evidence else ""
        model = EvalModel(
            self.claude,
            max_tokens=self.cfg.max_tokens,
            compact=self.cfg.compact,
            product_gates=self.cfg.product_gates,
        )
        Session = get_session_factory()
        started = time.monotonic()
        result: dict[str, Any] | None = None
        crashed: str | None = None
        with self.meter.scope("semantic") as usage:
            try:
                with Session() as db:
                    record = db.get(IngestionSession, session.session_id)
                    result = services.answer_question(
                        db,
                        record,
                        q.question,
                        claude=model,
                        extra_rules=[evidence] if evidence.strip() else [],
                    )
                    db.commit()
            except Exception as exc:  # a product bug is a result, not a reason to stop the run
                crashed = f"{type(exc).__name__}: {exc}"
                self.log(f"[semantic] {q.db_id} #{q.question_id} crashed:\n{traceback.format_exc()}")

        answered = bool(result and result.get("ok") and result.get("outcome") == "answered")
        if not answered and usage.api_errors:
            self._infra_failure("semantic", q, usage.last_api_error)
            return None

        sql = result["sql"] if answered else None
        scored = metric.score(
            sql,
            q.gold_sql,
            q.db_path,
            session.export_path,
            self.cfg.max_extra_columns,
            self.cfg.sql_timeout,
        )
        # Would the gold query itself still be right on the exported data?  Where it
        # is not, no model can pass by reading the data the way the gold does.
        preserved = metric.score(
            q.gold_sql, q.gold_sql, q.db_path, session.export_path, 0, self.cfg.sql_timeout
        )
        if preserved.gold_error:
            gold_on_export = "unavailable"
        elif preserved.predicted_error:
            gold_on_export = "not_runnable"  # BIRD's identifiers are not the export's
        elif preserved.strict:
            gold_on_export = "same"
        elif preserved.normalized:
            gold_on_export = "respelled"
        else:
            gold_on_export = "different"
        result = result or {}
        outcome = "answered" if answered else ("error" if crashed else result.get("outcome") or "rejected")
        record = self._base_record(
            "semantic",
            q,
            outcome=outcome,
            sql=sql,
            attempts=result.get("attempts") or [],
            error=crashed or (None if answered else result.get("error")),
            scored=scored,
            usage=usage.to_dict(),
            elapsed=time.monotonic() - started,
            explanation=result.get("explanation") or "",
            confidence=result.get("confidence"),
            source=result.get("source") or "generated",
        )

        # -- retrieval diagnostics: did the RAG put what the gold query needs in front of the model?
        retrieved = result.get("tables_used") or result.get("tables_considered") or []
        gold_tables, gold_columns = referenced_names(q.gold_sql, "sqlite")
        retrieved_original = session.names.original_tables(list(retrieved))
        sent = model.sent
        sent_columns = set(sent.get("column_names", []))
        if gold_columns and sent_columns:
            found = sum(1 for c in gold_columns if session.names.exported_column_names(c) & sent_columns)
            column_recall: float | None = round(found / len(gold_columns), 4)
        else:
            column_recall = None
        changed = [
            (f"{table}.{column}", kinds)
            for table, column, kinds in (session.build.get("drift") or {}).get("drifted_columns", [])
            if table.lower() in gold_tables and column.lower() in gold_columns
        ]
        drifted = [name for name, _ in changed]
        drift_kinds = sorted({kind for _, kinds in changed for kind in kinds})
        record.update(
            {
                "gold_on_export": gold_on_export,
                "gold_touches_drift": bool(drifted),
                "drift_kinds": drift_kinds,
                "drifted_columns_in_gold": drifted,
                "retrieved_tables": list(retrieved),
                "table_recall": (
                    round(len(gold_tables & retrieved_original) / len(gold_tables), 4)
                    if gold_tables
                    else None
                ),
                "context_tables": len(sent.get("tables", [])),
                "context_columns": sent.get("columns", 0),
                "column_recall": column_recall,
                "examples_sent": sent.get("examples", []),
                "rules_sent": sent.get("rules", 0),
                "examples_in_store": examples_in_store,
                "evidence_sent": evidence,
                "example_saved": False,
                "example_error": None,
            }
        )

        # -- a pass becomes an example for the questions that follow, as when a user saves one
        passed = bool(getattr(scored, self.cfg.save_on))
        if (
            self.cfg.save_examples
            and answered
            and passed
            and record["source"] != "saved_example"
        ):
            problem = self._save_example(session, q.question, sql, record["explanation"])
            record["example_saved"] = problem is None
            record["example_error"] = problem
        return record

    # -- driving ---------------------------------------------------------
    def run(self) -> int:
        self.prepare()
        arms = self.cfg.arms
        todo = {arm: self.pending(arm) for arm in arms}
        self.log("pending: " + ", ".join(f"{arm} {len(todo[arm])}" for arm in arms))
        if not any(todo.values()):
            self.log("nothing left to do")
            return 0

        # Longest lanes first: with fewer workers than databases, the biggest
        # database must not be the one left to start last.
        remaining = [q.db_id for q in todo.get("semantic", [])]
        sem_dbs = sorted(set(remaining), key=lambda db: (-remaining.count(db), db))
        futures = []
        sem_pool = ThreadPoolExecutor(max_workers=max(1, self.cfg.db_workers), thread_name_prefix="semantic")
        raw_pool = ThreadPoolExecutor(max_workers=max(1, self.cfg.raw_workers), thread_name_prefix="raw")
        interrupted = False
        try:
            if sem_dbs:
                # the export path is looked up per question through this map; lanes
                # fill it in as they finish building their database
                _install_query_timeouts(self._export_paths, self.cfg.sql_timeout)
            for db_id in sem_dbs:
                futures.append(sem_pool.submit(self._guard, self.semantic_lane, db_id))
            for q in todo.get("raw", []):
                futures.append(raw_pool.submit(self._guard, self.raw_question, q))
            pending = set(futures)
            while pending:
                try:
                    _, pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
                except KeyboardInterrupt:
                    if interrupted:
                        self.log("second interrupt — exiting now; everything finished so far is saved")
                        os._exit(130)
                    interrupted = True
                    self.stop.set()
                    self.log(
                        "interrupt — finishing the questions already in flight (Ctrl-C again to quit "
                        "immediately). Progress is saved."
                    )
        finally:
            sem_pool.shutdown(wait=True)
            raw_pool.shutdown(wait=True)
        return self.finish(interrupted)

    def build_only(self) -> int:
        """Build (or finish building) every database's semantic layer and stop.

        Useful before a long run: it spends the one-off build cost up front and
        prints each database's data-drift count, so a layer that came out wrong
        is found before any question is asked of it.
        """

        databases = sorted(set(self.cfg.databases))
        with ThreadPoolExecutor(max_workers=max(1, self.cfg.db_workers)) as pool:
            futures = {
                pool.submit(
                    session_build.build_session,
                    db_id,
                    claude=self.claude,
                    meter=self.meter,
                    work=self.work,
                    log=self.log,
                    rebuild=self.rebuild,
                ): db_id
                for db_id in databases
            }
            failed = 0
            for future, db_id in futures.items():
                try:
                    future.result()
                except CreditsExhausted as exc:
                    self.log(f"STOPPED: {exc}")
                    failed += 1
                except Exception as exc:
                    self.log(f"[{db_id}] build failed: {type(exc).__name__}: {exc}")
                    failed += 1
        totals = self.meter.totals().get("build", {})
        self.log(f"build tokens: {totals.get('total_tokens', 0):,} over {totals.get('calls', 0)} calls")
        return 1 if failed else 0

    def _guard(self, fn: Callable[..., None], *args: Any) -> None:
        try:
            fn(*args)
        except CreditsExhausted as exc:
            self.fatal = str(exc)
            self.stop.set()
        except Exception as exc:
            self.problems.append(f"{fn.__name__}{args!r}: {type(exc).__name__}: {exc}")
            self.log(f"{fn.__name__}{args!r} failed:\n{traceback.format_exc()}")

    def finish(self, interrupted: bool) -> int:
        outstanding = {arm: len(self.pending(arm)) for arm in self.cfg.arms}
        builds = session_build.read_manifest(self.work)
        _atomic_write(
            self.dir / "build.json",
            {db: builds.get(db) for db in sorted({q.db_id for q in self.questions}) if db in builds},
        )
        _atomic_write(self.dir / "usage.json", self.meter.totals())
        summary = report.write_outputs(self.dir)
        print("\n" + report.render(summary))
        total_left = sum(outstanding.values())
        elapsed = time.monotonic() - self._started
        self.log(f"session wall time {elapsed / 60:.1f} min; results in {self.dir}")
        if self.problems:
            self.log("problems:\n  " + "\n  ".join(self.problems))
        if self.fatal:
            self.log(f"STOPPED: {self.fatal}")
        if total_left:
            self.log(
                f"{total_left} (question, arm) pairs still to do ({outstanding}). Re-run the same "
                f"command to continue from where this stopped."
            )
            return 2
        return 0
