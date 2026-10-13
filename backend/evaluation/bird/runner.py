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

Order.  Questions are asked **database by database** — in the order the databases
are listed — and inside a database **simple, then moderate, then challenging**.
Both arms follow that order; a database is finished (and a copy of
``records.jsonl`` kept under ``checkpoints/``) before the next one starts.  With
``batch_size`` an invocation asks only the next ``batch_size`` unfinished questions
and stops; the same command, run again, asks the next ones.

Continuing.  ``continue_from="bird300"`` makes a new run that *is* that run, carried on: its
results are copied in, every other question of the chosen databases is added (no difficulty
quota), and the semantic arm's example store is rebuilt from the carried answers before the
first added question — the same restore a resume does — so the added questions are asked
with everything the earlier ones taught it.  The earlier run's directory is not touched.
Together they are the whole dev set of those databases, each question once, in one run.

Concurrency.  The raw arm's questions are independent and run across
``raw_workers`` threads.  The semantic arm is **stateful** — a question answered
correctly becomes an example for the questions after it — so a database is one
sequential lane in that fixed order (reproducible).  ``db_workers`` databases are
in flight at once; the default, 1, is strictly one database after another.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import os
import platform
import shutil
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
    #: The earlier run this one continues (not a setting of the experiment: how this run began,
    #: and only read when it begins).  Its results are carried in and every other question of
    #: ``databases`` is asked after them, with the examples its answers saved put back, so the two
    #: are one run.  The experiment's settings must be the earlier run's; ``per_difficulty`` is
    #: then taken from it, and ``limit`` keeps only the first N *added* questions (smoke tests).
    continue_from: str | None = None
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
    #: Databases in flight at once.  1 = one database finishes before the next starts.
    db_workers: int = 1
    max_infra_failures: int = 6
    #: Ask at most this many more questions in this invocation (both arms each), then
    #: stop; ``None`` asks everything that is left.  Where one invocation stops and the
    #: next begins changes no answer, so it is not part of the experiment's definition.
    batch_size: int | None = None
    #: Smallest share of a semantic layer's column descriptions the model must have
    #: written (``session_build.check_model_written``); 0 accepts any layer.
    min_model_written: float = session_build.DEFAULT_MIN_MODEL_WRITTEN


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


def _comparable(key: str, value: Any) -> Any:
    """A stored setting in the form the resume contract compares it in.

    Which databases a run draws from is part of the experiment; the *order* they are
    listed in is not — each has its own session, so no answer depends on which
    database went first.  Changing the order between two invocations is allowed.
    """

    if key == "databases" and isinstance(value, list):
        return sorted(value)
    return value


def _prompt_sha() -> str:
    """A short fingerprint of the SQL prompt both arms are given."""

    return hashlib.sha256(SQL_SYSTEM_PROMPT.encode()).hexdigest()[:12]


def _changed_settings(old: dict[str, Any], new: dict[str, Any], ignore: tuple[str, ...] = ()) -> dict[str, tuple]:
    """The settings of the experiment on which two configs disagree: ``{key: (old, new)}``."""

    return {
        k: (old.get(k), new.get(k))
        for k in CONDITION_KEYS
        if k not in ignore and _comparable(k, old.get(k)) != _comparable(k, new.get(k))
    }


def _read_questions(run_name: str, examples: list[dataset.BirdExample]) -> list[dataset.BirdExample]:
    """The questions an earlier run asked, in the order it stored them."""

    by_id = {(e.db_id, e.question_id): e for e in examples}
    rows = json.loads((RESULTS_DIR / run_name / "questions.json").read_text())
    return [by_id[(row["db_id"], row["question_id"])] for row in rows]


def draw_questions(cfg: RunConfig, examples: list[dataset.BirdExample]) -> list[dataset.BirdExample]:
    """The questions a new run asks, before the asking order is applied.

    A fresh run draws its quota of each difficulty.  A continuation is the earlier run's
    questions, as they were stored, followed by every other question of the databases.
    """

    if cfg.continue_from:
        earlier = _read_questions(cfg.continue_from, examples)
        added = dataset.rest_of(examples, {(e.db_id, e.question_id) for e in earlier}, cfg.databases, cfg.seed)
        return earlier + (added[: cfg.limit] if cfg.limit else added)
    sample = dataset.sample_by_difficulty(examples, cfg.per_difficulty, cfg.databases, cfg.seed)
    return sample[: cfg.limit] if cfg.limit else sample  # a random subset: taken before the ordering


def _install_query_timeouts(paths: dict[str, Path], timeout: float) -> None:
    """Give the product's SQLite query execution the deadline it does not have.

    ``app.export.runner`` bounds a runaway query only on PostgreSQL
    (``statement_timeout``).  On SQLite one bad cross join would hold a lane for
    as long as it likes.  ``answer_question`` calls ``explain`` and
    ``run_readonly`` as names in ``app.api.services.query``; these stand in for
    them with the same contract (same exceptions to the retry loop, same
    ``(columns, row dicts, truncated)`` result) plus a deadline, whose message
    goes back to the model like any other error.  They are installed on that
    module, where ``answer_question`` looks them up — set on the ``services``
    package they would never be called.
    """

    def explain(session_id: str, statement: str) -> None:
        sqlite_exec.explain(paths[session_id], statement, timeout)

    def run_readonly(session_id: str, statement: str, row_cap: int):
        columns, rows, truncated = sqlite_exec.run_readonly(
            paths[session_id], statement, row_cap, timeout
        )
        return columns, [dict(zip(columns, row)) for row in rows], truncated

    services.query.explain = explain  # type: ignore[assignment]
    services.query.run_readonly = run_readonly  # type: ignore[assignment]


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
    def _check_continuation(self, name: str) -> dict[str, Any]:
        """The earlier run's ``config.json`` if this run may continue it; otherwise ``SystemExit``.

        Everything is judged before anything is created, so a refused continuation leaves no
        half-made run behind.  The two parts are one experiment only if every setting that
        decides what an answer means is the same, including the SQL prompt.
        """

        source = RESULTS_DIR / name
        if not ((source / "config.json").exists() and (source / "questions.json").exists()):
            raise SystemExit(
                f"--continue-from {name!r}: {source} has no config.json and questions.json, so there is "
                "nothing to continue. Name a run that has been started."
            )
        earlier = json.loads((source / "config.json").read_text())
        old = earlier["config"]
        changed = _changed_settings(old, asdict(self.cfg), ignore=("per_difficulty", "limit"))
        if changed:
            lines = "\n".join(f"  {k}: '{name}' has {a!r}, this command gives {b!r}" for k, (a, b) in changed.items())
            hint = ""
            if {"data_revision", "retrieval"} & set(changed):
                hint = (
                    "\nThe pipeline itself changed since that run (data revision or retrieval settings): "
                    "its answers cannot be continued, start a fresh run instead."
                )
            raise SystemExit(
                f"--continue-from {name!r}: a continuation is one run, so it needs that run's settings, "
                f"and these differ:\n{lines}{hint}\n"
                "Give the flags that run was started with (for one made with --no-evidence, add --no-evidence)."
            )
        then = earlier.get("system_prompt_sha")
        if then and then != _prompt_sha():
            raise SystemExit(
                f"--continue-from {name!r}: the SQL prompt has changed since that run (then {then}, now "
                f"{_prompt_sha()}), so the two parts would not be one experiment. Start a fresh run."
            )
        self.cfg.per_difficulty = dict(old.get("per_difficulty") or {})  # that run's quota; it draws nothing here
        return earlier

    def _carry_records(self, name: str) -> int:
        """Put the earlier run's results into this run's ``records.jsonl``: clean lines, written atomically."""

        records = report.load_records(RESULTS_DIR / name / "records.jsonl")
        tmp = self.records_path.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.records_path)
        return len(records)

    def prepare(self) -> None:
        config_path = self.dir / "config.json"
        questions_path = self.dir / "questions.json"
        stored = json.loads(config_path.read_text()) if config_path.exists() else None
        earlier = (
            self._check_continuation(self.cfg.continue_from)
            if stored is None and self.cfg.continue_from
            else None
        )
        self.dir.mkdir(parents=True, exist_ok=True)

        if stored is not None:
            old = stored["config"]
            changed = _changed_settings(
                old, asdict(self.cfg), ignore=("per_difficulty", "limit") if old.get("continue_from") else ()
            )
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
            then = stored.get("system_prompt_sha")
            if then and then != _prompt_sha():
                raise SystemExit(
                    f"run '{self.cfg.run_name}' was begun under another SQL prompt (then {then}, now "
                    f"{_prompt_sha()}): the answers already on disk and the ones asked now would not be one "
                    "experiment. Start a new run (a new --run-name), or put the prompt back."
                )
            sample = json.loads(questions_path.read_text())
            by_id = {(e.db_id, e.question_id): e for e in dataset.load_all()}
            # Ordered afresh: a run begun before questions were asked database by
            # database stored them in a shuffle, and still finishes in the new order.
            self.questions = dataset.run_order(
                [by_id[(row["db_id"], row["question_id"])] for row in sample], self.cfg.databases
            )
            self.log(f"resuming run '{self.cfg.run_name}' ({len(self.questions)} questions)")
        else:
            examples = dataset.load_all()
            sample = dataset.run_order(draw_questions(self.cfg, examples), self.cfg.databases)
            self.questions = sample
            # The carried results first and the config last: the config's existence is what
            # makes this a run, so an interruption before it simply starts the copy again.
            carried = self._carry_records(self.cfg.continue_from) if earlier is not None else 0
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
                    "system_prompt_sha": _prompt_sha(),
                    "question_count": len(sample),
                    "by_difficulty": {
                        d: sum(1 for e in sample if e.difficulty == d) for d in dataset.DIFFICULTIES
                    },
                    "by_database": {
                        db: sum(1 for e in sample if e.db_id == db)
                        for db in sorted({e.db_id for e in sample})
                    },
                    **(
                        {
                            "continued_from": self.cfg.continue_from,
                            "carried_records": carried,
                            "carried_questions": earlier.get("question_count"),
                        }
                        if earlier is not None
                        else {}
                    ),
                },
            )
            if earlier is not None:
                self.log(
                    f"new run '{self.cfg.run_name}': the continuation of '{self.cfg.continue_from}' — "
                    f"{carried} of its answers carried in, {len(sample)} questions in all "
                    f"({len(sample) - int(earlier.get('question_count') or 0)} added)"
                )
            else:
                self.log(f"new run '{self.cfg.run_name}': {len(sample)} questions, sampled")

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

    @property
    def database_order(self) -> list[str]:
        """The databases of the run, in the order they are asked."""

        return list(dict.fromkeys(q.db_id for q in self.questions))

    def outstanding(self) -> list[dataset.BirdExample]:
        """Questions some arm has still to answer, in the order they are asked."""

        return [
            q
            for q in self.questions
            if any((arm, q.db_id, q.question_id) not in self.done for arm in self.cfg.arms)
        ]

    def batch(self) -> list[dataset.BirdExample]:
        """What this invocation asks: the next ``batch_size`` outstanding questions, or all of them."""

        left = self.outstanding()
        size = self.cfg.batch_size
        return left[:size] if size and size > 0 else left

    def preflight(self) -> list[str]:
        """Reasons not to start, found before a single question is asked or token spent.

        A semantic layer built under an older data revision, or one the model barely
        wrote, is found here for *every* database of the run — not hours later, when
        the lane that needs it finally starts.
        """

        if "semantic" not in self.cfg.arms:
            return []
        manifest = session_build.read_manifest(self.work)
        problems: list[str] = []
        for db_id in self.database_order:
            entry = manifest.get(db_id)
            if not entry:
                continue  # not built yet: built (and judged) when its lane starts
            if entry.get("session_id") and entry.get("stage") != "exported":
                # A build that was interrupted or rejected.  Left to its lane it would be
                # finished (and could fail again) hours into the run, after the other
                # databases had been paid for.
                problems.append(
                    f"{db_id}: its semantic layer was never finished (built as far as "
                    f"'{entry.get('stage')}'). Finish it first, so it is judged before any question "
                    f"is asked:\n    python -m evaluation.bird.run_eval --databases {db_id} "
                    "--prepare-only --min-model-written 0.9"
                )
                continue
            try:
                session_build.check_layer(db_id, entry, self.cfg.min_model_written)
            except session_build.BuildDegraded as exc:
                problems.append(str(exc))
        return problems

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

    def semantic_lane(self, db_id: str, only: list[dataset.BirdExample] | None = None) -> None:
        """Ask ``only`` (default: everything still unasked) of ``db_id``, one after another.

        ``lane`` is the database's whole share of the run, in asking order: the saved
        examples of *all* of its earlier answers are put back before the first question,
        whichever invocation gave them, so a batch boundary never changes what the
        model is shown.
        """

        lane = [q for q in self.questions if q.db_id == db_id]
        asked = lane if only is None else only
        todo = [q for q in asked if ("semantic", db_id, q.question_id) not in self.done]
        if not todo:
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
                min_model_written=self.cfg.min_model_written,
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

        for q in todo:
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
    def _announce(self, batch: list[dataset.BirdExample], outstanding: int) -> None:
        """What this invocation will ask, database by database, before anything is spent."""

        batch_note = f"batch of {self.cfg.batch_size}" if self.cfg.batch_size else "no batch limit"
        self.log(
            f"this invocation asks {len(batch)} of the {outstanding} outstanding questions "
            f"({batch_note}), both arms, in this order:"
        )
        for db_id in dict.fromkeys(q.db_id for q in batch):
            of_db = [q for q in batch if q.db_id == db_id]
            bands = ", ".join(
                f"{sum(1 for q in of_db if q.difficulty == d)} {d}"
                for d in dataset.DIFFICULTIES
                if any(q.difficulty == d for q in of_db)
            )
            self.log(f"    {db_id:<22} {len(of_db):>3}   ({bands})")

    def run_database(
        self, db_id: str, todo: dict[str, list[dataset.BirdExample]], raw_pool: ThreadPoolExecutor
    ) -> None:
        """Both arms over this invocation's questions of one database, then a checkpoint copy.

        The raw arm's questions are queued in asking order across ``raw_workers``
        threads while the semantic lane works through the same questions one by one;
        the unit is over only when both are, so the next database never starts early.
        """

        raw = [q for q in todo.get("raw", []) if q.db_id == db_id]
        semantic = [q for q in todo.get("semantic", []) if q.db_id == db_id]
        self.log(f"[{db_id}] starting — {len(raw)} raw, {len(semantic)} semantic question(s)")
        raw_futures = [raw_pool.submit(self._guard, self.raw_question, q) for q in raw]
        try:
            if semantic:
                self._guard(self.semantic_lane, db_id, semantic)
        finally:
            wait(raw_futures)
        left = sum(
            1
            for arm, questions in todo.items()
            for q in questions
            if q.db_id == db_id and (arm, db_id, q.question_id) not in self.done
        )
        saved = self._snapshot(db_id)
        self.log(
            f"[{db_id}] "
            + ("finished" if not left else f"stopped with {left} answer(s) still to get")
            + (f"; records copied to {saved.relative_to(self.dir)}" if saved else "")
        )

    def _snapshot(self, tag: str) -> Path | None:
        """A second copy of every answer so far, kept beside ``records.jsonl``.

        ``records.jsonl`` is already durable — each answer is appended and fsynced the
        moment it exists — so this is not what makes a crash survivable.  It is what
        makes the *file* survivable: an edit, a bad merge or a deleted line has a copy
        from the end of the last database to go back to.  ``checkpoints/`` is written
        atomically and never pruned.
        """

        if not self.records_path.exists():
            return None
        folder = self.dir / "checkpoints"
        folder.mkdir(exist_ok=True)
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        with self._io_lock:
            answers = len(self.done)
            target = folder / f"records-{stamp}-{answers:04d}answers-{tag}.jsonl"
            tmp = target.with_suffix(".tmp")
            shutil.copyfile(self.records_path, tmp)
            os.replace(tmp, target)
        return target

    def run(self) -> int:
        self.prepare()
        arms = self.cfg.arms
        batch = self.batch()
        todo = {
            arm: [q for q in batch if (arm, q.db_id, q.question_id) not in self.done] for arm in arms
        }
        self.log("pending: " + ", ".join(f"{arm} {len(self.pending(arm))}" for arm in arms))
        if not any(todo.values()):
            self.log("nothing left to do")
            return 0

        problems = self.preflight()
        if problems:
            self.log("NOT STARTED — nothing was asked and nothing was spent:\n  " + "\n  ".join(problems))
            return 1
        self._announce(batch, len(self.outstanding()))

        # One unit per database, in asking order.  With ``db_workers`` = 1 the next
        # database starts only when the previous one has finished both arms.
        databases = [db for db in self.database_order if any(q.db_id == db for q in batch)]
        futures = []
        unit_pool = ThreadPoolExecutor(max_workers=max(1, self.cfg.db_workers), thread_name_prefix="database")
        raw_pool = ThreadPoolExecutor(max_workers=max(1, self.cfg.raw_workers), thread_name_prefix="raw")
        interrupted = False
        try:
            if todo.get("semantic"):
                # the export path is looked up per question through this map; lanes
                # fill it in as they finish building their database
                _install_query_timeouts(self._export_paths, self.cfg.sql_timeout)
            for db_id in databases:
                futures.append(unit_pool.submit(self._guard, self.run_database, db_id, todo, raw_pool))
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
            unit_pool.shutdown(wait=True)
            raw_pool.shutdown(wait=True)
        return self.finish(interrupted, batch)

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
                    min_model_written=self.cfg.min_model_written,
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

    def finish(self, interrupted: bool, batch: list[dataset.BirdExample] | None = None) -> int:
        """Write the reports and say what is left.

        Exit codes: 0 — everything this invocation set out to ask has an answer (more
        batches may remain; that is not a failure); 2 — it stopped short (credit, an
        API failure, an interrupt, a layer that would not build), so the same command
        should be run again.
        """

        asked = self.questions if batch is None else batch
        outstanding = {arm: len(self.pending(arm)) for arm in self.cfg.arms}
        batch_left = sum(
            1
            for q in asked
            for arm in self.cfg.arms
            if (arm, q.db_id, q.question_id) not in self.done
        )
        builds = session_build.read_manifest(self.work)
        _atomic_write(
            self.dir / "build.json",
            {db: builds.get(db) for db in sorted({q.db_id for q in self.questions}) if db in builds},
        )
        _atomic_write(self.dir / "usage.json", self.meter.totals())
        summary = report.write_outputs(self.dir)
        print("\n" + report.render(summary))
        saved = self._snapshot("interrupted" if interrupted else "end")
        total_left = sum(outstanding.values())
        elapsed = time.monotonic() - self._started
        self.log(f"session wall time {elapsed / 60:.1f} min; results in {self.dir}")
        if saved:
            self.log(f"records copied to {saved.relative_to(self.dir)}")
        if self.problems:
            self.log("problems:\n  " + "\n  ".join(self.problems))
        if self.fatal:
            self.log(f"STOPPED: {self.fatal}")
        if batch_left:
            self.log(
                f"{batch_left} (question, arm) pairs of this invocation are still unanswered "
                f"({outstanding} left in the run). Fix what is reported above, then re-run the "
                "same command to continue from where this stopped."
            )
            return 2
        if total_left:
            by_database = {
                db: sum(
                    1
                    for q in self.outstanding()
                    if q.db_id == db
                )
                for db in self.database_order
            }
            self.log(
                f"this invocation is complete. {len(self.outstanding())} question(s) are still to ask "
                f"({', '.join(f'{db} {n}' for db, n in by_database.items() if n)}). "
                "Run the same command again for the next batch."
            )
            return 0
        self.log("the run is complete — every question has an answer from every arm.")
        return 0
