"""CLI entry point.

    cd backend
    python -m evaluation.bird.run_eval --run-name bird300 --batch-size 100

Runs the two arms — the **raw** baseline (table and column names only) and the
product's **semantic** pipeline (a real session: semantic layer, three-index
retrieval, saved examples) — over the default sample of 100 simple + 100
moderate + 100 challenging BIRD dev questions, with BIRD's ``evidence`` sent to
both, and scores each answer by execution accuracy against the gold SQL.

The questions are asked **database by database** (the order of ``--databases``),
and inside each database simple, then moderate, then challenging.  With
``--batch-size 100`` an invocation asks the next 100 unfinished questions and stops;
the identical command, run again, asks the next 100 — three runs cover the 300.

Continuing.  ``--continue-from bird300`` starts a new run that is that run, carried on: its
results are copied in, every other question of the same databases is added after them (no
difficulty quota), and the semantic arm's saved examples are put back before the first added
question, so the new questions are asked with everything the earlier ones taught it::

    python -m evaluation.bird.run_eval --run-name bird_all --continue-from bird300

Give it the settings that run was started with (``--no-evidence`` for one made with it): the
two parts are one experiment, and a different setting is refused.  The earlier run's own
directory is left exactly as it was.

Everything lands in ``results/<run-name>/`` (see ``runner.py``).  The run is
checkpointed after every answer, and a copy of ``records.jsonl`` is kept under
``checkpoints/`` as each database finishes.  If it stops for any reason — out of
credit, Ctrl-C, a crash — run the *same command* again and it continues.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import dataset


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("what to run")
    g.add_argument("--run-name", default="bird300", help="results/<run-name>/ (default: bird300). Re-using a name resumes it.")
    g.add_argument("--simple", type=int, default=100, help="simple questions (default 100)")
    g.add_argument("--moderate", type=int, default=100, help="moderate questions (default 100)")
    g.add_argument("--challenging", type=int, default=100, help="challenging questions (default 100)")
    g.add_argument(
        "--databases",
        nargs="+",
        default=list(dataset.DEFAULT_DATABASES),
        help="BIRD databases to draw from, in the order they are asked (one finishes before the next starts)",
    )
    g.add_argument(
        "--continue-from",
        default=None,
        metavar="RUN",
        help="begin this run as the continuation of an earlier one: its results are copied in, every other "
        "question of --databases is added after them, and the saved examples of its answers are put back "
        "before the first of them. Needs that run's settings (--no-evidence, ...); --simple/--moderate/"
        "--challenging come from it, and --limit keeps only the first N added questions",
    )
    g.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="ask only the next N unfinished questions (both arms), then stop; run the same command "
        "again for the next N. Default: everything that is left",
    )
    g.add_argument("--seed", type=int, default=0, help="sampling and ordering seed")
    g.add_argument("--arms", nargs="+", choices=("raw", "semantic"), default=["raw", "semantic"])
    g.add_argument("--limit", type=int, default=None, help="keep only the first N sampled questions (smoke tests)")
    g.add_argument("--prepare-only", action="store_true", help="build the semantic layers and stop (no questions asked)")
    g.add_argument("--rebuild-sessions", action="store_true", help="discard and rebuild each database's semantic layer")
    g.add_argument(
        "--no-evidence",
        action="store_true",
        help="send neither arm BIRD's evidence — the product's real condition (nobody writes a hint beside "
        "each question); use a new --run-name, the comparison with an evidence run is between runs",
    )

    m = p.add_argument_group("the model call")
    m.add_argument("--max-attempts", type=int, default=3, help="generate→validate→run attempts per question (product default 3)")
    m.add_argument("--max-tokens", type=int, default=8192, help="completion budget per SQL call, reasoning included")
    m.add_argument("--reasoning-effort", choices=("low", "medium", "high"), default=None, help="OpenRouter reasoning effort for SQL generation (default: the model's own)")
    m.add_argument("--no-compact", action="store_true", help="send the payload exactly as the product builds it (no token trimming)")

    s = p.add_argument_group("the semantic arm")
    s.add_argument("--no-examples", action="store_true", help="do not save passing answers as examples")
    s.add_argument("--save-on", choices=("strict", "normalized", "relaxed"), default="relaxed", help="verdict a passing answer needs to be saved as an example")
    s.add_argument("--product-gates", action="store_true", help="apply the product's ambiguity check and confidence gate (default: off, so both arms answer every question)")

    c = p.add_argument_group("scoring")
    c.add_argument("--primary-metric", choices=("strict", "normalized", "relaxed"), default="relaxed", help="verdict reported as 'correct'")
    c.add_argument("--max-extra-columns", type=int, default=3, help="surplus columns the relaxed verdict tolerates")
    c.add_argument("--sql-timeout", type=float, default=30.0, help="seconds a query may run")

    r = p.add_argument_group("running")
    r.add_argument("--raw-workers", type=int, default=4, help="raw-arm questions in flight")
    r.add_argument("--db-workers", type=int, default=1, help="databases in flight at once (1 = strictly one database after another)")
    r.add_argument(
        "--min-model-written",
        type=float,
        default=None,
        help="refuse to run on a semantic layer whose column descriptions the model wrote less than this "
        "share of (the rest are the mechanical template); 0 switches the check off (default: 0.5)",
    )
    r.add_argument("--max-infra-failures", type=int, default=6, help="API failures in a row before stopping")
    r.add_argument("--work-dir", type=Path, default=None, help="evaluation database/exports (default: evaluation/bird/work)")
    return p.parse_args(argv)


def config_from_args(args: argparse.Namespace, model: str):
    """The :class:`~evaluation.bird.runner.RunConfig` a command line asks for.

    Called after ``env.configure`` (the runner imports the product, which reads its settings
    at import), which is why the import is here and not at the top.
    """

    from . import session_build
    from .runner import RunConfig

    return RunConfig(
        run_name=args.run_name,
        seed=args.seed,
        per_difficulty={"simple": args.simple, "moderate": args.moderate, "challenging": args.challenging},
        continue_from=args.continue_from,
        databases=list(args.databases),
        arms=list(args.arms),
        model=model,
        max_attempts=args.max_attempts,
        max_tokens=args.max_tokens,
        reasoning_effort=args.reasoning_effort,
        compact=not args.no_compact,
        product_gates=args.product_gates,
        save_examples=not args.no_examples,
        save_on=args.save_on,
        primary_metric=args.primary_metric,
        max_extra_columns=args.max_extra_columns,
        sql_timeout=args.sql_timeout,
        limit=args.limit,
        use_evidence=not args.no_evidence,
        raw_workers=args.raw_workers,
        db_workers=args.db_workers,
        max_infra_failures=args.max_infra_failures,
        batch_size=args.batch_size,
        min_model_written=(
            session_build.DEFAULT_MIN_MODEL_WRITTEN
            if args.min_model_written is None
            else args.min_model_written
        ),
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # The product reads its settings while it is being imported, so the
    # environment is set first and everything from ``app`` is imported after.
    from . import env

    work = env.configure(args.work_dir, product_gates=args.product_gates)

    from app.semantics.claude_client import ClaudeClient

    from . import session_build
    from .runner import Run

    claude = ClaudeClient()
    if not claude.available:
        print(
            f"error: the language model is unavailable ({claude.last_error}).\n"
            "Set OPENROUTER_API_KEY (environment or .env). Without a model the semantic layer would "
            "be built from templates and no question could be answered, which is not what this "
            "benchmark measures.",
            file=sys.stderr,
        )
        return 1

    cfg = config_from_args(args, claude.model)

    session_build.prepare_database()
    run = Run(cfg, claude, work, rebuild=args.rebuild_sessions)
    run.log(f"model {claude.model}; evaluation workspace {work}")
    if args.prepare_only:
        return run.build_only()
    return run.run()


if __name__ == "__main__":
    sys.exit(main())
