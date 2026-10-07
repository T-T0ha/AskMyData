"""Check the whole harness end to end — for free.

    cd backend
    python -m evaluation.bird.selftest

No network, no API key, no credit.  A fake model that answers with BIRD's gold
SQL (and, for some questions, with a deliberately wrong or broken query) stands
in for OpenRouter; everything else is the real thing: the product's pipeline
builds a session for two small BIRD databases, the semantic arm asks it
questions through ``answer_question``, passing answers are saved as examples,
scores are computed against the gold, and the run is checkpointed.

The run is **interrupted partway** (the fake model raises the same stop signal
an HTTP 402 produces) and then resumed, which is the property that matters most
when real credit runs out.  Everything happens in a temporary directory.
"""

from __future__ import annotations

import itertools
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="bird_selftest_"))
    try:
        return _run(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _run(tmp: Path) -> int:
    from . import env

    work = env.configure(tmp / "work")

    from app.semantics.claude_client import DESCRIBE_DATASET_SYSTEM_PROMPT, SQL_SYSTEM_PROMPT, ClaudeClient

    from . import dataset, report, runner, session_build
    from .llm import CreditsExhausted

    if not dataset.DEV_JSON.exists():
        print("the BIRD dev set is not downloaded: python -m evaluation.bird.download", file=sys.stderr)
        return 1
    gold = {e.question: e for e in dataset.load_all()}

    class FakeModel(ClaudeClient):
        def __init__(self) -> None:
            super().__init__(api_key="")
            self.sql_payloads: list[dict] = []
            self.stop_after: int | None = None

        @property
        def available(self) -> bool:  # type: ignore[override]
            return True

        @property
        def model(self) -> str:  # type: ignore[override]
            return "selftest-fake"

        def _complete(self, system, user, max_tokens=None, json_mode=True, reasoning=True):  # noqa: ANN001
            if system != SQL_SYSTEM_PROMPT:
                return "{}"  # every labelling/description call degrades to the rule-based path
            payload = json.loads(user)
            self.sql_payloads.append(payload)
            if self.stop_after is not None and len(self.sql_payloads) > self.stop_after:
                raise CreditsExhausted("simulated 402: out of credit")
            example = gold[payload["question"]]
            if example.question_id % 6 == 0:
                sql = "SELECT 1"  # a wrong answer
            elif example.question_id % 5 == 0 and "previous_attempt" not in payload:
                sql = "SELECT nope FROM nowhere"  # broken first attempt, right after the error it is shown
            else:
                sql = example.gold_sql
            return json.dumps({"sql": sql, "explanation": "selftest", "confidence": 0.9})

    class DescribingModel(FakeModel):
        """Also writes the column descriptions, in the shape the product keeps."""

        def _complete(self, system, user, max_tokens=None, json_mode=True, reasoning=True):  # noqa: ANN001
            if system == DESCRIBE_DATASET_SYSTEM_PROMPT:
                tables = json.loads(user)["tables"]
                entries = [
                    {"table": t["table"], "name": c["name"], "description": f"The {c['name']} held in {t['table']}."}
                    for t in tables
                    for c in t["columns"]
                ]
                return json.dumps({"columns": entries})
            return super()._complete(system, user, max_tokens, json_mode, reasoning)

    # What is checked here is that a saved example *reaches* a later question; whether
    # two BIRD questions are alike enough to be shown to one another is the product's
    # floor (``query_example_min_similarity``), which a handful of fake-model questions
    # would rarely clear.
    from app.core.config import get_settings

    get_settings().query_example_min_similarity = 0.0

    runner.RESULTS_DIR = tmp / "results"
    session_build.prepare_database()
    fake = FakeModel()
    cfg = runner.RunConfig(
        run_name="selftest",
        model="selftest-fake",
        per_difficulty={"simple": 5, "moderate": 4, "challenging": 3},
        databases=["toxicology", "superhero"],
        raw_workers=2,
        db_workers=2,
        min_model_written=0.0,  # the fake writes no descriptions: every layer here is the template
    )
    total = sum(cfg.per_difficulty.values())
    records_path = runner.RESULTS_DIR / "selftest" / "records.jsonl"
    failures: list[str] = []

    def check(condition: bool, what: str) -> None:
        print(("  ok   " if condition else "  FAIL ") + what, file=sys.stderr)
        if not condition:
            failures.append(what)

    print("\n== pass 1: the model gives out after a few calls ==", file=sys.stderr)
    fake.stop_after = 14
    code = runner.Run(cfg, fake, work).run()
    partial = report.load_records(records_path)
    check(code == 2, "an interrupted run exits with code 2 (work remains)")
    check(0 < len(partial) < 2 * total, f"some, not all, answers were kept ({len(partial)}/{2 * total})")

    print("\n== pass 2: the same command, resumed ==", file=sys.stderr)
    fake.stop_after = None
    fake.sql_payloads.clear()
    code = runner.Run(cfg, fake, work).run()
    final = report.load_records(records_path)
    lines = [line for line in records_path.read_text().splitlines() if line.strip()]
    semantic = [r for r in final if r["arm"] == "semantic"]
    raw = [r for r in final if r["arm"] == "raw"]

    check(code == 0, "the resumed run finishes with code 0")
    check(len(raw) == total and len(semantic) == total, f"{total} questions answered by each arm")
    check(len(lines) == len(final), "no question was answered twice across the interruption")
    check(
        {r["difficulty"] for r in final} == {"simple", "moderate", "challenging"},
        "all three difficulty bands were asked",
    )
    check(any(r["correct"] for r in raw) and any(not r["correct"] for r in raw), "raw arm: passes and failures both scored")
    check(any(r["n_attempts"] > 1 for r in final), "a broken first attempt was retried with its error")
    check(all(r["attempts"] for r in final if r["outcome"] == "answered"), "every answer carries its attempts")
    check(sum(1 for r in semantic if r["example_saved"]) > 0, "passing semantic answers were saved as examples")
    check(
        any(r["examples_sent"] for r in semantic),
        "a later semantic question was shown a saved example (the retrieval index)",
    )
    with_evidence = [p for p in fake.sql_payloads if gold[p["question"]].evidence.strip()]
    check(
        bool(with_evidence)
        and all(p["business_rules"][0] == gold[p["question"]].evidence.strip() for p in with_evidence),
        "BIRD's evidence reached every model call that has any, first among the business rules",
    )
    check(
        all(not p["business_rules"] for p in fake.sql_payloads if not gold[p["question"]].evidence.strip()),
        "a question with no evidence is sent no business rule",
    )
    check(
        all(r["evidence_sent"] is not None for r in semantic) and all("evidence" in r for r in raw),
        "both arms record the evidence they were given",
    )
    check(
        all(r.get("table_recall") is not None for r in semantic if r["source"] == "generated"),
        "retrieval recall is recorded for the semantic arm",
    )
    check(
        all(r.get("gold_on_export") in {"same", "respelled", "different", "not_runnable"} for r in semantic),
        "every semantic answer records whether its gold query still gives the same answer on the export",
    )
    check(
        {"strict", "normalized", "relaxed"} <= set(final[0]),
        "all three execution-accuracy verdicts are stored per record",
    )
    check(
        all((runner.RESULTS_DIR / "selftest" / name).exists() for name in
            ("config.json", "questions.json", "summary.json", "records.csv", "paired.csv", "build.json")),
        "config, sample, summary, both CSV tables and the build report were written",
    )

    print("\n== pass 3: the same questions with the evidence switched off (--no-evidence) ==", file=sys.stderr)
    fake.sql_payloads.clear()
    code = runner.Run(
        runner.RunConfig(
            run_name="selftest_noevidence",
            model="selftest-fake",
            per_difficulty=cfg.per_difficulty,
            databases=cfg.databases,
            raw_workers=2,
            db_workers=2,
            min_model_written=0.0,
            use_evidence=False,
        ),
        fake,
        work,
    ).run()
    bare = report.load_records(runner.RESULTS_DIR / "selftest_noevidence" / "records.jsonl")
    check(code == 0 and len(bare) == 2 * total, "the run without evidence finishes and answers every question")
    check(
        bool(fake.sql_payloads) and all(not p["business_rules"] for p in fake.sql_payloads),
        "without evidence no model call carries a business rule",
    )
    check(
        all(r["evidence_sent"] == "" for r in bare if r["arm"] == "semantic"),
        "the semantic arm records that it was sent no evidence",
    )


    print("\n== pass 4: database by database, five questions per invocation ==", file=sys.stderr)
    fake.sql_payloads.clear()
    batches_dir = runner.RESULTS_DIR / "selftest_batches"
    batch_records = batches_dir / "records.jsonl"
    codes: list[int] = []
    added: list[int] = []
    seen = 0
    for _ in range(6):  # three are needed; the cap only guards against a loop that never ends
        code = runner.Run(
            runner.RunConfig(
                run_name="selftest_batches",
                model="selftest-fake",
                per_difficulty=cfg.per_difficulty,
                databases=["superhero", "toxicology"],  # against the alphabet: the order is the flag's
                raw_workers=1,  # one raw worker: its completion order is its asking order
                db_workers=1,
                min_model_written=0.0,
                batch_size=5,
            ),
            fake,
            work,
        ).run()
        now = len(report.load_records(batch_records)) if batch_records.exists() else 0
        codes.append(code)
        added.append(now - seen)
        seen = now
        if now == 2 * total:
            break
    ordered = [json.loads(line) for line in batch_records.read_text().splitlines() if line.strip()]
    final4 = report.load_records(batch_records)
    check(all(c == 0 for c in codes), f"every invocation that finishes its batch exits 0, more left or not ({codes})")
    check(added == [10, 10, 4], f"each invocation asks exactly its five questions of both arms, the last what remains ({added})")
    check(
        [db for db, _ in itertools.groupby(r["db_id"] for r in ordered)] == ["superhero", "toxicology"],
        "one database finishes both arms before the next starts, in the order they were listed",
    )
    ranks = {name: index for index, name in enumerate(dataset.DIFFICULTIES)}
    in_order = True
    for arm in ("semantic", "raw"):
        for db in ("superhero", "toxicology"):
            seq = [ranks[r["difficulty"]] for r in ordered if r["arm"] == arm and r["db_id"] == db]
            in_order = in_order and seq == sorted(seq)
    check(in_order, "inside a database, simple then moderate then challenging, in both arms")
    check(
        len(ordered) == len(final4) == 2 * total,
        "no question was answered twice across the invocations",
    )
    check(
        {(r["db_id"], r["question_id"]) for r in final4} == {(r["db_id"], r["question_id"]) for r in final},
        "the batches together asked exactly the questions the unbatched run did",
    )
    first_batch = ordered[:10]
    split = [
        db
        for db in ("superhero", "toxicology")
        if any(r["db_id"] == db for r in ordered[:10]) and any(r["db_id"] == db for r in ordered[10:])
    ]
    carried = True
    for db in split:
        saved_before = sum(1 for r in first_batch if r["arm"] == "semantic" and r["db_id"] == db and r["example_saved"])
        later = [r for r in ordered[10:] if r["arm"] == "semantic" and r["db_id"] == db]
        carried = carried and all(r["examples_in_store"] >= saved_before for r in later)
    check(
        bool(split) and carried,
        f"a database split across two invocations resumes with the examples the first one saved ({split})",
    )
    snapshots = sorted((batches_dir / "checkpoints").glob("records-*.jsonl"))
    check(
        len(snapshots) >= 3 and snapshots[-1].read_text() == batch_records.read_text(),
        f"a copy of the records is kept as each database ends ({len(snapshots)} copies, the last identical to the live file)",
    )

    print("\n== pass 4b: that run, continued with the other questions of its databases ==", file=sys.stderr)
    fake.sql_payloads.clear()
    source_text = batch_records.read_text()
    code = runner.Run(
        runner.RunConfig(
            run_name="selftest_more",
            model="selftest-fake",
            databases=["superhero", "toxicology"],
            continue_from="selftest_batches",
            limit=6,  # a smoke-sized slice of what is added; the real thing adds every other question
            raw_workers=1,
            db_workers=1,
            min_model_written=0.0,
        ),
        fake,
        work,
    ).run()
    more = report.load_records(runner.RESULTS_DIR / "selftest_more" / "records.jsonl")
    carried_keys = {(r["arm"], r["db_id"], r["question_id"]) for r in final4}
    carried = [r for r in more if (r["arm"], r["db_id"], r["question_id"]) in carried_keys]
    added = [r for r in more if (r["arm"], r["db_id"], r["question_id"]) not in carried_keys]
    check(
        code == 0 and len(carried) == 2 * total and len(added) == 12,
        f"every earlier answer is carried in and six more questions are asked, both arms "
        f"({len(carried)} carried, {len(added)} new, exit {code})",
    )
    check(batch_records.read_text() == source_text, "the earlier run's own records are exactly as they were")
    check(
        not {(r["db_id"], r["question_id"]) for r in added} & {(r["db_id"], r["question_id"]) for r in final4},
        "none of the added questions is one the earlier run asked",
    )
    saved_earlier: dict[str, int] = {}
    for r in final4:
        if r["arm"] == "semantic" and r["example_saved"]:
            saved_earlier[r["db_id"]] = saved_earlier.get(r["db_id"], 0) + 1
    first_store: dict[str, int] = {}
    for r in added:
        if r["arm"] == "semantic":
            first_store.setdefault(r["db_id"], r["examples_in_store"])
    check(
        bool(saved_earlier) and all(first_store.get(db, 0) >= n for db, n in saved_earlier.items()),
        f"the first added question of each database is asked with the examples the earlier run saved "
        f"(saved earlier {saved_earlier}; in store at the first added question {first_store})",
    )

    print("\n== pass 5: a layer the model barely wrote is refused before anything is asked ==", file=sys.stderr)
    fake.sql_payloads.clear()
    code = runner.Run(
        runner.RunConfig(
            run_name="selftest_degraded",
            model="selftest-fake",
            per_difficulty=cfg.per_difficulty,
            databases=cfg.databases,
            raw_workers=1,
            min_model_written=0.5,
        ),
        fake,
        work,
    ).run()
    degraded = runner.RESULTS_DIR / "selftest_degraded" / "records.jsonl"
    check(
        code == 1 and not fake.sql_payloads and not (degraded.exists() and degraded.read_text().strip()),
        "built layers whose descriptions are all template stop the run (exit 1) before a question is asked",
    )


    print("\n== pass 6: a rebuild judges the layer before keeping it ==", file=sys.stderr)
    from . import session_build as build

    rebuild_cfg = runner.RunConfig(
        run_name="selftest_rebuild", model="selftest-fake", databases=["superhero"], min_model_written=0.9
    )
    code = runner.Run(rebuild_cfg, fake, work, rebuild=True).build_only()
    entry = build.read_manifest(work)["superhero"]
    check(
        code == 1 and entry.get("stage") != "exported",
        "a model whose replies hold no usable descriptions: the rebuild fails (exit 1) and keeps nothing",
    )
    check(
        bool(list((work / "build_failures").glob("superhero-semantics-*.json"))),
        "and what the model said is saved under work/build_failures/ so the failure can be read",
    )
    check(
        "model_written" not in build.read_manifest(work)["superhero"],
        "the rejected attempt left no figures from the layer it replaced",
    )
    # The next command after a rejected attempt: no --rebuild-sessions, so the half-built
    # session is picked up at the description step instead of being made again.
    code = runner.Run(rebuild_cfg, DescribingModel(), work).build_only()
    entry = build.read_manifest(work)["superhero"]
    written = entry.get("model_written") or {}
    check(
        code == 0 and entry.get("stage") == "exported" and written.get("described_by_model") == written.get("columns") > 0,
        f"the same command again, now with a model that follows the format, finishes the half-built layer ({written})",
    )
    code = runner.Run(rebuild_cfg, DescribingModel(), work, rebuild=True).build_only()
    entry = build.read_manifest(work)["superhero"]
    written = entry.get("model_written") or {}
    check(
        code == 0
        and entry.get("stage") == "exported"
        and written.get("columns", 0) > 0
        and written.get("described_by_model") == written.get("columns"),
        f"a model that follows the format: the rebuild succeeds with every description written ({written})",
    )
    try:
        build.check_layer("superhero", entry, 0.9)
        passes_preflight = True
    except build.BuildDegraded:
        passes_preflight = False
    check(passes_preflight, "and the rebuilt layer passes the check a run makes before it starts")

    print(file=sys.stderr)
    if failures:
        print(f"selftest FAILED ({len(failures)} check(s)):", *failures, sep="\n  - ", file=sys.stderr)
        return 1
    print("selftest passed — the harness is sound; nothing in it spent a token.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
