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

    from app.semantics.claude_client import SQL_SYSTEM_PROMPT, ClaudeClient

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

    print(file=sys.stderr)
    if failures:
        print(f"selftest FAILED ({len(failures)} check(s)):", *failures, sep="\n  - ", file=sys.stderr)
        return 1
    print("selftest passed — the harness is sound; nothing in it spent a token.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
