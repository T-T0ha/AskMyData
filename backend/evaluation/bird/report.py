"""Reading a run's ``records.jsonl`` back as numbers.

Nothing here imports the product or touches a database — it works from the
records alone, so any run can be re-read at any time, under any metric:

    python -m evaluation.bird.report --run-name full
    python -m evaluation.bird.report --run-name full --primary strict
    python -m evaluation.bird.report --run-name full --max-extra-columns 0

Three execution-accuracy verdicts are stored for every record (see
:mod:`evaluation.bird.metric`); ``--primary`` chooses which one is called
"correct", and ``--max-extra-columns`` re-applies the surplus-column cap to the
relaxed verdict (the surplus is stored per record, so no query is re-run).

Besides ``summary.json`` a run directory gets two flat tables for pandas or a
spreadsheet: ``records.csv`` (one row per question × arm) and ``paired.csv``
(one row per question, the two arms side by side).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

RESULTS_DIR = Path(__file__).resolve().parent / "results"
METRICS = ("strict", "normalized", "relaxed")
#: The product's own confidence floor (``query_confidence_threshold``), for the
#: "what if the gate had been on" table.
CONFIDENCE_FLOOR = 0.45
CONFIDENCE_CUTS = (0.3, CONFIDENCE_FLOOR, 0.6, 0.75, 0.9)
EXAMPLE_BUCKETS = ((0, 0), (1, 5), (6, 20), (21, 10**9))


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------


def load_records(path: Path) -> list[dict[str, Any]]:
    """Every record in a ``records.jsonl``; the last one wins for a repeated key.

    A line cut short by a crash mid-write is skipped rather than fatal.
    """

    if not path.exists():
        return []
    latest: dict[tuple[str, str, int], dict[str, Any]] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            latest[(record["arm"], record["db_id"], record["question_id"])] = record
    return list(latest.values())


def verdicts(record: dict[str, Any], max_extra: int | None = None) -> dict[str, bool]:
    """The three verdicts of one record, optionally under a different surplus-column cap."""

    relaxed = bool(record.get("relaxed"))
    if max_extra is not None:
        extra = record.get("extra_columns")
        relaxed = bool(record.get("relaxed_match")) and extra is not None and extra <= max_extra
    return {
        "strict": bool(record.get("strict")),
        "normalized": bool(record.get("normalized")),
        "relaxed": relaxed or bool(record.get("normalized")),
    }


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


def _rate(flags: list[bool]) -> float | None:
    return sum(flags) / len(flags) if flags else None


def wilson(successes: int, n: int, z: float = 1.96) -> list[float] | None:
    """95% Wilson score interval for a proportion."""

    if n == 0:
        return None
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [round(max(0.0, centre - margin), 4), round(min(1.0, centre + margin), 4)]


def mcnemar_p(b: int, c: int) -> float:
    """Two-sided McNemar p-value (continuity-corrected chi-square, df=1)."""

    if b + c == 0:
        return 1.0
    statistic = max(abs(b - c) - 1, 0) ** 2 / (b + c)
    return math.erfc(math.sqrt(statistic / 2))


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]


def _usage(record: dict[str, Any]) -> dict[str, Any]:
    return record.get("usage") or {}


def _tokens(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ("calls", "prompt_tokens", "completion_tokens", "reasoning_tokens", "cached_tokens", "total_tokens")
    totals = {key: sum(_usage(r).get(key, 0) for r in rows) for key in keys}
    cost = sum(_usage(r).get("cost_usd", 0.0) for r in rows)
    n = len(rows) or 1
    return {
        **{f"{key}_total": value for key, value in totals.items()},
        **{f"{key}_avg": round(value / n, 1) for key, value in totals.items()},
        "cost_usd_total": round(cost, 6),
        "cost_usd_avg": round(cost / n, 6),
    }


def _accuracy_block(rows: list[dict[str, Any]], max_extra: int | None) -> dict[str, Any]:
    n = len(rows)
    out: dict[str, Any] = {"n": n}
    for name in METRICS:
        passed = sum(verdicts(r, max_extra)[name] for r in rows)
        out[name] = round(passed / n, 4) if n else None
    return out


def arm_stats(rows: list[dict[str, Any]], primary: str, max_extra: int | None) -> dict[str, Any]:
    n = len(rows)
    passed = [verdicts(r, max_extra)[primary] for r in rows]
    answered = [r for r in rows if r.get("outcome") == "answered"]
    elapsed = [r["elapsed_seconds"] for r in rows if r.get("elapsed_seconds") is not None]
    verdict_rows = [verdicts(r, max_extra) for r in rows]
    tokens = _tokens(rows)
    correct = sum(passed)
    stats: dict[str, Any] = {
        "n": n,
        "primary_metric": primary,
        "accuracy": round(correct / n, 4) if n else None,
        "accuracy_ci95": wilson(correct, n),
        "correct": correct,
        "by_metric": {
            name: round(sum(v[name] for v in verdict_rows) / n, 4) if n else None for name in METRICS
        },
        # How much each looser verdict adds over the stricter one.
        "rescued_by_value_normalisation": sum(
            1 for v in verdict_rows if v["normalized"] and not v["strict"]
        ),
        "rescued_by_extra_columns": sum(
            1 for v in verdict_rows if v["relaxed"] and not v["normalized"]
        ),
        "answered_rate": round(len(answered) / n, 4) if n else None,
        "no_sql_rate": round(sum(r.get("outcome") == "no_sql" for r in rows) / n, 4) if n else None,
        "valid_sql_rate": round(
            sum(1 for r in rows if r.get("predicted_sql") and not r.get("predicted_error")) / n, 4
        )
        if n
        else None,
        "accuracy_when_answered": (
            round(sum(verdicts(r, max_extra)[primary] for r in answered) / len(answered), 4)
            if answered
            else None
        ),
        "first_attempt_valid_rate": round(sum(bool(r.get("first_attempt_ok")) for r in rows) / n, 4) if n else None,
        "avg_attempts": round(_mean(r.get("n_attempts", 0) for r in rows) or 0, 3),
        "latency_seconds": {
            "avg": round(_mean(elapsed) or 0, 2),
            "median": round(statistics.median(elapsed), 2) if elapsed else None,
            "p90": round(_percentile(elapsed, 0.9) or 0, 2) if elapsed else None,
        },
        "tokens": tokens,
        "tokens_per_correct": round(tokens["total_tokens_total"] / correct, 1) if correct else None,
        "cost_usd_per_correct": round(tokens["cost_usd_total"] / correct, 6) if correct else None,
        "finish_reasons": _finish_reasons(rows),
    }
    return stats


def _finish_reasons(rows: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for r in rows:
        for reason, count in (_usage(r).get("finish_reasons") or {}).items():
            out[reason] += count
    return dict(out)


def _semantic_extras(rows: list[dict[str, Any]], primary: str, max_extra: int | None) -> dict[str, Any]:
    """Numbers only the product's pipeline has: retrieval quality, examples, confidence."""

    def column(name: str) -> list[float]:
        return [r[name] for r in rows if r.get(name) is not None]

    sources: dict[str, int] = defaultdict(int)
    for r in rows:
        sources[r.get("source") or "generated"] += 1

    by_store: dict[str, Any] = {}
    for low, high in EXAMPLE_BUCKETS:
        subset = [r for r in rows if low <= r.get("examples_in_store", 0) <= high]
        if subset:
            label = f"{low}" if low == high else (f"{low}+" if high >= 10**9 else f"{low}-{high}")
            by_store[label] = {
                "n": len(subset),
                "accuracy": round(sum(verdicts(r, max_extra)[primary] for r in subset) / len(subset), 4),
            }

    with_conf = [r for r in rows if r.get("confidence") is not None and r.get("outcome") == "answered"]
    selective = []
    for cut in CONFIDENCE_CUTS:
        kept = [r for r in with_conf if r["confidence"] >= cut]
        selective.append(
            {
                "min_confidence": cut,
                "coverage": round(len(kept) / len(rows), 4) if rows else None,
                "accuracy_of_kept": round(sum(verdicts(r, max_extra)[primary] for r in kept) / len(kept), 4)
                if kept
                else None,
            }
        )
    return {
        "table_recall_avg": round(_mean(column("table_recall")) or 0, 4) if column("table_recall") else None,
        "column_recall_avg": round(_mean(column("column_recall")) or 0, 4) if column("column_recall") else None,
        "context_tables_avg": round(_mean(column("context_tables")) or 0, 2) if column("context_tables") else None,
        "context_columns_avg": round(_mean(column("context_columns")) or 0, 1) if column("context_columns") else None,
        "questions_reading_a_changed_column": sum(1 for r in rows if r.get("gold_touches_drift")),
        "gold_on_export": dict(Counter(r["gold_on_export"] for r in rows if r.get("gold_on_export"))),
        "answered_from_saved_example": sources.get("saved_example", 0),
        "questions_shown_examples": sum(1 for r in rows if r.get("examples_sent")),
        "examples_saved": sum(1 for r in rows if r.get("example_saved")),
        "accuracy_by_examples_in_store": by_store,
        "selective_accuracy": selective,
        "withheld_by_product_gate": sum(
            1 for r in with_conf if r["confidence"] < CONFIDENCE_FLOOR
        ),
    }


def summarize(
    records: list[dict[str, Any]], primary: str = "relaxed", max_extra: int | None = None
) -> dict[str, Any]:
    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_arm[r["arm"]].append(r)

    arms: dict[str, Any] = {}
    for arm in sorted(by_arm):
        rows = by_arm[arm]
        stats = arm_stats(rows, primary, max_extra)
        stats["by_difficulty"] = {
            d: _accuracy_block([r for r in rows if r["difficulty"] == d], max_extra)
            for d in sorted({r["difficulty"] for r in rows})
        }
        stats["by_database"] = {
            db: _accuracy_block([r for r in rows if r["db_id"] == db], max_extra)
            for db in sorted({r["db_id"] for r in rows})
        }
        if arm == "semantic":
            stats["pipeline"] = _semantic_extras(rows, primary, max_extra)
        arms[arm] = stats

    return {
        "primary_metric": primary,
        "max_extra_columns_override": max_extra,
        "records": len(records),
        "arms": arms,
        "paired": _paired(by_arm, primary, max_extra),
    }


def _paired(by_arm: dict[str, list[dict[str, Any]]], primary: str, max_extra: int | None) -> dict[str, Any] | None:
    if "raw" not in by_arm or "semantic" not in by_arm:
        return None
    raw = {(r["db_id"], r["question_id"]): r for r in by_arm["raw"]}
    sem = {(r["db_id"], r["question_id"]): r for r in by_arm["semantic"]}
    common = sorted(set(raw) & set(sem))

    def table(keys: list[tuple[str, int]], metric: str) -> dict[str, Any]:
        both = sem_only = raw_only = neither = 0
        for key in keys:
            a, b = verdicts(raw[key], max_extra)[metric], verdicts(sem[key], max_extra)[metric]
            both += a and b
            sem_only += b and not a
            raw_only += a and not b
            neither += not a and not b
        return {
            "n": len(keys),
            "both_correct": both,
            "semantic_only": sem_only,
            "raw_only": raw_only,
            "neither": neither,
            "mcnemar_p": round(mcnemar_p(sem_only, raw_only), 6),
            "semantic_minus_raw": round((sem_only - raw_only) / len(keys), 4) if keys else None,
        }

    untouched = [k for k in common if not sem[k].get("gold_touches_drift")]
    touched = [k for k in common if sem[k].get("gold_touches_drift")]
    return {
        "n_paired": len(common),
        "by_metric": {m: table(common, m) for m in METRICS},
        # The semantic arm queries the product's *exported* data.  Where the export
        # changed a column the gold query reads (a '-' read as missing, a number
        # stored as text) a miss may be the data's doing rather than the SQL's.
        "by_data_drift": {
            "gold_reads_no_changed_column": table(untouched, primary),
            "gold_reads_a_changed_column": table(touched, primary),
            # values: 'na'/'-' read as missing; retyped: number stored as text;
            # reformatted: same value, different spelling.  A question can sit in several.
            **{
                f"gold_reads_a_{kind}_column": table(
                    [k for k in touched if kind in (sem[k].get("drift_kinds") or [])], primary
                )
                for kind in ("values", "retyped", "reformatted")
            },
        },
        # The cleanest fair-comparison slice: questions whose gold answer survives the
        # export.  "different" is a ceiling the semantic arm cannot be blamed for.
        "by_gold_on_export": {
            label: table([k for k in common if sem[k].get("gold_on_export") in members], primary)
            for label, members in (
                ("answer_preserved", {"same", "respelled"}),
                ("answer_changed", {"different"}),
                ("not_runnable_unchanged", {"not_runnable"}),
            )
        },
        "by_difficulty": {
            d: table([k for k in common if raw[k]["difficulty"] == d], primary)
            for d in sorted({raw[k]["difficulty"] for k in common})
        },
        "by_database": {
            db: table([k for k in common if k[0] == db], primary)
            for db in sorted({k[0] for k in common})
        },
    }


# ---------------------------------------------------------------------------
# flat tables
# ---------------------------------------------------------------------------

RECORD_COLUMNS = (
    "arm", "db_id", "question_id", "difficulty", "outcome", "source",
    "correct", "strict", "normalized", "relaxed", "relaxed_match", "extra_columns",
    "n_attempts", "first_attempt_ok", "confidence", "elapsed_seconds",
    "calls", "prompt_tokens", "completion_tokens", "reasoning_tokens", "cached_tokens", "total_tokens", "cost_usd",
    "predicted_rows", "gold_rows", "predicted_columns", "gold_columns",
    "table_recall", "column_recall", "context_tables", "context_columns", "gold_on_export", "gold_touches_drift", "drift_kinds",
    "examples_in_store", "examples_sent", "example_saved", "rules_sent",
    "predicted_error", "generation_error", "gold_error",
    "question", "evidence", "gold_sql", "predicted_sql",
)  # fmt: skip

PAIRED_FIELDS = (
    "outcome", "correct", "strict", "normalized", "relaxed", "extra_columns",
    "n_attempts", "total_tokens", "cost_usd", "elapsed_seconds", "predicted_sql", "predicted_error",
)  # fmt: skip


def _flat(record: dict[str, Any]) -> dict[str, Any]:
    flat = dict(record)
    usage = _usage(record)
    for key in ("calls", "prompt_tokens", "completion_tokens", "reasoning_tokens", "cached_tokens", "total_tokens", "cost_usd"):
        flat[key] = usage.get(key)
    flat["examples_sent"] = len(record.get("examples_sent") or []) if "examples_sent" in record else None
    if "drift_kinds" in record:
        flat["drift_kinds"] = "+".join(record["drift_kinds"])
    return flat


def write_tables(records: list[dict[str, Any]], directory: Path) -> None:
    ordered = sorted(records, key=lambda r: (r["db_id"], r["question_id"], r["arm"]))
    with (directory / "records.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=RECORD_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for record in ordered:
            writer.writerow(_flat(record))

    by_key: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in ordered:
        by_key[(record["db_id"], record["question_id"])][record["arm"]] = _flat(record)
    header = ["db_id", "question_id", "difficulty", "question", "evidence", "gold_sql"]
    header += [f"{arm}_{field}" for arm in ("raw", "semantic") for field in PAIRED_FIELDS]
    header += ["winner"]
    with (directory / "paired.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        for (db_id, question_id), arms in sorted(by_key.items()):
            any_arm = next(iter(arms.values()))
            row = {k: any_arm.get(k) for k in ("db_id", "question_id", "difficulty", "question", "evidence", "gold_sql")}
            for arm in ("raw", "semantic"):
                for field in PAIRED_FIELDS:
                    row[f"{arm}_{field}"] = arms.get(arm, {}).get(field)
            raw_ok, sem_ok = row.get("raw_correct"), row.get("semantic_correct")
            if "raw" in arms and "semantic" in arms:
                row["winner"] = "both" if raw_ok and sem_ok else "semantic" if sem_ok else "raw" if raw_ok else "neither"
            writer.writerow(row)


def write_outputs(
    directory: Path, primary: str | None = None, max_extra: int | None = None
) -> dict[str, Any]:
    """Write ``summary.json``, ``records.csv`` and ``paired.csv`` for a run directory."""

    config = {}
    config_path = directory / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text()).get("config", {})
    primary = primary or config.get("primary_metric", "relaxed")
    records = load_records(directory / "records.jsonl")
    summary = summarize(records, primary, max_extra)
    summary["run_name"] = directory.name
    (directory / "summary.json").write_text(json.dumps(summary, indent=2))
    write_tables(records, directory)
    return summary


# ---------------------------------------------------------------------------
# text
# ---------------------------------------------------------------------------


def _pct(value: float | None) -> str:
    return "  n/a " if value is None else f"{value:6.1%}"


def render(summary: dict[str, Any]) -> str:
    primary = summary["primary_metric"]
    lines = [f"BIRD dev evaluation — primary metric: {primary}", "=" * 64]
    for arm, s in summary["arms"].items():
        ci = s["accuracy_ci95"]
        interval = f"  (95% CI {ci[0]:.1%}–{ci[1]:.1%})" if ci else ""
        lines.append(f"\n[{arm}]  n={s['n']}   accuracy {_pct(s['accuracy']).strip()}{interval}")
        by = s["by_metric"]
        lines.append(
            f"  strict {_pct(by['strict'])}   normalized {_pct(by['normalized'])}   relaxed {_pct(by['relaxed'])}"
            f"   (+{s['rescued_by_value_normalisation']} by value normalisation, "
            f"+{s['rescued_by_extra_columns']} by extra columns)"
        )
        lines.append(
            f"  valid SQL {_pct(s['valid_sql_rate'])}   first attempt valid {_pct(s['first_attempt_valid_rate'])}"
            f"   avg attempts {s['avg_attempts']:.2f}   no-SQL {_pct(s['no_sql_rate'])}"
        )
        tokens = s["tokens"]
        lines.append(
            f"  tokens/question {tokens['total_tokens_avg']:,.0f} "
            f"(prompt {tokens['prompt_tokens_avg']:,.0f}, completion {tokens['completion_tokens_avg']:,.0f}, "
            f"of which reasoning {tokens['reasoning_tokens_avg']:,.0f}; "
            f"{tokens['cached_tokens_avg']:,.0f} of the prompt served from cache)   "
            f"tokens/correct {s['tokens_per_correct'] or 0:,.0f}   cost ${tokens['cost_usd_total']:.3f}"
        )
        lat = s["latency_seconds"]
        lines.append(f"  latency avg {lat['avg']}s  median {lat['median']}s  p90 {lat['p90']}s")
        for difficulty, d in s["by_difficulty"].items():
            lines.append(
                f"    {difficulty:<12} n={d['n']:<4} strict {_pct(d['strict'])}  normalized {_pct(d['normalized'])}  relaxed {_pct(d['relaxed'])}"
            )
        pipeline = s.get("pipeline")
        if pipeline:
            lines.append(
                f"  retrieval: table recall {_pct(pipeline['table_recall_avg'])}  column recall "
                f"{_pct(pipeline['column_recall_avg'])}  (≈{pipeline['context_columns_avg']} columns in "
                f"{pipeline['context_tables_avg']} tables sent)"
            )
            lines.append(
                f"  data: {pipeline['questions_reading_a_changed_column']} of {s['n']} questions read a column "
                "whose values the export changed (see build.json)"
            )
            on_export = pipeline.get("gold_on_export") or {}
            if on_export:
                lines.append(
                    "  gold query run on the exported data: "
                    + ", ".join(f"{name.replace('_', ' ')} {n}" for name, n in sorted(on_export.items()))
                    + "  (different = no model reading the data like the gold can pass)"
                )
            lines.append(
                f"  examples: saved {pipeline['examples_saved']}, shown to {pipeline['questions_shown_examples']} "
                f"questions, answered straight from one {pipeline['answered_from_saved_example']} times"
            )
            for bucket, b in pipeline["accuracy_by_examples_in_store"].items():
                lines.append(f"    with {bucket:>5} example(s) in store: n={b['n']:<4} accuracy {_pct(b['accuracy'])}")
            lines.append(
                f"  product gate (confidence < {CONFIDENCE_FLOOR}) would have withheld "
                f"{pipeline['withheld_by_product_gate']} answers"
            )
            for cut in pipeline["selective_accuracy"]:
                lines.append(
                    f"    confidence ≥ {cut['min_confidence']:<4} coverage {_pct(cut['coverage'])}  accuracy of those {_pct(cut['accuracy_of_kept'])}"
                )
    paired = summary.get("paired")
    if paired:
        lines.append(f"\n[paired: semantic vs raw]  n={paired['n_paired']}")
        for name, t in paired["by_metric"].items():
            lines.append(
                f"  {name:<10} both {t['both_correct']:<4} semantic-only {t['semantic_only']:<4} "
                f"raw-only {t['raw_only']:<4} neither {t['neither']:<4} "
                f"Δ {t['semantic_minus_raw']:+.1%}  McNemar p={t['mcnemar_p']:.4f}"
            )
        for label, t in paired.get("by_gold_on_export", {}).items():
            if t["n"]:
                lines.append(
                    f"    gold on export: {label.replace('_', ' '):<22} n={t['n']:<4} semantic-only {t['semantic_only']:<3} "
                    f"raw-only {t['raw_only']:<3} both {t['both_correct']:<3} Δ {t['semantic_minus_raw']:+.1%}"
                )
        for label, t in paired["by_data_drift"].items():
            if t["n"]:
                lines.append(
                    f"    {label.replace('_', ' '):<32} n={t['n']:<4} semantic-only {t['semantic_only']:<3} "
                    f"raw-only {t['raw_only']:<3} both {t['both_correct']:<3} Δ {t['semantic_minus_raw']:+.1%}"
                )
        for difficulty, t in paired["by_difficulty"].items():
            lines.append(
                f"    {difficulty:<12} n={t['n']:<4} semantic-only {t['semantic_only']:<3} raw-only {t['raw_only']:<3} Δ {t['semantic_minus_raw']:+.1%}  p={t['mcnemar_p']:.4f}"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--primary", choices=METRICS, default=None, help="verdict to call 'correct'")
    parser.add_argument("--max-extra-columns", type=int, default=None, help="re-apply the surplus-column cap")
    parser.add_argument("--write", action="store_true", help="also (re)write summary.json and the CSV tables")
    parser.add_argument("--json", action="store_true", help="print the summary as JSON instead of text")
    args = parser.parse_args()

    directory = RESULTS_DIR / args.run_name
    if not (directory / "records.jsonl").exists():
        sys.exit(f"no records at {directory / 'records.jsonl'}")
    if args.write:
        summary = write_outputs(directory, args.primary, args.max_extra_columns)
    else:
        config_path = directory / "config.json"
        config = json.loads(config_path.read_text()).get("config", {}) if config_path.exists() else {}
        summary = summarize(
            load_records(directory / "records.jsonl"),
            args.primary or config.get("primary_metric", "relaxed"),
            args.max_extra_columns,
        )
    print(json.dumps(summary, indent=2) if args.json else render(summary))


if __name__ == "__main__":
    main()
