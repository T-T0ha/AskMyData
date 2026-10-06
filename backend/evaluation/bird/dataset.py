"""Loading BIRD dev questions and picking a reproducible sample.

The full dev set is 1534 questions across 11 SQLite databases. Every question
here runs through *two* arms (bare-schema baseline, the product's pipeline),
each up to ``query_max_attempts`` model calls, so the evaluation uses a sample.

The default sample is **100 simple + 100 moderate + 100 challenging** questions
drawn from :data:`DEFAULT_DATABASES`.  BIRD dev has only 144 challenging
questions in total, so 100 of them cannot come from one or two databases; the
default set is the smallest one that holds them without the databases whose
largest table is several times the product's per-table ingestion cap (``card_games``,
``financial``, ``codebase_community``, ``debit_card_specializing``) or whose
width makes the one-off semantic-layer build very expensive
(``european_football_2``: 199 columns).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data" / "dev_20240627"
DEV_JSON = DATA_DIR / "dev.json"
DATABASES_DIR = DATA_DIR / "dev_databases"


@dataclass(frozen=True, slots=True)
class BirdExample:
    question_id: int
    db_id: str
    question: str
    evidence: str
    gold_sql: str
    difficulty: str

    @property
    def db_path(self) -> Path:
        return DATABASES_DIR / self.db_id / f"{self.db_id}.sqlite"


def load_all() -> list[BirdExample]:
    if not DEV_JSON.exists():
        raise FileNotFoundError(
            f"{DEV_JSON} not found — run `python -m evaluation.bird.download` first"
        )
    raw = json.loads(DEV_JSON.read_text())
    return [
        BirdExample(
            question_id=row["question_id"],
            db_id=row["db_id"],
            question=row["question"],
            evidence=row.get("evidence", ""),
            gold_sql=row["SQL"],
            difficulty=row.get("difficulty", "unknown"),
        )
        for row in raw
    ]


def stratified_sample(
    examples: list[BirdExample],
    size: int,
    seed: int = 0,
    databases: list[str] | None = None,
) -> list[BirdExample]:
    """``size`` questions, proportionally stratified by (db_id, difficulty).

    Deterministic for a fixed ``(size, seed, databases)`` — the same sample
    is used across arms and across reruns so results stay comparable and the
    exact question set is reproducible from the command line alone.
    """

    pool = [e for e in examples if databases is None or e.db_id in databases]
    if size >= len(pool):
        return pool

    groups: dict[tuple[str, str], list[BirdExample]] = {}
    for example in pool:
        groups.setdefault((example.db_id, example.difficulty), []).append(example)

    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)

    # Largest remainder method: guarantees every group's share rounds to
    # within one question of proportional, and the total is exactly `size`.
    shares = {key: len(group) * size / len(pool) for key, group in groups.items()}
    counts = {key: int(share) for key, share in shares.items()}
    remainder = size - sum(counts.values())
    ranked = sorted(shares, key=lambda key: shares[key] - counts[key], reverse=True)
    for key in ranked[:remainder]:
        counts[key] += 1

    sample: list[BirdExample] = []
    for key, group in groups.items():
        sample.extend(group[: counts[key]])
    rng.shuffle(sample)
    return sample


#: Databases the default evaluation draws from.  Challenging-question counts in
#: BIRD dev: toxicology 31, thrombosis_prediction 28, superhero 15, formula_1 14,
#: student_club 9, california_schools 5 — 102 in all, so a 100-question
#: challenging band takes almost every one of them.
DEFAULT_DATABASES: tuple[str, ...] = (
    "toxicology",
    "thrombosis_prediction",
    "superhero",
    "formula_1",
    "student_club",
    "california_schools",
)

DIFFICULTIES: tuple[str, ...] = ("simple", "moderate", "challenging")


def _proportional_counts(sizes: dict[str, int], total: int) -> dict[str, int]:
    """Split ``total`` across groups in proportion to ``sizes`` (largest remainder)."""

    pool = sum(sizes.values())
    if total >= pool:
        return dict(sizes)
    shares = {key: size * total / pool for key, size in sizes.items()}
    counts = {key: int(share) for key, share in shares.items()}
    remainder = total - sum(counts.values())
    for key in sorted(shares, key=lambda k: (shares[k] - counts[k], k), reverse=True)[:remainder]:
        counts[key] += 1
    return counts


def sample_by_difficulty(
    examples: list[BirdExample],
    per_difficulty: dict[str, int],
    databases: list[str] | tuple[str, ...] | None = None,
    seed: int = 0,
) -> list[BirdExample]:
    """``per_difficulty[d]`` questions of each difficulty ``d``, spread over the databases.

    Within a difficulty the databases contribute in proportion to how many
    questions of that difficulty they hold, so no database is over-represented
    relative to the dev set.  A band the chosen databases cannot fill is a
    configuration error and raises — a silently short band would make the
    "100 / 100 / 100" claim false.

    Deterministic for a fixed ``(per_difficulty, databases, seed)``.  Returned in
    a seeded shuffle, the order every database's questions are later asked in.
    """

    pool = [e for e in examples if databases is None or e.db_id in databases]
    rng = random.Random(seed)
    chosen: list[BirdExample] = []
    for difficulty in DIFFICULTIES:
        wanted = per_difficulty.get(difficulty, 0)
        if not wanted:
            continue
        groups: dict[str, list[BirdExample]] = {}
        for example in sorted(pool, key=lambda e: e.question_id):
            if example.difficulty == difficulty:
                groups.setdefault(example.db_id, []).append(example)
        available = sum(len(group) for group in groups.values())
        if available < wanted:
            raise ValueError(
                f"only {available} {difficulty!r} questions exist in "
                f"{sorted(databases) if databases else 'the dev set'}; {wanted} requested"
            )
        counts = _proportional_counts({db: len(group) for db, group in groups.items()}, wanted)
        for db_id in sorted(groups):
            group = list(groups[db_id])
            random.Random(f"{seed}:{difficulty}:{db_id}").shuffle(group)
            chosen.extend(group[: counts[db_id]])
    rng.shuffle(chosen)
    return chosen


def available_databases() -> list[str]:
    return sorted(p.name for p in DATABASES_DIR.iterdir() if p.is_dir())
