"""Loading BIRD dev questions and picking a reproducible sample.

The full dev set is 1534 questions across 11 SQLite databases. Every question
here runs through *two* arms (bare-schema baseline, the product's pipeline),
each up to ``query_max_attempts`` model calls, so the evaluation uses a sample.

The default sample is **100 simple + 100 moderate + 100 challenging** questions
drawn from :data:`DEFAULT_DATABASES`.  BIRD dev has only 144 challenging
questions in total, so 100 of them cannot come from one or two databases; 
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

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
#:
#: The order is the order a run asks them in, one database after another.  It is
#: chosen so that, with 100 questions per invocation, the first batch is exactly
#: formula_1 (53) + student_club (47) and the third is exactly the tail of
#: toxicology (56) + superhero (44); only toxicology is split, by four questions.
#: The order is operational only — it changes neither which questions are drawn
#: nor any database's result, since every database has its own session.
DEFAULT_DATABASES: tuple[str, ...] = (
    "formula_1",
    "student_club",
    "thrombosis_prediction",
    "california_schools",
    "toxicology",
    "superhero",
)

DIFFICULTIES: tuple[str, ...] = ("simple", "moderate", "challenging")


def run_order(
    examples: Sequence[BirdExample], databases: Sequence[str] = ()
) -> list[BirdExample]:
    """``examples`` in the order a run asks them: database by database, simple first.

    Databases come in the order they are listed in ``databases`` (any that are not
    listed follow, by name); inside one database the bands run simple → moderate →
    challenging, so the semantic arm's saved examples build up from the easy
    questions to the hard ones.  Within a (database, difficulty) band the incoming
    order is kept, so the seeded shuffle done upstream is still the order the band
    is asked in.  A pure reordering: nothing is added or dropped.
    """

    database_rank = {db: position for position, db in enumerate(databases)}
    difficulty_rank = {name: position for position, name in enumerate(DIFFICULTIES)}

    def key(item: tuple[int, BirdExample]) -> tuple[int, str, int, int]:
        position, example = item
        listed = example.db_id in database_rank
        return (
            database_rank[example.db_id] if listed else len(database_rank),
            "" if listed else example.db_id,
            difficulty_rank.get(example.difficulty, len(difficulty_rank)),
            position,
        )

    return [example for _, example in sorted(enumerate(examples), key=key)]


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


def rest_of(
    examples: list[BirdExample],
    asked: Iterable[tuple[str, int]],
    databases: Sequence[str] | None = None,
    seed: int = 0,
) -> list[BirdExample]:
    """Every question of ``databases`` that is not in ``asked`` — "the other questions".

    ``asked`` is ``(db_id, question_id)`` pairs, what an earlier run already put to the
    model.  No difficulty quota here: all of what is left is returned, in a seeded
    shuffle (the same role the shuffle plays in :func:`sample_by_difficulty`: the order
    inside a database and difficulty is not the order of ``dev.json``).  Together with
    that earlier run it is the whole dev set of those databases, each question once.
    """

    taken = set(asked)
    pool = sorted(
        (
            e
            for e in examples
            if (databases is None or e.db_id in databases) and (e.db_id, e.question_id) not in taken
        ),
        key=lambda e: e.question_id,
    )
    random.Random(seed).shuffle(pool)
    return pool


def available_databases() -> list[str]:
    return sorted(p.name for p in DATABASES_DIR.iterdir() if p.is_dir())
