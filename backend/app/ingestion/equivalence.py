"""Phase 0 — cross-sheet column name equivalence.

Two sheets that say ``Cust_ID`` and ``customer_id`` mean the same thing, and
nothing in the file says so.  Finding those pairs early gives the cleaning
plan (Phase 2) safe merge keys and gives foreign-key detection (Phase 3) a
prior.

Scoring is deliberately hybrid.  A pure cosine over MiniLM embeddings scores
``cust_id`` vs ``customer_id`` at only 0.57 — abbreviations are exactly where
sentence embeddings are weakest and where lexical matching is strongest — so
the final score blends the two and applies a small penalty when the two
columns' storage types disagree.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from app.core.config import get_settings
from app.semantics.embeddings import cosine_matrix, embed_column_names, get_embedder

EMBEDDING_WEIGHT = 0.55
LEXICAL_WEIGHT = 0.45
TYPE_MISMATCH_PENALTY = 0.15
#: Weak disconfirmation: two same-typed columns that genuinely mean the same
#: thing inside one workbook usually share at least a few values.
DISJOINT_VALUES_PENALTY = 0.08


@dataclass(slots=True)
class ColumnRef:
    table: str
    column: str

    @property
    def qualified(self) -> str:
        return f"{self.table}.{self.column}"


@dataclass(slots=True)
class EquivalenceCandidate:
    """A suggested "these two columns mean the same thing" pair."""

    left: ColumnRef
    right: ColumnRef
    score: float
    embedding_similarity: float
    lexical_similarity: float
    type_compatible: bool
    left_dtype: str
    right_dtype: str
    value_overlap: float | None = None
    confirmed: bool | None = None
    #: The two column-name embeddings ``embedding_similarity`` was computed
    #: from — carried alongside the scalar score so the caller can persist
    #: them (see ``app.db.models.EquivalenceCandidateRecord``) instead of
    #: discarding the prior the moment a score is derived from it.
    left_embedding: list[float] | None = None
    right_embedding: list[float] | None = None

    def explanation(self) -> str:
        parts = [
            f"name embedding similarity {self.embedding_similarity:.2f}",
            f"lexical similarity {self.lexical_similarity:.2f}",
        ]
        if self.value_overlap is not None:
            parts.append(f"value overlap {self.value_overlap:.0%}")
        if not self.type_compatible:
            parts.append(f"types differ ({self.left_dtype} vs {self.right_dtype})")
        return "; ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "left_table": self.left.table,
            "left_column": self.left.column,
            "right_table": self.right.table,
            "right_column": self.right.column,
            "score": round(self.score, 4),
            "embedding_similarity": round(self.embedding_similarity, 4),
            "lexical_similarity": round(self.lexical_similarity, 4),
            "value_overlap": (
                round(self.value_overlap, 4) if self.value_overlap is not None else None
            ),
            "type_compatible": self.type_compatible,
            "left_dtype": self.left_dtype,
            "right_dtype": self.right_dtype,
            "confirmed": self.confirmed,
            "explanation": self.explanation(),
            "left_embedding": self.left_embedding,
            "right_embedding": self.right_embedding,
        }


def _dtype_family(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    return "text"


def lexical_similarity(left: str, right: str) -> float:
    """Fuzzy similarity on the abbreviation-expanded names.

    ``WRatio`` alone rewards a shared substring too generously and
    ``token_sort_ratio`` alone punishes reordered/partial tokens too hard;
    averaging them separates ``customer identifier``/``order identifier``
    (0.69) from ``customer identifier``/``customer identifier`` (1.00) while
    still scoring ``email address``/``e mail`` highly.
    """

    return (fuzz.WRatio(left, right) + fuzz.token_sort_ratio(left, right)) / 200.0


def value_overlap(left: pd.Series, right: pd.Series, limit: int = 5000) -> float | None:
    """Jaccard-style overlap of the two columns' value sets.

    Reported as supporting evidence only — a name-equivalence suggestion is
    never made on values alone (that is Phase 3's job).
    """

    left_values = set(left.dropna().astype(str).head(limit))
    right_values = set(right.dropna().astype(str).head(limit))
    if not left_values or not right_values:
        return None
    smaller = min(len(left_values), len(right_values))
    return len(left_values & right_values) / smaller


def detect_equivalences(
    tables: Mapping[str, pd.DataFrame],
    threshold: float | None = None,
    include_value_overlap: bool = True,
) -> list[EquivalenceCandidate]:
    """Score every cross-sheet column pair and return those above threshold.

    Same-sheet pairs are skipped: a column cannot be equivalent to its own
    sibling for merge purposes.
    """

    settings = get_settings()
    threshold = settings.equivalence_threshold if threshold is None else threshold

    refs: list[ColumnRef] = []
    for table, df in tables.items():
        for column in df.columns:
            refs.append(ColumnRef(table, str(column)))
    if len(refs) < 2:
        return []

    vectors, normalized = embed_column_names([r.column for r in refs])
    similarity = cosine_matrix(vectors, vectors)

    candidates: list[EquivalenceCandidate] = []
    for i, j in combinations(range(len(refs)), 2):
        left, right = refs[i], refs[j]
        if left.table == right.table:
            continue
        embedding = float(similarity[i, j])
        lexical = lexical_similarity(normalized[i], normalized[j])
        left_series = tables[left.table][left.column]
        right_series = tables[right.table][right.column]
        left_family = _dtype_family(left_series)
        right_family = _dtype_family(right_series)
        compatible = left_family == right_family

        overlap = (
            value_overlap(left_series, right_series) if include_value_overlap else None
        )

        score = EMBEDDING_WEIGHT * embedding + LEXICAL_WEIGHT * lexical
        if not compatible:
            score -= TYPE_MISMATCH_PENALTY
        if compatible and overlap == 0.0:
            score -= DISJOINT_VALUES_PENALTY
        score = max(0.0, min(1.0, score))
        if score < threshold:
            continue

        candidates.append(
            EquivalenceCandidate(
                left=left,
                right=right,
                score=score,
                embedding_similarity=embedding,
                lexical_similarity=lexical,
                type_compatible=compatible,
                left_dtype=str(left_series.dtype),
                right_dtype=str(right_series.dtype),
                left_embedding=[float(v) for v in vectors[i]],
                right_embedding=[float(v) for v in vectors[j]],
                value_overlap=overlap,
            )
        )

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates


def suggestion_text(candidate: EquivalenceCandidate) -> str:
    """The confirmation prompt shown to the user."""

    return (
        f"{candidate.left.qualified} and {candidate.right.qualified} appear to refer "
        "to the same concept — confirm?"
    )


def embedding_backend() -> dict[str, Any]:
    """Which embedder is active, for display in the UI."""

    embedder = get_embedder()
    return {
        "name": embedder.name,
        "dim": embedder.dim,
        "semantic": embedder.is_semantic,
    }


def confirmed_pairs(
    candidates: Iterable[EquivalenceCandidate],
) -> list[tuple[ColumnRef, ColumnRef]]:
    """Pairs the user accepted — the signal handed to Phase 2 and Phase 3."""

    return [(c.left, c.right) for c in candidates if c.confirmed]
