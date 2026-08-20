"""Local sentence embeddings for column-name semantics.

``all-MiniLM-L6-v2`` (384-d) is loaded once, on CPU, and reused.  Two details
matter for accuracy on spreadsheet column names:

* Column names are not sentences.  ``humanize`` turns ``Cust_ID`` into
  ``cust id`` before encoding, which is much closer to the model's training
  distribution.
* MiniLM is weak on abbreviations — the very thing this feature exists to
  catch.  ``expand_abbreviations`` normalises the ~30 abbreviations that
  dominate business spreadsheets before encoding, so ``qty`` and ``quantity``
  become the same string rather than a 0.49 cosine.

If the model cannot be loaded (no download, offline machine) a deterministic
hashing embedder keeps the pipeline running; ``Embedder.is_semantic`` tells
callers which one they got so the UI can say so.
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
from typing import Iterable, Protocol

import numpy as np

from app.core.config import get_settings

logger = logging.getLogger(__name__)

_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_SPLIT_RE = re.compile(r"[^0-9a-zA-Z]+")

#: Abbreviations that appear constantly in business spreadsheet headers.
ABBREVIATIONS: dict[str, str] = {
    "id": "identifier",
    "ids": "identifiers",
    "cust": "customer",
    "custs": "customers",
    "qty": "quantity",
    "amt": "amount",
    "amnt": "amount",
    "no": "number",
    "num": "number",
    "nbr": "number",
    "addr": "address",
    "desc": "description",
    "dept": "department",
    "emp": "employee",
    "mgr": "manager",
    "prod": "product",
    "sku": "stock keeping unit",
    "inv": "invoice",
    "ord": "order",
    "txn": "transaction",
    "trans": "transaction",
    "pct": "percent",
    "perc": "percent",
    "avg": "average",
    "tot": "total",
    "bal": "balance",
    "dt": "date",
    "dob": "date of birth",
    "yr": "year",
    "mo": "month",
    "tel": "telephone",
    "ph": "phone",
    "fk": "foreign key",
    "pk": "primary key",
    "ref": "reference",
    "vat": "value added tax",
    "disc": "discount",
    "wt": "weight",
}


def humanize(name: str) -> str:
    """``"Cust_ID"`` / ``"custID"`` -> ``"cust id"``."""

    spaced = _CAMEL_RE.sub(" ", str(name))
    tokens = [t for t in _SPLIT_RE.split(spaced) if t]
    return " ".join(t.lower() for t in tokens)


def expand_abbreviations(text: str) -> str:
    """Rewrite known abbreviations token by token."""

    return " ".join(ABBREVIATIONS.get(token, token) for token in text.split())


def normalize_column_name(name: str) -> str:
    """The canonical string used for both embedding and lexical comparison."""

    return expand_abbreviations(humanize(name))


class Embedder(Protocol):
    dim: int
    name: str
    is_semantic: bool

    def encode(self, texts: Iterable[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """all-MiniLM-L6-v2, CPU only, L2-normalised output."""

    is_semantic = True

    def __init__(self, model_name: str, dim: int) -> None:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        # CPU is both the documented deployment target and a guard against
        # mismatched CUDA builds on developer machines.
        self._model = SentenceTransformer(model_name, device="cpu")
        self.dim = dim
        self.name = model_name

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        items = list(texts)
        if not items:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = self._model.encode(
            items,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


class HashingEmbedder:
    """Deterministic character-trigram hashing fallback.

    Not semantic — it only captures surface form — but it keeps every downstream
    stage (equivalence scoring, storage, API shape) exercisable without a model
    download.  ``is_semantic = False`` so the UI can flag reduced accuracy.
    """

    is_semantic = False
    name = "hashing-trigram-fallback"

    def __init__(self, dim: int) -> None:
        self.dim = dim

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        padded = f"  {text.lower().strip()}  "
        grams = [padded[i : i + 3] for i in range(max(1, len(padded) - 2))]
        for token in text.lower().split():
            grams.append(f"__{token}__")
        for gram in grams:
            digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else vector

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        items = list(texts)
        if not items:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self._vector(t) for t in items])


_embedder: Embedder | None = None
_lock = threading.Lock()


def get_embedder(force_fallback: bool = False) -> Embedder:
    """Process-wide singleton; loading MiniLM takes a few seconds."""

    global _embedder
    if _embedder is not None and not force_fallback:
        return _embedder
    settings = get_settings()
    with _lock:
        if _embedder is not None and not force_fallback:
            return _embedder
        if force_fallback:
            _embedder = HashingEmbedder(settings.embedding_dim)
            return _embedder
        try:
            _embedder = SentenceTransformerEmbedder(
                settings.embedding_model, settings.embedding_dim
            )
        except Exception as exc:  # pragma: no cover - depends on environment
            if not settings.allow_embedding_fallback:
                raise
            logger.warning(
                "sentence-transformers unavailable (%s); using hashing fallback", exc
            )
            _embedder = HashingEmbedder(settings.embedding_dim)
        return _embedder


def reset_embedder() -> None:
    """Test hook — drops the cached model."""

    global _embedder
    _embedder = None


def embed_column_names(names: Iterable[str]) -> tuple[np.ndarray, list[str]]:
    """Encode column names after humanising + abbreviation expansion."""

    normalized = [normalize_column_name(n) for n in names]
    return get_embedder().encode(normalized), normalized


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity for L2-normalised inputs."""

    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    return np.clip(a @ b.T, -1.0, 1.0)
