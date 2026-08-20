"""Session-scoped table storage with snapshots.

LangGraph checkpoints must stay JSON-serialisable, so DataFrames never enter
the graph state.  They live here instead, keyed by session, and the graph
carries only table *names*.  Every execution node snapshots before it runs,
which is what makes the "Revert" button in the validation interrupt cheap and
exact.
"""

from __future__ import annotations

import pickle
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import pandas as pd

from app.core.config import get_settings

PICKLE_PROTOCOL = 5


@dataclass(slots=True)
class TableMeta:
    name: str
    row_count: int
    column_count: int
    columns: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "columns": self.columns,
        }


class TableStore:
    """DataFrames for one ingestion session, mirrored to disk."""

    def __init__(self, session_id: str, root: Path | None = None) -> None:
        settings = get_settings()
        self.session_id = session_id
        self.root = (root or settings.upload_dir).parent / "sessions" / session_id
        self.root.mkdir(parents=True, exist_ok=True)
        self._tables: dict[str, pd.DataFrame] = {}
        self._lock = threading.RLock()

    # -- access ----------------------------------------------------------
    @property
    def _current_path(self) -> Path:
        return self.root / "current.pkl"

    def _snapshot_path(self, tag: str) -> Path:
        return self.root / f"snapshot-{tag}.pkl"

    def put(self, name: str, df: pd.DataFrame) -> None:
        with self._lock:
            self._tables[name] = df

    def put_many(self, tables: Mapping[str, pd.DataFrame]) -> None:
        with self._lock:
            self._tables.update(tables)

    def get(self, name: str) -> pd.DataFrame:
        with self._lock:
            if name not in self._tables:
                raise KeyError(f"unknown table {name!r}; have {sorted(self._tables)}")
            return self._tables[name]

    def has(self, name: str) -> bool:
        return name in self._tables

    def drop(self, name: str) -> None:
        with self._lock:
            self._tables.pop(name, None)

    def names(self) -> list[str]:
        return sorted(self._tables)

    def tables(self) -> dict[str, pd.DataFrame]:
        with self._lock:
            return dict(self._tables)

    def meta(self) -> list[TableMeta]:
        return [
            TableMeta(name, int(len(df)), int(len(df.columns)), [str(c) for c in df.columns])
            for name, df in sorted(self._tables.items())
        ]

    def __iter__(self) -> Iterator[tuple[str, pd.DataFrame]]:
        return iter(self.tables().items())

    def __len__(self) -> int:
        return len(self._tables)

    # -- persistence -----------------------------------------------------
    def persist(self) -> None:
        with self._lock, self._current_path.open("wb") as handle:
            pickle.dump(self._tables, handle, protocol=PICKLE_PROTOCOL)

    def load(self) -> bool:
        if not self._current_path.exists():
            return False
        with self._lock, self._current_path.open("rb") as handle:
            self._tables = pickle.load(handle)
        return True

    def snapshot(self, tag: str) -> None:
        """Freeze the current tables under ``tag`` (used before each step)."""

        with self._lock, self._snapshot_path(tag).open("wb") as handle:
            pickle.dump(self._tables, handle, protocol=PICKLE_PROTOCOL)

    def has_snapshot(self, tag: str) -> bool:
        return self._snapshot_path(tag).exists()

    def restore(self, tag: str) -> bool:
        path = self._snapshot_path(tag)
        if not path.exists():
            return False
        with self._lock, path.open("rb") as handle:
            self._tables = pickle.load(handle)
        self.persist()
        return True

    def delete(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)
        self._tables.clear()


_stores: dict[str, TableStore] = {}
_registry_lock = threading.Lock()


def get_store(session_id: str) -> TableStore:
    """Process-wide store registry; reloads from disk after a restart."""

    with _registry_lock:
        store = _stores.get(session_id)
        if store is None:
            store = TableStore(session_id)
            store.load()
            _stores[session_id] = store
        return store


def drop_store(session_id: str) -> None:
    """Erase everything on disk that belongs to one session.

    The store is constructed even when the registry has never seen it: after a
    restart the pickles are on disk with nothing in memory pointing at them, so
    a delete that only cleared the registry would leave a deleted user's data
    lying in ``var/sessions``.  The uploaded originals go with it — they are as
    much the user's data as the tables parsed out of them.
    """

    with _registry_lock:
        store = _stores.pop(session_id, None)
    (store or TableStore(session_id)).delete()
    shutil.rmtree(get_settings().upload_dir / session_id, ignore_errors=True)
