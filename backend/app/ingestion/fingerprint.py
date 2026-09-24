"""Content fingerprints for cross-session duplicate detection.

Nothing stops a user from clicking "New session" and uploading (or
reconnecting to) the same data a second time — the two sessions would look
identical afterwards, and the account ends up with what looks like two
different datasets that are actually one. These fingerprints give a source a
stable identity so :func:`app.api.services.find_duplicate_session` can catch
that before it happens, independent of filename or which table subset was
picked.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from sqlalchemy.engine import make_url


def file_fingerprint(path: Path) -> str:
    """A content hash — a re-uploaded copy under a new filename is still a duplicate."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return f"file:{digest.hexdigest()}"


def database_fingerprint(url: str, tables: list[str] | None) -> str:
    """Identity of a live connection: dialect, host, database and the chosen
    tables — never the password, so this never has to keep a secret."""

    parsed = make_url(url)
    identity = "|".join(
        [
            (parsed.drivername or "").split("+")[0],
            parsed.host or "",
            str(parsed.port or ""),
            parsed.database or "",
            parsed.username or "",
            ",".join(sorted(tables)) if tables else "*",
        ]
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return f"db:{digest}"
