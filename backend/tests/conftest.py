"""Shared test fixtures.

The suite runs against a temporary SQLite database and a temporary upload
directory, so it never touches a developer's real session data and needs no
PostgreSQL server.  The production target is still PostgreSQL + pgvector; the
dialect-aware column type in ``app.db.models`` is what lets both work.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest

os.environ.setdefault("ALLOW_EMBEDDING_FALLBACK", "true")


@pytest.fixture(scope="session", autouse=True)
def _isolated_environment(tmp_path_factory: pytest.TempPathFactory) -> None:
    """Point config at a scratch database/upload dir before anything imports it."""

    root = tmp_path_factory.mktemp("semanticlayer")
    os.environ["DATABASE_URL"] = f"sqlite:///{root / 'test.db'}"
    os.environ["UPLOAD_DIR"] = str(root / "uploads")
    os.environ["ANTHROPIC_API_KEY"] = ""
    # scrypt at the production work factor costs ~60 ms and ~16 MB per hash;
    # every authenticated test pays it at least twice.  The algorithm under
    # test is the same one either way — only the cost differs — and the
    # parameters live in the hash, so this cannot mask a verification bug.
    os.environ["AUTH_SCRYPT_N"] = "1024"

    from app.core.config import get_settings

    get_settings.cache_clear()
    get_settings()


@pytest.fixture(scope="session")
def messy_workbook() -> Path:
    from tests.fixtures.make_fixtures import build_messy_workbook

    return build_messy_workbook()


@pytest.fixture(scope="session")
def clean_workbook() -> Path:
    from tests.fixtures.make_fixtures import build_clean_workbook

    return build_clean_workbook()


@pytest.fixture(scope="session")
def relational_workbook() -> Path:
    from tests.fixtures.make_fixtures import build_relational_workbook

    return build_relational_workbook()


@pytest.fixture(scope="session")
def relational_tables(relational_workbook: Path):
    """The relational fixture as ``{table: DataFrame}``, straight from Phase 0."""

    from app.ingestion.loader import load_workbook_sheets

    return {r.name: r.dataframe for r in load_workbook_sheets(relational_workbook)}


@pytest.fixture(scope="session")
def messy_sheets(messy_workbook: Path):
    from app.ingestion.loader import load_workbook_sheets

    return {r.name: r for r in load_workbook_sheets(messy_workbook)}


@pytest.fixture
def anon_client():
    """A client with no credentials — for the auth tests themselves."""

    from fastapi.testclient import TestClient

    from app.auth.service import reset_throttle
    from app.db.base import init_db
    from app.main import app

    init_db()
    reset_throttle()
    with TestClient(app) as test_client:
        yield test_client


def register_client(client, email: str, password: str = "correct-horse-battery"):
    """Create an account on ``client`` and leave it signed in.  Returns the token."""

    response = client.post(
        "/api/auth/register",
        json={"email": email, "password": password, "name": email.split("@")[0]},
    )
    assert response.status_code == 201, response.text
    token = response.json()["access_token"]
    client.headers["Authorization"] = f"Bearer {token}"
    return token


@pytest.fixture
def client(anon_client):
    """The default client: signed in as a fresh account.

    Every dataset endpoint is owner-scoped, so a test that wants to exercise a
    pipeline phase needs an account first.  Making that the default keeps the
    phase tests about the phase.
    """

    register_client(anon_client, f"tester-{uuid.uuid4().hex[:8]}@example.test")
    return anon_client


@pytest.fixture
def other_client():
    """A second signed-in account, for proving datasets do not leak between them."""

    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as test_client:
        register_client(test_client, f"other-{uuid.uuid4().hex[:8]}@example.test")
        yield test_client
