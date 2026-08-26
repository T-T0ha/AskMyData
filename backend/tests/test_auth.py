"""Authentication, and the ownership boundary it exists to enforce.

Two halves.  The first is the account lifecycle — register, sign in, sign out,
rotate a password, delete an account.  The second is the part that actually
protects anything: that one user cannot see, read, modify or delete another
user's dataset, and cannot learn that it exists.
"""

from __future__ import annotations

import uuid

import pytest

from app.auth import passwords, service
from tests.conftest import register_client

PASSWORD = "correct-horse-battery"


def _email() -> str:
    return f"user-{uuid.uuid4().hex[:10]}@example.test"


# ---------------------------------------------------------------------------
# password hashing
# ---------------------------------------------------------------------------


def test_hashing_is_salted_and_verifiable():
    first = passwords.hash_password(PASSWORD)
    second = passwords.hash_password(PASSWORD)

    assert first != second, "two hashes of one password must differ — the salt is per-hash"
    assert PASSWORD not in first
    assert passwords.verify_password(PASSWORD, first)
    assert passwords.verify_password(PASSWORD, second)
    assert not passwords.verify_password(PASSWORD + "x", first)


def test_a_corrupt_hash_fails_the_login_rather_than_the_request():
    for broken in ["", "not-a-hash", "scrypt$bad$8$1$xx$yy", "md5$1$1$1$aa$bb"]:
        assert passwords.verify_password(PASSWORD, broken) is False


def test_stored_parameters_travel_with_the_hash(monkeypatch):
    """A hash made under weaker parameters still verifies, and is flagged for upgrade."""

    from app.core.config import get_settings

    settings = get_settings()
    weak = passwords.hash_password(PASSWORD)
    assert not passwords.needs_rehash(weak)

    monkeypatch.setattr(settings, "auth_scrypt_n", settings.auth_scrypt_n * 2)
    assert passwords.verify_password(PASSWORD, weak), "old hashes must keep working"
    assert passwords.needs_rehash(weak), "but should be marked for re-hashing"


def test_short_passwords_are_rejected():
    assert passwords.password_problem("short") is not None
    assert passwords.password_problem("   " + " " * 20) is not None
    assert passwords.password_problem(PASSWORD) is None


# ---------------------------------------------------------------------------
# registration and sign-in
# ---------------------------------------------------------------------------


def test_register_returns_a_usable_token_and_never_the_hash(anon_client):
    response = anon_client.post(
        "/api/auth/register",
        json={"email": _email(), "password": PASSWORD, "name": "Toha"},
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["token_type"] == "bearer"
    assert payload["user"]["name"] == "Toha"
    assert "password" not in response.text and "password_hash" not in response.text

    me = anon_client.get(
        "/api/auth/me", headers={"Authorization": f"Bearer {payload['access_token']}"}
    )
    assert me.status_code == 200
    assert me.json()["user"]["id"] == payload["user"]["id"]


def test_email_is_normalised_so_one_address_is_one_account(anon_client):
    email = _email()
    assert anon_client.post(
        "/api/auth/register", json={"email": email.upper(), "password": PASSWORD}
    ).status_code == 201

    duplicate = anon_client.post(
        "/api/auth/register", json={"email": f"  {email}  ", "password": PASSWORD}
    )
    assert duplicate.status_code == 409

    # ...and the mixed-case spelling signs in to the account it created.
    assert anon_client.post(
        "/api/auth/login", json={"email": email.upper(), "password": PASSWORD}
    ).status_code == 200


@pytest.mark.parametrize(
    "body,status",
    [
        ({"email": "not-an-email", "password": PASSWORD}, 422),
        ({"email": _email(), "password": "short"}, 422),
    ],
)
def test_registration_refuses_bad_input(anon_client, body, status):
    assert anon_client.post("/api/auth/register", json=body).status_code == status


def test_wrong_password_and_unknown_account_are_indistinguishable(anon_client):
    email = _email()
    anon_client.post("/api/auth/register", json={"email": email, "password": PASSWORD})

    wrong = anon_client.post("/api/auth/login", json={"email": email, "password": "wrong-password"})
    unknown = anon_client.post("/api/auth/login", json={"email": _email(), "password": PASSWORD})

    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json()["detail"] == unknown.json()["detail"], (
        "distinguishable errors tell an attacker which addresses have accounts"
    )


def test_repeated_failures_are_throttled(anon_client):
    from app.core.config import get_settings

    email = _email()
    anon_client.post("/api/auth/register", json={"email": email, "password": PASSWORD})

    limit = get_settings().auth_max_failed_logins
    for _ in range(limit):
        assert anon_client.post(
            "/api/auth/login", json={"email": email, "password": "nope-nope-nope"}
        ).status_code == 401

    blocked = anon_client.post("/api/auth/login", json={"email": email, "password": PASSWORD})
    assert blocked.status_code == 429, "the correct password must not slip past the throttle"

    service.reset_throttle()
    assert anon_client.post(
        "/api/auth/login", json={"email": email, "password": PASSWORD}
    ).status_code == 200


def test_login_records_the_moment(anon_client):
    email = _email()
    created = anon_client.post(
        "/api/auth/register", json={"email": email, "password": PASSWORD}
    ).json()
    assert created["user"]["last_login_at"] is None

    signed_in = anon_client.post(
        "/api/auth/login", json={"email": email, "password": PASSWORD}
    ).json()
    assert signed_in["user"]["last_login_at"] is not None


# ---------------------------------------------------------------------------
# tokens
# ---------------------------------------------------------------------------


def test_protected_endpoints_refuse_missing_and_bogus_tokens(anon_client):
    assert anon_client.get("/api/auth/me").status_code == 401
    assert anon_client.get("/api/sessions").status_code == 401

    bad = anon_client.get("/api/auth/me", headers={"Authorization": "Bearer not-a-real-token"})
    assert bad.status_code == 401
    assert bad.headers.get("www-authenticate") == "Bearer"


def test_logout_revokes_only_the_token_it_was_called_with(anon_client):
    email = _email()
    anon_client.post("/api/auth/register", json={"email": email, "password": PASSWORD})
    laptop = anon_client.post(
        "/api/auth/login", json={"email": email, "password": PASSWORD}
    ).json()["access_token"]
    phone = anon_client.post(
        "/api/auth/login", json={"email": email, "password": PASSWORD}
    ).json()["access_token"]

    assert anon_client.post(
        "/api/auth/logout", headers={"Authorization": f"Bearer {laptop}"}
    ).status_code == 204

    assert anon_client.get(
        "/api/auth/me", headers={"Authorization": f"Bearer {laptop}"}
    ).status_code == 401
    assert anon_client.get(
        "/api/auth/me", headers={"Authorization": f"Bearer {phone}"}
    ).status_code == 200


def test_an_expired_token_stops_working(anon_client, monkeypatch):
    from app.core.config import get_settings

    monkeypatch.setattr(get_settings(), "auth_token_ttl_hours", -1)
    email = _email()
    token = anon_client.post(
        "/api/auth/register", json={"email": email, "password": PASSWORD}
    ).json()["access_token"]

    assert anon_client.get(
        "/api/auth/me", headers={"Authorization": f"Bearer {token}"}
    ).status_code == 401


def test_changing_the_password_signs_other_devices_out(anon_client):
    email = _email()
    anon_client.post("/api/auth/register", json={"email": email, "password": PASSWORD})
    here = anon_client.post(
        "/api/auth/login", json={"email": email, "password": PASSWORD}
    ).json()["access_token"]
    elsewhere = anon_client.post(
        "/api/auth/login", json={"email": email, "password": PASSWORD}
    ).json()["access_token"]

    changed = anon_client.post(
        "/api/auth/password",
        json={"current_password": PASSWORD, "new_password": "a-much-better-secret"},
        headers={"Authorization": f"Bearer {here}"},
    )
    assert changed.status_code == 200
    assert changed.json()["revoked"] >= 1

    assert anon_client.get(
        "/api/auth/me", headers={"Authorization": f"Bearer {elsewhere}"}
    ).status_code == 401
    assert anon_client.get(
        "/api/auth/me", headers={"Authorization": f"Bearer {here}"}
    ).status_code == 200
    assert anon_client.post(
        "/api/auth/login", json={"email": email, "password": "a-much-better-secret"}
    ).status_code == 200


def test_changing_the_password_requires_the_current_one(anon_client):
    email = _email()
    token = anon_client.post(
        "/api/auth/register", json={"email": email, "password": PASSWORD}
    ).json()["access_token"]

    refused = anon_client.post(
        "/api/auth/password",
        json={"current_password": "not-it-either", "new_password": "a-much-better-secret"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert refused.status_code == 401


def test_live_logins_are_listed_with_the_current_one_marked(anon_client):
    email = _email()
    first = anon_client.post(
        "/api/auth/register", json={"email": email, "password": PASSWORD}
    ).json()["access_token"]
    anon_client.post("/api/auth/login", json={"email": email, "password": PASSWORD})

    logins = anon_client.get(
        "/api/auth/sessions", headers={"Authorization": f"Bearer {first}"}
    ).json()["logins"]
    assert len(logins) == 2
    assert sum(1 for login in logins if login["current"]) == 1
    assert all("token" not in str(login) for login in logins)


# ---------------------------------------------------------------------------
# ownership — the point of the module
# ---------------------------------------------------------------------------


def test_a_dataset_belongs_to_the_account_that_made_it(client):
    created = client.post("/api/sessions", json={"name": "mine"}).json()
    me = client.get("/api/auth/me").json()["user"]

    assert created["user_id"] == me["id"]
    assert [s["id"] for s in client.get("/api/sessions").json()["sessions"]] == [created["id"]]


def test_another_users_dataset_is_reported_as_missing_not_forbidden(client, other_client):
    mine = client.post("/api/sessions", json={"name": "mine"}).json()["id"]

    # Nothing of mine appears in their listing...
    assert other_client.get("/api/sessions").json()["sessions"] == []

    # ...and every way of addressing it directly answers 404, which is also
    # what a genuinely unknown id gets.  403 would confirm the id is real.
    for method, path in [
        ("get", f"/api/sessions/{mine}"),
        ("get", f"/api/sessions/{mine}/triage"),
        ("get", f"/api/sessions/{mine}/semantics"),
        ("get", f"/api/sessions/{mine}/equivalences"),
        ("get", f"/api/sessions/{mine}/cleaning/state"),
        ("get", f"/api/sessions/{mine}/tables/customers/preview"),
        ("get", f"/api/sessions/{mine}/export/versions"),
        ("get", f"/api/sessions/{mine}/export/documentation"),
        ("post", f"/api/sessions/{mine}/semantics"),
        ("post", f"/api/sessions/{mine}/cleaning/start"),
        ("delete", f"/api/sessions/{mine}"),
    ]:
        response = getattr(other_client, method)(path)
        assert response.status_code == 404, f"{method.upper()} {path} leaked: {response.status_code}"

    # And it is still there afterwards.
    assert client.get(f"/api/sessions/{mine}").status_code == 200


def test_every_dataset_endpoint_refuses_an_anonymous_caller(anon_client):
    for method, path in [
        ("get", "/api/sessions"),
        ("post", "/api/sessions"),
        ("get", "/api/sessions/anything"),
        ("get", "/api/sessions/anything/triage"),
        ("post", "/api/sessions/anything/semantics"),
        ("post", "/api/sessions/anything/cleaning/start"),
        ("post", "/api/sources/inspect"),
    ]:
        response = anon_client.request(
            method.upper(), path, json={} if method == "post" else None
        )
        assert response.status_code == 401, f"{method.upper()} {path} was reachable anonymously"


def test_status_and_vocabulary_stay_public(anon_client):
    """The sign-in screen shows whether the backend is up, before anyone is signed in."""

    assert anon_client.get("/health").status_code == 200
    assert anon_client.get("/api/status").status_code == 200
    assert anon_client.get("/api/vocabulary").status_code == 200
    assert anon_client.get("/api/auth/config").status_code == 200


def test_uploaded_data_does_not_leak_between_accounts(client, other_client, messy_workbook):
    session_id = client.post("/api/sessions", json={"name": "private"}).json()["id"]
    with messy_workbook.open("rb") as handle:
        client.post(f"/api/sessions/{session_id}/upload", files={"file": (messy_workbook.name, handle)})

    assert client.get(f"/api/sessions/{session_id}/tables/customers/preview").status_code == 200
    assert (
        other_client.get(f"/api/sessions/{session_id}/tables/customers/preview").status_code == 404
    ), "a table preview must not be readable by an account that does not own the dataset"


# ---------------------------------------------------------------------------
# account deletion (FR-15)
# ---------------------------------------------------------------------------


def test_deleting_an_account_takes_its_datasets_and_files_with_it(client, messy_workbook):
    from app.core.config import get_settings

    session_id = client.post("/api/sessions", json={"name": "doomed"}).json()["id"]
    with messy_workbook.open("rb") as handle:
        client.post(f"/api/sessions/{session_id}/upload", files={"file": (messy_workbook.name, handle)})

    uploads = get_settings().upload_dir / session_id
    working = get_settings().upload_dir.parent / "sessions" / session_id
    assert uploads.exists()

    deleted = client.post("/api/auth/delete", json={"password": PASSWORD})
    assert deleted.status_code == 200
    assert deleted.json()["deleted_datasets"] == 1

    assert not uploads.exists(), "the uploaded originals are the user's data too"
    assert not working.exists(), "the working tables outlive the database row unless removed"
    assert client.get("/api/auth/me").status_code == 401


def test_deleting_an_account_requires_the_password(client):
    refused = client.post("/api/auth/delete", json={"password": "not-the-password"})
    assert refused.status_code == 401
    assert client.get("/api/auth/me").status_code == 200
