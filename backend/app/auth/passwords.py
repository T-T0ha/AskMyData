"""Password hashing.

scrypt from the standard library, rather than bcrypt or Argon2 from PyPI.
The reason is not that scrypt is better than Argon2 — it is not — but that it
is *there*: the platform already carries a large dependency tree, and a login
that cannot be verified because a native wheel failed to build is a worse
outcome than a hash function one generation behind the state of the art.
scrypt is memory-hard, which is the property that matters against GPU attack
and the one PBKDF2 lacks.

Every hash carries its own parameters::

    scrypt$16384$8$1$<salt-b64>$<digest-b64>

so raising the cost is a configuration change, not a migration: old hashes
keep verifying under the parameters they were made with, and
:func:`needs_rehash` tells the caller when to re-hash a password it has just
seen in plaintext — which is only ever at login.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets

from app.core.config import get_settings

ALGORITHM = "scrypt"
SALT_BYTES = 16
KEY_BYTES = 32


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def _unb64(text: str) -> bytes:
    return base64.b64decode(text.encode("ascii"))


def _derive(password: str, salt: bytes, n: int, r: int, p: int) -> bytes:
    # maxmem must be given explicitly: OpenSSL's own default is 32 MB and
    # refuses anything larger, so a raised work factor would fail at runtime
    # rather than simply cost more.  128 * n * r is scrypt's own formula for
    # the memory it needs; the doubling is headroom for the p parallelism.
    return hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=n,
        r=r,
        p=p,
        dklen=KEY_BYTES,
        maxmem=128 * n * r * 2 + (1 << 20),
    )


def hash_password(password: str) -> str:
    """Encode a password for storage.  Never returns the same string twice."""

    settings = get_settings()
    n, r, p = settings.auth_scrypt_n, settings.auth_scrypt_r, settings.auth_scrypt_p
    salt = secrets.token_bytes(SALT_BYTES)
    digest = _derive(password, salt, n, r, p)
    return f"{ALGORITHM}${n}${r}${p}${_b64(salt)}${_b64(digest)}"


def verify_password(password: str, encoded: str) -> bool:
    """Constant-time check of a password against a stored hash.

    Returns ``False`` rather than raising on a malformed or unknown-algorithm
    hash: a corrupt row must fail the login, not the request.
    """

    try:
        algorithm, n, r, p, salt, digest = encoded.split("$")
        if algorithm != ALGORITHM:
            return False
        expected = _unb64(digest)
        actual = _derive(password, _unb64(salt), int(n), int(r), int(p))
    except (ValueError, TypeError, MemoryError):
        return False
    return hmac.compare_digest(expected, actual)


def needs_rehash(encoded: str) -> bool:
    """True when a stored hash was made with weaker parameters than current."""

    settings = get_settings()
    try:
        algorithm, n, r, p, _salt, _digest = encoded.split("$")
    except ValueError:
        return True
    if algorithm != ALGORITHM:
        return True
    return (int(n), int(r), int(p)) != (
        settings.auth_scrypt_n,
        settings.auth_scrypt_r,
        settings.auth_scrypt_p,
    )


def password_problem(password: str) -> str | None:
    """Why this password is unacceptable, or ``None`` if it is fine.

    Length only.  Composition rules ("one digit, one symbol") push users
    towards short predictable passwords that satisfy a checker rather than an
    attacker, and this project has no reason to repeat that.
    """

    settings = get_settings()
    minimum = settings.auth_min_password_length
    if len(password) < minimum:
        return f"password must be at least {minimum} characters"
    if len(password.encode("utf-8")) > 1024:
        # scrypt cost is independent of input length, but an unbounded field
        # is still an unbounded field.
        return "password must be at most 1024 bytes"
    if password.strip() == "":
        return "password cannot be only whitespace"
    return None
