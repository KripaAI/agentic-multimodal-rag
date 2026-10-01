"""Argon2id password hashing and the password policy (spec §7.6, NIST SP 800-63B, LLD §5.8)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

_HASHER = PasswordHasher()  # argon2-cffi's recommended Argon2id parameters


class PolicyError(ValueError):
    pass


def hash_password(password: str) -> str:
    return _HASHER.hash(password)


def verify_password(encoded: str, password: str) -> bool:
    try:
        return _HASHER.verify(encoded, password)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


def needs_rehash(encoded: str) -> bool:
    """True when the hash uses older parameters than the current ones (upgraded at the next sign-in)."""
    try:
        return _HASHER.check_needs_rehash(encoded)
    except InvalidHashError:
        return True


@lru_cache(maxsize=1)
def _dummy_hash() -> str:
    return hash_password("dummy password for equal timing")


def verify_dummy(password: str) -> None:
    """Spend the same time as a real check, for unknown or disabled accounts (no timing leak)."""
    verify_password(_dummy_hash(), password)


@lru_cache(maxsize=4)
def load_common(path: Path) -> frozenset[str]:
    if not path.is_file():
        raise FileNotFoundError(f"common-password list not found: {path} (run scripts/fetch_common_passwords.py)")
    return frozenset(line.strip().lower() for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()
                     if line.strip())


def check_policy(password: str, min_length: int, common: frozenset[str] | set[str], email: str) -> None:
    """Length and a blocklist; no forced character mixing and no expiry (NIST SP 800-63B)."""
    if len(password) < min_length:
        raise PolicyError(f"Use at least {min_length} characters.")
    if password.lower() in common:
        raise PolicyError("That password is too common. Choose another.")
    local = email.split("@")[0].lower()
    if email.lower() in password.lower() or (len(local) >= 4 and local in password.lower()):
        raise PolicyError("The password must not contain your email address.")
