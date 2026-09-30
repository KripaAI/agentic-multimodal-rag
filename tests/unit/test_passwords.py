"""Password hashing and policy (spec §7.6, NIST SP 800-63B). Test-first."""

from __future__ import annotations

import pytest

from mmrag.auth.passwords import PolicyError, check_policy, hash_password, needs_rehash, verify_password

pytestmark = pytest.mark.unit

COMMON = {"password1234", "qwertyuiop12"}


def test_argon2id_hash_verifies_and_never_contains_the_password():
    h = hash_password("correct horse battery")
    assert h.startswith("$argon2id$") and "correct horse" not in h
    assert verify_password(h, "correct horse battery") and not verify_password(h, "wrong horse battery")


def test_the_same_password_gets_a_different_hash_each_time():
    assert hash_password("same password 123") != hash_password("same password 123")


def test_a_garbage_hash_never_verifies():
    assert not verify_password("not-a-hash", "anything at all")


def test_old_parameters_are_flagged_for_rehash():
    from argon2 import PasswordHasher

    weak = PasswordHasher(time_cost=1, memory_cost=8, parallelism=1).hash("some long password")
    assert needs_rehash(weak) and not needs_rehash(hash_password("some long password"))


@pytest.mark.parametrize("pw, why", [("short", "at least 12"), ("Password1234", "common"),
                                     ("alice@example.com-x", "email")])
def test_policy_rejects(pw, why):
    with pytest.raises(PolicyError, match=why):
        check_policy(pw, min_length=12, common=COMMON, email="alice@example.com")


def test_policy_needs_no_symbols_or_digits():
    # NIST: length and a blocklist, not forced character mixing
    check_policy("a long lowercase passphrase", min_length=12, common=COMMON, email="a@b.c")
