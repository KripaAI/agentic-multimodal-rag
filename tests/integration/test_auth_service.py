"""Sign-in, sessions, lockout, limits and account admin against a real database (spec §7.6,
LLD §5.8, plan Phase 7 acceptance criteria). Test-first."""

from __future__ import annotations

import logging

import pytest

import mmrag.db as db
from mmrag.auth import service
from mmrag.auth.service import GENERIC_ERROR, LimitReached
from mmrag.config import load_settings

pytestmark = pytest.mark.integration
PW = "correct horse battery staple"


@pytest.fixture
def settings(fresh_db_url, base_config, write_config, tmp_path):
    common = tmp_path / "common.txt"
    common.write_text("password1234\nqwertyuiop12\n", encoding="utf-8")
    base_config["auth"]["common_passwords_file"] = str(common)
    base_config["observability"].update(enabled=False, log_file=None)
    s = load_settings(write_config(base_config), {"DATABASE_URL": fresh_db_url})
    db.migrate(s)
    return s


@pytest.fixture
def user(settings):
    temp = service.add_user(settings, "Alice@Example.com ", role="user")
    token = service.login(settings, "alice@example.com", temp, "10.0.0.1").token
    service.change_password(settings, token, temp, PW)
    return settings


def _sql(settings, q, *a):
    with db.connect(settings) as conn:
        r = conn.execute(q, a).fetchall() if q.lstrip().upper().startswith("SELECT") else conn.execute(q, a)
        conn.commit()
        return r


def test_new_accounts_get_a_temporary_password_and_must_change_it(settings):
    temp = service.add_user(settings, "bob@example.com")
    assert len(temp) >= 16
    result = service.login(settings, "BOB@example.com", temp, "10.0.0.1")
    assert result.ok and result.must_change_password
    assert service.require_user(settings, result.token).email == "bob@example.com"


def test_wrong_password_and_unknown_email_get_the_same_message(user):
    wrong = service.login(user, "alice@example.com", "not the password!!", "10.0.0.2")
    unknown = service.login(user, "nobody@example.com", "not the password!!", "10.0.0.2")
    assert not wrong.ok and not unknown.ok and wrong.message == unknown.message == GENERIC_ERROR


def test_five_failures_lock_the_account_even_for_the_right_password(user):
    for _ in range(5):
        service.login(user, "alice@example.com", "wrong password here", "10.0.0.3")
    locked = service.login(user, "alice@example.com", PW, "10.0.0.3")
    assert not locked.ok and locked.message == GENERIC_ERROR
    assert _sql(user, "SELECT event_type FROM auth_events WHERE event_type = 'lockout'")
    _sql(user, "UPDATE users SET locked_until = now() - interval '1 second'")  # 15 minutes later
    assert service.login(user, "alice@example.com", PW, "10.0.0.3").ok


def test_unlock_clears_a_lockout(user):
    for _ in range(5):
        service.login(user, "alice@example.com", "wrong password here", "10.0.0.4")
    service.unlock(user, "alice@example.com")
    assert service.login(user, "alice@example.com", PW, "10.0.0.4").ok


def test_one_ip_with_many_failures_is_throttled(user):
    user.auth.ip_max_failed_attempts  # 20 in config
    for n in range(user.auth.ip_max_failed_attempts):
        service.login(user, f"guess{n}@example.com", "wrong password here", "10.9.9.9")
    assert not service.login(user, "alice@example.com", PW, "10.9.9.9").ok  # right password, throttled IP
    assert service.login(user, "alice@example.com", PW, "10.0.0.5").ok  # another IP is fine


def test_logout_and_password_change_end_every_session(user):
    # spec §7.6: logout, password change, reset and disable revoke all of the user's sessions
    t1 = service.login(user, "alice@example.com", PW, "10.0.0.6").token
    t2 = service.login(user, "alice@example.com", PW, "10.0.0.6").token
    service.logout(user, t1)
    assert service.require_user(user, t1) is None and service.require_user(user, t2) is None
    t3 = service.login(user, "alice@example.com", PW, "10.0.0.6").token
    service.change_password(user, t3, PW, "another good passphrase")
    assert service.require_user(user, t3) is None


def test_idle_and_absolute_expiry(user):
    token = service.login(user, "alice@example.com", PW, "10.0.0.7").token
    _sql(user, "UPDATE sessions SET last_seen_at = now() - interval '9 hours'")
    assert service.require_user(user, token) is None
    token = service.login(user, "alice@example.com", PW, "10.0.0.7").token
    _sql(user, "UPDATE sessions SET expires_at = now() - interval '1 second' WHERE revoked_at IS NULL")
    assert service.require_user(user, token) is None


def test_a_disabled_user_cannot_sign_in_and_loses_sessions(user):
    token = service.login(user, "alice@example.com", PW, "10.0.0.8").token
    service.disable(user, "alice@example.com")
    assert service.require_user(user, token) is None
    assert service.login(user, "alice@example.com", PW, "10.0.0.8").message == GENERIC_ERROR
    service.enable(user, "alice@example.com")
    assert service.login(user, "alice@example.com", PW, "10.0.0.8").ok


def test_password_change_needs_the_current_password_and_a_new_one(user):
    token = service.login(user, "alice@example.com", PW, "10.0.0.9").token
    with pytest.raises(ValueError):
        service.change_password(user, token, "not the current one", "another good passphrase")
    with pytest.raises(ValueError, match="differ"):
        service.change_password(user, token, PW, PW)
    with pytest.raises(ValueError, match="common"):
        service.change_password(user, token, PW, "password1234")


def test_reset_password_revokes_sessions_and_forces_a_change(user):
    token = service.login(user, "alice@example.com", PW, "10.0.0.10").token
    temp = service.reset_password(user, "alice@example.com")
    assert service.require_user(user, token) is None
    assert service.login(user, "alice@example.com", temp, "10.0.0.10").must_change_password


def test_daily_limits_block_the_question_before_any_api_call(user):
    uid = service.require_user(user, service.login(user, "alice@example.com", PW, "10.0.0.11").token).user_id
    service.check_limits(user, uid)  # nothing asked yet
    _sql(user, "INSERT INTO query_log (user_id, thread_id, trace_id, question, model, rounds, tokens_in, tokens_out, "
               "cost_usd, latency_ms, validator_result, answer_json) "
               "SELECT %s, 't', 'x', 'q', 'm', 1, 1, 1, %s, 1, 'ok', '{}'", uid, user.auth.daily_cost_limit_usd)
    with pytest.raises(LimitReached, match="resets"):
        service.check_limits(user, uid)


def test_no_password_or_token_is_stored_or_logged(user, caplog):
    caplog.set_level(logging.DEBUG)
    result = service.login(user, "alice@example.com", PW, "10.0.0.12")
    dump = " ".join(str(r) for r in _sql(user, "SELECT * FROM users") + _sql(user, "SELECT * FROM sessions")
                    + _sql(user, "SELECT * FROM auth_events"))
    assert PW not in dump and result.token not in dump
    assert PW not in caplog.text and result.token not in caplog.text


def test_every_attempt_is_audited(user):
    service.login(user, "alice@example.com", "wrong password here", "10.0.0.13")
    service.login(user, "alice@example.com", PW, "10.0.0.13")
    events = [r[0] for r in _sql(user, "SELECT event_type FROM auth_events WHERE ip = '10.0.0.13' ORDER BY event_id")]
    assert events == ["login_failure", "login_success"]
