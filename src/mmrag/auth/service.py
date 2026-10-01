"""Sign-in, sessions, limits and account administration (spec §7.6, LLD §5.8).

Every sign-in failure returns GENERIC_ERROR, whatever the reason (unknown email, wrong password,
disabled, locked, throttled IP), and unknown or disabled accounts are checked against a dummy
hash so every attempt takes the same time. Passwords and tokens are never logged or stored.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from mmrag import db
from mmrag.auth import sessions
from mmrag.auth.passwords import check_policy, hash_password, load_common, needs_rehash, verify_dummy, verify_password
from mmrag.config import Settings
from mmrag.obs import get_logger, get_tracer

GENERIC_ERROR = "Invalid email or password, or too many attempts. Try again later."
_log = get_logger("mmrag.auth")


@dataclass
class User:
    user_id: str
    email: str
    role: str
    must_change_password: bool

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"


@dataclass
class LoginResult:
    ok: bool
    message: str = ""
    token: str | None = None
    must_change_password: bool = False


class LimitReached(RuntimeError):
    pass


def _norm(email: str) -> str:
    return email.strip().lower()


def _event(conn, event_type: str, user_id=None, email=None, ip=None) -> None:
    conn.execute("INSERT INTO auth_events (user_id, email_attempted, event_type, ip) VALUES (%s, %s, %s, %s)",
                 (user_id, email, event_type, ip))


# ---------------------------------------------------------------- sign-in and sessions

def login(settings: Settings, email: str, password: str, ip: str | None) -> LoginResult:
    email, a = _norm(email), settings.auth
    with get_tracer("mmrag.auth").start_as_current_span("auth.login"), db.connect(settings) as conn:
        failures_from_ip = conn.execute(
            "SELECT count(*) FROM auth_events WHERE event_type = 'login_failure' AND ip = %s "
            "AND created_at > now() - make_interval(mins => %s)", (ip, a.lockout_minutes)).fetchone()[0] if ip else 0
        row = conn.execute("SELECT user_id, password_hash, status, locked_until > now(), must_change_password "
                           "FROM users WHERE email = %s", (email,)).fetchone()

        def fail(user_id=None) -> LoginResult:
            _event(conn, "login_failure", user_id, email, ip)
            conn.commit()
            _log.info("sign-in failed")  # no email, password or token in logs
            return LoginResult(ok=False, message=GENERIC_ERROR)

        if failures_from_ip >= a.ip_max_failed_attempts:
            verify_dummy(password)
            return fail(row[0] if row else None)
        if row is None or row[2] != "active":
            verify_dummy(password)
            return fail(row[0] if row else None)
        user_id, encoded, _, locked, must_change = row
        if locked:
            verify_dummy(password)
            return fail(user_id)
        if not verify_password(encoded, password):
            attempts = conn.execute("UPDATE users SET failed_attempts = failed_attempts + 1 WHERE user_id = %s "
                                    "RETURNING failed_attempts", (user_id,)).fetchone()[0]
            if attempts >= a.max_failed_attempts:
                conn.execute("UPDATE users SET locked_until = now() + make_interval(mins => %s), failed_attempts = 0 "
                             "WHERE user_id = %s", (a.lockout_minutes, user_id))
                _event(conn, "lockout", user_id, email, ip)
            return fail(user_id)
        if needs_rehash(encoded):
            conn.execute("UPDATE users SET password_hash = %s WHERE user_id = %s", (hash_password(password), user_id))
        conn.execute("UPDATE users SET failed_attempts = 0, locked_until = NULL, last_login_at = now() "
                     "WHERE user_id = %s", (user_id,))
        token = sessions.create(conn, user_id, settings)
        _event(conn, "login_success", user_id, email, ip)
        conn.commit()
        return LoginResult(ok=True, token=token, must_change_password=must_change)


def require_user(settings: Settings, token: str | None) -> User | None:
    """Called on every Streamlit rerun; None means: clear the state and show the sign-in screen."""
    with db.connect(settings) as conn:
        row = sessions.lookup(conn, token, settings)
        conn.commit()
    return User(str(row[0]), row[1], row[2], row[3]) if row else None


def logout(settings: Settings, token: str | None) -> None:
    with db.connect(settings) as conn:
        row = sessions.lookup(conn, token, settings)
        if row:
            sessions.revoke_all(conn, row[0])
            _event(conn, "logout", row[0], row[1])
        conn.commit()


def change_password(settings: Settings, token: str | None, current: str, new: str) -> None:
    """Needs the current password; the new one must pass the policy and differ. Ends every session."""
    with db.connect(settings) as conn:
        row = sessions.lookup(conn, token, settings)
        if row is None:
            raise PermissionError("not signed in")
        user_id, email = row[0], row[1]
        encoded = conn.execute("SELECT password_hash FROM users WHERE user_id = %s", (user_id,)).fetchone()[0]
        if not verify_password(encoded, current):
            raise ValueError("The current password is not correct.")
        if new == current:
            raise ValueError("The new password must differ from the current one.")
        check_policy(new, settings.auth.min_password_length, _common(settings), email)
        conn.execute("UPDATE users SET password_hash = %s, must_change_password = false WHERE user_id = %s",
                     (hash_password(new), user_id))
        sessions.revoke_all(conn, user_id)
        _event(conn, "password_change", user_id, email)
        conn.commit()


def _common(settings: Settings) -> frozenset[str]:
    return load_common(settings.resolve(settings.auth.common_passwords_file))


# ---------------------------------------------------------------- per-user daily limits

def usage_today(settings: Settings, user_id: str) -> tuple[int, float, datetime]:
    """(questions, cost in US$, when the day resets) in `auth.limits_timezone`."""
    tz = ZoneInfo(settings.auth.limits_timezone)
    start = datetime.now(tz).replace(hour=0, minute=0, second=0, microsecond=0)
    with db.connect(settings) as conn:
        n, cost = conn.execute("SELECT count(*), coalesce(sum(cost_usd), 0) FROM query_log "
                               "WHERE user_id = %s AND created_at >= %s", (user_id, start)).fetchone()
    return n, float(cost), start + timedelta(days=1)


def check_limits(settings: Settings, user_id: str) -> None:
    """Before each question: over either daily limit raises LimitReached, and the agent is not run."""
    with get_tracer("mmrag.auth").start_as_current_span("auth.check_limits") as span:
        n, cost, reset = usage_today(settings, user_id)
        span.set_attribute("mmrag.questions_today", n)
        span.set_attribute("mmrag.cost_today_usd", cost)
        a = settings.auth
        hit = n >= a.daily_question_limit or cost >= a.daily_cost_limit_usd
        span.set_attribute("mmrag.limit_hit", hit)
        if hit:
            what = (f"{a.daily_question_limit} questions" if n >= a.daily_question_limit
                    else f"US${a.daily_cost_limit_usd:.2f} of usage")
            raise LimitReached(f"You have reached today's limit of {what}. It resets at "
                               f"{reset:%H:%M} {a.limits_timezone} ({reset:%Y-%m-%d}).")


# ---------------------------------------------------------------- account administration (CLI only)

def _temporary_password() -> str:
    return secrets.token_urlsafe(12)  # 16 characters; shown once, stored only as a hash


def _user_id(conn, email: str) -> str:
    row = conn.execute("SELECT user_id FROM users WHERE email = %s", (_norm(email),)).fetchone()
    if row is None:
        raise ValueError(f"no account for {_norm(email)}")
    return row[0]


def add_user(settings: Settings, email: str, role: str = "user") -> str:
    """Creates the account and returns its temporary password (the caller shows it once)."""
    temp = _temporary_password()
    with db.connect(settings) as conn:
        uid = conn.execute("INSERT INTO users (email, password_hash, role, must_change_password) "
                           "VALUES (%s, %s, %s, true) RETURNING user_id",
                           (_norm(email), hash_password(temp), role)).fetchone()[0]
        _event(conn, "user_created", uid, _norm(email))
        conn.commit()
    return temp


def reset_password(settings: Settings, email: str) -> str:
    temp = _temporary_password()
    with db.connect(settings) as conn:
        uid = _user_id(conn, email)
        conn.execute("UPDATE users SET password_hash = %s, must_change_password = true, failed_attempts = 0, "
                     "locked_until = NULL WHERE user_id = %s", (hash_password(temp), uid))
        sessions.revoke_all(conn, uid)
        _event(conn, "password_reset", uid, _norm(email))
        conn.commit()
    return temp


def _set_status(settings: Settings, email: str, status: str, event: str) -> None:
    with db.connect(settings) as conn:
        uid = _user_id(conn, email)
        conn.execute("UPDATE users SET status = %s WHERE user_id = %s", (status, uid))
        if status == "disabled":
            sessions.revoke_all(conn, uid)
        _event(conn, event, uid, _norm(email))
        conn.commit()


def disable(settings: Settings, email: str) -> None:
    _set_status(settings, email, "disabled", "user_disabled")


def enable(settings: Settings, email: str) -> None:
    _set_status(settings, email, "active", "user_enabled")


def unlock(settings: Settings, email: str) -> None:
    with db.connect(settings) as conn:
        uid = _user_id(conn, email)
        conn.execute("UPDATE users SET locked_until = NULL, failed_attempts = 0 WHERE user_id = %s", (uid,))
        _event(conn, "user_unlocked", uid, _norm(email))
        conn.commit()


def list_users(settings: Settings) -> list[tuple]:
    with db.connect(settings) as conn:
        return conn.execute("SELECT email, role, status, locked_until > now(), last_login_at FROM users "
                            "ORDER BY email").fetchall()
