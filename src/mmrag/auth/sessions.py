"""Server-side sessions (spec §7.6, LLD §5.8): a random 256-bit token lives only in the browser
tab's Streamlit state; the database stores its SHA-256 hash."""

from __future__ import annotations

import hashlib
import secrets

from mmrag.config import Settings

TOUCH_EVERY_S = 60  # last_seen_at is written at most once a minute


def new_token() -> str:
    return secrets.token_urlsafe(32)  # 256 bits


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def create(conn, user_id: str, settings: Settings) -> str:
    token = new_token()
    conn.execute("INSERT INTO sessions (user_id, token_hash, expires_at) "
                 "VALUES (%s, %s, now() + make_interval(secs => %s))",
                 (user_id, token_hash(token), settings.auth.session_absolute_days * 86400))
    return token


def lookup(conn, token: str | None, settings: Settings):
    """(user_id, email, role, must_change_password) for a live session of an active user, else None.
    Live = not revoked, before its absolute expiry, and used within the idle limit."""
    if not token:
        return None
    row = conn.execute(
        "SELECT s.session_id, u.user_id, u.email, u.role, u.must_change_password, "
        "       extract(epoch FROM now() - s.last_seen_at) "
        "FROM sessions s JOIN users u USING (user_id) "
        "WHERE s.token_hash = %s AND s.revoked_at IS NULL AND s.expires_at > now() "
        "  AND s.last_seen_at > now() - make_interval(secs => %s) AND u.status = 'active'",
        (token_hash(token), settings.auth.session_idle_hours * 3600)).fetchone()
    if row is None:
        return None
    if row[5] > TOUCH_EVERY_S:
        conn.execute("UPDATE sessions SET last_seen_at = now() WHERE session_id = %s", (row[0],))
    return row[1:5]


def revoke_all(conn, user_id: str) -> None:
    conn.execute("UPDATE sessions SET revoked_at = now() WHERE user_id = %s AND revoked_at IS NULL", (user_id,))
