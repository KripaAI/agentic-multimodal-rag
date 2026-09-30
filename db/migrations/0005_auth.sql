-- Phase 7: sign-in (spec §7.6, LLD §4.1, §5.8). Passwords and session tokens are never stored:
-- only Argon2id hashes and SHA-256 token hashes.

CREATE TABLE users (
    user_id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    email                text NOT NULL UNIQUE CHECK (email = lower(email)),
    password_hash        text NOT NULL,                 -- Argon2id encoded (salt and parameters included)
    role                 text NOT NULL DEFAULT 'user' CHECK (role IN ('admin', 'user')),
    status               text NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'disabled')),
    must_change_password boolean NOT NULL DEFAULT true,
    failed_attempts      int NOT NULL DEFAULT 0,
    locked_until         timestamptz,
    created_at           timestamptz NOT NULL DEFAULT now(),
    last_login_at        timestamptz
);

CREATE TABLE sessions (
    session_id   uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id      uuid NOT NULL REFERENCES users ON DELETE CASCADE,
    token_hash   text NOT NULL UNIQUE,                  -- SHA-256 of the 256-bit token
    created_at   timestamptz NOT NULL DEFAULT now(),
    last_seen_at timestamptz NOT NULL DEFAULT now(),
    expires_at   timestamptz NOT NULL,                  -- absolute expiry
    revoked_at   timestamptz
);
CREATE INDEX sessions_user ON sessions (user_id) WHERE revoked_at IS NULL;

CREATE TABLE auth_events (
    event_id        bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    user_id         uuid REFERENCES users ON DELETE SET NULL,
    email_attempted text,
    event_type      text NOT NULL CHECK (event_type IN ('login_success', 'login_failure', 'lockout', 'logout',
                        'password_change', 'password_reset', 'user_created', 'user_disabled', 'user_enabled',
                        'user_unlocked')),
    ip              inet,
    created_at      timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX auth_events_ip_failures ON auth_events (ip, created_at) WHERE event_type = 'login_failure';
CREATE INDEX auth_events_created ON auth_events (created_at);

-- query_log.user_id exists since Phase 4 without a constraint; the users table exists now.
ALTER TABLE query_log ADD CONSTRAINT query_log_user_fk FOREIGN KEY (user_id) REFERENCES users ON DELETE SET NULL;
CREATE INDEX query_log_user_day ON query_log (user_id, created_at);
