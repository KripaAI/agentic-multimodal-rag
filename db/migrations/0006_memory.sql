-- Phase 9: long-term memory (spec §7.8, LLD §5.11, constitution P13).
--
-- The memories themselves live in LangGraph's `store` table, which `PostgresStore.setup()`
-- creates on first use (D15); it is not managed here, because its schema belongs to the
-- library (it already indexes `prefix`, which carries the user's namespace). What belongs to
-- the application is the per-user switch (FR-25).

ALTER TABLE users ADD COLUMN memory_enabled boolean NOT NULL DEFAULT true;

-- Account deletion (FR-25: "deleting an account deletes its memories") needs an audit event
-- of its own; the user row is gone by then, so only the email is kept.
ALTER TABLE auth_events DROP CONSTRAINT auth_events_event_type_check;
ALTER TABLE auth_events ADD CONSTRAINT auth_events_event_type_check CHECK (event_type IN (
    'login_success', 'login_failure', 'lockout', 'logout', 'password_change', 'password_reset',
    'user_created', 'user_disabled', 'user_enabled', 'user_unlocked', 'user_deleted'));
