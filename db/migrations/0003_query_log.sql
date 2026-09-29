-- Phase 4: one row per question (spec §5.5, LLD §4.1).
-- user_id has no foreign key yet: the users table arrives with sign-in in Phase 7, whose
-- migration adds the constraint. LangGraph's checkpoint tables are created by the
-- checkpointer's own setup(), not here.

CREATE TABLE query_log (
    query_id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id          uuid,
    thread_id        text NOT NULL,
    trace_id         text NOT NULL,             -- 32 hex characters; opens the trace in Phoenix
    question         text NOT NULL,
    qtype            text,
    model            text NOT NULL,
    rounds           int  NOT NULL,
    tool_calls       jsonb NOT NULL DEFAULT '[]',
    tokens_in        int  NOT NULL,
    tokens_out       int  NOT NULL,
    cost_usd         numeric(10, 6),            -- null when the model has no price in config
    latency_ms       int  NOT NULL,
    validator_result text NOT NULL CHECK (validator_result IN ('ok', 'repaired', 'dropped_blocks')),
    answer_json      jsonb NOT NULL,            -- the hydrated answer (restores chat history later)
    created_at       timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX query_log_thread ON query_log (thread_id, created_at);
CREATE INDEX query_log_created ON query_log (created_at);
