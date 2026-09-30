-- Phase 6: evaluation runs and per-question scores (spec §9, LLD §4.1, §5.10).

CREATE TABLE eval_runs (
    run_id             uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    started_at         timestamptz NOT NULL DEFAULT now(),
    finished_at        timestamptz,
    git_commit         text,
    config_hash        text NOT NULL,
    agent_model        text NOT NULL,
    judge_model        text NOT NULL,
    ragas_version      text NOT NULL,             -- judge prompt version (own RAGAS-style judge, Phase 6)
    golden_set_version text NOT NULL,
    is_baseline        boolean NOT NULL DEFAULT false,
    label              text,                      -- e.g. "rerank on"
    summary            jsonb,                     -- mean per metric, per question type, cost, latency
    passed             boolean,                   -- regression gate
    trace_id           text
);
CREATE INDEX eval_runs_baseline ON eval_runs (is_baseline, started_at);

CREATE TABLE eval_results (
    run_id      uuid NOT NULL REFERENCES eval_runs (run_id) ON DELETE CASCADE,
    question_id text NOT NULL,
    metric      text NOT NULL,
    score       numeric,                          -- 0-1; null when the metric does not apply
    details     jsonb NOT NULL DEFAULT '{}',
    trace_id    text,
    PRIMARY KEY (run_id, question_id, metric)
);
