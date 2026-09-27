"""Command-line entry points (LLD §3.9). Phase 0: `check`, `db migrate`, `models`."""

from __future__ import annotations

import argparse
import sys
from typing import Callable

from opentelemetry import trace

from mmrag.config import ConfigError, Settings, get_settings
from mmrag.obs import get_logger, get_tracer, init_telemetry, shutdown_telemetry

_log = get_logger("mmrag.cli")


# ---------------------------------------------------------------- check

def _check_database(settings: Settings) -> str:
    from mmrag import db

    with db.connect(settings) as conn:
        server = conn.execute("SHOW server_version").fetchone()[0]
        row = conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()
        if row is None:
            raise RuntimeError("pgvector not enabled; run `mmrag db migrate`")
        # HNSW smoke test on a temporary table (dropped at session end).
        conn.execute("CREATE TEMP TABLE _phase0_hnsw (id int, v vector(3))")
        conn.execute("INSERT INTO _phase0_hnsw VALUES (1, '[1,0,0]'), (2, '[0,1,0]'), (3, '[0,0,1]')")
        conn.execute("CREATE INDEX ON _phase0_hnsw USING hnsw (v vector_cosine_ops)")
        nearest = conn.execute(
            "SELECT id FROM _phase0_hnsw ORDER BY v <=> '[0.9,0.1,0]' LIMIT 1"
        ).fetchone()[0]
        if nearest != 1:
            raise RuntimeError(f"HNSW query returned id {nearest}, expected 1")
    return f"PostgreSQL {server}, pgvector {row[0]}, HNSW index OK"


def _check_openai(settings: Settings) -> str:
    from mmrag.llm import get_client

    client = get_client(settings)
    n_models = sum(1 for _ in client.models.list())
    emb = client.embeddings.create(
        model=settings.embed.model, input="phase 0 check", dimensions=settings.embed.dims
    )
    got = len(emb.data[0].embedding)
    if got != settings.embed.dims:
        raise RuntimeError(f"embedding has {got} dims, config says {settings.embed.dims}")
    msg = f"{n_models} models visible; {settings.embed.model} returns {got} dims"
    if settings.agent.model:
        chat = client.chat.completions.create(
            model=settings.agent.model,
            messages=[{"role": "user", "content": "Reply with the single word: ready"}],
            max_completion_tokens=200,
        )
        msg += f"; {settings.agent.model} replied {chat.choices[0].message.content!r}"
    else:
        msg += "; agent.model not set yet (choose one with `mmrag models`)"
    return msg


def cmd_check(settings: Settings, args: argparse.Namespace) -> int:
    tracer = get_tracer("mmrag.cli")
    checks: list[tuple[str, Callable[[Settings], str], bool]] = [
        ("database", _check_database, True),
        ("openai", _check_openai, settings.secrets.openai_api_key is not None),
    ]
    failed = False
    with tracer.start_as_current_span("phase0.check") as root:
        trace_id = format(root.get_span_context().trace_id, "032x")
        print("config    OK   config.yaml + .env loaded and validated")
        for name, fn, enabled in checks:
            if not enabled:
                print(f"{name:<9} SKIP OPENAI_API_KEY not set in .env")
                continue
            with tracer.start_as_current_span(f"check.{name}") as span:
                try:
                    detail = fn(settings)
                    print(f"{name:<9} OK   {detail}")
                    _log.info("check %s passed: %s", name, detail)
                except Exception as e:
                    failed = True
                    span.record_exception(e)
                    span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                    print(f"{name:<9} FAIL {type(e).__name__}: {e}")
                    _log.error("check %s failed: %s", name, e)
    obs = settings.observability
    if not obs.enabled:
        print("telemetry     disabled (observability.enabled: false)")
    else:
        endpoint = obs.otlp_traces_endpoint
        print(f"telemetry     trace {trace_id} -> {endpoint or 'no exporter configured'}")
        if endpoint:
            print("              open http://localhost:6006 to view it in Phoenix")
    return 1 if failed else 0


# ---------------------------------------------------------------- db / models

def cmd_db_migrate(settings: Settings, args: argparse.Namespace) -> int:
    from mmrag import db

    applied = db.migrate(settings)
    print("Applied: " + ", ".join(applied) if applied else "Database is up to date.")
    return 0


def cmd_models(settings: Settings, args: argparse.Namespace) -> int:
    from mmrag.llm import get_client

    ids = sorted(m.id for m in get_client(settings).models.list())
    embedding = [i for i in ids if "embedding" in i]
    other = [i for i in ids if "embedding" not in i]
    print("Embedding models:\n  " + "\n  ".join(embedding))
    print("\nOther models (set agent.model / eval.judge_model in config.yaml):\n  " + "\n  ".join(other))
    return 0


# ---------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mmrag")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="verify config, database, telemetry and OpenAI access")
    sub.add_parser("models", help="list OpenAI models visible to your API key")
    db_parser = sub.add_parser("db", help="database commands")
    db_sub = db_parser.add_subparsers(dest="db_command", required=True)
    db_sub.add_parser("migrate", help="apply pending SQL migrations")
    args = parser.parse_args(argv)

    try:
        settings = get_settings()
    except ConfigError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 2
    init_telemetry(settings, service_name="mmrag-cli")

    handlers = {
        "check": cmd_check,
        "models": cmd_models,
        "db": {"migrate": cmd_db_migrate}.get(getattr(args, "db_command", None)),
    }
    try:
        return handlers[args.command](settings, args)
    finally:
        shutdown_telemetry()


if __name__ == "__main__":
    sys.exit(main())
