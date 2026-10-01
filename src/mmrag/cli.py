"""Command-line entry points (LLD §3.9).

Phase 0: `check`, `db migrate`, `models`. Phase 1: `profile`, `ingest parse`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

from opentelemetry import trace

from mmrag.config import PROJECT_ROOT, ConfigError, Settings, get_settings
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


def cmd_db_backup(settings: Settings, args: argparse.Namespace) -> int:
    import datetime

    from mmrag.db.backup import backup

    out = Path(args.out) if args.out else (settings.resolve(settings.paths.data_dir) / "backups"
                                           / f"mmrag-{datetime.datetime.now():%Y%m%d-%H%M%S}.dump")
    path = backup(settings, out)
    print(f"Backup: {path} ({path.stat().st_size / 1e6:.1f} MB)")
    return 0


def cmd_db_restore(settings: Settings, args: argparse.Namespace) -> int:
    from mmrag.db.backup import restore

    name = restore(settings, Path(args.dump), args.into)
    print(f"Restored into database {name}. To use it, point DATABASE_URL in .env at {name}.")
    return 0


def cmd_db_verify_restore(settings: Settings, args: argparse.Namespace) -> int:
    """Restore into a scratch database, run the retrieval test queries on both, then drop it."""
    import uuid

    from mmrag.db.backup import compare_search, drop_database, restore
    from mmrag.retrieval.report import load_queries

    scratch = f"mmrag_verify_{uuid.uuid4().hex[:8]}"
    restore(settings, Path(args.dump), scratch)
    try:
        queries = [(q.query, q.collection) for q in load_queries(PROJECT_ROOT / "eval" / "retrieval_queries.yaml")]
        diffs = compare_search(settings, scratch, queries)
    finally:
        drop_database(settings, scratch)
    if diffs:
        for q, coll, a, b in diffs:
            print(f"DIFFERENT: {coll} '{q}': live {a[:3]}… vs restored {b[:3]}…")
        return 1
    print(f"Restore verified: {len(queries)} search queries return identical results on the restored copy.")
    return 0


def cmd_models(settings: Settings, args: argparse.Namespace) -> int:
    from mmrag.llm import get_client

    ids = sorted(m.id for m in get_client(settings).models.list())
    embedding = [i for i in ids if "embedding" in i]
    other = [i for i in ids if "embedding" not in i]
    print("Embedding models:\n  " + "\n  ".join(embedding))
    print("\nOther models (set agent.model / eval.judge_model in config.yaml):\n  " + "\n  ".join(other))
    return 0


# ---------------------------------------------------------------- profile / ingest

def _pdf_paths(settings: Settings, names: list[str]) -> list[Path]:
    """The named PDFs, or every PDF in `paths.pdf_dir` when none are named."""
    pdf_dir = settings.resolve(settings.paths.pdf_dir)
    if not names:
        return sorted(pdf_dir.glob("*.pdf"))
    paths = []
    for name in names:
        path = Path(name) if Path(name).is_file() else pdf_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"PDF not found: {name} (looked in {pdf_dir})")
        paths.append(path)
    return paths


def cmd_profile(settings: Settings, args: argparse.Namespace) -> int:
    """Profile PDFs; print a summary table and write `data/elements/corpus_profile.json`."""
    from mmrag.ingest.parse import profile_pdf

    try:
        paths = _pdf_paths(settings, args.pdfs)
    except FileNotFoundError as e:
        print(e)
        return 1
    if not paths:
        print(f"No PDFs found in {settings.resolve(settings.paths.pdf_dir)}")
        return 1
    profiles = []
    print(f"{'file':<50} {'pages':>5} {'images':>6} {'tables':>6} {'vector-heavy':>12} {'textless':>8}")
    for path in paths:
        p = profile_pdf(path, settings.parse)
        profiles.append(p)
        print(f"{path.name[:50]:<50} {len(p.pages):>5} {sum(x.images for x in p.pages):>6} "
              f"{sum(x.tables for x in p.pages):>6} {sum(x.vector_heavy for x in p.pages):>12} "
              f"{sum(x.textless for x in p.pages):>8}")
    out = settings.resolve(settings.paths.data_dir) / "elements" / "corpus_profile.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("[\n" + ",\n".join(p.model_dump_json() for p in profiles) + "\n]\n", encoding="utf-8")
    print(f"\nProfile written to {out}")
    return 0


def cmd_ingest_parse(settings: Settings, args: argparse.Namespace) -> int:
    """Parse one PDF, write its outputs and print their paths and counts."""
    from collections import Counter

    from mmrag.ingest.parse import parse_document, write_outputs

    try:
        (path,) = _pdf_paths(settings, [args.pdf])
    except FileNotFoundError as e:
        print(e)
        return 1
    result = parse_document(path, settings)
    files = write_outputs(result, path, settings)

    print(f"{result.source_file}  (doc_id {result.doc_id}, {result.page_count} pages)\n")
    status = Counter((e.type, e.status) for e in result.elements)
    print(f"  {'element':<14} {'ok':>6} {'skipped':>8} {'review':>7}")
    for etype in ("text", "image", "vector_figure", "table", "scanned_page"):
        print(f"  {etype:<14} {status[(etype, 'ok')]:>6} {status[(etype, 'skipped')]:>8} "
              f"{status[(etype, 'needs_review')]:>7}")
    low = sum(t.low_confidence for t in result.tables)
    print(f"\n  low-confidence tables: {low}")
    for kind, count in sorted(Counter(s.kind for s in result.skips).items()):
        print(f"  skipped {kind}: {count}")
    for name, count in sorted(Counter(r.filter for r in result.rejected).items()):
        print(f"  rejected {name}: {count}")
    print()
    for name, file in files.items():
        print(f"  {name:<13} {file}")
    return 0


# ---------------------------------------------------------------- obs (Phase 8)

def cmd_obs(settings: Settings, args: argparse.Namespace) -> int:
    from mmrag.obs.ops import check_alerts, cleanup

    if args.obs_command == "cleanup":
        r = settings.retention
        if not args.yes:
            print(f"This deletes query history older than {r.query_log_days} days, sign-in events older than "
                  f"{r.auth_events_days} days, memories older than {settings.memory.retention_days} days "
                  f"and old sessions. Re-run with --yes to confirm.")
            return 1
        for table, n in cleanup(settings).items():
            print(f"  {table:18} {n} deleted")
        return 0
    alerts = check_alerts(settings)
    for a in alerts:
        print(f"ALERT: {a}")
    if not alerts:
        print("No alerts.")
    return 1 if alerts else 0  # non-zero lets a scheduled task notify


# ---------------------------------------------------------------- doc (Phase 8)

def cmd_doc(settings: Settings, args: argparse.Namespace) -> int:
    """Manage the library one PDF at a time; the other PDFs are never re-processed."""
    from mmrag.index import library

    c = args.doc_command
    if c == "list":
        docs = library.list_documents(settings)
        for d in docs:
            print(f"{d.source_file:58} v{d.version}  {d.text:>4} text {d.figure:>4} figure {d.table:>4} table  "
                  f"indexed {d.ingested_at:%Y-%m-%d}")
        print(f"{len(docs)} documents")
        return 0
    if c == "remove":
        if not args.yes:
            print(f"This deletes {args.name} and all its search data. Re-run with --yes to confirm.")
            return 1
        library.remove_document(settings, args.name)
        print(f"Removed {args.name} (its PDF is kept in {settings.paths.pdf_dir}/removed/)")
        return 0
    try:
        dest = library.add_pdf(settings, Path(args.path)) if c == "add" \
            else library.replace_pdf(settings, args.name, Path(args.path))
    except (FileExistsError, FileNotFoundError) as e:
        print(e)
        return 1
    print(f"{'Added' if c == 'add' else 'Replaced'} {dest.name}. Parsing it now...\n")
    rc = cmd_ingest_parse(settings, argparse.Namespace(pdf=dest.name))
    if c == "replace":
        print("\nSearch still answers from the previous version until the index step below swaps it in.")
    print(f"\nNext steps for {dest.name}:\n"
          f"  1. Check the review sheet printed above.\n"
          f"  2. Figure descriptions on the GPU: mmrag caption bundle|push|pull|import \"{dest.name}\"\n"
          f"  3. Index it: mmrag ingest index \"{dest.name}\"")
    return rc


# ---------------------------------------------------------------- main

# ---------------------------------------------------------------- index and search (Phase 3)

def cmd_ingest_index(settings: Settings, args: argparse.Namespace) -> int:
    """Enrich, chunk, embed and write one parsed (and captioned) PDF to PostgreSQL."""
    from mmrag.index.pipeline import index_document

    try:
        (pdf,) = _pdf_paths(settings, [args.pdf])
        report = index_document(pdf, settings)
    except FileNotFoundError as e:
        print(e)
        return 1
    methods = {m: sum(lk.method == m for lk in report.links) for m in ("explicit", "deictic", "related")}
    review = sum(c.status != "ok" for c in report.doc.captions.values())
    print(f"{report.doc.source_file}: {report.result}")
    print(f"  search documents: {report.counts['text']} text, {report.counts['figure']} figure, "
          f"{report.counts['table']} table")
    print(f"  links: {methods['explicit']} explicit, {methods['deictic']} deictic, {methods['related']} related; "
          f"{len(report.unlinked)} unlinked")
    print(f"  captions needing review after the label check: {review}")
    print(f"  link report: {report.link_report}")
    return 0


def cmd_compare_summaries(settings: Settings, args: argparse.Namespace) -> int:
    """Summarize every table with several models, side by side, for the owner to choose."""
    from mmrag.index.document import load_document
    from mmrag.index.reports import write_summary_comparison
    from mmrag.llm import complete, get_client

    (pdf,) = _pdf_paths(settings, [args.pdf])
    doc = load_document(pdf, settings)
    client = get_client(settings)
    from mmrag.ingest.enrich import _SUMMARY_PROMPT

    results = {}
    for model in [m.strip() for m in args.models.split(",")]:
        results[model] = {}
        for eid, t in doc.tables.items():
            prompt = _SUMMARY_PROMPT.format(title=t.title or "(none)", columns=" | ".join(t.columns),
                                            rows="\n".join(" | ".join(r) for r in t.rows[:25]))
            results[model][eid] = complete(client, model, prompt)
        print(f"  {model}: {len(doc.tables)} tables summarized")
    out = settings.resolve(settings.paths.data_dir) / "elements" / doc.doc_id / "summary_comparison.html"
    print(f"Comparison page: {write_summary_comparison(doc, results, out)}")
    return 0


def cmd_search(settings: Settings, args: argparse.Namespace) -> int:
    from mmrag.retrieval.hybrid import COLLECTIONS, search

    for coll in (COLLECTIONS if args.collection == "all" else [args.collection]):
        print(f"\n== {coll}")
        for n, h in enumerate(search(settings, args.query, coll, args.k), 1):
            pages = sorted({loc["page"] for loc in h.locations})
            ranks = f"sem {h.semantic_rank or '-'} / kw {h.keyword_rank or '-'}"
            print(f"{n:>2}. {h.score:.4f} ({ranks}) {h.source_file} p{','.join(map(str, pages))}  "
                  f"{' '.join(h.dense_text.split())[:110]}")
            for r in h.related:
                print(f"      related {r['type']} p{r['page']} ({r['method']}): {' '.join((r['title'] or '').split())[:80]}")
    return 0


def cmd_search_report(settings: Settings, args: argparse.Namespace) -> int:
    from mmrag.retrieval.report import run_retrieval_report

    out, passed, total = run_retrieval_report(settings, Path(args.queries))
    print(f"Expected result in the top 5 for {passed}/{total} queries ({passed / total:.0%}; gate: 80%)")
    print(f"Report: {out}")
    return 0 if passed / total >= 0.8 else 1


# ---------------------------------------------------------------- ask (Phase 4)

def cmd_ask(settings: Settings, args: argparse.Namespace) -> int:
    """Answer one question with the agent; write the answer page; print its path, the thread
    id (for follow-ups), cost, latency and the trace id."""
    from mmrag.agent.graph import run_query
    from mmrag.agent.render import write_answer_page

    run = run_query(args.question, settings, thread_id=args.thread, model=args.model)
    page = write_answer_page(args.question, run, settings)
    cost = f"${run.cost_usd:.4f}" if run.cost_usd is not None else "cost n/a (add the model to `pricing`)"
    print(f"Answer page: {page}")
    print(f"thread {run.thread_id} (continue with --thread {run.thread_id}), {run.model}, {run.rounds} rounds, "
          f"{len(run.tool_calls)} tool calls, {cost}, {run.latency_ms / 1000:.1f} s, validator "
          f"{run.validator_result}, trace {run.trace_id}")
    for n in run.answer.notices:
        print(f"  notice: {n}")
    return 0


def cmd_ask_batch(settings: Settings, args: argparse.Namespace) -> int:
    """Answer every question in a YAML file with each model in --models; write each answer
    page and a comparison page (answers side by side, cost, latency, validator results)."""
    from mmrag.agent.batch import load_questions, run_batch, write_comparison_page
    from mmrag.agent.graph import run_query
    from mmrag.agent.render import write_answer_page

    questions = load_questions(Path(args.questions))
    models = [m.strip() for m in args.models.split(",")] if args.models else [settings.agent.model]
    unpriced = [m for m in models if m not in settings.pricing]
    if unpriced:
        print(f"Note: no price in config for {', '.join(unpriced)}; their cost is not counted toward --max-cost")
    results = run_batch(questions, models, settings, ask=run_query, page=write_answer_page, max_cost=args.max_cost)
    failed = sum(1 for out in results.values() for o in out.values() if o.error)
    print(f"{len(questions)} questions x {len(models)} models, {failed} failed or skipped")
    print(f"Comparison page: {write_comparison_page(questions, results, settings)}")
    return 1 if failed else 0


# ---------------------------------------------------------------- user (Phase 7)

def cmd_user(settings: Settings, args: argparse.Namespace) -> int:
    """Account administration (CLI only; there is no public sign-up). Temporary passwords are
    shown once here and stored only as Argon2id hashes."""
    from mmrag.auth import service

    c = args.user_command
    if c == "events":
        from collections import Counter

        events = service.recent_events(settings, args.days)
        for when, kind, email, ip in events[:args.limit]:
            print(f"{when:%Y-%m-%d %H:%M}  {kind:16} {email or '-':32} {ip or '-'}")
        print(f"\n{len(events)} events in {args.days} days: "
              + ", ".join(f"{k} {n}" for k, n in Counter(e[1] for e in events).most_common()))
        return 0
    if c == "list":
        for email, role, status, locked, last in service.list_users(settings):
            print(f"{email:32} {role:6} {status:9} {'LOCKED' if locked else '':7} last sign-in {last or '-'}")
        return 0
    if c == "delete":
        email = args.email.strip().lower()
        if not args.yes:
            print(f"This deletes the account {email}, its sessions and everything it remembers. "
                  f"Its questions stay in the history without a user. Re-run with --yes to confirm.")
            return 1
        print(f"Deleted {email} and {service.delete_user(settings, email)['memories']} memories.")
        return 0
    if c in ("add", "reset-password"):
        temp = service.add_user(settings, args.email, role=args.role) if c == "add" \
            else service.reset_password(settings, args.email)
        print(f"Temporary password for {args.email.strip().lower()} (shown once; they must change it at sign-in):")
        print(f"  {temp}")
        return 0
    {"disable": service.disable, "enable": service.enable, "unlock": service.unlock}[c](settings, args.email)
    print(f"{c}d {args.email.strip().lower()}" if c != "unlock" else f"unlocked {args.email.strip().lower()}")
    return 0


# ---------------------------------------------------------------- memory (Phase 9)

def _memory_llm(settings: Settings):
    """The low-cost extractor used for statements and episode summaries."""
    from mmrag.agent.llm import OpenAILLM
    from mmrag.llm import get_client

    model = settings.memory.model or settings.agent.summary_model or settings.agent.model
    if not model:
        raise SystemExit("memory.model is not set in config.yaml")
    return OpenAILLM(get_client(settings), model)


def cmd_memory(settings: Settings, args: argparse.Namespace) -> int:
    """See, delete and switch off what the assistant remembers about a user (FR-25).

    Memory is per user, so every command takes an email. Nothing here is evidence: these notes
    only shape how a question is understood and how an answer is presented (P13).
    """
    from mmrag import db, memory

    if not settings.memory.enabled:
        print("Memory is switched off for the whole application (memory.enabled in config.yaml).")
        return 1
    with db.connect(settings) as conn:
        row = conn.execute("SELECT user_id::text, memory_enabled FROM users WHERE email = %s",
                           (args.email.strip().lower(),)).fetchone()
    if row is None:
        print(f"No account for {args.email.strip().lower()} (see `mmrag user list`).", file=sys.stderr)
        return 2
    user_id, enabled = row
    c = args.memory_command

    if c in ("on", "off"):
        memory.set_enabled(settings, user_id, c == "on")
        print(f"Memory switched {c} for {args.email.strip().lower()}."
              + ("" if c == "on" else " Nothing more is stored or recalled; use `forget-all` to delete what is kept."))
        return 0

    with memory.open_store(settings) as store:
        if c == "list":
            items = memory.list_memories(store, user_id)
            for m in items:
                when = (m.updated_at or "")[:16].replace("T", " ")
                print(f"{m.kind:9} {m.key:28} {when:16} {m.text}")
            print(f"\n{len(items)} memories for {args.email.strip().lower()}"
                  f" (memory is {'on' if enabled else 'OFF'} for this account)")
            return 0
        if c == "delete":
            if not any(m.key == args.key for m in memory.list_memories(store, user_id, args.kind)):
                print(f"No {args.kind} memory with key {args.key!r} (see `mmrag memory list`).", file=sys.stderr)
                return 2
            memory.delete(store, user_id, args.kind, args.key)
            print(f"Deleted {args.kind} memory {args.key}.")
            return 0
        if c == "forget-all":
            if not args.yes:
                print("This deletes every memory of this account. Re-run with --yes to confirm.")
                return 1
            print(f"Deleted {memory.forget_all(store, user_id)} memories.")
            return 0
        # summarize: write the episode summaries of this account's finished conversations
        from mmrag.memory.episodes import summarize_idle

        written = summarize_idle(settings, store, _memory_llm(settings), args.idle_minutes, user_id)
        for m in written:
            print(f"{m.key:28} {m.text}")
        print(f"{len(written)} conversation(s) summarised.")
        return 0


# ---------------------------------------------------------------- eval (Phase 6)

def _judge(settings: Settings):
    from mmrag.eval.judge import Metrics, OpenAIJudge
    from mmrag.llm import get_client

    if not settings.eval.judge_model:
        raise SystemExit("eval.judge_model is not set in config.yaml")
    return Metrics(OpenAIJudge(get_client(settings), settings.eval.judge_model, settings.embed.model,
                               settings.embed.dims))


def cmd_eval(settings: Settings, args: argparse.Namespace) -> int:
    """Phase 6: draft golden questions, promote the chosen ones, run and report the evaluation."""
    from mmrag.eval.dataset import load_golden_set

    golden = settings.resolve(settings.eval.golden_set)
    drafts_file = golden.with_name("drafts.jsonl")
    data_dir = settings.resolve(settings.paths.data_dir)
    if args.eval_command == "draft-questions":
        from mmrag import db
        from mmrag.eval.drafts import draft_questions, write_drafts

        drafts = draft_questions(settings, _judge(settings).judge, seed=args.seed)
        with db.connect(settings) as conn:
            assets = dict(conn.execute("SELECT element_id, asset_path FROM elements WHERE asset_path IS NOT NULL"))
        page = data_dir / "eval" / "drafts_review.html"
        write_drafts(drafts, drafts_file, page, data_dir, assets)
        print(f"{len(drafts)} drafts -> {drafts_file}\nReview page: {page}")
        return 0
    if args.eval_command == "promote":
        from mmrag.eval.drafts import coverage, promote

        n = promote(drafts_file, golden, [i.strip() for i in args.ids.split(",") if i.strip()])
        print(f"Golden set: {n} questions -> {golden}")
        for book, types in coverage(golden).items():
            print(f"  {sum(types.values()):>3}  {book}  " + ", ".join(f"{t} {c}" for t, c in sorted(types.items())))
        return 0
    if args.eval_command == "retrieval":
        from mmrag.agent.validator import DbStore
        from mmrag.eval.retrieval_eval import apply_overrides, retrieval_scores
        from mmrag.retrieval.hybrid import search

        tuned = apply_overrides(settings, args.set or [])
        items, _ = load_golden_set(golden)
        out = retrieval_scores(items, lambda q, coll: search(tuned, q, coll, 10), DbStore(settings).locations)
        label = ", ".join(args.set or []) or "current config"
        print(f"{label}: " + " · ".join(f"{k} {v}" for k, v in out["overall"].items()))
        for coll, st in out["by_collection"].items():
            print(f"  {coll:7} " + " · ".join(f"{k} {v}" for k, v in st.items()))
        if args.show_misses:
            print("  missed: " + ", ".join(q for q, r in out["per_question"].items() if r["rank"] is None))
        return 0
    if args.eval_command == "report":
        from mmrag.eval.report import write_report

        print(f"Report: {write_report(settings, args.run_id)}")
        return 0
    # run
    from mmrag.agent.graph import run_query
    from mmrag.agent.validator import DbStore
    from mmrag.eval.report import write_report
    from mmrag.eval.runner import run_eval

    items, version = load_golden_set(golden)
    if args.only:
        wanted = set(args.only.split(","))
        items = [i for i in items if i.question_id in wanted or i.question_id in {x.follows for x in items
                                                                                 if x.question_id in wanted}]
    if args.limit:
        items = items[:args.limit]
    from mmrag.eval.retrieval_eval import apply_overrides

    tuned = apply_overrides(settings, args.set or [])  # e.g. search.rerank=true, agent.prompt_version=v2
    label = " ".join(x for x in [args.label, f"[{', '.join(args.set)}]" if args.set else ""] if x) or None
    model = args.model or tuned.agent.model
    if model == tuned.eval.judge_model:
        raise SystemExit("the judge model must differ from the agent model (D14)")
    print(f"Evaluating {len(items)} questions (golden set {version}) with {model}; judge {tuned.eval.judge_model}"
          + (f"; overrides {args.set}" if args.set else ""))
    from mmrag.eval.runner import keep_awake

    with keep_awake():  # a laptop dozing off would freeze the run mid-question
        out = run_eval(tuned, items, version, run_query, _judge(tuned), DbStore(tuned), model,
                       baseline=args.baseline, label=label, max_cost=args.max_cost)
    print(f"Spent ${out.summary.get('spent_usd', 0):.3f} (agent + judge)")
    print(f"\n{'PASSED' if out.passed else 'FAILED'} · run {out.run_id}")
    for f in out.failures:
        print(f"  gate: {f}")
    print(f"Report: {write_report(settings, out.run_id)}")
    return 0 if out.passed else 1


# ---------------------------------------------------------------- caption (Phase 2)

def _caption_dirs(settings: Settings, pdf: Path) -> tuple[str, Path]:
    from mmrag.ingest.ids import doc_id

    did = doc_id(pdf)
    return did, settings.resolve(settings.paths.data_dir) / "captions" / did


def _latest_run(caption_dir: Path, run: str | None) -> Path:
    if run:
        return Path(run)
    runs = sorted((caption_dir / "runs").glob("*")) if (caption_dir / "runs").is_dir() else []
    if not runs:
        raise FileNotFoundError(f"no pulled runs in {caption_dir / 'runs'}; run `mmrag caption pull` first")
    return runs[-1]


def cmd_caption(settings: Settings, args: argparse.Namespace) -> int:
    """Phase 2: bundle figures, run the VLM on Kaggle, compare models, import captions."""
    import datetime

    from mmrag.ingest import kaggle_job
    from mmrag.ingest.bundle import build_jobs, write_bundle
    from mmrag.ingest.caption_import import cache_path, import_captions
    from mmrag.ingest.caption_report import MODEL_ORDER, build_pilot_report
    from mmrag.ingest.captions import CaptionCache
    from mmrag.ingest.models import Element, Table

    try:
        (pdf,) = _pdf_paths(settings, [args.pdf])
    except FileNotFoundError as e:
        print(e)
        return 1
    did, cdir = _caption_dirs(settings, pdf)
    bundle = cdir / "bundle"

    if args.caption_command == "bundle":
        data = settings.resolve(settings.paths.data_dir)
        el_file, tab_file = data / "elements" / did / "elements.jsonl", data / "tables" / did / "tables.jsonl"
        if not el_file.is_file():
            print(f"No parse output for {pdf.name}; run `mmrag ingest parse {pdf.name}` first")
            return 1
        elements = [Element.model_validate_json(x) for x in el_file.read_text(encoding="utf-8").splitlines() if x]
        tables = [Table.model_validate_json(x) for x in tab_file.read_text(encoding="utf-8").splitlines() if x] \
            if tab_file.is_file() else []
        cache = None if args.pilot else CaptionCache(cache_path(settings))
        jobs = build_jobs(elements, tables, settings, cache=cache)
        if args.elements:
            wanted = {f"{did}:{e.strip()}" for e in args.elements.split(",")}
            jobs = [j for j in jobs if j.element_id in wanted]
        if not jobs:
            print("Nothing to caption: every figure is already cached for this model and prompt version.")
            return 0
        # vLLM's install may replace the GPU machine's torch, so the vLLM path runs last.
        models = ["3b", "nf4-7b", "awq-7b"] if args.pilot else [settings.caption.model_path]
        write_bundle(jobs, settings, bundle, models=models)
        print(f"Bundle: {len(jobs)} job(s) for {', '.join(models)} -> {bundle}")
        return 0

    if args.caption_command == "push":
        if not (bundle / "run.json").is_file():
            print(f"No bundle in {bundle}; run `mmrag caption bundle {pdf.name}` first")
            return 1
        ref = kaggle_job.push(bundle, did, settings, cdir)
        print(f"Started Kaggle job {ref}\nCheck progress with `mmrag caption status {pdf.name}` "
              f"or at https://www.kaggle.com/code/{ref}")
        return 0

    if args.caption_command == "status":
        print(kaggle_job.status(did, settings))
        return 0

    if args.caption_command == "pull":
        run = cdir / "runs" / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        kaggle_job.pull(did, settings, run)
        print(f"Downloaded to {run}")
        summary = run / "summary.json"
        if summary.is_file():
            for m, rep in json.loads(summary.read_text(encoding="utf-8")).items():
                print(f"  {m:<7} " + ("FAILED" if rep.get("failed") else f"{rep['ok']}/{rep['jobs']} valid, "
                                      f"{rep['seconds_per_figure']}s per figure"))
        return 0

    if args.caption_command == "review":
        from mmrag.ingest.caption_report import build_caption_review

        print(f"Caption review page: {build_caption_review(did, settings)}")
        return 0

    try:
        run = _latest_run(cdir, args.run)
    except FileNotFoundError as e:
        print(e)
        return 1
    if args.caption_command == "report":
        out = build_pilot_report(run, bundle, cdir / "pilot_report.html")
        print(f"Comparison page: {out}")
        return 0
    if args.caption_command == "import":
        model = args.model or settings.caption.model_path
        result = import_captions(run / model / "captions.jsonl", did, settings)
        print(f"Imported {result.ok} caption(s); {result.needs_review} need review -> {result.path}")
        for e in result.errors:
            print(f"  invalid record, not imported: {e}")
        return 0
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mmrag")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="verify config, database, telemetry and OpenAI access")
    sub.add_parser("models", help="list OpenAI models visible to your API key")
    db_parser = sub.add_parser("db", help="database commands")
    db_sub = db_parser.add_subparsers(dest="db_command", required=True)
    db_sub.add_parser("migrate", help="apply pending SQL migrations")
    db_backup = db_sub.add_parser("backup", help="dump the whole database (default: data/backups/)")
    db_backup.add_argument("--out")
    db_restore = db_sub.add_parser("restore", help="restore a dump into a NEW database (never overwrites)")
    db_restore.add_argument("dump")
    db_restore.add_argument("--into", required=True, help="name of the new database")
    db_verify = db_sub.add_parser("verify-restore", help="restore into a scratch database and compare searches")
    db_verify.add_argument("dump")
    profile_parser = sub.add_parser("profile", help="report pages, images, tables and textless pages per PDF")
    profile_parser.add_argument("pdfs", nargs="*", help="PDF files (default: all in paths.pdf_dir)")
    ingest_parser = sub.add_parser("ingest", help="ingestion commands")
    ingest_sub = ingest_parser.add_subparsers(dest="ingest_command", required=True)
    parse_parser = ingest_sub.add_parser("parse", help="parse one PDF and build its review sheet")
    parse_parser.add_argument("pdf", help="PDF file name in paths.pdf_dir, or a path")
    index_parser = ingest_sub.add_parser("index", help="enrich, chunk, embed and write one PDF to the database")
    index_parser.add_argument("pdf", help="PDF file name in paths.pdf_dir, or a path")
    cmp_parser = ingest_sub.add_parser("compare-summaries", help="table summaries from several models, side by side")
    cmp_parser.add_argument("pdf", help="PDF file name in paths.pdf_dir, or a path")
    cmp_parser.add_argument("--models", default="gpt-4o-mini,gpt-5.4-mini,gpt-5.4-nano")
    search_parser = sub.add_parser("search", help="hybrid search over the indexed documents")
    search_parser.add_argument("query")
    search_parser.add_argument("--collection", choices=["all", "text", "figure", "table"], default="all")
    search_parser.add_argument("-k", type=int, default=5)
    ask_parser = sub.add_parser("ask", help="answer a question with the agent (writes an HTML answer page)")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--thread", help="continue an earlier conversation (thread id printed by ask)")
    ask_parser.add_argument("--model", help="override agent.model for this question")
    batch_parser = sub.add_parser("ask-batch", help="answer a question file, optionally with several models")
    batch_parser.add_argument("questions", help="YAML file of questions")
    batch_parser.add_argument("--models", help="comma-separated models to compare (default: agent.model)")
    batch_parser.add_argument("--max-cost", type=float, default=3.0,
                              help="stop once priced spend passes this many US$ (default 3)")
    report_parser = sub.add_parser("search-report", help="run the fixed retrieval test queries (Phase 3 gate)")
    report_parser.add_argument("--queries", default=str(PROJECT_ROOT / "eval" / "retrieval_queries.yaml"))
    obs_parser = sub.add_parser("obs", help="operations: retention cleanup and alerts (Phase 8)")
    obs_sub = obs_parser.add_subparsers(dest="obs_command", required=True)
    o_clean = obs_sub.add_parser("cleanup", help="delete history, sign-in events and sessions past retention")
    o_clean.add_argument("--yes", action="store_true", help="confirm the deletion")
    obs_sub.add_parser("alerts", help="check cost, removed-parts rate and failed sign-ins; exit 1 on an alert")
    doc_parser = sub.add_parser("doc", help="manage the PDF library one document at a time (Phase 8)")
    doc_sub = doc_parser.add_subparsers(dest="doc_command", required=True)
    doc_sub.add_parser("list", help="indexed PDFs with their counts")
    d_add = doc_sub.add_parser("add", help="copy a new PDF into the library and parse it")
    d_add.add_argument("path")
    d_rep = doc_sub.add_parser("replace", help="put a new version of a PDF in place of the old one")
    d_rep.add_argument("name", help="the PDF's file name in the library")
    d_rep.add_argument("path", help="the new version")
    d_rm = doc_sub.add_parser("remove", help="delete a PDF and all its search data")
    d_rm.add_argument("name")
    d_rm.add_argument("--yes", action="store_true", help="confirm the deletion")
    user_parser = sub.add_parser("user", help="account administration (Phase 7)")
    user_sub = user_parser.add_subparsers(dest="user_command", required=True)
    u_add = user_sub.add_parser("add", help="create an account with a temporary password (shown once)")
    u_add.add_argument("--role", choices=["user", "admin"], default="user")
    for name, helptext in [("reset-password", "new temporary password; ends every session"),
                           ("disable", "block sign-in and end every session"), ("enable", "allow sign-in again"),
                           ("unlock", "clear a lockout")]:
        user_sub.add_parser(name, help=helptext)
    for name in ("add", "reset-password", "disable", "enable", "unlock"):
        user_sub.choices[name].add_argument("email")
    u_del = user_sub.add_parser("delete", help="delete an account, its sessions and its memories")
    u_del.add_argument("email")
    u_del.add_argument("--yes", action="store_true", help="confirm the deletion")
    user_sub.add_parser("list", help="email, role, status, lockout, last sign-in")
    u_ev = user_sub.add_parser("events", help="sign-in audit log (newest first) with a summary")
    u_ev.add_argument("--days", type=int, default=7)
    u_ev.add_argument("--limit", type=int, default=50, help="rows to print")
    mem_parser = sub.add_parser("memory", help="see, delete and switch off long-term memory (Phase 9)")
    mem_sub = mem_parser.add_subparsers(dest="memory_command", required=True)
    mem_sub.add_parser("list", help="everything remembered about this account")
    m_del = mem_sub.add_parser("delete", help="delete one memory")
    m_del.add_argument("kind", choices=["semantic", "episodic"])
    m_del.add_argument("key", help="the key shown by `mmrag memory list`")
    m_all = mem_sub.add_parser("forget-all", help="delete every memory of this account")
    m_all.add_argument("--yes", action="store_true", help="confirm the deletion")
    mem_sub.add_parser("on", help="switch memory on for this account")
    mem_sub.add_parser("off", help="switch memory off (stops storing and recalling)")
    m_sum = mem_sub.add_parser("summarize", help="summarise finished conversations (run on a schedule)")
    m_sum.add_argument("--idle-minutes", type=int, default=None,
                       help="a conversation counts as finished after this long (default: memory.idle_minutes)")
    for name in ("list", "delete", "forget-all", "on", "off", "summarize"):
        mem_sub.choices[name].add_argument("email", help="the account the memories belong to")

    eval_parser = sub.add_parser("eval", help="golden-set evaluation (Phase 6)")
    eval_sub = eval_parser.add_subparsers(dest="eval_command", required=True)
    e_draft = eval_sub.add_parser("draft-questions", help="draft candidate golden questions for review")
    e_draft.add_argument("--seed", type=int, default=7)
    e_promote = eval_sub.add_parser("promote", help="copy the chosen drafts into the golden set")
    e_promote.add_argument("ids", help="comma-separated draft ids, in order")
    e_run = eval_sub.add_parser("run", help="answer and score the golden set; non-zero exit on regression")
    e_run.add_argument("--baseline", action="store_true", help="store this run as the new baseline")
    e_run.add_argument("--label", help="what changed, e.g. 'reranker on'")
    e_run.add_argument("--model", help="override agent.model")
    e_run.add_argument("--limit", type=int, help="only the first N questions (smoke test)")
    e_run.add_argument("--only", help="comma-separated question ids (their parent questions are included)")
    e_run.add_argument("--max-cost", type=float, help="stop the run once agent + judge spend reaches this many US$")
    e_run.add_argument("--set", action="append", metavar="SECTION.FIELD=VALUE",
                       help="override a setting for this run, e.g. search.rerank=true (repeatable; shown in the label)")
    e_retr = eval_sub.add_parser("retrieval", help="search-only scores on the golden set (compare search settings)")
    e_retr.add_argument("--set", action="append", metavar="SECTION.FIELD=VALUE",
                        help="override a setting for this run, e.g. search.keyword_mode=any (repeatable)")
    e_retr.add_argument("--show-misses", action="store_true")
    e_report = eval_sub.add_parser("report", help="write the report for a run")
    e_report.add_argument("run_id")
    caption_parser = sub.add_parser("caption", help="VLM figure captioning on Kaggle (Phase 2)")
    caption_sub = caption_parser.add_subparsers(dest="caption_command", required=True)
    c_bundle = caption_sub.add_parser("bundle", help="package figures and context for the GPU job")
    c_bundle.add_argument("--pilot", action="store_true", help="run all three model paths, ignoring the cache")
    c_bundle.add_argument("--elements", help="only these elements, e.g. p3:vector_figure:2,p19:table:1")
    caption_sub.add_parser("push", help="upload the bundle and start the Kaggle GPU job")
    caption_sub.add_parser("status", help="show the Kaggle job's status")
    caption_sub.add_parser("pull", help="download the Kaggle job's results")
    c_report = caption_sub.add_parser("report", help="build the model comparison page from a pulled run")
    c_import = caption_sub.add_parser("import", help="validate and store captions from a pulled run")
    c_import.add_argument("--model", choices=["awq-7b", "nf4-7b", "3b"], help="default: caption.model_path")
    caption_sub.add_parser("review", help="page of all imported captions for the owner's spot-check")
    for p in (c_bundle, caption_sub.choices["push"], caption_sub.choices["status"], caption_sub.choices["pull"],
              c_report, c_import, caption_sub.choices["review"]):
        p.add_argument("pdf", help="PDF file name in paths.pdf_dir, or a path")
    for p in (c_report, c_import):
        p.add_argument("--run", help="a pulled run folder (default: the latest)")
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
        "db": {"migrate": cmd_db_migrate, "backup": cmd_db_backup, "restore": cmd_db_restore,
               "verify-restore": cmd_db_verify_restore}.get(getattr(args, "db_command", None)),
        "profile": cmd_profile,
        "ingest": {"parse": cmd_ingest_parse, "index": cmd_ingest_index,
                   "compare-summaries": cmd_compare_summaries}.get(getattr(args, "ingest_command", None)),
        "search": cmd_search,
        "ask": cmd_ask,
        "ask-batch": cmd_ask_batch,
        "search-report": cmd_search_report,
        "caption": cmd_caption,
        "eval": cmd_eval,
        "user": cmd_user,
        "doc": cmd_doc,
        "obs": cmd_obs,
        "memory": cmd_memory,
    }
    try:
        return handlers[args.command](settings, args)
    except Exception as e:  # noqa: BLE001 - known failures get a clear message instead of a traceback
        message = _friendly_error(e)
        if message is None:
            raise
        print(f"Error: {message}", file=sys.stderr)
        return 2
    finally:
        shutdown_telemetry()


def _friendly_error(e: Exception) -> str | None:
    """A one-line explanation and fix for failures a user can act on; None for real bugs."""
    import psycopg

    from mmrag.llm import MissingApiKey

    name, text = type(e).__name__, str(e)
    if isinstance(e, ImportError) and "Application Control policy" in text:
        return ("Windows Smart App Control blocked a library this command needs. Run it in Linux instead: "
                "docker compose run --rm ingest <the same command>, e.g. docker compose run --rm ingest "
                "ingest index \"file.pdf\" (see docs/07-operations.md).")
    if isinstance(e, psycopg.OperationalError):
        return ("Cannot reach the database. Start it with `docker compose up -d` and check DATABASE_URL in .env.")
    if isinstance(e, MissingApiKey):
        return "OPENAI_API_KEY is not set. Add it to .env."
    if "insufficient_quota" in text or "credit_balance_exhausted" in text:
        return "Your OpenAI account has no credits left. Add credits at platform.openai.com/settings/organization/billing."
    if name == "AuthenticationError":
        return "OpenAI rejected the API key. Check OPENAI_API_KEY in .env."
    if name in ("APIConnectionError", "APITimeoutError"):
        return "Cannot reach OpenAI (network problem or timeout). Check the connection and try again."
    if name == "RateLimitError":
        return "OpenAI rate limit reached. Wait a minute and try again."
    return None


if __name__ == "__main__":
    sys.exit(main())
