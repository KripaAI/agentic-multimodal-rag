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
    report_parser = sub.add_parser("search-report", help="run the fixed retrieval test queries (Phase 3 gate)")
    report_parser.add_argument("--queries", default=str(PROJECT_ROOT / "eval" / "retrieval_queries.yaml"))
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
        "db": {"migrate": cmd_db_migrate}.get(getattr(args, "db_command", None)),
        "profile": cmd_profile,
        "ingest": {"parse": cmd_ingest_parse, "index": cmd_ingest_index,
                   "compare-summaries": cmd_compare_summaries}.get(getattr(args, "ingest_command", None)),
        "search": cmd_search,
        "search-report": cmd_search_report,
        "caption": cmd_caption,
    }
    try:
        return handlers[args.command](settings, args)
    finally:
        shutdown_telemetry()


if __name__ == "__main__":
    sys.exit(main())
