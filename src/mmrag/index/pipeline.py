"""Phase 3 for one PDF: enrich → chunk → embed → write (plan Phase 3, LLD §3.5–3.8).

Traced as one `ingest.document` trace with enrich, chunk, embed and write spans (P12).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from mmrag.config import Settings
from mmrag.index.chunk import Chunk, build_figure_docs, build_table_docs, chunk_text
from mmrag.index.document import Link, LoadedDoc, load_document
from mmrag.index.embed import Embedder
from mmrag.index.reports import write_link_report
from mmrag.index.writer import write_document
from mmrag.ingest.enrich import cross_check_labels, find_links, skip_figures, summarize_tables
from mmrag.obs import get_tracer


@dataclass
class IndexReport:
    doc: LoadedDoc
    chunks: list[Chunk]
    links: list[Link]
    unlinked: list[str]
    label_shares: dict[str, float]
    result: str
    link_report: Path | None = None
    counts: dict[str, int] = field(default_factory=dict)


def summary_cache(settings: Settings) -> Path:
    return settings.resolve(settings.paths.data_dir) / "cache" / "summaries.jsonl"


def index_document(pdf: Path, settings: Settings, embedder: Embedder | None = None, chat=None) -> IndexReport:
    """`chat(model, prompt) -> str` defaults to OpenAI; tests pass fakes for chat and the embedder."""
    embedder = embedder or Embedder(settings)
    if chat is None:
        from mmrag.llm import complete, get_client

        client = get_client(settings)
        chat = lambda model, prompt: complete(client, model, prompt).text  # noqa: E731
    tracer = get_tracer("mmrag.index")
    with tracer.start_as_current_span("ingest.document") as root:
        doc = load_document(pdf, settings)
        root.set_attribute("mmrag.doc_id", doc.doc_id)
        root.set_attribute("mmrag.source_file", doc.source_file)

        with tracer.start_as_current_span("ingest.enrich") as span:
            span.set_attribute("mmrag.figures_skipped", skip_figures(doc, settings.enrich.skip_figures_for))
            shares = cross_check_labels(doc, settings.enrich)
            summaries = summarize_tables(list(doc.tables.values()), settings.agent.summary_model, chat,
                                         summary_cache(settings))
            for eid, summary in summaries.items():
                doc.tables[eid] = doc.tables[eid].model_copy(update={"summary": summary})
            links, unlinked = find_links(doc, embedder.embed, settings.enrich)
            span.set_attribute("mmrag.links", len(links))
            span.set_attribute("mmrag.unlinked", len(unlinked))
            span.set_attribute("mmrag.captions_needs_review", sum(c.status != "ok" for c in doc.captions.values()))

        with tracer.start_as_current_span("index.chunk") as span:
            chunks = chunk_text(doc, settings.chunk) + build_figure_docs(doc, links) + build_table_docs(doc)
            for coll in ("text", "figure", "table"):
                span.set_attribute(f"mmrag.chunks.{coll}", sum(c.collection == coll for c in chunks))

        vectors = embedder.embed([c.dense_text for c in chunks])
        result = write_document(settings, doc, chunks, vectors, links)
        root.set_attribute("mmrag.write.result", result)

    report = IndexReport(doc=doc, chunks=chunks, links=links, unlinked=unlinked, label_shares=shares, result=result,
                         counts={c: sum(ch.collection == c for ch in chunks) for c in ("text", "figure", "table")})
    report.link_report = write_link_report(report, settings)
    return report
