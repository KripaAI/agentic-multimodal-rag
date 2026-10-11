"""Embedding on a Kaggle GPU (gpu_job/embed/embed.py): the bundle going up, the cache coming back.

The job runs this project's own `ingest index --no-db`, so it embeds exactly the texts a local run
would, and returns them as embedding-cache lines. Merging them into the local cache makes the
following local `ingest index` a pure database write. PostgreSQL, the PDFs and the UI stay here.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import yaml

from mmrag.config import PROJECT_ROOT, Settings
from mmrag.fsutil import empty_dir, jsonl_lines
from mmrag.index.document import load_document
from mmrag.index.pipeline import summary_cache
from mmrag.ingest.enrich import summarize_tables

GPU_BATCH_SIZE = 32  # texts per forward pass on the GPU: 128 chunks of up to 800 tokens ran a T4 out of memory


def config_chain(config_file: Path) -> list[Path]:
    """The config file and every file it extends, in order."""
    chain, path = [], config_file.resolve()
    while path not in chain:
        chain.append(path)
        base = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("extends")
        if not base:
            break
        path = (path.parent / base).resolve()
    return chain


def cache_file(settings: Settings) -> Path:
    """The local embedding cache (as Embedder names it)."""
    e = settings.embed
    return settings.resolve(settings.paths.data_dir) / "cache" / "embeddings" / f"{e.model.replace('/', '--')}-{e.dims}.jsonl"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


@dataclass
class BundleInfo:
    path: Path
    documents: int
    megabytes: float


def build_bundle(pdfs: list[Path], settings: Settings, config_file: Path, out: Path, chat=None) -> BundleInfo:
    """Write `out/` (project.bundle, manifest.json, run.json) for the PDFs, which must be parsed (and, if
    wanted, captioned and transcribed) already. Table summaries are made here first, so the GPU job
    never needs an OpenAI key; `chat` defaults to OpenAI and is only called for an uncached table."""
    if chat is None:
        from mmrag.llm import complete, get_client

        clients = []

        def chat(model, prompt):
            if not clients:
                clients.append(get_client(settings))
            return complete(clients[0], model, prompt).text

    data = settings.resolve(settings.paths.data_dir)
    manifest = {}
    for pdf in pdfs:
        doc = load_document(pdf, settings)  # fails clearly if the PDF is not parsed
        summarize_tables(list(doc.tables.values()), settings.agent.summary_model, chat, summary_cache(settings))
        manifest[pdf.name] = {"doc_id": doc.doc_id, "sha256": _sha256(pdf)}

    chain = config_chain(config_file)
    empty_dir(out)
    # A zip archive, but not named .zip: Kaggle unpacks .zip files in a dataset, and the job would not find it.
    with zipfile.ZipFile(out / "project.bundle", "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted((PROJECT_ROOT / "src" / "mmrag").rglob("*.py")):
            z.write(f, f.relative_to(PROJECT_ROOT).as_posix())
        for f in chain:  # config files sit at the project root; `extends` names them relative to each other
            z.write(f, f.name)
        wanted = [summary_cache(settings)]
        for entry in manifest.values():
            did = entry["doc_id"]
            wanted += [data / "elements" / did / "elements.jsonl", data / "tables" / did / "tables.jsonl"]
            # Every record indexing reads beside the captions: captions, transcripts, chart placements.
            # (Listing them by name once left charts.jsonl out: the job then embedded figure texts
            # without placements, and the laptop had to embed the real ones on its CPU.)
            wanted += sorted((data / "captions" / did).glob("*.jsonl"))
        for f in wanted:  # under data/ wherever paths.data_dir is; the job's config points there
            if f.is_file():
                z.write(f, "data/" + f.relative_to(data).as_posix())
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    run = {"task": "embed", "config": chain[0].name, "model": settings.embed.model,
           "dims": settings.embed.dims, "batch_size": GPU_BATCH_SIZE, "documents": len(manifest)}
    (out / "run.json").write_text(json.dumps(run, indent=1) + "\n", encoding="utf-8")
    size = sum(f.stat().st_size for f in out.iterdir() if f.is_file()) / 1e6
    return BundleInfo(path=out, documents=len(manifest), megabytes=round(size, 1))


def merge_cache(pulled: Path, settings: Settings) -> tuple[int, int]:
    """Add the pulled cache lines whose key the local cache lacks. Returns (added, already there).
    Never replaces the local file: it holds the vectors of every other indexed document too."""
    target = cache_file(settings)
    have = set()
    if target.is_file():
        for line in jsonl_lines(target):
            try:
                have.add(json.loads(line)["k"])
            except (ValueError, KeyError, TypeError):
                continue  # a damaged line: Embedder skips it too
    new, seen = [], 0
    for line in jsonl_lines(pulled):
        row = json.loads(line)
        if len(row["v"]) != settings.embed.dims:
            raise ValueError(f"pulled vectors have {len(row['v'])} dims, the config says {settings.embed.dims}")
        if row["k"] in have:
            seen += 1
            continue
        have.add(row["k"])
        new.append(line)
    if new:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as f:
            f.write("".join(line + "\n" for line in new))
    return len(new), seen


def bundle_key(settings: Settings) -> str:
    """One Kaggle dataset and kernel per database: mmrag-embed-<database name>."""
    from psycopg.conninfo import conninfo_to_dict

    name = settings.database_name or conninfo_to_dict(settings.secrets.database_url.get_secret_value())["dbname"]
    return name.replace("_", "-").lower()


def bundle_dir(settings: Settings) -> Path:
    return settings.resolve(settings.paths.data_dir) / "embed" / bundle_key(settings)
