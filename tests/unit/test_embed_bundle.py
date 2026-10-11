"""Embedding on a Kaggle GPU: the bundle, the merge back into the local cache, the Kaggle names."""

from __future__ import annotations

import json
import zipfile

import pytest

from mmrag.config import PROJECT_ROOT
from mmrag.index import embed_bundle
from mmrag.ingest import kaggle_job
from mmrag.ingest.ids import doc_id
from mmrag.ingest.models import Element

pytestmark = pytest.mark.unit


def test_config_chain_follows_extends():
    names = [p.name for p in embed_bundle.config_chain(PROJECT_ROOT / "config.astro.yaml")]
    assert names == ["config.astro.yaml", "config.gemma.yaml", "config.yaml"]


def _row(key, dims, value=0.1):
    return json.dumps({"k": key, "v": [value] * dims})


def test_merge_adds_only_new_keys_and_never_replaces(parse_settings, tmp_path):
    dims = parse_settings.embed.dims
    local = embed_bundle.cache_file(parse_settings)
    local.parent.mkdir(parents=True)
    local.write_text(_row("a", dims, 0.5) + "\n", encoding="utf-8")
    pulled = tmp_path / "embeddings.jsonl"
    pulled.write_text(_row("a", dims, 0.9) + "\n" + _row("b", dims) + "\n", encoding="utf-8")
    assert embed_bundle.merge_cache(pulled, parse_settings) == (1, 1)
    rows = {json.loads(x)["k"]: json.loads(x)["v"][0] for x in local.read_text(encoding="utf-8").split("\n") if x}
    assert rows == {"a": 0.5, "b": 0.1}  # the local vector for "a" is kept
    assert embed_bundle.merge_cache(pulled, parse_settings) == (0, 2)  # importing twice adds nothing


def test_merge_refuses_vectors_of_another_size(parse_settings, tmp_path):
    pulled = tmp_path / "embeddings.jsonl"
    pulled.write_text(_row("a", parse_settings.embed.dims + 1) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dims"):
        embed_bundle.merge_cache(pulled, parse_settings)


def test_bundle_holds_code_config_and_parse_outputs_not_pdfs(parse_settings, tmp_path):
    pdf = tmp_path / "book.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    did = doc_id(pdf)
    data = parse_settings.resolve(parse_settings.paths.data_dir)
    (data / "elements" / did).mkdir(parents=True)
    e = Element(element_id=f"{did}:p1:text:1", doc_id=did, source_file="book.pdf", page=1, bbox=(0, 0, 9, 9),
                type="text", text="Saturn rules Capricorn.", content_hash="t1")
    (data / "elements" / did / "elements.jsonl").write_text(e.model_dump_json() + "\n", encoding="utf-8")
    (data / "captions" / did / "runs").mkdir(parents=True)
    for name in ("captions.jsonl", "transcripts.jsonl", "charts.jsonl", "runs/old.jsonl"):
        (data / "captions" / did / name).write_text("", encoding="utf-8")  # empty: valid, no records

    info = embed_bundle.build_bundle([pdf], parse_settings, PROJECT_ROOT / "config.gemma.yaml", tmp_path / "out",
                                     chat=lambda m, p: pytest.fail("no tables, so no summaries"))
    names = zipfile.ZipFile(info.path / "project.bundle").namelist()
    assert "src/mmrag/cli.py" in names and "config.gemma.yaml" in names and "config.yaml" in names
    assert not any(n.endswith(".pdf") for n in names)
    for name in ("captions.jsonl", "transcripts.jsonl", "charts.jsonl"):  # everything indexing reads
        assert f"data/captions/{did}/{name}" in names
    assert f"data/captions/{did}/runs/old.jsonl" not in names  # pulled runs stay home
    manifest = json.loads((info.path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest == {"book.pdf": {"doc_id": did, "sha256": embed_bundle._sha256(pdf)}}
    run = json.loads((info.path / "run.json").read_text(encoding="utf-8"))
    assert run["config"] == "config.gemma.yaml" and run["batch_size"] == embed_bundle.GPU_BATCH_SIZE


def test_embedding_has_its_own_kaggle_job():
    assert kaggle_job.script_for("embed").name == "embed.py"
    assert kaggle_job.script_for("caption") == kaggle_job.script_for("transcribe") == kaggle_job.JOB_SCRIPT
    meta = kaggle_job.kernel_metadata("u", "mmrag-astro", "embed")
    assert meta["id"] == "u/mmrag-embedder-mmrag-astro" and meta["code_file"] == "embed.py"
    assert meta["dataset_sources"] == ["u/mmrag-embed-mmrag-astro"] and meta["enable_gpu"] is True
    assert kaggle_job.kernel_slug("d" * 16) == f"mmrag-captioner-{'d' * 16}"  # captioning unchanged


def test_the_newest_run_is_chosen_by_time_not_name(tmp_path):
    import os

    from mmrag.cli import _newest_run

    for name, t in [("20261011-120000", 100), ("local-test", 50), ("20261011-130000", 200)]:
        (tmp_path / name).mkdir()
        os.utime(tmp_path / name, (t, t))
    assert _newest_run(tmp_path).name == "20261011-130000"
    with pytest.raises(ValueError):
        _newest_run(tmp_path / "missing")


def test_sources_are_shown_once_per_page():
    from mmrag.agent.answer import source_pages

    rows = [{"source_file": "a.pdf", "page": 67, "element_id": f"x:p67:text:{n}"} for n in range(9)]
    rows += [{"source_file": "b.pdf", "page": 30, "element_id": "y:p30:text:1"},
             {"source_file": "a.pdf", "page": 68, "element_id": "x:p68:text:1"}]
    assert source_pages(rows) == [("a.pdf", 67, 9), ("b.pdf", 30, 1), ("a.pdf", 68, 1)]
