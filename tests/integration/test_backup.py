"""Backup and a tested restore (plan Phase 8 task 4: a restore returns identical search results).
Test-first. Needs pg_dump on PATH or the docker compose `db` service."""

from __future__ import annotations

import pytest

import mmrag.db as db
from mmrag.config import load_settings
from mmrag.db.backup import available, backup, compare_search, drop_database, restore
from mmrag.index.chunk import Chunk
from mmrag.index.document import LoadedDoc
from mmrag.index.writer import write_document
from mmrag.ingest.models import Element

pytestmark = [pytest.mark.integration, pytest.mark.skipif(not available(), reason="no pg_dump / docker compose db")]


@pytest.fixture
def settings(fresh_db_url, base_config, write_config):
    base_config["observability"].update(enabled=False, log_file=None)
    s = load_settings(write_config(base_config), {"DATABASE_URL": fresh_db_url})
    db.migrate(s)
    dims = s.embed.dims
    for n, text in enumerate(["grouped query attention shrinks the KV cache", "LoRA trains low-rank adapters"]):
        did = f"{n}" * 16
        el = Element(element_id=f"{did}:p1:text:1", doc_id=did, source_file=f"d{n}.pdf", page=1, bbox=(0, 0, 9, 9),
                     type="text", text=text, content_hash="t")
        write_document(s, LoadedDoc(doc_id=did, source_file=f"d{n}.pdf", content_hash=did, elements=[el]),
                       [Chunk(chunk_id=f"{did}:text:1", collection="text", element_ids=[el.element_id], dense_text=text,
                              keyword_text=text)],
                       [[1.0 if i == n else 0.0 for i in range(dims)]], [])
    return s


def test_backup_restore_gives_identical_search_results(settings, tmp_path):
    dump = backup(settings, tmp_path / "mmrag.dump")
    assert dump.stat().st_size > 1000
    target = restore(settings, dump, "mmrag_restore_check")
    try:
        queries = [("KV cache", "text"), ("adapters", "text"), ("nothing matches this", "text")]
        diffs = compare_search(settings, target, queries,
                               embed_query=lambda q: [1.0] + [0.0] * (settings.embed.dims - 1))
        assert diffs == []  # same chunks, same order, for every query
    finally:
        drop_database(settings, "mmrag_restore_check")


def test_restore_never_overwrites_an_existing_database(settings, tmp_path):
    dump = backup(settings, tmp_path / "mmrag.dump")
    live = settings.secrets.database_url.get_secret_value().rsplit("/", 1)[1].split("?")[0]
    with pytest.raises(ValueError, match="already exists"):
        restore(settings, dump, live)
