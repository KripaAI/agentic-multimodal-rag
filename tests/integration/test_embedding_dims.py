"""Phase 10: a database per embedding size. Migrations build the configured size, and a config
whose size differs from its database's is refused."""

from __future__ import annotations

import psycopg
import pytest

import mmrag.db as db
from mmrag.config import PROJECT_ROOT, load_settings

pytestmark = pytest.mark.integration


def _settings(url: str, dims: int | None = None):
    s = load_settings(PROJECT_ROOT / "config.yaml", {"DATABASE_URL": url})
    return s.model_copy(update={"embed": s.embed.model_copy(update={"dims": dims})}) if dims else s


def test_migrate_builds_the_configured_vector_size(fresh_db_url):
    s = _settings(fresh_db_url, 768)
    db.migrate(s)
    with db.connect(s) as conn:
        assert db.embedding_dims(conn) == 768
        chunk = "[" + ",".join(["0.1"] * 768) + "]"
        conn.execute("INSERT INTO documents (doc_id, source_file, content_hash, build_hash, version)"
                     " VALUES ('d', 'd.pdf', 's', 'h', 1)")
        conn.execute("INSERT INTO search_chunks (chunk_id, doc_id, collection, element_ids, dense_text,"
                     " keyword_text, embedding, embedding_model) VALUES ('c', 'd', 'text', '{e}', 't', 't',"
                     " %s::vector, 'm')", (chunk,))


def test_default_config_still_builds_1536(fresh_db_url):
    s = _settings(fresh_db_url)
    db.migrate(s)
    with db.connect(s) as conn:
        assert db.embedding_dims(conn) == 1536


def test_config_with_other_dims_is_refused(fresh_db_url):
    db.migrate(_settings(fresh_db_url))  # a 1536 database
    with pytest.raises(RuntimeError, match="1536-dim embeddings.*embed.dims is 768"):
        db.migrate(_settings(fresh_db_url, 768))


def test_create_database_is_idempotent(fresh_db_url):
    name = fresh_db_url.rsplit("/", 1)[1] + "_new"
    s = _settings(fresh_db_url.rsplit("/", 1)[0] + "/" + name)
    try:
        assert db.create_database(s) is True
        assert db.create_database(s) is False
    finally:
        with psycopg.connect(fresh_db_url, autocommit=True) as conn:
            conn.execute(f'DROP DATABASE IF EXISTS "{name}"')
