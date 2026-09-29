"""Migrations and pgvector against a real, throwaway PostgreSQL database."""

from __future__ import annotations

import psycopg
import pytest

import mmrag.db as db
from mmrag.config import PROJECT_ROOT, load_settings

pytestmark = pytest.mark.integration


def _settings(url: str):
    return load_settings(PROJECT_ROOT / "config.yaml", {"DATABASE_URL": url})


def test_migrate_applies_then_is_a_no_op(fresh_db_url):
    s = _settings(fresh_db_url)
    applied = db.migrate(s)
    assert applied[:2] == ["0001_extensions.sql", "0002_documents.sql"]
    assert db.migrate(s) == []
    with db.connect(s) as conn:
        versions = [r[0] for r in conn.execute("SELECT version FROM schema_migrations ORDER BY version")]
        ext = conn.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'").fetchone()
    assert versions[:2] == ["0001", "0002"]
    assert ext is not None


def test_phase3_schema(fresh_db_url):
    """Spec §5.5 / LLD §4: tables, generated keyword columns, partial HNSW indexes, cascade."""
    s = _settings(fresh_db_url)
    db.migrate(s)
    with db.connect(s) as conn:
        tables = {r[0] for r in conn.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")}
        assert {"documents", "elements", "figure_captions", "doc_tables", "element_links", "search_chunks"} <= tables
        indexes = {r[0]: r[1] for r in conn.execute(
            "SELECT indexname, indexdef FROM pg_indexes WHERE tablename = 'search_chunks'")}
        hnsw = [d for d in indexes.values() if "hnsw" in d]
        assert len(hnsw) == 3 and all("WHERE" in d for d in hnsw)  # one partial index per collection
        assert sum("gin" in d for d in indexes.values()) == 2  # tsv_english and tsv_simple

        conn.execute("INSERT INTO documents (doc_id, source_file, content_hash, build_hash, version) "
                     "VALUES ('d1', 'x.pdf', 'h', 'b', 1)")
        conn.execute("INSERT INTO elements (element_id, doc_id, page, bbox, type, content_hash) "
                     "VALUES ('d1:p1:text:1', 'd1', 1, '{0,0,1,1}', 'text', 'h')")
        conn.execute("INSERT INTO search_chunks (chunk_id, doc_id, collection, element_ids, dense_text, "
                     "keyword_text, embedding, embedding_model) VALUES ('d1:text:1', 'd1', 'text', "
                     "'{d1:p1:text:1}', 'LoRA trains adapters', 'LoRA trains adapters', %s, 'm')",
                     ("[" + ",".join(["0.1"] * s.embed.dims) + "]",))
        english, simple = conn.execute("SELECT tsv_english::text, tsv_simple::text FROM search_chunks").fetchone()
        assert "'train'" in english  # stemmed
        assert "'lora'" in simple and "'trains'" in simple  # exact words kept
        conn.execute("DELETE FROM documents WHERE doc_id = 'd1'")
        assert conn.execute("SELECT count(*) FROM elements").fetchone()[0] == 0
        assert conn.execute("SELECT count(*) FROM search_chunks").fetchone()[0] == 0


def test_failed_migration_rolls_back_completely(fresh_db_url, tmp_path, monkeypatch):
    (tmp_path / "0001_ok.sql").write_text("CREATE TABLE good (id int);")
    (tmp_path / "0002_bad.sql").write_text("CREATE TABLE half_done (id int); SELECT no_such_column;")
    monkeypatch.setattr(db, "MIGRATIONS_DIR", tmp_path)
    s = _settings(fresh_db_url)
    with pytest.raises(psycopg.errors.UndefinedColumn):
        db.migrate(s)
    with db.connect(s, register_vectors=False) as conn:
        versions = [r[0] for r in conn.execute("SELECT version FROM schema_migrations")]
        tables = {r[0] for r in conn.execute(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public'")}
    assert versions == ["0001"]
    assert "good" in tables and "half_done" not in tables


def test_hnsw_nearest_neighbour(fresh_db_url):
    s = _settings(fresh_db_url)
    db.migrate(s)
    with db.connect(s) as conn:
        conn.execute("CREATE TABLE v (id int, e vector(3))")
        conn.execute("INSERT INTO v VALUES (1,'[1,0,0]'), (2,'[0,1,0]'), (3,'[0,0,1]')")
        conn.execute("CREATE INDEX ON v USING hnsw (e vector_cosine_ops)")
        conn.execute("SET enable_seqscan = off")  # force the HNSW index
        plan = " ".join(r[0] for r in conn.execute(
            "EXPLAIN SELECT id FROM v ORDER BY e <=> '[0,0.9,0.1]' LIMIT 1"))
        nearest = conn.execute("SELECT id FROM v ORDER BY e <=> '[0,0.9,0.1]' LIMIT 1").fetchone()[0]
    assert "v_e_idx" in plan
    assert nearest == 2
