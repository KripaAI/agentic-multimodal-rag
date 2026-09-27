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
    assert db.migrate(s) == ["0001_extensions.sql"]
    assert db.migrate(s) == []
    with db.connect(s) as conn:
        versions = [r[0] for r in conn.execute("SELECT version FROM schema_migrations")]
        ext = conn.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'").fetchone()
    assert versions == ["0001"]
    assert ext is not None


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
