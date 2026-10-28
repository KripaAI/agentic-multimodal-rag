"""Migration file discovery (LLD §3.9 `db migrate`)."""

from __future__ import annotations

import pytest

import mmrag.db as db

pytestmark = pytest.mark.unit


def test_migrations_are_ordered_by_number(tmp_path, monkeypatch):
    for name in ("0002_second.sql", "0001_first.sql", "0010_tenth.sql"):
        (tmp_path / name).write_text("SELECT 1;")
    monkeypatch.setattr(db, "MIGRATIONS_DIR", tmp_path)
    assert [v for v, _ in db._migration_files()] == ["0001", "0002", "0010"]


def test_badly_named_migration_rejected(tmp_path, monkeypatch):
    (tmp_path / "1_bad.sql").write_text("SELECT 1;")
    monkeypatch.setattr(db, "MIGRATIONS_DIR", tmp_path)
    with pytest.raises(RuntimeError, match="Badly named"):
        db._migration_files()


def test_project_migrations_are_well_named():
    assert db._migration_files()[0][0] == "0001"


def test_vector_size_comes_from_config(parse_settings):
    sql = (db.MIGRATIONS_DIR / "0002_documents.sql").read_text(encoding="utf-8")
    assert "{{embed_dims}}" in sql and "vector(1536)" not in sql
    rendered = db.render(sql, parse_settings.model_copy(update={
        "embed": parse_settings.embed.model_copy(update={"dims": 768})}))
    assert "vector(768)" in rendered and "{{" not in rendered
