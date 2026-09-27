"""PostgreSQL access and schema migrations (LLD §3.9, §4)."""

from __future__ import annotations

import re
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector

from mmrag.config import PROJECT_ROOT, Settings
from mmrag.obs import get_logger

MIGRATIONS_DIR = PROJECT_ROOT / "db" / "migrations"
_MIGRATION_NAME = re.compile(r"^(\d{4})_[a-z0-9_]+\.sql$")
_log = get_logger("mmrag.db")


def connect(settings: Settings, *, register_vectors: bool = True) -> psycopg.Connection:
    conn = psycopg.connect(settings.secrets.database_url.get_secret_value(), connect_timeout=10)
    if register_vectors:
        try:
            register_vector(conn)
        except psycopg.ProgrammingError:
            # The `vector` type doesn't exist until migration 0001 has run.
            conn.rollback()
    return conn


def _migration_files() -> list[tuple[str, Path]]:
    files = []
    for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
        m = _MIGRATION_NAME.match(path.name)
        if not m:
            raise RuntimeError(f"Badly named migration file: {path.name}")
        files.append((m.group(1), path))
    return files


def migrate(settings: Settings) -> list[str]:
    """Apply pending migrations in order, each in its own transaction."""
    applied_now = []
    with connect(settings, register_vectors=False) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            " version text PRIMARY KEY,"
            " applied_at timestamptz NOT NULL DEFAULT now())"
        )
        conn.commit()
        done = {row[0] for row in conn.execute("SELECT version FROM schema_migrations")}
        for version, path in _migration_files():
            if version in done:
                continue
            with conn.transaction():
                conn.execute(path.read_text(encoding="utf-8"))
                conn.execute("INSERT INTO schema_migrations (version) VALUES (%s)", (version,))
            _log.info("Applied migration %s", path.name)
            applied_now.append(path.name)
    return applied_now
