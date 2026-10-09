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


def connect(
    settings: Settings, *, register_vectors: bool = True, autocommit: bool = False
) -> psycopg.Connection:
    conn = psycopg.connect(
        settings.secrets.database_url.get_secret_value(), connect_timeout=10, autocommit=autocommit
    )
    if register_vectors:
        try:
            register_vector(conn)
        except psycopg.ProgrammingError:
            # The `vector` type doesn't exist until migration 0001 has run.
            conn.rollback()
    return conn


def create_database(settings: Settings) -> bool:
    """Create the database named in the configured URL, on the same server, if it does not exist
    yet. Connects to the server's `postgres` database to do it. True when it was created."""
    from psycopg import sql
    from psycopg.conninfo import conninfo_to_dict, make_conninfo

    url = settings.secrets.database_url.get_secret_value()
    name = conninfo_to_dict(url)["dbname"]
    with psycopg.connect(make_conninfo(url, dbname="postgres"), connect_timeout=10, autocommit=True) as conn:
        if conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (name,)).fetchone():
            return False
        conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    _log.info("Created database %s", name)
    return True


def _migration_files() -> list[tuple[str, Path]]:
    files = []
    for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
        m = _MIGRATION_NAME.match(path.name)
        if not m:
            raise RuntimeError(f"Badly named migration file: {path.name}")
        files.append((m.group(1), path))
    return files


def migrate(settings: Settings) -> list[str]:
    """Apply pending migrations in order, each in its own transaction.

    Autocommit mode makes every `conn.transaction()` a real top-level
    transaction, so a failing migration never undoes earlier ones.
    """
    applied_now = []
    with connect(settings, register_vectors=False, autocommit=True) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            " version text PRIMARY KEY,"
            " applied_at timestamptz NOT NULL DEFAULT now())"
        )
        done = {row[0] for row in conn.execute("SELECT version FROM schema_migrations")}
        for version, path in _migration_files():
            if version in done:
                continue
            with conn.transaction():
                conn.execute(render(path.read_text(encoding="utf-8"), settings))
                conn.execute("INSERT INTO schema_migrations (version) VALUES (%s)", (version,))
            _log.info("Applied migration %s", path.name)
            applied_now.append(path.name)
        check_embedding_dims(conn, settings)
    return applied_now


def render(sql_text: str, settings: Settings) -> str:
    """Fill a migration's placeholders from settings. `{{embed_dims}}` is the vector size."""
    return sql_text.replace("{{embed_dims}}", str(settings.embed.dims))


def embedding_dims(conn: psycopg.Connection) -> int | None:
    """The vector size of `search_chunks.embedding` (pgvector keeps it as the type modifier), or
    None before migration 0002."""
    row = conn.execute(
        "SELECT atttypmod FROM pg_attribute"
        " WHERE attrelid = to_regclass('search_chunks') AND attname = 'embedding' AND NOT attisdropped"
    ).fetchone()
    return row[0] if row else None


class EmbeddingDimsMismatch(RuntimeError):
    """The config's embed.dims differs from the database's vector size."""


def check_embedding_dims(conn: psycopg.Connection, settings: Settings) -> None:
    """Refuse a config whose embedding size differs from the database's: every insert and search
    would fail, or a config would be pointed at another model's database by mistake."""
    dims = embedding_dims(conn)
    if dims is not None and dims != settings.embed.dims:
        raise EmbeddingDimsMismatch(
            f"this database stores {dims}-dim embeddings, but the config's embed.dims is "
            f"{settings.embed.dims} ({settings.embed.model}). Use the config that matches this database, "
            "or set database_name in the config to a database of its own.")
