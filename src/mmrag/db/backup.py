"""Backup and tested restore (plan Phase 8 task 4).

`pg_dump`/`pg_restore` run from PATH when installed (a managed database, CI), otherwise inside the
docker compose `db` container. A restore always goes into a NEW database, never over the live one;
`compare_search` then proves the copy answers searches exactly like the original.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict
from pydantic import SecretStr

from mmrag.config import PROJECT_ROOT, Settings

COMPOSE = ["docker", "compose", "-f", str(PROJECT_ROOT / "docker-compose.yml"), "exec", "-T", "db"]


def _url(settings: Settings) -> str:
    return settings.secrets.database_url.get_secret_value()


def _with_db(url: str, dbname: str) -> str:
    return urlunsplit(urlsplit(url)._replace(path="/" + dbname))


@lru_cache(maxsize=1)
def _use_docker() -> bool:
    if shutil.which("pg_dump") and shutil.which("pg_restore"):
        return False
    r = subprocess.run([*COMPOSE, "pg_dump", "--version"], capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        raise RuntimeError("pg_dump not found: install the PostgreSQL client tools or start `docker compose up -d`")
    return True


def available() -> bool:
    try:
        _use_docker()
        return True
    except (RuntimeError, OSError, subprocess.TimeoutExpired):
        return False


def _tool(settings: Settings, tool: str, dbname: str) -> tuple[list[str], dict]:
    """The command and environment to run a PostgreSQL client tool against `dbname`."""
    info = conninfo_to_dict(_url(settings))
    if _use_docker():  # inside the container: local socket, the server's own user
        return [*COMPOSE, tool, "-U", info["user"], "-d", dbname], dict(os.environ)
    env = {**os.environ, "PGPASSWORD": info.get("password", "")}  # never on the command line
    return [tool, "-h", info.get("host", "localhost"), "-p", str(info.get("port", 5432)), "-U", info["user"],
            "-d", dbname], env


def backup(settings: Settings, out: Path) -> Path:
    """A compressed custom-format dump of the whole database (schema, data, extensions)."""
    dbname = conninfo_to_dict(_url(settings))["dbname"]
    cmd, env = _tool(settings, "pg_dump", dbname)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        r = subprocess.run([*cmd, "--format=custom", "--no-owner"], stdout=f, stderr=subprocess.PIPE, env=env,
                           timeout=3600)
    if r.returncode != 0:
        out.unlink(missing_ok=True)
        raise RuntimeError(f"pg_dump failed: {r.stderr.decode(errors='replace')[-500:]}")
    return out


def restore(settings: Settings, dump: Path, target: str) -> str:
    """Restore `dump` into a new database `target` (refuses an existing one). Returns its name."""
    with psycopg.connect(_url(settings), autocommit=True) as conn:
        if conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (target,)).fetchone():
            raise ValueError(f"database {target} already exists; restore never overwrites a database")
        conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(target)))
    cmd, env = _tool(settings, "pg_restore", target)
    with dump.open("rb") as f:
        r = subprocess.run([*cmd, "--no-owner", "--exit-on-error"], stdin=f, capture_output=True, env=env,
                           timeout=3600)
    if r.returncode != 0:
        drop_database(settings, target)
        raise RuntimeError(f"pg_restore failed: {r.stderr.decode(errors='replace')[-500:]}")
    return target


def drop_database(settings: Settings, name: str) -> None:
    live = conninfo_to_dict(_url(settings))["dbname"]
    if name == live:
        raise ValueError("refusing to drop the live database")
    with psycopg.connect(_url(settings), autocommit=True) as conn:
        conn.execute(sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(name)))


def compare_search(settings: Settings, target: str, queries: list[tuple[str, str]], embed_query=None,
                   k: int = 10) -> list[tuple]:
    """Run each (query, collection) on the live and the restored database; return the differences
    (an empty list means identical results, same chunks in the same order)."""
    from mmrag.retrieval.hybrid import search

    copy = settings.model_copy(update={"secrets": settings.secrets.model_copy(
        update={"database_url": SecretStr(_with_db(_url(settings), target))})})
    diffs = []
    for q, coll in queries:
        a = [h.chunk_id for h in search(settings, q, coll, k, embed_query=embed_query)]
        b = [h.chunk_id for h in search(copy, q, coll, k, embed_query=embed_query)]
        if a != b:
            diffs.append((q, coll, a, b))
    return diffs
