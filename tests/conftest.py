"""Shared fixtures (LLD §10)."""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import psycopg
import pytest
import yaml
from dotenv import load_dotenv

from mmrag.config import PROJECT_ROOT, load_settings

load_dotenv(PROJECT_ROOT / ".env")

UNIT_ENV = {"DATABASE_URL": "postgresql://user:pw@localhost:5433/unit"}


@pytest.fixture
def base_config() -> dict:
    """The project's config.yaml as a dict, for tests to modify."""
    return yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text(encoding="utf-8"))


@pytest.fixture
def write_config(tmp_path):
    """Write a config dict to a temporary YAML file and return its path."""

    def _write(cfg: dict) -> Path:
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        return path

    return _write


@pytest.fixture
def parse_settings(base_config, write_config, tmp_path):
    """Project settings with outputs redirected to a temporary data dir and telemetry off."""
    base_config["paths"]["data_dir"] = str(tmp_path / "data")
    base_config["observability"].update(enabled=False, log_file=None)
    return load_settings(write_config(base_config), UNIT_ENV)


def _url_with_db(url: str, dbname: str) -> str:
    return urlunsplit(urlsplit(url)._replace(path="/" + dbname))


@pytest.fixture
def fresh_db_url():
    """A brand-new, empty test database; dropped after the test."""
    admin_url = os.environ.get("DATABASE_URL")
    if not admin_url:
        pytest.fail("DATABASE_URL not set; integration tests need .env")
    name = f"mmrag_test_{uuid.uuid4().hex[:8]}"
    try:
        with psycopg.connect(admin_url, autocommit=True, connect_timeout=5) as conn:
            conn.execute(f'CREATE DATABASE "{name}"')
    except psycopg.OperationalError as e:
        pytest.fail(f"PostgreSQL not reachable; start it with `docker compose up -d` ({e})")
    yield _url_with_db(admin_url, name)
    with psycopg.connect(admin_url, autocommit=True) as conn:
        conn.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
