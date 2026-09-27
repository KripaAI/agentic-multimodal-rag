"""`mmrag check` exit codes and output (LLD §3.9), run as a real subprocess."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

import mmrag.db as db
from mmrag.config import PROJECT_ROOT, load_settings

pytestmark = pytest.mark.integration


@pytest.fixture
def check_env(base_config, write_config):
    """Environment for running the CLI against a test config (telemetry off)."""
    base_config["observability"].update(enabled=False, log_file=None)
    env = {**os.environ, "MMRAG_CONFIG": str(write_config(base_config)), "OPENAI_API_KEY": ""}
    return env


def _run_check(env):
    return subprocess.run([sys.executable, "-m", "mmrag.cli", "check"],
                          capture_output=True, text=True, env=env, timeout=120)


def test_check_passes_on_healthy_database(check_env, fresh_db_url):
    db.migrate(load_settings(PROJECT_ROOT / "config.yaml", {"DATABASE_URL": fresh_db_url}))
    r = _run_check({**check_env, "DATABASE_URL": fresh_db_url})
    assert r.returncode == 0, r.stdout + r.stderr
    assert "database  OK" in r.stdout
    assert "openai    SKIP" in r.stdout
    assert "telemetry     disabled" in r.stdout


def test_check_fails_when_pgvector_missing(check_env, fresh_db_url):
    r = _run_check({**check_env, "DATABASE_URL": fresh_db_url})  # not migrated
    assert r.returncode == 1
    assert "database  FAIL" in r.stdout and "mmrag db migrate" in r.stdout


def test_check_fails_when_database_unreachable(check_env):
    r = _run_check({**check_env, "DATABASE_URL": "postgresql://u:p@127.0.0.1:1/none"})
    assert r.returncode == 1
    assert "database  FAIL" in r.stdout


def test_invalid_config_exits_2(check_env, base_config, write_config):
    base_config["embed"]["dims"] = 99999
    r = _run_check({**check_env, "MMRAG_CONFIG": str(write_config(base_config))})
    assert r.returncode == 2
    assert "Configuration error" in r.stderr
