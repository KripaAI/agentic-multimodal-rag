"""`mmrag memory` and `mmrag user delete` as real subprocesses (plan Phase 9 tasks 6, 9).

These are the controls FR-25 promises, so they are tested the way a user meets them: through
the command line, with a real database and no API key — seeing and deleting what is
remembered must never depend on reaching OpenAI.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

import mmrag.db as db
from mmrag.auth import service
from mmrag.config import load_settings

pytestmark = pytest.mark.integration


def _cli(fresh_db_url, base_config, write_config, api_key=""):
    base_config["observability"].update(enabled=False, log_file=None)
    config_file = write_config(base_config)
    settings = load_settings(config_file, {"DATABASE_URL": fresh_db_url})
    db.migrate(settings)
    service.add_user(settings, "a@example.com")
    env = {**os.environ, "MMRAG_CONFIG": str(config_file), "DATABASE_URL": fresh_db_url, "OPENAI_API_KEY": api_key}

    def run(*args):
        return subprocess.run([sys.executable, "-m", "mmrag.cli", *args],
                              capture_output=True, text=True, encoding="utf-8", env=env, timeout=300)

    return run, settings


@pytest.fixture
def cli(fresh_db_url, base_config, write_config):
    """No API key: seeing and deleting memories must work without one."""
    return _cli(fresh_db_url, base_config, write_config)


@pytest.fixture
def cli_with_key(fresh_db_url, base_config, write_config):
    """Writing a memory embeds it, so this one needs a real key (one embedding call, ~0.00001 US$)."""
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        pytest.skip("OPENAI_API_KEY not set; storing a memory needs the embedder")
    return _cli(fresh_db_url, base_config, write_config, api_key=key)


def test_listing_memories_needs_no_api_key(cli):
    run, _ = cli
    r = run("memory", "list", "a@example.com")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "0 memories for a@example.com" in r.stdout and "memory is on" in r.stdout


def test_switching_memory_off_and_on_again(cli):
    run, settings = cli
    assert run("memory", "off", "a@example.com").returncode == 0
    assert "0 memories for a@example.com (memory is OFF" in run("memory", "list", "a@example.com").stdout
    run("memory", "on", "a@example.com")
    with db.connect(settings) as conn:
        assert conn.execute("SELECT memory_enabled FROM users").fetchone()[0] is True


def test_an_unknown_account_is_an_error_not_an_empty_list(cli):
    run, _ = cli
    r = run("memory", "list", "nobody@example.com")
    assert r.returncode == 2 and "No account" in r.stderr


def test_forgetting_everything_asks_for_confirmation_first(cli):
    run, _ = cli
    assert run("memory", "forget-all", "a@example.com").returncode == 1
    r = run("memory", "forget-all", "a@example.com", "--yes")
    assert r.returncode == 0 and "Deleted 0 memories" in r.stdout


def test_deleting_an_unknown_memory_says_so(cli):
    run, _ = cli
    r = run("memory", "delete", "semantic", "nothing", "a@example.com")
    assert r.returncode == 2 and "No semantic memory" in r.stderr


def test_deleting_an_account_asks_for_confirmation_first(cli):
    run, settings = cli
    assert run("user", "delete", "a@example.com").returncode == 1
    with db.connect(settings) as conn:
        assert conn.execute("SELECT count(*) FROM users").fetchone()[0] == 1
    r = run("user", "delete", "a@example.com", "--yes")
    assert r.returncode == 0 and "Deleted a@example.com" in r.stdout
    with db.connect(settings) as conn:
        assert conn.execute("SELECT count(*) FROM users").fetchone()[0] == 0


# ---------------------------------------------------------------- eval run --as-user (Phase 9 task 8)

def test_an_evaluation_with_memory_on_refuses_to_be_the_baseline(cli):
    run, _ = cli
    r = run("eval", "run", "--as-user", "a@example.com", "--baseline")
    assert r.returncode != 0 and "baseline measures the agent without memory" in r.stdout + r.stderr


def test_an_evaluation_with_memory_on_refuses_an_unknown_or_switched_off_account(cli):
    run, _ = cli
    assert "no account for nobody@example.com" in run("eval", "run", "--as-user", "nobody@example.com").stderr
    run("memory", "off", "a@example.com")
    assert "memory is switched off" in run("eval", "run", "--as-user", "a@example.com").stderr


def test_an_empty_memory_would_measure_the_baseline_so_it_is_refused(cli):
    run, _ = cli
    r = run("eval", "run", "--as-user", "a@example.com", "--limit", "1")
    assert r.returncode != 0 and "remembers nothing yet" in r.stderr
    assert "--warmup" in r.stderr  # ...and says how to fill it


def test_a_seeded_note_makes_the_comparison_runnable(cli_with_key):
    """`memory add` exists so the evaluation account can be given the notes real use produces."""
    run, _ = cli_with_key
    r = run("memory", "add", "output_format", "Prefers charts to tables.", "a@example.com")
    assert r.returncode == 0 and "output_format" in r.stdout
    assert "Prefers charts to tables." in run("memory", "list", "a@example.com").stdout
    assert "1 memories" in run("memory", "list", "a@example.com").stdout
