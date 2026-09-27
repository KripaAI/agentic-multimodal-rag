"""`mmrag profile` and `mmrag ingest parse` (plan Phase 1), run as real subprocesses. No services needed."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

import pytest

from mmrag.config import PROJECT_ROOT

pytestmark = pytest.mark.unit

FIXTURE = PROJECT_ROOT / "tests" / "fixtures" / "pages" / "transformers_p003.pdf"


@pytest.fixture
def cli_env(base_config, write_config, tmp_path):
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    shutil.copy(FIXTURE, pdf_dir / "sample.pdf")
    base_config["paths"].update(pdf_dir=str(pdf_dir), data_dir=str(tmp_path / "data"))
    base_config["observability"].update(enabled=False, log_file=None)
    env = {**os.environ, "MMRAG_CONFIG": str(write_config(base_config)), "OPENAI_API_KEY": ""}
    return env, tmp_path / "data"


def _run(env, *args):
    return subprocess.run([sys.executable, "-m", "mmrag.cli", *args],
                          capture_output=True, text=True, encoding="utf-8", env=env, timeout=300)


def test_ingest_parse_writes_outputs(cli_env):
    env, data = cli_env
    r = _run(env, "ingest", "parse", "sample.pdf")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "vector_figure" in r.stdout and "review_sheet" in r.stdout
    (doc_dir,) = (data / "elements").iterdir()
    assert (doc_dir / "elements.jsonl").is_file()
    assert (doc_dir / "review_sheet.html").is_file()


def test_ingest_parse_unknown_file_fails(cli_env):
    env, _ = cli_env
    r = _run(env, "ingest", "parse", "missing.pdf")
    assert r.returncode == 1
    assert "not found" in r.stdout + r.stderr


def test_profile_writes_report(cli_env):
    env, data = cli_env
    r = _run(env, "profile")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "sample.pdf" in r.stdout
    report = json.loads((data / "elements" / "corpus_profile.json").read_text(encoding="utf-8"))
    assert report[0]["source_file"] == "sample.pdf"
    assert len(report[0]["pages"]) == 1
