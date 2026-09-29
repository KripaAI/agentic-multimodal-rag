"""Waiting for a freshly uploaded Kaggle dataset (no network: the CLI call is faked). Test-first."""

from __future__ import annotations

from types import SimpleNamespace as NS

import pytest

from mmrag.ingest import kaggle_job

pytestmark = pytest.mark.unit


def test_a_new_dataset_answers_403_for_a_moment_then_ready(monkeypatch):
    replies = iter([RuntimeError("403 Client Error: Forbidden"), NS(stdout="running"), NS(stdout="ready")])

    def fake(*args):
        r = next(replies)
        if isinstance(r, Exception):
            raise r
        return r

    monkeypatch.setattr(kaggle_job, "_kaggle", fake)
    monkeypatch.setattr("time.sleep", lambda s: None)
    kaggle_job._wait_until_ready("u/d")  # no exception
    with pytest.raises(StopIteration):
        next(replies)  # all three replies were used


def test_other_errors_are_not_swallowed(monkeypatch):
    def fake(*args):
        raise RuntimeError("401 Client Error: Unauthorized")

    monkeypatch.setattr(kaggle_job, "_kaggle", fake)
    monkeypatch.setattr("time.sleep", lambda s: None)
    with pytest.raises(RuntimeError, match="401"):
        kaggle_job._wait_until_ready("u/d")


def test_an_error_printed_with_exit_code_0_is_still_an_error(monkeypatch):
    # `kaggle datasets create` prints "Dataset creation error: ..." and exits 0.
    out = NS(returncode=0, stdout="Upload successful\nDataset creation error: title already in use", stderr="")
    monkeypatch.setattr(kaggle_job.subprocess, "run", lambda *a, **k: out)
    with pytest.raises(RuntimeError, match="creation error"):
        kaggle_job._kaggle("datasets", "create", "-p", "x")


def test_bundle_datasets_use_the_mmrag_bundle_address():
    # mmrag-caption-<doc_id> addresses can get stuck on Kaggle after a failed create (Phase 5).
    meta = kaggle_job.dataset_metadata("someone", "abc")
    assert meta["id"] == "someone/mmrag-bundle-abc" and meta["title"] == "mmrag bundle abc"


def test_the_cli_runs_in_utf8_mode(monkeypatch):
    # On Windows the CLI crashed printing the job log ('charmap' codec) after downloading.
    seen = {}

    def run(cmd, **kw):
        seen.update(kw)
        return NS(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(kaggle_job.subprocess, "run", run)
    kaggle_job._kaggle("kernels", "output", "u/k")
    assert seen["env"]["PYTHONIOENCODING"] == "utf-8" and seen["env"]["PYTHONUTF8"] == "1"
