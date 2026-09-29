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
