"""Embeddings: batching, cache, retries (LLD §3.7). A fake client; no network. Written test-first."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mmrag.index.embed import Embedder

pytestmark = pytest.mark.unit


class ApiError(Exception):
    def __init__(self, status_code):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class FakeClient:
    def __init__(self, fail_first=()):
        self.calls = []
        self.fail = list(fail_first)
        self.embeddings = SimpleNamespace(create=self._create)

    def _create(self, model, input, dimensions):
        if self.fail:
            raise self.fail.pop(0)
        self.calls.append((model, list(input), dimensions))
        return SimpleNamespace(data=[SimpleNamespace(embedding=[float(len(t))] * dimensions) for t in input])


@pytest.fixture
def embedder_for(parse_settings, tmp_path):
    def make(client):
        return Embedder(parse_settings, client=client, cache_path=tmp_path / "emb.jsonl", sleep=lambda s: None)
    return make


def test_vectors_have_the_configured_dims(embedder_for, parse_settings):
    client = FakeClient()
    vecs = embedder_for(client).embed(["a", "bb"])
    assert [len(v) for v in vecs] == [parse_settings.embed.dims] * 2
    assert vecs[1][0] == 2.0  # order preserved
    assert client.calls[0][0] == parse_settings.embed.model and client.calls[0][2] == parse_settings.embed.dims


def test_cached_texts_are_not_sent_again(embedder_for):
    first = FakeClient()
    embedder_for(first).embed(["same", "other"])
    second = FakeClient()
    vecs = embedder_for(second).embed(["same", "new", "same"])  # a fresh embedder reads the cache file
    assert second.calls == [(second.calls[0][0], ["new"], second.calls[0][2])]
    assert vecs[0] == vecs[2]


def test_requests_are_batched(embedder_for, parse_settings):
    client = FakeClient()
    n = parse_settings.embed.batch_size * 2 + 1
    embedder_for(client).embed([f"t{i}" for i in range(n)])
    assert [len(c[1]) for c in client.calls] == [parse_settings.embed.batch_size] * 2 + [1]


def test_rate_limits_and_server_errors_are_retried(embedder_for):
    client = FakeClient(fail_first=[ApiError(429), ApiError(503)])
    assert len(embedder_for(client).embed(["x"])) == 1


def test_bad_requests_are_not_retried(embedder_for):
    client = FakeClient(fail_first=[ApiError(400)])
    with pytest.raises(ApiError):
        embedder_for(client).embed(["x"])


def test_gives_up_after_three_attempts(embedder_for):
    client = FakeClient(fail_first=[ApiError(429)] * 3)
    with pytest.raises(ApiError):
        embedder_for(client).embed(["x"])
