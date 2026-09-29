"""Embeddings with batching, a content cache and retries (LLD §3.7, spec §6.5).

Cache key: SHA-256 of (model, dims, text), stored one JSON line per vector in
data/cache/embeddings/{model}-{dims}.jsonl. Only cache misses are sent to OpenAI.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path
from typing import Callable

import openai

from mmrag.config import Settings
from mmrag.obs import get_logger, get_tracer

_log = get_logger("mmrag.index.embed")
ATTEMPTS = 3
_RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}


def _retryable(e: Exception) -> bool:
    if isinstance(e, (openai.APIConnectionError,)):  # includes timeouts
        return True
    return getattr(e, "status_code", None) in _RETRY_STATUS


class Embedder:
    def __init__(self, settings: Settings, client=None, cache_path: Path | None = None,
                 sleep: Callable[[float], None] = time.sleep):
        self.model, self.dims, self.batch = settings.embed.model, settings.embed.dims, settings.embed.batch_size
        if client is None:
            from mmrag.llm import get_client
            client = get_client(settings)
        self.client = client
        self.sleep = sleep
        self.cache_path = cache_path or (settings.resolve(settings.paths.data_dir) / "cache" / "embeddings"
                                         / f"{self.model}-{self.dims}.jsonl")
        self._cache: dict[str, list[float]] = {}
        if self.cache_path.is_file():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                if line:
                    row = json.loads(line)
                    self._cache[row["k"]] = row["v"]

    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self.model}\n{self.dims}\n{text}".encode("utf-8")).hexdigest()

    def _request(self, texts: list[str]) -> list[list[float]]:
        for attempt in range(1, ATTEMPTS + 1):
            try:
                resp = self.client.embeddings.create(model=self.model, input=texts, dimensions=self.dims)
                return [d.embedding for d in resp.data]
            except Exception as e:
                if attempt == ATTEMPTS or not _retryable(e):
                    raise
                wait = 2 ** attempt + random.random()  # exponential backoff with jitter
                _log.warning("embedding request failed (%s); retry %d in %.1fs", e, attempt, wait)
                self.sleep(wait)
        raise AssertionError("unreachable")

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Vectors in input order; each distinct uncached text is sent once."""
        keys = [self._key(t) for t in texts]
        missing = list(dict.fromkeys(k for k in keys if k not in self._cache))
        text_of = dict(zip(keys, texts))
        with get_tracer("mmrag.index").start_as_current_span("index.embed") as span:
            span.set_attribute("mmrag.embed.texts", len(texts))
            span.set_attribute("mmrag.embed.cache_hits", len(texts) - len(missing))
            for i in range(0, len(missing), self.batch):
                batch = missing[i:i + self.batch]
                vectors = self._request([text_of[k] for k in batch])
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                with self.cache_path.open("a", encoding="utf-8") as f:
                    for k, v in zip(batch, vectors):
                        self._cache[k] = v
                        f.write(json.dumps({"k": k, "v": v}) + "\n")
        return [self._cache[k] for k in keys]
