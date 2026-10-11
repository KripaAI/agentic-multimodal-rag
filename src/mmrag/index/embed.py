"""Embeddings with batching, a content cache and retries (LLD §3.7, spec §6.5).

Two providers (`embed.provider`): the OpenAI embeddings API, or a local sentence-transformers model
(Phase 10: EmbeddingGemma 2). Documents and queries must always be embedded by the same model, so
both go through this class.

`kind` says what a text is for. OpenAI embeds every kind the same way. EmbeddingGemma expects a
task prompt in front of the text (model card), so a local model gets `LOCAL_PREFIXES[kind]`.

Cache key: SHA-256 of (model, dims, text as sent), stored one JSON line per vector in
data/cache/embeddings/{model}-{dims}.jsonl ("/" in the model name becomes "--"). Only cache misses
are embedded.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from functools import lru_cache
from pathlib import Path
from typing import Callable, Literal

import openai

from mmrag.fsutil import jsonl_lines
from mmrag.config import Settings
from mmrag.obs import get_logger, get_tracer

_log = get_logger("mmrag.index.embed")
ATTEMPTS = 3
_RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}

Kind = Literal["query", "document", "similarity"]
# EmbeddingGemma task prompts, from the model card. Text only: images take no prompt.
LOCAL_PREFIXES: dict[str, str] = {
    "query": "task: search result | query: ",     # a search query
    "document": "title: none | text: ",            # a passage to be found
    "similarity": "task: sentence similarity | query: ",  # symmetric: both sides use the same prompt
}


def _retryable(e: Exception) -> bool:
    if isinstance(e, (openai.APIConnectionError,)):  # includes timeouts
        return True
    return getattr(e, "status_code", None) in _RETRY_STATUS


@lru_cache(maxsize=2)
def load_local_model(model: str, dims: int):
    """A sentence-transformers model, text tower only. Loaded on first use and kept for the process:
    loading takes tens of seconds, and indexing a library builds one Embedder per document."""
    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        if "Application Control policy" in str(e):
            raise  # installed but blocked by Windows Smart App Control: the CLI explains that case
        raise RuntimeError("embed.provider is 'local': install requirements-local-embed.txt") from e
    # The model card: bfloat16 or float32, never float16. CPUs get float32.
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    _log.info("loading local embedding model %s (%d dims, %s)", model, dims, dtype)
    return SentenceTransformer(model, truncate_dim=dims, model_kwargs={"torch_dtype": dtype},
                               config_kwargs={"vision_config": None, "audio_config": None})


class Embedder:
    def __init__(self, settings: Settings, client=None, cache_path: Path | None = None,
                 sleep: Callable[[float], None] = time.sleep, local_model=None):
        """`client` (OpenAI) and `local_model` (anything with sentence-transformers' `encode`) are
        for tests; by default they are built from settings, the local model only on first use."""
        self.provider = settings.embed.provider
        self.model, self.dims, self.batch = settings.embed.model, settings.embed.dims, settings.embed.batch_size
        if client is None and self.provider == "openai":
            from mmrag.llm import get_client
            client = get_client(settings)
        self.client = client
        self._local = local_model
        self.sleep = sleep
        self.cache_path = cache_path or (settings.resolve(settings.paths.data_dir) / "cache" / "embeddings"
                                         / f"{self.model.replace('/', '--')}-{self.dims}.jsonl")
        self._cache: dict[str, list[float]] = {}
        if self.cache_path.is_file():
            damaged = 0
            for line in jsonl_lines(self.cache_path):
                if line:
                    try:
                        row = json.loads(line)
                        self._cache[row["k"]] = row["v"]
                    except (ValueError, KeyError, TypeError):
                        damaged += 1  # e.g. two processes appended at once; the text is simply re-embedded
            if damaged:
                _log.warning("embedding cache: skipped %d damaged line(s) in %s", damaged, self.cache_path)

    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self.model}\n{self.dims}\n{text}".encode("utf-8")).hexdigest()

    def _prepare(self, text: str, kind: Kind) -> str:
        return LOCAL_PREFIXES[kind] + text if self.provider == "local" else text

    def _local_model(self):
        if self._local is None:
            self._local = load_local_model(self.model, self.dims)
        return self._local

    def _encode_local(self, texts: list[str]) -> list[list[float]]:
        vectors = self._local_model().encode(texts, batch_size=self.batch, normalize_embeddings=True,
                                             convert_to_numpy=True)
        out = [[float(x) for x in v] for v in vectors]
        if out and len(out[0]) != self.dims:
            raise RuntimeError(f"{self.model} returned {len(out[0])} dims, config says {self.dims}")
        return out

    def _request(self, texts: list[str]) -> list[list[float]]:
        if self.provider == "local":
            return self._encode_local(texts)
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

    def embed(self, texts: list[str], kind: Kind = "document") -> list[list[float]]:
        """Vectors in input order; each distinct uncached text is sent once."""
        texts = [self._prepare(t, kind) for t in texts]
        keys = [self._key(t) for t in texts]
        missing = list(dict.fromkeys(k for k in keys if k not in self._cache))
        text_of = dict(zip(keys, texts))
        with get_tracer("mmrag.index").start_as_current_span("index.embed") as span:
            span.set_attribute("mmrag.embed.provider", self.provider)
            span.set_attribute("mmrag.embed.texts", len(texts))
            span.set_attribute("mmrag.embed.cache_hits", len(texts) - len(missing))
            for i in range(0, len(missing), self.batch):
                batch = missing[i:i + self.batch]
                vectors = self._request([text_of[k] for k in batch])
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                for k, v in zip(batch, vectors):
                    self._cache[k] = v
                with self.cache_path.open("a", encoding="utf-8") as f:  # one write per batch: less interleaving
                    f.write("".join(json.dumps({"k": k, "v": v}) + "\n" for k, v in zip(batch, vectors)))
        return [self._cache[k] for k in keys]
