"""Identifiers and content hashes (LLD §3.1). Deterministic: built test-first (W5)."""

from __future__ import annotations

import hashlib
from pathlib import Path

from mmrag.ingest.models import ElementType


def doc_id(pdf_path: Path) -> str:
    """First 16 hex characters of SHA-256 of the file bytes."""
    h = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def element_id(doc: str, page: int, etype: ElementType, n: int) -> str:
    """`{doc_id}:p{page}:{type}:{n}`, where `n` is the element's order on the page (1-based)."""
    return f"{doc}:p{page}:{etype}:{n}"


def content_hash(content: str | bytes) -> str:
    """Full SHA-256 hex digest of element text (UTF-8) or image bytes; the cache key."""
    if isinstance(content, str):
        content = content.encode("utf-8")
    return hashlib.sha256(content).hexdigest()
