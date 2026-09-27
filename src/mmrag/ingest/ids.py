"""Identifiers and content hashes (LLD §3.1). Deterministic: built test-first (W5)."""

from __future__ import annotations

from pathlib import Path

from mmrag.ingest.models import ElementType


def doc_id(pdf_path: Path) -> str:
    """First 16 hex characters of SHA-256 of the file bytes."""
    raise NotImplementedError


def element_id(doc: str, page: int, etype: ElementType, n: int) -> str:
    """`{doc_id}:p{page}:{type}:{n}`, where `n` is the element's order on the page (1-based)."""
    raise NotImplementedError


def content_hash(content: str | bytes) -> str:
    """Full SHA-256 hex digest of element text (UTF-8) or image bytes; the cache key."""
    raise NotImplementedError
