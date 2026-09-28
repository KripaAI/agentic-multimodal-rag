"""File-system helpers."""

from __future__ import annotations

from pathlib import Path


def empty_dir(path: Path) -> Path:
    """Create `path`, or delete every file below it while keeping the folders.

    Deleting and re-creating a folder fails on Windows while OneDrive or an indexer
    holds it open ("Access is denied"); deleting its files does not.
    """
    path.mkdir(parents=True, exist_ok=True)
    for item in sorted(path.rglob("*"), reverse=True):
        if item.is_file() or item.is_symlink():
            item.unlink()
    return path
