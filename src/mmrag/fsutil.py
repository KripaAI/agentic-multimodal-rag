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


def jsonl_lines(path: Path) -> list[str]:
    """The non-blank lines of a JSONL file. Split on "\\n" only: str.splitlines() also splits on
    characters such as \\x85 and \\u2028, which JSON keeps unescaped inside strings, and so cut a
    record in two (a PDF whose text contained \\x85 could not be indexed)."""
    return [line for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]
