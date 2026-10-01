"""The PDF library, one document at a time (plan Phase 8 task 1).

- add: copy a new PDF into `paths.pdf_dir`; it is then parsed, captioned and indexed on its own.
- replace: archive the old file and put the new one under the same name. Search keeps serving
  the old version until `ingest index` writes the new one, which swaps it in one transaction
  (writer.write_document).
- remove: delete the document from the database (cascade: elements, captions, tables, links,
  search chunks) and move its PDF out of the library. Nothing else is touched.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from mmrag import db
from mmrag.config import Settings


@dataclass
class DocInfo:
    source_file: str
    doc_id: str
    version: int
    ingested_at: datetime
    text: int
    figure: int
    table: int


def _library(settings: Settings) -> Path:
    return settings.resolve(settings.paths.pdf_dir)


def list_documents(settings: Settings) -> list[DocInfo]:
    with db.connect(settings) as conn:
        rows = conn.execute(
            "SELECT d.source_file, d.doc_id, d.version, d.ingested_at, "
            "count(*) FILTER (WHERE c.collection = 'text'), count(*) FILTER (WHERE c.collection = 'figure'), "
            "count(*) FILTER (WHERE c.collection = 'table') "
            "FROM documents d LEFT JOIN search_chunks c USING (doc_id) GROUP BY 1, 2, 3, 4 ORDER BY 1").fetchall()
    return [DocInfo(*r) for r in rows]


def add_pdf(settings: Settings, source: Path) -> Path:
    dest = _library(settings) / source.name
    if dest.exists():
        raise FileExistsError(f"{source.name} is already in the library; use `mmrag doc replace` for a new version")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
    return dest


def replace_pdf(settings: Settings, name: str, new: Path) -> Path:
    lib = _library(settings)
    current = lib / name
    if not current.is_file():
        raise FileNotFoundError(f"{name} is not in the library ({lib})")
    archive = lib / "replaced" / f"{current.stem}.{datetime.now():%Y%m%d-%H%M%S}{current.suffix}"
    archive.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(current), archive)
    shutil.copy2(new, current)
    return current


def remove_document(settings: Settings, name: str) -> None:
    with db.connect(settings) as conn:
        deleted = conn.execute("DELETE FROM documents WHERE source_file = %s RETURNING doc_id", (name,)).fetchall()
        conn.commit()
    pdf = _library(settings) / name
    if not deleted and not pdf.exists():
        raise ValueError(f"{name} is not in the library")
    if pdf.exists():
        target = _library(settings) / "removed" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(pdf), target)
