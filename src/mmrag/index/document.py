"""One document's Phase 1–2 outputs, loaded from data/ for enrichment and indexing."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from mmrag.config import Settings
from mmrag.ingest.captions import CaptionRecord, load_records
from mmrag.ingest.ids import doc_id as compute_doc_id
from mmrag.ingest.models import Element, Table


class Link(BaseModel):
    """A paragraph that discusses a figure or table (spec §6.3)."""

    model_config = ConfigDict(extra="forbid")

    target_id: str  # the figure or table
    text_element_id: str
    method: Literal["explicit", "deictic", "related"]
    score: float


@dataclass
class LoadedDoc:
    doc_id: str
    source_file: str
    content_hash: str  # full SHA-256 of the PDF
    elements: list[Element]  # reading order
    tables: dict[str, Table] = field(default_factory=dict)
    captions: dict[str, CaptionRecord] = field(default_factory=dict)

    def texts(self) -> list[Element]:
        return [e for e in self.elements if e.type == "text" and e.status == "ok" and e.text]

    def figures(self) -> list[Element]:
        return [e for e in self.elements if e.type in ("vector_figure", "image", "scanned_page") and e.status == "ok"]

    def table_elements(self) -> list[Element]:
        return [e for e in self.elements if e.type == "table" and e.element_id in self.tables]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_document(pdf: Path, settings: Settings) -> LoadedDoc:
    """Read elements, tables and captions written by `ingest parse` and `caption import`."""
    did = compute_doc_id(pdf)
    data = settings.resolve(settings.paths.data_dir)
    el_file = data / "elements" / did / "elements.jsonl"
    if not el_file.is_file():
        raise FileNotFoundError(f"no parse output for {pdf.name}; run `mmrag ingest parse {pdf.name}` first")
    elements = [Element.model_validate_json(x) for x in el_file.read_text(encoding="utf-8").splitlines() if x]
    elements.sort(key=lambda e: (e.page, e.bbox[1], e.bbox[0]))
    tab_file = data / "tables" / did / "tables.jsonl"
    tables = [Table.model_validate_json(x) for x in tab_file.read_text(encoding="utf-8").splitlines() if x] \
        if tab_file.is_file() else []
    cap_file = data / "captions" / did / "captions.jsonl"
    captions = load_records(cap_file)[0] if cap_file.is_file() else []
    return LoadedDoc(doc_id=did, source_file=pdf.name, content_hash=_sha256(pdf), elements=elements,
                     tables={t.element_id: t for t in tables}, captions={c.element_id: c for c in captions})
